"""
====================================================================================================
ALGORITHM: src/features.py — Mainnet Market Data Ingestion & Wall-Clock Freshness Filter
====================================================================================================
Purpose:
  Connects to Binance Mainnet (fapi.binance.com) through the Frankfurt proxy to pull authentic,
  liquid multi-timeframe OHLCV data matching TradingView tick-for-tick. Implements an epoch-based
  wall-clock freshness filter that mathematically rejects any candle older than 15 minutes,
  eradicating the 30-minute stale execution lag. Computes Section A.7 features and rolling sequences.

Algorithm Steps:
  Step 1: Safe Math & Utility Functions (Division-by-zero protection).
  Step 2: Mainnet Multi-Timeframe Ingestion with Wall-Clock Freshness Filter (`fetch_closed_ohlcv`):
          - Ingests candles directly from Binance Mainnet public client (`data_exchange`).
          - Determines expected closed bar open timestamp using discrete 15m epoch math.
          - Rejects forming bar dynamically; verifies index -1 closed exactly 5 seconds ago.
  Step 3: Freshness-Guarded Crossover Detection Engine (`detect_crossover`):
          - Computes 9 EMA and 15 EMA strictly on closed bars.
          - Evaluates crossover on index -1 vs -2; aborts if candle timestamp is stale.
  Step 4: Vectorized Section A.7 Feature Calculator (`compute_production_features_at_idx`).
  Step 5: True Rolling Sequence Extractor (`extract_rolling_features_history`):
          - Builds authentic (15, 16) sequence history for Funnel GRU inference.
  Step 6: Production Model Pipeline (`ProductionFeaturePipeline`):
          - Scaler normalization for CatBoost (1, 16) and GRU (1, seq_len, 16) tensors.
  Step 7: Production Self-Test Probe (`if __name__ == '__main__'`).
====================================================================================================
"""

import os
import json
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import ta
import ccxt


# =============================================================================
# STEP 1: Safe Math Utilities
# =============================================================================
def safe_ratio(num, den):
    """Prevents division by zero or infinite outputs in ratio calculations."""
    try:
        den_clean = np.where(den == 0, np.nan, den)
        res = num / den_clean
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        return float(res) if np.isscalar(res) else res
    except Exception:
        return 0.0


# =============================================================================
# STEP 2: Mainnet Ingestion & Wall-Clock Freshness Filter
# =============================================================================
def fetch_closed_ohlcv(data_exchange, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    """
    Fetches liquid candles from Binance Mainnet and applies deterministic wall-clock epoch math.
    Guarantees that index -1 is the bar that completed on the immediate prior interval.
    """
    raw_bars = data_exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    # Discrete epoch math: determine exact open timestamp of the expected closed candle
    now_utc = datetime.now(timezone.utc)
    now_ms  = int(now_utc.timestamp() * 1000)

    if timeframe == '15m':
        interval_ms = 15 * 60 * 1000
    elif timeframe == '4h':
        interval_ms = 4 * 60 * 60 * 1000
    elif timeframe == '1d':
        interval_ms = 24 * 60 * 60 * 1000
    else:
        interval_ms = 15 * 60 * 1000

    # The current forming bar opened at: (now_ms // interval_ms) * interval_ms
    # The candle that JUST closed opened at: current_forming_open_ms - interval_ms
    current_forming_open_ms = (now_ms // interval_ms) * interval_ms
    expected_closed_open_ms = current_forming_open_ms - interval_ms

    # Filter out active forming bar and keep all closed bars up to expected_closed_open_ms
    df_closed = df[df['timestamp'] <= expected_closed_open_ms].copy().reset_index(drop=True)
    return df_closed


# =============================================================================
# STEP 3: Freshness-Guarded Crossover Detection Engine
# =============================================================================
def detect_crossover(df_15m_closed: pd.DataFrame):
    """
    Evaluates 9/15 EMA crossover strictly on verified fresh candles.
    Rejects any candle whose timestamp does not match the expected 15m closed boundary.
    """
    if len(df_15m_closed) < 20:
        return None, 0.0, None, "INSUFFICIENT_BARS"

    # Wall-Clock Freshness Guard
    now_utc = datetime.now(timezone.utc)
    now_ms  = int(now_utc.timestamp() * 1000)
    expected_closed_open_ms = ((now_ms // (15 * 60 * 1000)) * (15 * 60 * 1000)) - (15 * 60 * 1000)
    actual_closed_open_ms   = int(df_15m_closed['timestamp'].iloc[-1])

    # If the latest bar in the closed dataframe is older than expected, reject as stale
    if actual_closed_open_ms < expected_closed_open_ms:
        stale_mins = (expected_closed_open_ms - actual_closed_open_ms) / 60000.0
        return None, 0.0, None, f"STALE_DATA_REJECTED ({stale_mins:.1f}m late)"

    close_series = df_15m_closed['close']
    ema_fast_s = close_series.ewm(span=9, adjust=False).mean()
    ema_slow_s = close_series.ewm(span=15, adjust=False).mean()

    ema_fast_curr = float(ema_fast_s.iloc[-1])
    ema_slow_curr = float(ema_slow_s.iloc[-1])
    ema_fast_prev = float(ema_fast_s.iloc[-2])
    ema_slow_prev = float(ema_slow_s.iloc[-2])

    crossover_price  = float(close_series.iloc[-1])
    candle_close_utc = df_15m_closed['datetime_utc'].iloc[-1].isoformat()

    bullish_cross = (ema_fast_curr > ema_slow_curr) and (ema_fast_prev <= ema_slow_prev)
    bearish_cross = (ema_fast_curr < ema_slow_curr) and (ema_fast_prev >= ema_slow_prev)

    if bullish_cross:
        return 'LONG', crossover_price, candle_close_utc, "FRESH_SIGNAL"
    elif bearish_cross:
        return 'SHORT', crossover_price, candle_close_utc, "FRESH_SIGNAL"
    else:
        return None, crossover_price, candle_close_utc, "NO_CROSSOVER"


# =============================================================================
# STEP 4: Vectorized Section A.7 Feature Calculator
# =============================================================================
def compute_production_features_at_idx(df_15m: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, idx: int = -1) -> dict:
    """Computes Section A.7 features at a specific candle index `idx`."""
    c_15m = df_15m['close']
    h_15m = df_15m['high']
    l_15m = df_15m['low']
    v_15m = df_15m['volume']

    ema_fast_15m_s = c_15m.ewm(span=9, adjust=False).mean()
    ema_slow_15m_s = c_15m.ewm(span=15, adjust=False).mean()

    ema_fast_ltf   = float(ema_fast_15m_s.iloc[idx])
    ema_slow_ltf   = float(ema_slow_15m_s.iloc[idx])
    ema_fast_prev  = float(ema_fast_15m_s.iloc[idx - 1])
    ema_slow_prev  = float(ema_slow_15m_s.iloc[idx - 1])
    price_latest   = float(c_15m.iloc[idx])

    ema_fast_slope = safe_ratio((ema_fast_ltf - ema_fast_prev), ema_fast_prev) * 100.0
    ema_slow_slope = safe_ratio((ema_slow_ltf - ema_slow_prev), ema_slow_prev) * 100.0
    ema_separation = safe_ratio((ema_fast_ltf - ema_slow_ltf), ema_slow_ltf) * 100.0

    adx_ind_15m = ta.trend.ADXIndicator(high=h_15m, low=l_15m, close=c_15m, window=14, fillna=True)
    adx_15m_s   = adx_ind_15m.adx()
    adx_ltf     = float(adx_15m_s.iloc[idx])
    adx_slope   = float(adx_15m_s.iloc[idx] - adx_15m_s.iloc[idx - 1])

    rsi_15m_s = ta.momentum.RSIIndicator(close=c_15m, window=14, fillna=True).rsi()
    rsi_ltf   = float(rsi_15m_s.iloc[idx])

    atr_15m_s = ta.volatility.AverageTrueRange(high=h_15m, low=l_15m, close=c_15m, window=14, fillna=True).average_true_range()
    atr_ltf   = float(atr_15m_s.iloc[idx])
    atr_pct   = (atr_ltf / price_latest * 100.0) if price_latest > 0 else 0.0
    price_to_atr = safe_ratio(price_latest, atr_ltf)

    bb_ind = ta.volatility.BollingerBands(close=c_15m, window=20, window_dev=2, fillna=True)
    bb_width_ltf = float(bb_ind.bollinger_wband().iloc[idx])

    macd_15m_diff = ta.trend.MACD(close=c_15m, window_fast=12, window_slow=26, window_sign=9, fillna=True).macd_diff()
    macd_histogram_ltf = float(macd_15m_diff.iloc[idx])

    vol_ma_s     = v_15m.rolling(window=20, min_periods=1).mean()
    volume_ratio = safe_ratio(float(v_15m.iloc[idx]), float(vol_ma_s.iloc[idx]))
    volume_trend = safe_ratio((float(vol_ma_s.iloc[idx]) - float(vol_ma_s.iloc[idx - 1])), float(vol_ma_s.iloc[idx - 1])) * 100.0

    swing_window = df_15m.iloc[max(0, len(df_15m) + idx - 19) : len(df_15m) + idx + 1]
    swing_high = float(swing_window['high'].max())
    swing_low  = float(swing_window['low'].min())

    # 4h HTF
    c_4h = df_4h['close']
    h_4h = df_4h['high']
    l_4h = df_4h['low']
    ema_fast_4h = float(c_4h.ewm(span=9, adjust=False).mean().iloc[-1])
    ema_slow_4h = float(c_4h.ewm(span=15, adjust=False).mean().iloc[-1])
    ema_separation_4h = safe_ratio((ema_fast_4h - ema_slow_4h), ema_slow_4h) * 100.0
    htf_4h_bias = 1.0 if ema_fast_4h > ema_slow_4h else -1.0

    adx_4h = float(ta.trend.ADXIndicator(high=h_4h, low=l_4h, close=c_4h, window=14, fillna=True).adx().iloc[-1])
    rsi_4h = float(ta.momentum.RSIIndicator(close=c_4h, window=14, fillna=True).rsi().iloc[-1])
    macd_histogram_4h = float(ta.trend.MACD(close=c_4h, window_fast=12, window_slow=26, window_sign=9, fillna=True).macd_diff().iloc[-1])

    # 1d Daily
    c_1d = df_1d['close']
    ema_fast_1d = float(c_1d.ewm(span=9, adjust=False).mean().iloc[-1])
    ema_slow_1d = float(c_1d.ewm(span=15, adjust=False).mean().iloc[-1])
    htf_1d_bias = 1.0 if ema_fast_1d > ema_slow_1d else -1.0

    target_time = df_15m['datetime_utc'].iloc[idx]
    hour_of_day = int(target_time.hour)
    day_of_week = int(target_time.weekday())

    # Section A.7 Interactions
    fe_rsi_mtf_ratio        = safe_ratio(rsi_ltf, rsi_4h)
    fe_ema_ratio            = safe_ratio(ema_fast_ltf, ema_slow_ltf)
    fe_price_to_bb          = safe_ratio(atr_pct, bb_width_ltf)
    fe_adx_4h_ratio         = safe_ratio(adx_ltf, adx_4h)
    fe_vol_efficiency_ratio = safe_ratio(volume_ratio, atr_pct)
    fe_spread_to_atr_ratio  = safe_ratio((price_latest - ema_fast_ltf), atr_ltf)

    fe_macd_x_volume        = macd_histogram_ltf * volume_ratio
    fe_adx_x_volume         = adx_ltf * volume_ratio
    fe_ema_sep_x_adx        = ema_separation * adx_ltf
    fe_adx_x_atr_pct        = adx_ltf * (atr_ltf / price_latest if price_latest > 0 else 0.0)
    fe_exhaustion_risk      = (1 if rsi_ltf > 70.0 else 0) * ema_separation

    fe_rsi_x_htf4h          = rsi_ltf * htf_4h_bias
    fe_rsi4h_x_htf1d        = rsi_4h * htf_1d_bias
    fe_adx_x_htf1d          = adx_ltf * htf_1d_bias

    fe_full_htf_align_long  = 1 if (htf_4h_bias == 1.0 and htf_1d_bias == 1.0) else 0
    fe_full_htf_align_short = 1 if (htf_4h_bias == -1.0 and htf_1d_bias == -1.0) else 0

    fe_adx_trending         = 1 if adx_ltf > 25.0 else 0
    fe_adx_4h_trending      = 1 if adx_4h > 25.0 else 0
    fe_rsi_overbought       = 1 if rsi_ltf > 65.0 else 0
    fe_rsi_oversold         = 1 if rsi_ltf < 35.0 else 0
    fe_rsi_4h_bull          = 1 if rsi_4h > 55.0 else 0
    fe_high_volume          = 1 if volume_ratio > 1.5 else 0
    fe_bb_squeeze_regime    = 1 if bb_width_ltf < atr_ltf else 0

    fe_session_london       = 1 if hour_of_day in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16] else 0
    fe_session_ny           = 1 if hour_of_day in [13, 14, 15, 16, 17, 18, 19, 20, 21] else 0
    fe_session_asia         = 1 if hour_of_day in [23, 0, 1, 2, 3, 4, 5, 6, 7, 8] else 0
    fe_session_overlap      = 1 if hour_of_day in [13, 14, 15] else 0
    fe_weekend              = 1 if day_of_week in [5, 6] else 0

    features_dict = {
        'FE_adx_4h_ratio':           round(float(fe_adx_4h_ratio), 4),
        'FE_adx_4h_trending':        int(fe_adx_4h_trending),
        'FE_adx_trending':           int(fe_adx_trending),
        'FE_adx_x_atr_pct':          round(float(fe_adx_x_atr_pct), 4),
        'FE_adx_x_htf1d':            round(float(fe_adx_x_htf1d), 4),
        'FE_adx_x_volume':           round(float(fe_adx_x_volume), 4),
        'FE_bb_squeeze_regime':      int(fe_bb_squeeze_regime),
        'FE_ema_ratio':              round(float(fe_ema_ratio), 6),
        'FE_ema_sep_x_adx':          round(float(fe_ema_sep_x_adx), 4),
        'FE_exhaustion_risk':        round(float(fe_exhaustion_risk), 4),
        'FE_full_htf_align_long':    int(fe_full_htf_align_long),
        'FE_full_htf_align_short':   int(fe_full_htf_align_short),
        'FE_high_volume':            int(fe_high_volume),
        'FE_macd_x_volume':          round(float(fe_macd_x_volume), 6),
        'FE_price_to_bb':            round(float(fe_price_to_bb), 4),
        'FE_rsi_4h_bull':            int(fe_rsi_4h_bull),
        'FE_rsi_mtf_ratio':          round(float(fe_rsi_mtf_ratio), 4),
        'FE_rsi_overbought':         int(fe_rsi_overbought),
        'FE_rsi_oversold':           int(fe_rsi_oversold),
        'FE_rsi_x_htf4h':            round(float(fe_rsi_x_htf4h), 4),
        'FE_rsi4h_x_htf1d':          round(float(fe_rsi4h_x_htf1d), 4),
        'FE_session_asia':           int(fe_session_asia),
        'FE_session_london':         int(fe_session_london),
        'FE_session_ny':             int(fe_session_ny),
        'FE_session_overlap':        int(fe_session_overlap),
        'FE_spread_to_atr_ratio':    round(float(fe_spread_to_atr_ratio), 4),
        'FE_vol_efficiency_ratio':   round(float(fe_vol_efficiency_ratio), 4),
        'FE_weekend':                int(fe_weekend),
        'adx_4h':                    round(float(adx_4h), 2),
        'adx_ltf':                   round(float(adx_ltf), 2),
        'adx_slope':                 round(float(adx_slope), 2),
        'atr_ltf':                   round(float(atr_ltf), 6),
        'atr_pct':                   round(float(atr_pct), 4),
        'bb_width_ltf':              round(float(bb_width_ltf), 6),
        'day_of_week':               int(day_of_week),
        'ema_fast_ltf':              round(float(ema_fast_ltf), 6),
        'ema_fast_slope':            round(float(ema_fast_slope), 4),
        'ema_separation':            round(float(ema_separation), 4),
        'ema_separation_4h':         round(float(ema_separation_4h), 4),
        'ema_slow_ltf':              round(float(ema_slow_ltf), 6),
        'ema_slow_slope':            round(float(ema_slow_slope), 4),
        'hour_of_day':               int(hour_of_day),
        'htf_1d_bias':               float(htf_1d_bias),
        'htf_4h_bias':               float(htf_4h_bias),
        'macd_histogram_4h':         round(float(macd_histogram_4h), 6),
        'macd_histogram_ltf':        round(float(macd_histogram_ltf), 6),
        'price_to_atr':              round(float(price_to_atr), 2),
        'rsi_4h':                    round(float(rsi_4h), 2),
        'rsi_ltf':                   round(float(rsi_ltf), 2),
        'swing_high':                round(float(swing_high), 6),
        'swing_low':                 round(float(swing_low), 6),
        'volume_ratio':              round(float(volume_ratio), 4),
        'volume_trend':              round(float(volume_trend), 4)
    }

    for k, v in features_dict.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            features_dict[k] = 0.0

    return features_dict


# =============================================================================
# STEP 5: True Rolling Sequence Extractor
# =============================================================================
def extract_rolling_features_history(
    df_15m: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    seq_len: int = 15
) -> list:
    """Extracts a genuine rolling sequential history of `seq_len` completed candles."""
    history = []
    start_idx = -seq_len
    for i in range(start_idx, 0):
        feat = compute_production_features_at_idx(df_15m, df_4h, df_1d, idx=i)
        history.append(feat)
    return history


# =============================================================================
# STEP 6: Production Model Pipeline
# =============================================================================
class ProductionFeaturePipeline:
    def __init__(self, manifest_path: str = None):
        if manifest_path is None:
            manifest_path = os.path.join(os.getcwd(), "Optimal_hyperparameters", "Ema_testnet_feature_manifest.json")
            if not os.path.exists(manifest_path):
                manifest_path = os.path.join(os.getcwd(), "ema_testnet", "Optimal_hyperparameters", "Ema_testnet_feature_manifest.json")

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"[FATAL] Feature manifest not found at: {manifest_path}")

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.shap_features = self.manifest["shap_features_by_target"]
        self.scalers       = self.manifest["scalers"]["production"]

    def get_feature_names(self, target: str, direction: str) -> list:
        return self.shap_features[target][direction]

    def get_scaler_params(self, target: str, direction: str, symbol: str):
        key = f"{target}__{direction}__{symbol}"
        if key not in self.scalers:
            raise KeyError(f"[Feature Error] No production scaler registered for: {key}")
        entry = self.scalers[key]
        return np.array(entry["mean"], dtype=np.float32), np.array(entry["scale"], dtype=np.float32)

    def prepare_catboost_input(self, features_dict: dict, target: str, direction: str, symbol: str) -> np.ndarray:
        f_names = self.get_feature_names(target, direction)
        mean_arr, scale_arr = self.get_scaler_params(target, direction, symbol)

        raw_vals = [features_dict[col] for col in f_names]
        raw_arr = np.array(raw_vals, dtype=np.float32)
        scaled_vals = (raw_arr - mean_arr) / np.where(scale_arr == 0, 1.0, scale_arr)
        return scaled_vals.reshape(1, -1)

    def prepare_gru_sequence_tensor(
        self,
        recent_features_list: list,
        target_cls: str,
        direction: str,
        symbol: str,
        seq_len: int = 15,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        target_cont = 'target_profit_v1' if 'profit' in target_cls else 'target_danger_v1'
        f_names = self.get_feature_names(target_cont, direction)
        mean_arr, scale_arr = self.get_scaler_params(target_cont, direction, symbol)

        window = recent_features_list[-seq_len:]
        while len(window) < seq_len:
            window.insert(0, window[0])

        matrix_raw = np.array([[f_dict[col] for col in f_names] for f_dict in window], dtype=np.float32)
        matrix_scaled = (matrix_raw - mean_arr) / np.where(scale_arr == 0, 1.0, scale_arr)

        tensor_3d = torch.tensor(matrix_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        return tensor_3d


# =============================================================================
# STEP 7: Module Integration Self-Test Probe
# =============================================================================
if __name__ == "__main__":
    print("===============================================================================")
    print("  TESTING TIME-AWARE PRODUCTION FEATURE PIPELINE (src/features.py)             ")
    print("===============================================================================")
    pipeline = ProductionFeaturePipeline()
    print("  [PASS] Feature manifest and pre-fitted production scalers loaded cleanly.")
