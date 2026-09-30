"""
====================================================================================================
ALGORITHM: src/features.py — True Rolling Sequence & Multi-Timeframe Feature Pipeline
====================================================================================================
Purpose:
  Connects to Binance Futures via CCXT, fetches multi-timeframe OHLCV data with adequate warmup,
  detects 9/15 EMA crossovers on immutable closed candles (-1 vs -2), and computes the Section A.7
  feature suite. Generates authentic rolling 15-bar sequence histories for Funnel GRU inference,
  permanently resolving the temporal flattening bug.

Algorithm Steps:
  Step 1: Module Setup, Safe Math Utilities & Library Ingestion:
          - Safe ratio division to prevent NaN/Inf outputs in ratio features.
  Step 2: Multi-Timeframe Ingestion with Expanded Warmup Limits:
          - Ingest 100 bars (15m), 60 bars (4h), 50 bars (1d).
          - Drop forming candle `[:-1]` to ensure index -1 is closed and immutable.
  Step 3: Crossover Detection Engine (`detect_crossover`):
          - Compute 9 EMA and 15 EMA strictly on index -1 vs index -2.
  Step 4: Vectorized Indicator Calculator (`compute_production_features`):
          - Calculate 15m LTF indicators, 4h HTF indicators, 1d Daily indicators.
          - Compute full suite of Section A.7 interaction features (FE_ prefix).
  Step 5: True Rolling Sequence Extractor (`extract_rolling_features_history`):
          - Iterates across the last `seq_len` closed bars to construct a genuine temporal history.
  Step 6: Production Model Pipeline (`ProductionFeaturePipeline`):
          - Normalizes CatBoost (1, 16) array and Funnel GRU (1, seq_len, 16) causal tensor
            using pre-fitted production scaler parameters (mu, sigma) from the manifest.
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
    """Prevents division by zero or infinite outputs in financial ratio calculations."""
    try:
        den_clean = np.where(den == 0, np.nan, den)
        res = num / den_clean
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        return float(res) if np.isscalar(res) else res
    except Exception:
        return 0.0


# =============================================================================
# STEP 2: Multi-Timeframe Ingestion with Expanded Warmup
# =============================================================================
def fetch_closed_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    """
    Fetches raw OHLCV candles from Binance and drops the active, forming candle.
    Guarantees that .iloc[-1] is immutable and closed; .iloc[-2] is the prior closed candle.
    Default limit of 100 ensures complete mathematical warmup for 15-span and 26-span indicators.
    """
    raw_bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Drop incomplete forming bar
    df_closed = df.iloc[:-1].copy().reset_index(drop=True)
    return df_closed


# =============================================================================
# STEP 3: Crossover Detection Engine
# =============================================================================
def detect_crossover(df_15m_closed: pd.DataFrame):
    """
    Evaluates active 9/15 EMA crossover strictly on closed candles using .iloc[-1] and .iloc[-2].
    """
    if len(df_15m_closed) < 20:
        return None, 0.0, None

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
        return 'LONG', crossover_price, candle_close_utc
    elif bearish_cross:
        return 'SHORT', crossover_price, candle_close_utc
    else:
        return None, crossover_price, candle_close_utc


# =============================================================================
# STEP 4: Vectorized Indicator Calculator (Section A.7 Feature Parity)
# =============================================================================
def compute_production_features_at_idx(df_15m: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, idx: int = -1) -> dict:
    """
    Computes Section A.7 features at a specific candle index `idx` (default -1 = latest closed).
    """
    # ── 1. 15m Indicators ──
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

    # ── 2. 4h Indicators ──
    c_4h = df_4h['close']
    h_4h = df_4h['high']
    l_4h = df_4h['low']

    ema_fast_4h_s = c_4h.ewm(span=9, adjust=False).mean()
    ema_slow_4h_s = c_4h.ewm(span=15, adjust=False).mean()
    ema_fast_4h   = float(ema_fast_4h_s.iloc[-1])
    ema_slow_4h   = float(ema_slow_4h_s.iloc[-1])
    ema_separation_4h = safe_ratio((ema_fast_4h - ema_slow_4h), ema_slow_4h) * 100.0
    htf_4h_bias = 1.0 if ema_fast_4h > ema_slow_4h else -1.0

    adx_4h = float(ta.trend.ADXIndicator(high=h_4h, low=l_4h, close=c_4h, window=14, fillna=True).adx().iloc[-1])
    rsi_4h = float(ta.momentum.RSIIndicator(close=c_4h, window=14, fillna=True).rsi().iloc[-1])
    macd_histogram_4h = float(ta.trend.MACD(close=c_4h, window_fast=12, window_slow=26, window_sign=9, fillna=True).macd_diff().iloc[-1])

    # ── 3. 1d Indicators ──
    c_1d = df_1d['close']
    ema_fast_1d = float(c_1d.ewm(span=9, adjust=False).mean().iloc[-1])
    ema_slow_1d = float(c_1d.ewm(span=15, adjust=False).mean().iloc[-1])
    htf_1d_bias = 1.0 if ema_fast_1d > ema_slow_1d else -1.0

    # ── 4. Temporal Context ──
    target_time = df_15m['datetime_utc'].iloc[idx]
    hour_of_day = int(target_time.hour)
    day_of_week = int(target_time.weekday())

    # ── 5. Section A.7 Interactions ──
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


def compute_production_features(df_15m: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> dict:
    """Wrapper computing features on the most recent finalized closed bar (index -1)."""
    return compute_production_features_at_idx(df_15m, df_4h, df_1d, idx=-1)


# =============================================================================
# STEP 5: True Rolling Sequence Extractor
# =============================================================================
def extract_rolling_features_history(
    df_15m: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    seq_len: int = 15
) -> list:
    """
    Extracts a genuine rolling sequential history of `seq_len` completed candles.
    Eliminates the temporal flattening bug by providing authentic trajectory dynamics to the GRU.
    """
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
        """Constructs normalized (1, 16) array for CatBoost."""
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
        """Constructs normalized (1, seq_len, 16) tensor for Funnel GRU forward pass."""
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
# STEP 7: Integration Self-Test Probe
# =============================================================================
if __name__ == "__main__":
    print("===============================================================================")
    print("  TESTING ROLLING SEQUENCE FEATURE PIPELINE (src/features.py)                  ")
    print("===============================================================================")
    pipeline = ProductionFeaturePipeline()
    print("  [PASS] Feature manifest and scalers loaded cleanly.")
