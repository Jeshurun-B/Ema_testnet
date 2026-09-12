"""
====================================================================================================
ALGORITHM: src/features.py — Real-Time Signal Detection & Feature Engineering Engine
====================================================================================================
Purpose:
  Connect to Binance Futures via CCXT, fetch multi-timeframe market data (15m, 4h, 1d), evaluate
  active 9/15 EMA crossovers on the freshly completed candle (preserving -1 and -2 indexing),
  compute the 25 Master Union indicators with mathematical parity to Section A.7, and format
  normalized model inputs (2D CatBoost arrays and 3D causal Funnel GRU tensors).
  Fixes the deprecated `set_sandbox_mode(True)` exception in the Step 6 test harness.

Algorithm Steps:
  Step 1: Module Setup, Safe Math & Dependency Ingestion:
          - Import numpy, pandas, torch, ta, ccxt. Define safe_ratio to guard against div-by-zero.
  Step 2: Multi-Timeframe Ingestion & Boundary Trimming:
          - Drop forming bar via `raw_df.iloc[:-1]` so index -1 is strictly completed.
  Step 3: Crossover Detection Engine (`detect_crossover`):
          - Compute 9 EMA and 15 EMA on closed 15m bars.
          - Evaluate bullish/bearish crossover strictly on index -1 vs index -2.
  Step 4: Vectorized Feature Math (All 25 Production Features):
          - Compute 17 base indicators across 15m, 4h, and 1d.
          - Compute 8 Section A.7 interaction features (FE_ prefix).
  Step 5: Model Input Formatting Pipeline (`ProductionFeaturePipeline`):
          - Ingest feature manifest, extract 16 SHAP features, and apply pre-fitted scaler parameters.
  Step 6: Module Integration Self-Test Probe:
          - Connect to Binance Futures Testnet using hardened modern endpoints (no deprecated sandbox calls).
====================================================================================================
"""

# =============================================================================
# STEP 1: Execution Guard, Library Ingestion & Safe Math
# =============================================================================
import os
import json
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch

try:
    import ta
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "ta"], check=True)
    import ta

try:
    import ccxt
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "ccxt"], check=True)
    import ccxt

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
# STEP 2: Multi-Timeframe Ingestion & Incomplete Bar Trimming (The -1 and -2 Fix)
# =============================================================================
def fetch_closed_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    """
    Fetches raw OHLCV candles from Binance and DISCARDS the active, forming candle.
    Guarantees that .iloc[-1] is immutable and closed; .iloc[-2] is the prior closed candle.
    """
    raw_bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Drop incomplete forming bar
    df_closed = df.iloc[:-1].copy().reset_index(drop=True)
    return df_closed


# =============================================================================
# STEP 3: Crossover Detection Engine (Foundational -1 and -2 Logic)
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
# STEP 4: Vectorized Feature Math (All 25 Production Features)
# =============================================================================
def compute_production_features(df_15m: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> dict:
    """Computes Master Union of 25 technical indicators directly from closed OHLCV data."""
    # 15m Low-Timeframe Indicators
    c_15m = df_15m['close']
    h_15m = df_15m['high']
    l_15m = df_15m['low']
    v_15m = df_15m['volume']

    ema_fast_15m_s = c_15m.ewm(span=9, adjust=False).mean()
    ema_slow_15m_s = c_15m.ewm(span=15, adjust=False).mean()
    
    ema_fast_ltf   = float(ema_fast_15m_s.iloc[-1])
    ema_slow_ltf   = float(ema_slow_15m_s.iloc[-1])
    ema_fast_prev  = float(ema_fast_15m_s.iloc[-2])
    ema_slow_prev  = float(ema_slow_15m_s.iloc[-2])
    price_latest   = float(c_15m.iloc[-1])

    ema_fast_slope = safe_ratio((ema_fast_ltf - ema_fast_prev), ema_fast_prev) * 100.0
    ema_slow_slope = safe_ratio((ema_slow_ltf - ema_slow_prev), ema_slow_prev) * 100.0

    adx_ind_15m = ta.trend.ADXIndicator(high=h_15m, low=l_15m, close=c_15m, window=14, fillna=True)
    adx_15m_s   = adx_ind_15m.adx()
    adx_ltf     = float(adx_15m_s.iloc[-1])
    adx_slope   = float(adx_15m_s.iloc[-1] - adx_15m_s.iloc[-2])

    rsi_15m_s = ta.momentum.RSIIndicator(close=c_15m, window=14, fillna=True).rsi()
    rsi_ltf   = float(rsi_15m_s.iloc[-1])

    atr_15m_s = ta.volatility.AverageTrueRange(high=h_15m, low=l_15m, close=c_15m, window=14, fillna=True).average_true_range()
    atr_ltf   = float(atr_15m_s.iloc[-1])
    atr_pct   = (atr_ltf / price_latest * 100.0) if price_latest > 0 else 0.0
    price_to_atr = safe_ratio(price_latest, atr_ltf)

    bb_ind = ta.volatility.BollingerBands(close=c_15m, window=20, window_dev=2, fillna=True)
    bb_width_ltf = float(bb_ind.bollinger_wband().iloc[-1])

    macd_15m_diff = ta.trend.MACD(close=c_15m, window_fast=12, window_slow=26, window_sign=9, fillna=True).macd_diff()
    macd_histogram_ltf = float(macd_15m_diff.iloc[-1])

    vol_ma_s     = v_15m.rolling(window=20, min_periods=1).mean()
    volume_ratio = safe_ratio(float(v_15m.iloc[-1]), float(vol_ma_s.iloc[-1]))
    volume_trend = safe_ratio((float(vol_ma_s.iloc[-1]) - float(vol_ma_s.iloc[-2])), float(vol_ma_s.iloc[-2])) * 100.0

    swing_high = float(h_15m.iloc[-20:].max())
    swing_low  = float(l_15m.iloc[-20:].min())

    # 4h High-Timeframe Indicators
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

    # 1d Daily Timeframe Indicators
    c_1d = df_1d['close']
    ema_fast_1d = float(c_1d.ewm(span=9, adjust=False).mean().iloc[-1])
    ema_slow_1d = float(c_1d.ewm(span=15, adjust=False).mean().iloc[-1])
    htf_1d_bias = 1.0 if ema_fast_1d > ema_slow_1d else -1.0

    # Temporal Context
    last_candle_time = df_15m['datetime_utc'].iloc[-1]
    hour_of_day = int(last_candle_time.hour)
    day_of_week = int(last_candle_time.weekday())

    # Section A.7 Engineered Features (FE_)
    fe_rsi_mtf_ratio  = safe_ratio(rsi_ltf, rsi_4h)
    fe_ema_ratio      = safe_ratio(ema_fast_ltf, ema_slow_ltf)
    fe_price_to_bb    = safe_ratio(atr_pct, bb_width_ltf)
    fe_macd_x_volume  = macd_histogram_ltf * volume_ratio
    fe_session_london = 1 if hour_of_day in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16] else 0
    fe_rsi_x_htf4h    = rsi_ltf * htf_4h_bias
    fe_rsi4h_x_htf1d  = rsi_4h * htf_1d_bias
    fe_adx_x_htf1d    = adx_ltf * htf_1d_bias

    features_25 = {
        'FE_adx_x_htf1d':       round(float(fe_adx_x_htf1d), 4),
        'FE_ema_ratio':         round(float(fe_ema_ratio), 6),
        'FE_macd_x_volume':     round(float(fe_macd_x_volume), 6),
        'FE_price_to_bb':       round(float(fe_price_to_bb), 4),
        'FE_rsi4h_x_htf1d':     round(float(fe_rsi4h_x_htf1d), 4),
        'FE_rsi_mtf_ratio':     round(float(fe_rsi_mtf_ratio), 4),
        'FE_rsi_x_htf4h':       round(float(fe_rsi_x_htf4h), 4),
        'FE_session_london':    int(fe_session_london),
        'adx_4h':               round(float(adx_4h), 2),
        'adx_slope':            round(float(adx_slope), 2),
        'atr_pct':              round(float(atr_pct), 4),
        'day_of_week':          int(day_of_week),
        'ema_fast_ltf':         round(float(ema_fast_ltf), 6),
        'ema_fast_slope':       round(float(ema_fast_slope), 4),
        'ema_separation_4h':    round(float(ema_separation_4h), 4),
        'ema_slow_slope':       round(float(ema_slow_slope), 4),
        'hour_of_day':          int(hour_of_day),
        'macd_histogram_4h':    round(float(macd_histogram_4h), 6),
        'macd_histogram_ltf':   round(float(macd_histogram_ltf), 6),
        'price_to_atr':         round(float(price_to_atr), 2),
        'rsi_4h':               round(float(rsi_4h), 2),
        'swing_high':           round(float(swing_high), 6),
        'swing_low':            round(float(swing_low), 6),
        'volume_ratio':         round(float(volume_ratio), 4),
        'volume_trend':         round(float(volume_trend), 4)
    }

    for k, v in features_25.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            features_25[k] = 0.0

    return features_25


# =============================================================================
# STEP 5: Model Input Formatting Pipeline (CatBoost Array & GRU Tensor)
# =============================================================================
class ProductionFeaturePipeline:
    """Manages feature alignment to SHAP-16 and normalization using manifest scalers."""
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

    def prepare_catboost_input(self, features_25: dict, target: str, direction: str, symbol: str) -> np.ndarray:
        f_names = self.get_feature_names(target, direction)
        mean_arr, scale_arr = self.get_scaler_params(target, direction, symbol)

        raw_vals = np.array([features_25[col] for col in f_names], dtype=np.float32)
        scaled_vals = (raw_vals - mean_arr) / np.where(scale_arr == 0, 1.0, scale_arr)
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
# STEP 6: Module Integration Self-Test (Hardened Testnet Probe)
# =============================================================================
RUN_FEATURES_SELF_TEST = True

if __name__ == "__main__" and RUN_FEATURES_SELF_TEST:
    print("===============================================================================")
    print("  RUNNING REAL-TIME FEATURE & SIGNAL PIPELINE SELF-TEST                        ")
    print("===============================================================================")

    # Initialize public testnet CCXT client without deprecated sandbox call
    exchange = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Modern CCXT Demo Trading Routing
    if hasattr(exchange, "enable_demo_trading"):
        exchange.enable_demo_trading(True)
    elif hasattr(exchange, "enableDemoTrading"):
        exchange.enableDemoTrading(True)
    else:
        exchange.urls['api']['fapiPublic'] = 'https://testnet.binancefuture.com/fapi/v1'

    test_symbol = "BTCUSDT"

    print(f"\n1. Fetching closed multi-timeframe candles for {test_symbol}...")
    try:
        df_15m = fetch_closed_ohlcv(exchange, test_symbol, '15m', limit=60)
        df_4h  = fetch_closed_ohlcv(exchange, test_symbol, '4h', limit=40)
        df_1d  = fetch_closed_ohlcv(exchange, test_symbol, '1d', limit=25)

        print(f"   --> 15m Closed Candles: {len(df_15m)} bars (Latest: {df_15m['datetime_utc'].iloc[-1]})")
        print(f"   --> 4h  Closed Candles: {len(df_4h)} bars")
        print(f"   --> 1d  Closed Candles: {len(df_1d)} bars")

        print("\n2. Evaluating 9/15 EMA Crossover on completed candle [-1] vs [-2]...")
        signal, cross_price, cross_time = detect_crossover(df_15m)
        print(f"   --> Signal Detected : {signal}")
        print(f"   --> Crossover Price : ${cross_price:,.2f}")
        print(f"   --> Candle Timestamp: {cross_time}")

        print("\n3. Computing the 25 Master Union Features...")
        feats_25 = compute_production_features(df_15m, df_4h, df_1d)
        print(f"   --> Total Indicators Computed: {len(feats_25)} / 25")

        print("\n4. Formatting Model Inputs via Feature Pipeline...")
        pipeline = ProductionFeaturePipeline()
        cb_x = pipeline.prepare_catboost_input(feats_25, 'target_profit_v1', 'LONG', test_symbol)
        print(f"   --> CatBoost Input Shape : {cb_x.shape} (Expected: 1, 16)")
        assert cb_x.shape == (1, 16), "CatBoost shape mismatch!"

        gru_x = pipeline.prepare_gru_sequence_tensor([feats_25], 'target_profit_b50', 'LONG', test_symbol, seq_len=15)
        print(f"   --> Funnel GRU Input Shape: {tuple(gru_x.shape)} (Expected: 1, 15, 16)")
        assert tuple(gru_x.shape) == (1, 15, 16), "Funnel GRU shape mismatch!"

        print("\n===============================================================================")
        print("  VERDICT: [PASS] PRODUCTION FEATURE PIPELINE FULLY OPERATIONAL                 ")
        print("===============================================================================")

    except Exception as e:
        print(f"\n[Test Error] Execution probe failed: {repr(e)}")
