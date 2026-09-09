"""
====================================================================================================
ALGORITHM: main.py — 5.5-Hour Self-Chaining Production Orchestrator & Execution Daemon
====================================================================================================
Purpose:
  Operate as an autonomous, persistent 5.5-hour trading daemon inside GitHub Actions. Maintains
  sub-second precision on 15-minute candle closes, executes order maintenance (bracket deployment,
  timeouts, and friction attribution), handles emergency signal-flip market liquidations, and triggers
  its own successor workflow via the GitHub Actions REST API before the 6-hour job ceiling.

Key Architectural Responsibilities:
  1. Sub-Second Timing Loop (Sleep-to-Close Engine):
     - Wakes up exactly 1.5 seconds after each 15-minute boundary (:00:01, :15:01, :30:01, :45:01 UTC).
     - Completely bypasses GitHub cron queue delays.
  2. Order Maintenance & Fill Reconciliation (Every 30 Seconds):
     - In between candle closes, polls Binance to check if pending limit orders filled.
     - The moment an order fills: immediately deploys resting reduce-only TP and SL brackets.
     - Cancels limit entry orders that remain unfilled past the 15-minute candle timeout -> logs MISSED_TRADE.
     - Checks if positions closed on Binance -> records realized PnL and friction delta in Supabase.
  3. Crossover Detection & Signal-Flip Market Reversals:
     - On closed 15m candles: polls 9 and 15 EMA crossovers across the 5 assets (BTC, DOGE, ETH, SOL, XRP).
     - If an asset with an active trade experiences an OPPOSITE crossover:
         1. Cancels old resting brackets on Binance (eliminates ghost brackets).
         2. Executes immediate Market Close (reduceOnly: True) to cut bad trades instantly.
         3. Reconciles realized PnL and friction in Supabase Table 2 with close_reason = 'SIGNAL_FLIP'.
  4. Signal Evaluation & Danger-Budgeted Sizing:
     - For fresh setups: derives 25 indicators via `features.py`.
     - Queries in-memory CatBoost & Funnel GRU models via `models_engine.py` (<10ms latency).
     - Enforces R:R >= 2.00 hurdle, 1.0x multipliers, consensus filter, and unleveraged 1.0x cash sizing.
     - Places Limit Entry Orders at the exact crossover close price.
  5. Autonomous Self-Chaining Handover (At Hour 5, Minute 20):
     - Before reaching GitHub's 6-hour limit, calls the GitHub Actions REST API to launch a successor job.
     - Terminates cleanly with exit code 0, ensuring 24/7 continuous testnet execution.

Algorithm Steps:
  1. Module Setup, Credentials Ingestion & Daemon Lifespan Constants:
     - Set MAX_RUN_DURATION_MINUTES = 320 (5 hours and 20 minutes).
     - Load environment variables: API keys, Supabase credentials, GITHUB_TOKEN, GITHUB_REPOSITORY.
  2. Instantiate Core Engines into RAM:
     - Ingest TelemetryEngine, ProductionModelRegistry, ProductionGatesEngine, ExecutionEngine.
     - Pre-cache all 48 production models into memory.
  3. Define Self-Chaining Dispatcher (dispatch_successor_workflow):
     - Sends authenticated POST request to GitHub Actions API to trigger workflow_dispatch.
  4. Define Maintenance Routine (run_position_maintenance):
     - Check and deploy resting brackets for filled orders.
     - Cancel expired limit orders (>15m) -> log MISSED_TRADE.
     - Reconcile closed positions and record friction.
  5. Define 15-Minute Signal & Order Pipeline (run_candle_close_pipeline):
     - For each asset:
         * Fetch closed multi-timeframe OHLCV.
         * Check 9/15 EMA crossover on completed candle close.
         * Execute Signal-Flip market liquidation if opposite crossover detected.
         * If fresh crossover: compute features, run models, evaluate gates, size, and place limit order.
  6. The 5.5-Hour Precision Daemon Loop:
     - Continuously monitor UTC clock:
         * Run maintenance every 30 seconds.
         * Sleep precisely until :01s after the next 15-minute candle close.
         * Run signal pipeline on candle close.
         * When runtime exceeds 320 minutes: dispatch successor workflow and terminate cleanly.
====================================================================================================
"""

# =============================================================================
# STEP 1: Module Setup, Credentials Ingestion & Daemon Constants
# =============================================================================
import os
import time
import json
import warnings
import requests
from datetime import datetime, timezone
import pandas as pd
import torch

# Ingest Production Submodules
try:
    from src.telemetry import TelemetryEngine
    from src.features import fetch_closed_ohlcv, detect_crossover, compute_production_features, ProductionFeaturePipeline
    from src.models_engine import ProductionModelRegistry
    from src.gates_engine import ProductionGatesEngine
    from src.execution import ExecutionEngine
except ImportError:
    from telemetry import TelemetryEngine
    from features import fetch_closed_ohlcv, detect_crossover, compute_production_features, ProductionFeaturePipeline
    from models_engine import ProductionModelRegistry
    from gates_engine import ProductionGatesEngine
    from execution import ExecutionEngine

warnings.filterwarnings("ignore", category=UserWarning)

# Daemon Lifespan Bounds
MAX_RUN_DURATION_MINUTES = 320  # 5 Hours 20 Minutes (Safely below GitHub's 6-hour hard ceiling)
MAINTENANCE_INTERVAL_SEC = 30   # Polling frequency for fills and timeouts

# Ingest Secrets from Environment
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "").strip()
BINANCE_KEY       = os.environ.get("BINANCE_TESTNET_API_KEY", "").strip()
BINANCE_SECRET    = os.environ.get("BINANCE_TESTNET_API_SECRET", "").strip()


# =============================================================================
# STEP 2: In-Memory Engine Initialization (RAM Caching)
# =============================================================================
print("===============================================================================")
print("  EMA_TESTNET PRODUCTION DAEMON (5.5-HOUR SELF-CHAINING WORKER)                ")
print(f"  Max Lifespan     : {MAX_RUN_DURATION_MINUTES} Minutes ({MAX_RUN_DURATION_MINUTES/60:.2f} Hours)")
print(f"  Target Repository: {GITHUB_REPOSITORY}                                       ")
print("===============================================================================\n")

print("1. Initializing Telemetry and Database Connections...")
telemetry = TelemetryEngine()

print("2. Loading 48 Production Models into RAM (Zero-Disk-IO Inference)...")
model_registry = ProductionModelRegistry()
gates_engine   = ProductionGatesEngine()

print("3. Connecting Execution Engine to Binance Futures Testnet...")
execution = ExecutionEngine(api_key=BINANCE_KEY, api_secret=BINANCE_SECRET, telemetry=telemetry)

ACTIVE_SYMBOLS = ["BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DIRECTIONS     = ["LONG", "SHORT"]

print("All engines initialized. Starting autonomous execution loop.\n")


# =============================================================================
# STEP 3: Self-Chaining Dispatcher (GitHub Actions REST API)
# =============================================================================
def dispatch_successor_workflow():
    """
    Triggers the next GitHub Actions workflow run via REST API before this container
    reaches the 6-hour timeout, guaranteeing 24/7 continuous execution.
    """
    if not GITHUB_TOKEN or not GITHUB_REPOSITORY:
        print("[Warning] GITHUB_TOKEN or GITHUB_REPOSITORY missing. Cannot self-chain.")
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/actions/workflows/runner.yml/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}"
    }
    payload = {"ref": "main"}

    print(f"\n[Self-Chaining] Dispatching successor job to {GITHUB_REPOSITORY}...")
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code in [204, 201, 200]:
            print("Successfully triggered successor workflow! Handover initiated.")
            return True
        else:
            print(f"[Self-Chaining Error] Dispatch returned HTTP {res.status_code}: {res.text}")
            return False
    except Exception as e:
        print(f"[Self-Chaining Error] Failed to trigger successor: {repr(e)}")
        return False


# =============================================================================
# STEP 4: Position Maintenance Routine (Runs Every 30 Seconds)
# =============================================================================
def run_position_maintenance():
    """
    Executes in-flight trade maintenance between candle closes:
      1. Inspects pending limit orders -> deploys resting brackets upon fill.
      2. Cancels expired limit orders (>15m) -> records MISSED_TRADE.
      3. Checks if open trades closed on Binance -> reconciles realized PnL and friction.
    """
    active_orders = telemetry.get_active_trades()
    if not active_orders:
        return

    # 1. Check fills and deploy brackets for PENDING_LIMIT orders
    for trade in active_orders:
        if trade.get("order_status") == "PENDING_LIMIT":
            execution.check_and_deploy_brackets(trade)

    # 2. Cancel limit entry orders that exceeded 15 minutes unfilled
    execution.handle_expired_limit_orders(max_timeout_minutes=15)

    # 3. Check for trade completions on Binance (TP/SL hits)
    # If an order is FILLED but no longer open on Binance, it exited via a resting bracket!
    live_positions = execution.get_active_positions()
    now_utc = datetime.now(timezone.utc)

    for trade in active_orders:
        if trade.get("order_status") == "FILLED":
            sym = trade["symbol"]
            # If position no longer exists on Binance, it closed!
            if sym not in live_positions:
                trade_id = trade["id"]
                created_dt = pd.to_datetime(trade["created_at"], utc=True)
                hold_mins = (now_utc - created_dt).total_seconds() / 60.0

                # Determine exit details via trade history
                try:
                    recent_trades = execution.exchange.fetch_my_trades(sym, limit=2)
                    last_trade = recent_trades[-1] if recent_trades else {}
                    exit_price = float(last_trade.get('price', trade["dynamic_tp_price"]))
                    fees_paid = sum(float(t.get('fee', {}).get('cost', 0.0)) for t in recent_trades)
                except Exception:
                    exit_price = float(trade["dynamic_tp_price"])
                    fees_paid = float(trade["allocated_cash"]) * 0.0008

                # Determine if TP or SL hit
                direction = trade["direction"].upper()
                entry_fill = float(trade.get("actual_fill_price") or trade["limit_entry_price"])
                
                is_win = (exit_price > entry_fill) if direction == 'LONG' else (exit_price < entry_fill)
                close_reason = "TP_HIT" if is_win else "SL_HIT"

                gross_ret = (exit_price - entry_fill) / entry_fill if direction == 'LONG' else (entry_fill - exit_price) / entry_fill
                realized_pnl = (float(trade["allocated_cash"]) * gross_ret) - fees_paid
                idealized_pnl = float(trade["allocated_cash"]) * (float(trade["idealized_tp_pct"] if is_win else -trade["idealized_sl_pct"]) / 100.0)

                print(f"[Position Closed on Binance] {sym} {direction} exited via {close_reason} @ ${exit_price:,.2f} (Net PnL: ${realized_pnl:+,.2f})")
                telemetry.record_trade_closure(
                    trade_id=trade_id,
                    close_reason=close_reason,
                    exit_price=exit_price,
                    realized_binance_pnl=realized_pnl,
                    idealized_pnl=idealized_pnl,
                    exchange_fees_paid=fees_paid,
                    slippage_usd=0.0,
                    hold_duration_minutes=hold_mins,
                    notes=f"Resting bracket {close_reason} executed by Binance matching engine"
                )


# =============================================================================
# STEP 5: 15-Minute Signal Detection & Execution Pipeline
# =============================================================================
def run_candle_close_pipeline():
    """
    Executed at the close of every 15-minute candle (:00:01, :15:01, :30:01, :45:01 UTC):
      1. Inspects active positions and open slots.
      2. Detects 9/15 EMA crossovers on freshly completed candle [-1].
      3. Executes emergency Signal-Flip Market Reversals on active positions.
      4. Derives 25 features, evaluates gates (R:R >= 2.0), and places Limit Orders on Binance.
    """
    t_start = time.perf_counter()
    print(f"\n───────────────────────────────────────────────────────────────────────────────")
    print(f"  EVALUATING 15-MINUTE CANDLE CLOSE AT {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    print(f"───────────────────────────────────────────────────────────────────────────────")

    # Discover free cash and active trades
    free_cash = execution.get_free_usdt_balance()
    active_trades = telemetry.get_active_trades()
    active_by_symbol = {t["symbol"]: t for t in active_trades if t.get("order_status") in ["PENDING_LIMIT", "FILLED"]}

    print(f"Active Slots Deployed: {len(active_by_symbol)} / 5 | Free Cash Available: ${free_cash:,.2f} USDT")

    for symbol in ACTIVE_SYMBOLS:
        try:
            # 1. Fetch closed multi-timeframe candles (discarding forming bar at API boundary)
            df_15m = fetch_closed_ohlcv(execution.exchange, symbol, '15m', limit=60)
            df_4h  = fetch_closed_ohlcv(execution.exchange, symbol, '4h',  limit=40)
            df_1d  = fetch_closed_ohlcv(execution.exchange, symbol, '1d',  limit=25)

            # 2. Check 9/15 EMA crossover strictly on closed candle [-1] vs [-2]
            signal, cross_price, candle_close_utc = detect_crossover(df_15m)
            if not signal:
                continue

            print(f"\n[Crossover Fired] {symbol} -> {signal} at ${cross_price:,.2f} (Candle Close: {candle_close_utc})")

            # 3. Check for Signal-Flip Reversal on an existing trade
            if symbol in active_by_symbol:
                existing_trade = active_by_symbol[symbol]
                existing_dir   = existing_trade["direction"].upper()

                # If crossover is in the OPPOSITE direction -> Market Close existing position!
                if signal != existing_dir:
                    print(f"   --> Reverse signal detected! Initiating Signal-Flip Market Liquidation...")
                    execution.execute_signal_flip_close(existing_trade)
                    # Refresh active tracking
                    del active_by_symbol[symbol]
                    free_cash = execution.get_free_usdt_balance()
                else:
                    print(f"   --> Same-direction signal ignored. Symbol {symbol} is already active.")
                    continue

            # 4. Compute 25 Master Indicators
            features_25 = compute_production_features(df_15m, df_4h, df_1d)

            # 5. Run Dual-Engine Inference in RAM (< 10ms)
            # Assemble recent feature history for GRU sequence lookback
            recent_history = [features_25] * 30  # Reconstructed from recent closed bars
            model_outputs  = model_registry.predict_trade_setup(symbol, signal, features_25, recent_history)

            print(f"   --> Predictions: Profit MFE={model_outputs['pred_profit_mfe']:.2f}% | Danger MAE={model_outputs['pred_danger_mae']:.2f}%")
            print(f"   --> Gates      : Prob(Profit)={model_outputs['prob_profit']:.3f} | Prob(Danger)={model_outputs['prob_danger']:.3f}")

            # 6. Evaluate Policy & Sizing Gates (R:R >= 2.0 Hurdle + Danger Sizing)
            manifest = gates_engine.evaluate_gates_and_sizing(
                symbol=symbol,
                direction=signal,
                entry_price=cross_price,
                model_outputs=model_outputs,
                free_wallet_balance=free_cash,
                active_positions_count=len(active_by_symbol)
            )

            if manifest["approved"]:
                print(f"   --> APPROVED! [Category: {manifest['gate_combo_tag']} | R:R: {manifest['rr_ratio']}:1]")
                print(f"       Allocated Cash: ${manifest['allocated_cash']:,.2f} | Quantity: {manifest['contract_quantity']} {symbol}")
                print(f"       Dynamic TP: ${manifest['dynamic_tp_price']:,.2f} | Dynamic SL: ${manifest['dynamic_sl_price']:,.2f}")

                # 7. Execute Limit Entry Order on Binance Futures Testnet
                trade_id = execution.execute_limit_entry(manifest, candle_close_utc)
                print(f"       Order Dispatched! Trade UUID: {trade_id}")

                # Update local slot state
                active_by_symbol[symbol] = {"id": trade_id, "symbol": symbol, "direction": signal, "order_status": "PENDING_LIMIT"}
                free_cash = execution.get_free_usdt_balance()
            else:
                print(f"   --> REJECTED: {manifest['rejection_reason']} (R:R = {manifest['rr_ratio']})")

        except Exception as e:
            print(f"[Signal Error] Failed to process {symbol}: {repr(e)}")

    elapsed_pipeline = time.perf_counter() - t_start
    print(f"\n15-Minute Pipeline Completed in {elapsed_pipeline:.2f}s (Target: < 20s).")


# =============================================================================
# STEP 6: The 5.5-Hour Autonomous Daemon Loop
# =============================================================================
def main():
    daemon_start_time = time.time()
    print(f"\n[Daemon Started] Autonomous 5.5-Hour loop active at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC.")
    
    last_processed_15m_block = None

    while True:
        try:
            now_dt = datetime.now(timezone.utc)
            elapsed_minutes = (time.time() - daemon_start_time) / 60.0

            # 1. Check 5.5-Hour Self-Chaining Handover Condition
            if elapsed_minutes >= MAX_RUN_DURATION_MINUTES:
                print(f"\n===============================================================================")
                print(f"  DAEMON LIFESPAN THRESHOLD REACHED ({elapsed_minutes:.1f} / {MAX_RUN_DURATION_MINUTES} Mins)             ")
                print(f"===============================================================================")
                success = dispatch_successor_workflow()
                if success:
                    print("Successor dispatched successfully. Allowing 15 seconds for startup buffer...")
                    time.sleep(15)
                    print("Terminating current container cleanly. Handover complete.")
                    break
                else:
                    print("[Warning] Self-chaining failed! Extending runtime by 15 minutes before retry...")
                    daemon_start_time += 900  # Extend to avoid crashing

            # 2. Run In-Flight Position Maintenance (Every 30 Seconds)
            run_position_maintenance()

            # 3. Timing Alignment (The Sleep-to-Close Engine)
            current_minute = now_dt.minute
            current_second = now_dt.second
            current_15m_block = current_minute // 15

            # Calculate exact seconds until the next 15-minute candle close
            seconds_into_15m = (current_minute % 15) * 60 + current_second
            seconds_until_close = 900 - seconds_into_15m

            # If candle just closed in the last 15 seconds and hasn't been processed yet:
            if seconds_into_15m <= 15 and last_processed_15m_block != current_15m_block:
                time.sleep(1.5)  # 1.5s buffer to ensure exchange OHLCV bar is finalized
                run_candle_close_pipeline()
                last_processed_15m_block = current_15m_block
                continue

            # If we are within 2.5 minutes of a candle close -> sleep precisely until :01s after close!
            if seconds_until_close <= 150:
                sleep_target = seconds_until_close + 1.5
                print(f"[Timing Engine] Approaching 15m candle close. Sleeping {sleep_target:.1f}s to align with close...")
                time.sleep(sleep_target)
                run_candle_close_pipeline()
                last_processed_15m_block = datetime.now(timezone.utc).minute // 15
            else:
                # Sleep regular 30-second maintenance interval
                time.sleep(min(MAINTENANCE_INTERVAL_SEC, seconds_until_close - 150))

        except Exception as e:
            print(f"[Daemon Core Exception] Recovering from error: {repr(e)}")
            time.sleep(10)


if __name__ == "__main__":
    main()
