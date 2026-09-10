"""
====================================================================================================
ALGORITHM: main.py — Autonomous Production Daemon with Rejection Telemetry
====================================================================================================
Purpose:
  Operates as an autonomous 5.5-hour trading daemon inside GitHub Actions. Wakes up on closed
  15-minute candles, runs in-flight order maintenance every 30s, evaluates crossovers, logs approved
  orders to Table 1, logs REJECTED signals to Table 2, and self-chains at 5 hours 20 minutes.

Key Updates:
  - When manifest["approved"] is False, calls `telemetry.record_rejected_signal(...)`, logging
    the rejection reason, predicted MFE/MAE, and R:R directly into Supabase Table 2!

Algorithm Steps:
  1. Module Setup & Secrets Ingestion.
  2. RAM Model Loading (CatBoost + Funnel GRU).
  3. Self-Chaining Handover Trigger via GitHub API.
  4. In-Flight Position Maintenance (Every 30s).
  5. 15-Minute Pipeline (Signal Detection, Reversal Market Exit, Sizing, and Order/Rejection Logging).
  6. Autonomous 5.5-Hour Execution Loop.
====================================================================================================
"""

# =============================================================================
# STEP 1: Module Setup & Force Unbuffered Output
# =============================================================================
import os
import sys
import time
import json
import warnings
import requests
from datetime import datetime, timezone
import pandas as pd
import torch

# Force immediate real-time line buffering on stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

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

MAX_RUN_DURATION_MINUTES = 320
MAINTENANCE_INTERVAL_SEC = 30

GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "").strip()
BINANCE_KEY       = os.environ.get("BINANCE_TESTNET_API_KEY", "").strip()
BINANCE_SECRET    = os.environ.get("BINANCE_TESTNET_API_SECRET", "").strip()


# =============================================================================
# STEP 2: In-Memory Engine Initialization
# =============================================================================
print("===============================================================================")
print("  EMA_TESTNET PRODUCTION DAEMON (AUTONOMOUS 5.5-HOUR WORKER)                   ")
print(f"  Max Lifespan     : {MAX_RUN_DURATION_MINUTES} Minutes ({MAX_RUN_DURATION_MINUTES/60:.2f} Hours)")
print(f"  Target Repository: {GITHUB_REPOSITORY}                                       ")
print("===============================================================================\n")

print("1. Initializing Telemetry and Database Connections...")
telemetry = TelemetryEngine()

print("2. Loading 48 Production Models into RAM...")
model_registry = ProductionModelRegistry()
gates_engine   = ProductionGatesEngine()

print("3. Connecting Execution Engine to Binance Futures Testnet...")
execution = ExecutionEngine(api_key=BINANCE_KEY, api_secret=BINANCE_SECRET, telemetry=telemetry)

ACTIVE_SYMBOLS = ["BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
print("All engines initialized. Starting autonomous execution loop.\n")


# =============================================================================
# STEP 3: Self-Chaining Dispatcher (GitHub Actions REST API)
# =============================================================================
def dispatch_successor_workflow():
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
# STEP 4: Position Maintenance Routine (Every 30 Seconds)
# =============================================================================
def run_position_maintenance():
    active_orders = telemetry.get_active_trades()
    if not active_orders:
        return

    for trade in active_orders:
        if trade.get("order_status") == "PENDING_LIMIT":
            execution.check_and_deploy_brackets(trade)

    execution.handle_expired_limit_orders(max_timeout_minutes=15)

    live_positions = execution.get_active_positions()
    now_utc = datetime.now(timezone.utc)

    for trade in active_orders:
        if trade.get("order_status") == "FILLED":
            sym = trade["symbol"]
            if sym not in live_positions:
                trade_id = trade["id"]
                created_dt = pd.to_datetime(trade["created_at"], utc=True)
                hold_mins = (now_utc - created_dt).total_seconds() / 60.0

                try:
                    recent_trades = execution.exchange.fetch_my_trades(sym, limit=2)
                    last_trade = recent_trades[-1] if recent_trades else {}
                    exit_price = float(last_trade.get('price', trade["dynamic_tp_price"]))
                    fees_paid = sum(float(t.get('fee', {}).get('cost', 0.0)) for t in recent_trades)
                except Exception:
                    exit_price = float(trade["dynamic_tp_price"])
                    fees_paid = float(trade["allocated_cash"]) * 0.0008

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
# STEP 5: 15-Minute Pipeline (With Rejection Telemetry to Supabase)
# =============================================================================
def run_candle_close_pipeline():
    t_start = time.perf_counter()
    print(f"\n───────────────────────────────────────────────────────────────────────────────")
    print(f"  EVALUATING 15-MINUTE CANDLE CLOSE AT {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    print(f"───────────────────────────────────────────────────────────────────────────────")

    free_cash = execution.get_free_usdt_balance()
    active_trades = telemetry.get_active_trades()
    active_by_symbol = {t["symbol"]: t for t in active_trades if t.get("order_status") in ["PENDING_LIMIT", "FILLED"]}

    print(f"Active Slots Deployed: {len(active_by_symbol)} / 5 | Free Cash Available: ${free_cash:,.2f} USDT")

    for symbol in ACTIVE_SYMBOLS:
        try:
            df_15m = fetch_closed_ohlcv(execution.exchange, symbol, '15m', limit=60)
            df_4h  = fetch_closed_ohlcv(execution.exchange, symbol, '4h',  limit=40)
            df_1d  = fetch_closed_ohlcv(execution.exchange, symbol, '1d',  limit=25)

            signal, cross_price, candle_close_utc = detect_crossover(df_15m)
            if not signal:
                continue

            print(f"\n[Crossover Fired] {symbol} -> {signal} at ${cross_price:,.2f} (Candle Close: {candle_close_utc})")

            # Signal-Flip Reversal Check
            if symbol in active_by_symbol:
                existing_trade = active_by_symbol[symbol]
                existing_dir   = existing_trade["direction"].upper()

                if signal != existing_dir:
                    print(f"   --> Reverse signal detected! Initiating Signal-Flip Market Liquidation...")
                    execution.execute_signal_flip_close(existing_trade)
                    del active_by_symbol[symbol]
                    free_cash = execution.get_free_usdt_balance()
                else:
                    print(f"   --> Same-direction signal ignored. Symbol {symbol} is already active.")
                    continue

            # Compute features & model predictions
            features_25 = compute_production_features(df_15m, df_4h, df_1d)
            recent_history = [features_25] * 30
            model_outputs  = model_registry.predict_trade_setup(symbol, signal, features_25, recent_history)

            print(f"   --> Predictions: Profit MFE={model_outputs['pred_profit_mfe']:.2f}% | Danger MAE={model_outputs['pred_danger_mae']:.2f}%")
            print(f"   --> Gates      : Prob(Profit)={model_outputs['prob_profit']:.3f} | Prob(Danger)={model_outputs['prob_danger']:.3f}")

            # Gate policy evaluation
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

                trade_id = execution.execute_limit_entry(manifest, candle_close_utc)
                print(f"       Order Dispatched! Trade UUID: {trade_id}")

                active_by_symbol[symbol] = {"id": trade_id, "symbol": symbol, "direction": signal, "order_status": "PENDING_LIMIT"}
                free_cash = execution.get_free_usdt_balance()
            else:
                print(f"   --> REJECTED: {manifest['rejection_reason']} (R:R = {manifest['rr_ratio']})")
                
                # ── LOG REJECTION TELEMETRY DIRECTLY TO SUPABASE TABLE 2 ──
                telemetry.record_rejected_signal(symbol, signal, cross_price, manifest)
                print(f"       Rejection telemetry logged to Supabase Table 2 (testnet_trade_log).")

        except Exception as e:
            print(f"[Signal Error] Failed to process {symbol}: {repr(e)}")

    elapsed_pipeline = time.perf_counter() - t_start
    print(f"\n15-Minute Pipeline Completed in {elapsed_pipeline:.2f}s (Target: < 20s).")


# =============================================================================
# STEP 6: Autonomous 5.5-Hour Execution Loop
# =============================================================================
def main():
    daemon_start_time = time.time()
    print(f"\n[Daemon Started] Autonomous 5.5-Hour loop active at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC.")
    last_processed_15m_block = None

    while True:
        try:
            now_dt = datetime.now(timezone.utc)
            elapsed_minutes = (time.time() - daemon_start_time) / 60.0

            if elapsed_minutes >= MAX_RUN_DURATION_MINUTES:
                print(f"\n[Lifespan Reached] {elapsed_minutes:.1f} / {MAX_RUN_DURATION_MINUTES} Mins elapsed. Handover initiated.")
                success = dispatch_successor_workflow()
                if success:
                    time.sleep(15)
                    break
                else:
                    daemon_start_time += 900

            run_position_maintenance()

            current_minute = now_dt.minute
            current_second = now_dt.second
            current_15m_block = current_minute // 15

            seconds_into_15m = (current_minute % 15) * 60 + current_second
            seconds_until_close = 900 - seconds_into_15m

            if seconds_into_15m <= 15 and last_processed_15m_block != current_15m_block:
                time.sleep(1.5)
                run_candle_close_pipeline()
                last_processed_15m_block = current_15m_block
                continue

            if seconds_until_close <= 150:
                sleep_target = seconds_until_close + 1.5
                print(f"[Timing Engine] Approaching 15m candle close. Sleeping {sleep_target:.1f}s to align with close...")
                time.sleep(sleep_target)
                run_candle_close_pipeline()
                last_processed_15m_block = datetime.now(timezone.utc).minute // 15
            else:
                time.sleep(min(MAINTENANCE_INTERVAL_SEC, seconds_until_close - 150))

        except Exception as e:
            print(f"[Daemon Core Exception] Recovering: {repr(e)}")
            time.sleep(10)


if __name__ == "__main__":
    main()
            time.sleep(10)


if __name__ == "__main__":
    main()
