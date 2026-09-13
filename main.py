"""
====================================================================================================
ALGORITHM: main.py — Always-Awake Production Daemon with Silent Heartbeat & Pure Crossover Exits
====================================================================================================
Purpose:
  Autonomous 5.5-hour continuous trading engine. Operates on a silent, non-blocking 10-second
  heartbeat loop: monitors fills, brackets, and timeouts in the background without terminal clutter,
  and evaluates completed 15-minute candles at T+5.0s past the close. Strictly enforces pure
  crossover exits: active trades terminate early IF AND ONLY IF a formal, completed opposite
  9/15 EMA crossover is registered.

Key Architectural Invariants:
  1. Silent Vigilance Heartbeat (Every 10s):
     - Background maintenance loop checks order fills, timeouts, and bracket executions every 10s.
     - Emits zero terminal spam during normal idle checks. Outputs appear ONLY when an event occurs
       (fill, bracket deployment, timeout cancellation, or trade exit).
  2. Pure Event-Driven Crossover Exits (No Continuous Regime Invalidation):
     - An active trade is protected by its native, exchange-side Dynamic TP and Dynamic SL brackets.
     - It terminates early IF AND ONLY IF a completed 15-minute candle registers a formal opposite
       9/15 EMA crossover (Index [-1] vs [-2]).
     - On reversal: instantly purges resting brackets on Binance and issues a Market Close order.
  3. Surgical Candle Alignment (+5.0s Settlement Window):
     - Evaluates completed bars at minute % 15 == 0 and second >= 5.0s.
     - Block tracking tuple guarantees each 15-minute bar is evaluated strictly once.
  4. Autonomous Self-Chaining at 320 Minutes:
     - Dispatches successor runner via GitHub Actions REST API at 5h 20m for continuous 24/7 uptime.

Algorithm Steps:
  Step 1: Module Setup, Real-Time Line Buffering & Dependency Ingestion:
          - Import standard, networking, pandas, and CCXT libraries.
          - Reconfigure stdout for immediate unbuffered streaming.
  Step 2: In-Memory Engine Initialization:
          - Initialize TelemetryEngine, ProductionModelRegistry, ProductionGatesEngine, and ExecutionEngine.
          - Load all 48 models into RAM once (eliminating disk I/O).
  Step 3: Self-Chaining Handover Dispatcher:
          - Trigger successor workflow via GitHub REST API before GitHub Actions 6-hour hard timeout.
  Step 4: Silent In-Flight Position Maintenance Routine:
          - Checks limit fills -> deploys resting brackets.
          - Checks timeouts (>15m) -> cancels unfilled orders.
          - Checks bracket executions -> reconciles realized PnL and archives to Supabase Table 2.
          - Produces terminal output ONLY when an action is executed.
  Step 5: 15-Minute Pipeline with Pure Crossover Exits:
          - Ingests completed candles (15m, 4h, 1d) via the proxy tunnel.
          - Detects 9/15 EMA crossover on completed candle [-1] vs [-2].
          - Reconciles live positions: if an opposite crossover is confirmed, liquidates immediately.
          - If setup is valid: computes 25 features, runs RAM inference, evaluates $R:R \ge 2.0$ hurdle,
            and either routes a quantized limit entry or logs rejection telemetry to Table 2.
  Step 6: Master Continuous Heartbeat Loop:
          - Loops silently every 10 seconds.
          - Executes the 15-minute evaluation pipeline precisely at T+5.0s.
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
HEARTBEAT_INTERVAL_SEC   = 10   # Silent background tick interval
SETTLEMENT_BUFFER_SEC    = 5.0  # 5.0s buffer for Binance candle aggregation

GITHUB_TOKEN      = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "").strip()
BINANCE_KEY       = os.environ.get("BINANCE_TESTNET_API_KEY", "").strip()
BINANCE_SECRET    = os.environ.get("BINANCE_TESTNET_API_SECRET", "").strip()
BINANCE_PROXY     = os.environ.get("BINANCE_PROXY_URL", "").strip()


# =============================================================================
# STEP 2: In-Memory Engine Initialization
# =============================================================================
print("===============================================================================")
print("  EMA_TESTNET PRODUCTION DAEMON (SILENT HEARTBEAT & PURE CROSSOVER EXITS)      ")
print(f"  Max Lifespan     : {MAX_RUN_DURATION_MINUTES} Minutes ({MAX_RUN_DURATION_MINUTES/60:.2f} Hours)")
print(f"  Heartbeat Tick   : Every {HEARTBEAT_INTERVAL_SEC} Seconds (Silent Mode)                      ")
print(f"  Target Repository: {GITHUB_REPOSITORY}                                       ")
print("===============================================================================\n")

print("1. Initializing Telemetry and Database Connections...")
telemetry = TelemetryEngine()

print("2. Loading 48 Production Models into RAM...")
model_registry = ProductionModelRegistry()
gates_engine   = ProductionGatesEngine()

print("3. Connecting Execution Engine to Binance Futures Testnet...")
execution = ExecutionEngine(
    api_key=BINANCE_KEY,
    api_secret=BINANCE_SECRET,
    proxy_url=BINANCE_PROXY,
    telemetry=telemetry
)

ACTIVE_SYMBOLS = ["BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
last_evaluated_15m_block = None

print("All systems initialized successfully. Continuous event loop active.\n")


# =============================================================================
# STEP 3: Self-Chaining Dispatcher (GitHub Actions REST API)
# =============================================================================
def dispatch_successor_workflow():
    """Dispatches next 5.5-hour workflow runner via GitHub Actions REST API."""
    if not GITHUB_TOKEN or not GITHUB_REPOSITORY:
        print("[Warning] GITHUB_TOKEN/GH_PAT missing. Relying on scheduled cron triggers.")
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
            print("Successfully dispatched successor workflow! Clean handover complete.")
            return True
        else:
            print(f"[Self-Chaining Notice] Dispatch returned HTTP {res.status_code}: {res.text}")
            return False
    except Exception as e:
        print(f"[Self-Chaining Error] Dispatch exception: {repr(e)}")
        return False


# =============================================================================
# STEP 4: Silent In-Flight Position Maintenance (Emits Output Only on Events)
# =============================================================================
def run_position_maintenance():
    """
    Monitors in-flight orders silently every 10 seconds.
    Emits terminal output ONLY when an action occurs (fill, timeout, or bracket exit).
    """
    active_orders = telemetry.get_active_trades()
    if not active_orders:
        return

    # 1. Deploy native brackets for newly filled limit entry orders
    for trade in active_orders:
        if trade.get("order_status") == "PENDING_LIMIT":
            execution.check_and_deploy_brackets(trade)

    # 2. Cancel limit entry orders that exceeded 15 wall-clock minutes
    execution.handle_expired_limit_orders(max_timeout_minutes=15)

    # 3. Check for bracket fills (closures) on Binance
    live_positions = execution.get_active_positions()
    now_utc = datetime.now(timezone.utc)

    for trade in active_orders:
        if trade.get("order_status") == "FILLED":
            sym = trade["symbol"]
            # With normalized keys ('SOLUSDT'), this check is strictly accurate
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
# STEP 5: 15-Minute Pipeline (Pure Event-Driven Crossover Exits)
# =============================================================================
def run_candle_close_pipeline():
    """
    Evaluates completed 15m candle close across the 5 assets.
    Enforces Pure Crossover Exits: active trades terminate early IF AND ONLY IF
    a formal, completed opposite 9/15 EMA crossover is registered.
    """
    t_start = time.perf_counter()
    eval_time_str = datetime.now(timezone.utc).strftime('%H:%M:%S')

    print(f"\n───────────────────────────────────────────────────────────────────────────────")
    print(f"  EVALUATING 15-MINUTE CANDLE CLOSE AT {eval_time_str} UTC")
    print(f"───────────────────────────────────────────────────────────────────────────────")

    free_cash = execution.get_free_usdt_balance()
    active_trades = telemetry.get_active_trades()
    active_by_symbol = {t["symbol"]: t for t in active_trades if t.get("order_status") in ["PENDING_LIMIT", "FILLED"]}
    live_binance_positions = execution.get_active_positions()

    print(f"Active Slots Deployed: {len(active_by_symbol)} / 5 | Free Cash Available: ${free_cash:,.2f} USDT")

    for symbol in ACTIVE_SYMBOLS:
        try:
            df_15m = fetch_closed_ohlcv(execution.exchange, symbol, '15m', limit=60)
            df_4h  = fetch_closed_ohlcv(execution.exchange, symbol, '4h',  limit=40)
            df_1d  = fetch_closed_ohlcv(execution.exchange, symbol, '1d',  limit=25)

            # Detect formal 9/15 EMA Crossover on completed candle [-1] vs [-2]
            signal, cross_price, candle_close_utc = detect_crossover(df_15m)

            # ── PURE CROSSOVER EXIT ENFORCEMENT ──
            # If NO crossover is registered on this bar, do nothing.
            # Active trades remain protected by their resting TP and SL brackets.
            if not signal:
                continue

            # Check if this asset is currently active in Supabase or live on Binance
            has_db_trade = symbol in active_by_symbol
            has_live_pos = symbol in live_binance_positions

            if has_db_trade or has_live_pos:
                active_record = active_by_symbol.get(symbol)
                pos_dir = ""
                if active_record:
                    pos_dir = active_record["direction"].upper()
                elif has_live_pos:
                    pos_dir = live_binance_positions[symbol]["side"].upper()

                # 1. Opposite Crossover Registered -> IMMEDIATE SIGNAL FLIP CLOSE!
                if signal != pos_dir:
                    print(f"\n[Crossover Reversal Detected] Confirmed {signal} crossover opposing active {pos_dir}! Liquidating immediately...")
                    if active_record:
                        execution.execute_signal_flip_close(active_record)
                        del active_by_symbol[symbol]
                    elif has_live_pos:
                        # Orphaned position recovery: liquidate directly on Binance
                        mock_trade = {
                            "id": f"RECON_{int(time.time())}",
                            "symbol": symbol,
                            "direction": pos_dir,
                            "contract_quantity": live_binance_positions[symbol]["contracts"],
                            "limit_entry_price": live_binance_positions[symbol]["entry_price"],
                            "actual_fill_price": live_binance_positions[symbol]["entry_price"],
                            "allocated_cash": live_binance_positions[symbol]["contracts"] * live_binance_positions[symbol]["entry_price"],
                            "created_at": datetime.now(timezone.utc).isoformat()
                        }
                        execution.execute_signal_flip_close(mock_trade)

                    free_cash = execution.get_free_usdt_balance()
                else:
                    # Same-direction crossover on existing position -> Ignore (already entered)
                    print(f"   --> {symbol} already positioned in {pos_dir}. Repeat signal ignored.")
                    continue

            print(f"\n[Crossover Fired] {symbol} -> {signal} at ${cross_price:,.2f} (Candle Close: {candle_close_utc})")

            # Extract 25 master indicators & compute dual-engine model predictions
            features_25 = compute_production_features(df_15m, df_4h, df_1d)
            recent_history = [features_25] * 30
            model_outputs  = model_registry.predict_trade_setup(symbol, signal, features_25, recent_history)

            print(f"   --> Predictions: Profit MFE={model_outputs['pred_profit_mfe']:.2f}% | Danger MAE={model_outputs['pred_danger_mae']:.2f}%")
            print(f"   --> Gates      : Prob(Profit)={model_outputs['prob_profit']:.3f} | Prob(Danger)={model_outputs['prob_danger']:.3f}")

            # Risk gates & asymmetrical parity hurdle (R:R >= 2.0)
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

                active_by_symbol[symbol] = {
                    "id": trade_id,
                    "symbol": symbol,
                    "direction": signal,
                    "order_status": "PENDING_LIMIT",
                    "contract_quantity": manifest["contract_quantity"],
                    "limit_entry_price": manifest["entry_price"],
                    "allocated_cash": manifest["allocated_cash"],
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                free_cash = execution.get_free_usdt_balance()
            else:
                print(f"   --> REJECTED: {manifest['rejection_reason']} (R:R = {manifest['rr_ratio']})")
                telemetry.record_rejected_signal(symbol, signal, cross_price, manifest)
                print(f"       Rejection telemetry logged to Supabase Table 2 (testnet_trade_log).")

        except Exception as e:
            print(f"[Signal Error] Failed to process {symbol}: {repr(e)}")

    elapsed_pipeline = time.perf_counter() - t_start
    print(f"\n15-Minute Pipeline Completed in {elapsed_pipeline:.2f}s (Target: < 20s).")


# =============================================================================
# STEP 6: Master Continuous Heartbeat Loop (Silent Vigilance Mode)
# =============================================================================
def main():
    """
    Master daemon loop. Polles maintenance silently every 10s and triggers
    the 15-minute pipeline at T+5.0s past candle close.
    """
    global last_evaluated_15m_block
    daemon_start_time = time.time()
    print(f"[Daemon Started] Continuous Silent Heartbeat active at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC.")

    while True:
        try:
            now_dt = datetime.now(timezone.utc)
            elapsed_minutes = (time.time() - daemon_start_time) / 60.0

            # 1. Self-Chaining Lifespan Check at 320 Mins (5h 20m)
            if elapsed_minutes >= MAX_RUN_DURATION_MINUTES:
                print(f"\n[Lifespan Reached] {elapsed_minutes:.1f} / {MAX_RUN_DURATION_MINUTES} Mins elapsed. Handover initiated.")
                success = dispatch_successor_workflow()
                if success:
                    time.sleep(15)
                    break
                else:
                    daemon_start_time += 900

            # 2. Silent 10-Second In-Flight Order Maintenance (Outputs only on events)
            run_position_maintenance()

            # 3. Surgical Candle Close Detection (At minute :00, :15, :30, :45 when second >= 5.0)
            current_minute = now_dt.minute
            current_second = now_dt.second
            current_15m_block = (now_dt.year, now_dt.month, now_dt.day, now_dt.hour, current_minute // 15)

            is_candle_close_window = (current_minute % 15 == 0) and (current_second >= SETTLEMENT_BUFFER_SEC)

            if is_candle_close_window and (last_evaluated_15m_block != current_15m_block):
                run_candle_close_pipeline()
                last_evaluated_15m_block = current_15m_block

            # 4. Silent Sleep for 10 Seconds (Always awake, zero console spam)
            time.sleep(HEARTBEAT_INTERVAL_SEC)

        except Exception as e:
            print(f"[Daemon Heartbeat Exception] Recovering: {repr(e)}")
            time.sleep(10)


if __name__ == "__main__":
    main()
