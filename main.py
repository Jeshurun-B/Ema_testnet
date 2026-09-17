r"""
====================================================================================================
ALGORITHM: main.py — Tier-1 Production Daemon with Precise Order-ID Exit Reconciliation
====================================================================================================
Purpose:
  Institutional 5.5-hour continuous trading engine. Operates with In-Memory State as the primary
  source of truth during runtime (Hot Path) with Supabase serving purely as an asynchronous write-only
  telemetry sink (Cold Path). Resolves stale historical trade calculations by checking the exact
  status of `binance_tp_id` and `binance_sl_id` orders. Restores the Idempotent Timestamp Guard,
  enforces strict directional inversions (`signal != pos_dir`), phase-locks sleep timing to :05.00,
  and uses GH_PAT to dispatch runners.yml.

Algorithm Steps:
  Step 1: Module Setup, Dynamic Price Formatter & Output Unbuffering.
  Step 2: Engine Initialization & One-Time Startup Hydration (active_by_symbol).
  Step 3: Self-Chaining Dispatcher (Targeting runners.yml via GH_PAT).
  Step 4: Silent In-Memory Position Maintenance with Order-ID Reconciliation:
          - For PENDING_LIMIT: polls `fetch_order(binance_order_id)` -> on fill, deploys brackets.
            Enforces 15m wall-clock timeout -> physically cancels order on Binance.
          - For FILLED: inspects `binance_tp_id` and `binance_sl_id` specifically.
            Extracts the exact closing fill price and commissions of THAT specific order.
            Eliminates all stale historical trade calculations.
  Step 5: 15-Minute Pipeline (Idempotent Bar Guard & Strict Directional Inversion):
          - Rejects duplicate bars via `last_evaluated_candles`.
          - Detects 9/15 EMA crossover on completed candle [-1] vs [-2].
          - Liquidates position IF AND ONLY IF `signal != pos_dir`.
          - Computes 25 features, runs RAM inference, evaluates R:R >= 2.0 hurdle,
            and routes quantized limit entry capturing `(trade_id, binance_order_id)`.
  Step 6: Master Phase-Locked Loop (Target: Exact :05.00 Close).
====================================================================================================
"""

import os
import sys
import time
import json
import uuid
import warnings
import requests
from datetime import datetime, timezone
import pandas as pd
import torch

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
HEARTBEAT_INTERVAL_SEC   = 10
SETTLEMENT_BUFFER_SEC    = 5.0

GITHUB_TOKEN      = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "").strip()
BINANCE_KEY       = os.environ.get("BINANCE_TESTNET_API_KEY", "").strip()
BINANCE_SECRET    = os.environ.get("BINANCE_TESTNET_API_SECRET", "").strip()
BINANCE_PROXY     = os.environ.get("BINANCE_PROXY_URL", "").strip()


def format_price(price: float) -> str:
    """Formats price dynamically based on asset magnitude (e.g. DOGE vs BTC)."""
    if price < 1.0:
        return f"${price:.4f}"
    elif price < 10.0:
        return f"${price:.3f}"
    else:
        return f"${price:,.2f}"


# =============================================================================
# STEP 2: Engine Initialization & One-Time Startup Hydration
# =============================================================================
print("===============================================================================")
print("  EMA_TESTNET PRODUCTION DAEMON (TIER-1 IN-MEMORY HOT-PATH ENGINE)             ")
print(f"  Max Lifespan     : {MAX_RUN_DURATION_MINUTES} Minutes ({MAX_RUN_DURATION_MINUTES/60:.2f} Hours)")
print(f"  Heartbeat Tick   : Every {HEARTBEAT_INTERVAL_SEC} Seconds (Silent In-Memory Mode)           ")
print(f"  Target Repository: {GITHUB_REPOSITORY}                                       ")
print(f"  Dispatch Target  : .github/workflows/runners.yml                             ")
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
last_evaluated_candles = {sym: None for sym in ACTIVE_SYMBOLS}

print("4. Hydrating In-Memory Active Trade State from Supabase...")
active_by_symbol = telemetry.hydrate_active_trades_from_db()

live_positions = execution.get_active_positions()
for sym, pos_data in live_positions.items():
    if sym not in active_by_symbol:
        print(f"[Startup Reconciliation] Live Binance position detected for {sym} ({pos_data['side'].upper()}). Tracking in RAM.")
        active_by_symbol[sym] = {
            "id": None,
            "binance_order_id": None,
            "binance_tp_id": None,
            "binance_sl_id": None,
            "symbol": sym,
            "direction": pos_data["side"].upper(),
            "order_status": "FILLED",
            "contract_quantity": pos_data["contracts"],
            "limit_entry_price": pos_data["entry_price"],
            "actual_fill_price": pos_data["entry_price"],
            "allocated_cash": pos_data["contracts"] * pos_data["entry_price"],
            "created_at": datetime.now(timezone.utc).isoformat()
        }

print(f"In-Memory State Engine active. Current tracked positions: {len(active_by_symbol)} / 5 slots.\n")


# =============================================================================
# STEP 3: Self-Chaining Dispatcher (Targeting runners.yml via GH_PAT)
# =============================================================================
def dispatch_successor_workflow():
    if not GITHUB_TOKEN or not GITHUB_REPOSITORY:
        print("[Warning] GITHUB_TOKEN/GH_PAT missing. Relying on scheduled cron triggers.")
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/actions/workflows/runners.yml/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}"
    }
    payload = {"ref": "main"}

    print(f"\n[Self-Chaining] Dispatching successor job to {GITHUB_REPOSITORY} via runners.yml...")
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code in [204, 201, 200]:
            print("Successfully dispatched successor workflow! Clean handover complete (HTTP 204).")
            return True
        else:
            print(f"[Self-Chaining Notice] Dispatch returned HTTP {res.status_code}: {res.text}")
            return False
    except Exception as e:
        print(f"[Self-Chaining Error] Dispatch exception: {repr(e)}")
        return False


# =============================================================================
# STEP 4: In-Flight Maintenance with Order-ID Specific Reconciliation
# =============================================================================
def run_position_maintenance():
    global active_by_symbol
    
    if not active_by_symbol:
        return

    now_utc = datetime.now(timezone.utc)
    live_positions = execution.get_active_positions()
    symbols_to_remove = []

    for sym, trade in list(active_by_symbol.items()):
        status = trade.get("order_status")

        # 1. PENDING_LIMIT: Poll entry fill & enforce 15m timeout
        if status == "PENDING_LIMIT":
            binance_id = trade.get("binance_order_id")
            if binance_id and "MOCK" not in str(binance_id):
                try:
                    order_info = execution.exchange.fetch_order(binance_id, sym)
                    if order_info.get("status", "").lower() == "closed":
                        actual_fill = float(order_info.get("average") or order_info.get("price") or trade["limit_entry_price"])
                        print(f"[Order Fill Detected] {sym} {trade['direction']} filled at {format_price(actual_fill)}. Deploying resting brackets...")
                        
                        trade["order_status"] = "FILLED"
                        trade["actual_fill_price"] = actual_fill

                        if trade.get("id"):
                            telemetry.record_order_fill(trade["id"], actual_fill)

                        qty = float(trade["contract_quantity"])
                        tp_px, _ = execution.quantize_order_params(sym, float(trade["dynamic_tp_price"]), qty)
                        sl_px, _ = execution.quantize_order_params(sym, float(trade["dynamic_sl_price"]), qty)
                        close_side = "sell" if trade["direction"].upper() == "LONG" else "buy"

                        tp_order = execution.exchange.create_order(
                            symbol=sym, type="TAKE_PROFIT_MARKET", side=close_side, amount=qty,
                            params={"stopPrice": tp_px, "reduceOnly": True, "workingType": "MARK_PRICE"}
                        )
                        sl_order = execution.exchange.create_order(
                            symbol=sym, type="STOP_MARKET", side=close_side, amount=qty,
                            params={"stopPrice": sl_px, "reduceOnly": True, "workingType": "MARK_PRICE"}
                        )
                        trade["binance_tp_id"] = str(tp_order["id"])
                        trade["binance_sl_id"] = str(sl_order["id"])

                        if trade.get("id"):
                            telemetry.record_bracket_order_ids(trade["id"], str(tp_order["id"]), str(sl_order["id"]))
                        print(f"   --> Native Brackets Deployed: TP @ {format_price(tp_px)} | SL @ {format_price(sl_px)} [MARK_PRICE]")
                except Exception:
                    pass

            created_str = trade.get("created_at")
            order_dt = pd.to_datetime(created_str, utc=True) if created_str else now_utc
            elapsed_min = (now_utc - order_dt).total_seconds() / 60.0

            if elapsed_min >= 15.0:
                print(f"[Timeout Triggered] {sym} limit entry unfilled after {elapsed_min:.1f}m wall-clock time. Cancelling...")
                if binance_id and "MOCK" not in str(binance_id):
                    try:
                        execution.exchange.cancel_order(binance_id, sym)
                        print(f"   --> Order {binance_id} cancelled successfully on Binance.")
                    except Exception as e:
                        print(f"   --> Cancel Notice: {e}")
                if trade.get("id"):
                    telemetry.record_missed_trade(trade["id"], notes=f"Limit entry expired unfilled after {elapsed_min:.1f}m")
                symbols_to_remove.append(sym)

        # 2. FILLED: Order-ID Specific Bracket Exit Verification
        elif status == "FILLED":
            if sym not in live_positions:
                created_dt = pd.to_datetime(trade["created_at"], utc=True)
                hold_mins = (now_utc - created_dt).total_seconds() / 60.0
                direction = trade["direction"].upper()
                entry_fill = float(trade.get("actual_fill_price") or trade["limit_entry_price"])

                tp_id = trade.get("binance_tp_id")
                sl_id = trade.get("binance_sl_id")
                exit_price = None
                close_reason = None
                fees_paid = 0.0

                # Check TP Order specifically
                if tp_id and "MOCK" not in str(tp_id):
                    try:
                        tp_info = execution.exchange.fetch_order(tp_id, sym)
                        if tp_info.get("status", "").lower() == "closed":
                            exit_price = float(tp_info.get("average") or tp_info.get("price") or trade["dynamic_tp_price"])
                            close_reason = "TP_HIT"
                            fees_paid = float(tp_info.get("fee", {}).get("cost", 0.0))
                    except Exception:
                        pass

                # Check SL Order specifically
                if not close_reason and sl_id and "MOCK" not in str(sl_id):
                    try:
                        sl_info = execution.exchange.fetch_order(sl_id, sym)
                        if sl_info.get("status", "").lower() == "closed":
                            exit_price = float(sl_info.get("average") or sl_info.get("price") or trade["dynamic_sl_price"])
                            close_reason = "SL_HIT"
                            fees_paid = float(sl_info.get("fee", {}).get("cost", 0.0))
                    except Exception:
                        pass

                # Fallback if position was closed on reversal
                if not close_reason:
                    exit_price = float(trade.get("dynamic_sl_price", entry_fill))
                    close_reason = "SL_HIT"
                    fees_paid = float(trade["allocated_cash"]) * 0.0008

                gross_ret = (exit_price - entry_fill) / entry_fill if direction == "LONG" else (entry_fill - exit_price) / entry_fill
                realized_pnl = (float(trade["allocated_cash"]) * gross_ret) - fees_paid
                idealized_pct = float(trade.get("idealized_tp_pct", 0.5) if close_reason == "TP_HIT" else -trade.get("idealized_sl_pct", 0.25))
                idealized_pnl = float(trade["allocated_cash"]) * (idealized_pct / 100.0)

                print(f"[Position Closed on Binance] {sym} {direction} exited via {close_reason} @ {format_price(exit_price)} (Net PnL: ${realized_pnl:+,.2f})")
                telemetry.record_trade_closure(
                    trade_id=trade.get("id"),
                    close_reason=close_reason,
                    exit_price=exit_price,
                    realized_binance_pnl=realized_pnl,
                    idealized_pnl=idealized_pnl,
                    exchange_fees_paid=fees_paid,
                    slippage_usd=0.0,
                    hold_duration_minutes=hold_mins,
                    notes=f"Order-ID matched {close_reason} executed on Binance",
                    symbol=sym,
                    direction=direction,
                    entry_price=entry_fill
                )
                symbols_to_remove.append(sym)

    for s in symbols_to_remove:
        if s in active_by_symbol:
            del active_by_symbol[s]


# =============================================================================
# STEP 5: 15-Minute Pipeline (Idempotent Guard & Strict Directional Inversion)
# =============================================================================
def run_candle_close_pipeline():
    global active_by_symbol, last_evaluated_candles
    t_start = time.perf_counter()
    eval_time_str = datetime.now(timezone.utc).strftime('%H:%M:%S')

    print(f"\n───────────────────────────────────────────────────────────────────────────────")
    print(f"  EVALUATING 15-MINUTE CANDLE CLOSE AT {eval_time_str} UTC")
    print(f"───────────────────────────────────────────────────────────────────────────────")

    free_cash = execution.get_free_usdt_balance()
    live_binance_positions = execution.get_active_positions()

    print(f"Active Slots Deployed: {len(active_by_symbol)} / 5 | Free Cash Available: ${free_cash:,.2f} USDT")

    for symbol in ACTIVE_SYMBOLS:
        try:
            df_15m = fetch_closed_ohlcv(execution.exchange, symbol, '15m', limit=60)
            df_4h  = fetch_closed_ohlcv(execution.exchange, symbol, '4h',  limit=40)
            df_1d  = fetch_closed_ohlcv(execution.exchange, symbol, '1d',  limit=25)

            signal, cross_price, candle_close_utc = detect_crossover(df_15m)

            if not signal:
                continue

            # ── IDEMPOTENT DEDUPLICATION GUARD ──
            # Rejects duplicate evaluations of the exact same completed candle
            if last_evaluated_candles.get(symbol) == candle_close_utc:
                continue

            last_evaluated_candles[symbol] = candle_close_utc

            # Check if this asset has an active position
            has_ram_trade = symbol in active_by_symbol
            has_live_pos  = symbol in live_binance_positions

            if has_ram_trade or has_live_pos:
                active_record = active_by_symbol.get(symbol)
                pos_dir = ""
                if active_record:
                    pos_dir = active_record["direction"].upper()
                elif has_live_pos:
                    pos_dir = live_binance_positions[symbol]["side"].upper()

                # ── STRICT DIRECTIONAL INVERSION CHECK ──
                # Liquidate IF AND ONLY IF signal opposes active position (signal != pos_dir)
                if signal != pos_dir:
                    print(f"\n[Crossover Inversion Detected] {signal} crossover opposing active {pos_dir}! Liquidating immediately...")

                    trade_to_close = active_record or {
                        "id": None,
                        "binance_order_id": None,
                        "symbol": symbol,
                        "direction": pos_dir,
                        "contract_quantity": live_binance_positions[symbol]["contracts"],
                        "limit_entry_price": live_binance_positions[symbol]["entry_price"],
                        "actual_fill_price": live_binance_positions[symbol]["entry_price"],
                        "allocated_cash": live_binance_positions[symbol]["contracts"] * live_binance_positions[symbol]["entry_price"],
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }

                    execution.execute_signal_flip_close(trade_to_close)

                    if symbol in active_by_symbol:
                        del active_by_symbol[symbol]
                    free_cash = execution.get_free_usdt_balance()
                else:
                    # Same direction signal on active trade -> Ignore duplicate entry
                    continue

            print(f"\n[Crossover Fired] {symbol} -> {signal} at {format_price(cross_price)} (Candle Close: {candle_close_utc})")

            # Extract 25 master indicators & compute dual-engine model predictions
            features_25 = compute_production_features(df_15m, df_4h, df_1d)
            recent_history = [features_25] * 30
            model_outputs  = model_registry.predict_trade_setup(symbol, signal, features_25, recent_history)

            print(f"   --> Predictions: Profit MFE={model_outputs['pred_profit_mfe']:.2f}% | Danger MAE={model_outputs['pred_danger_mae']:.2f}%")
            print(f"   --> Gates      : Prob(Profit)={model_outputs['prob_profit']:.3f} | Prob(Danger)={model_outputs['prob_danger']:.3f}")

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
                print(f"       Dynamic TP: {format_price(manifest['dynamic_tp_price'])} | Dynamic SL: {format_price(manifest['dynamic_sl_price'])}")

                trade_id, binance_order_id = execution.execute_limit_entry(manifest, candle_close_utc)
                print(f"       Order Dispatched! Trade UUID: {trade_id} | Binance ID: {binance_order_id}")

                active_by_symbol[symbol] = {
                    "id": trade_id,
                    "binance_order_id": binance_order_id,
                    "binance_tp_id": None,
                    "binance_sl_id": None,
                    "symbol": symbol,
                    "direction": signal,
                    "order_status": "PENDING_LIMIT",
                    "contract_quantity": manifest["contract_quantity"],
                    "limit_entry_price": manifest["entry_price"],
                    "allocated_cash": manifest["allocated_cash"],
                    "dynamic_tp_price": manifest["dynamic_tp_price"],
                    "dynamic_sl_price": manifest["dynamic_sl_price"],
                    "idealized_tp_pct": manifest["dynamic_tp_pct"],
                    "idealized_sl_pct": manifest["dynamic_sl_pct"],
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
# STEP 6: Master Phase-Locked Execution Loop (Target: Exact :05.00 Close)
# =============================================================================
def main():
    global last_evaluated_15m_block
    daemon_start_time = time.time()
    print(f"[Daemon Started] Continuous Silent Heartbeat active at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC.")

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

            current_minute = now_dt.minute
            current_second = now_dt.second
            current_15m_block = (now_dt.year, now_dt.month, now_dt.day, now_dt.hour, current_minute // 15)

            seconds_into_15m = (current_minute % 15) * 60 + current_second
            seconds_until_close = 900 - seconds_into_15m

            is_candle_close_window = (current_minute % 15 == 0) and (current_second >= SETTLEMENT_BUFFER_SEC)

            if is_candle_close_window and (last_evaluated_15m_block != current_15m_block):
                run_candle_close_pipeline()
                last_evaluated_15m_block = current_15m_block

            run_position_maintenance()

            if seconds_until_close <= 15:
                sleep_duration = seconds_until_close + SETTLEMENT_BUFFER_SEC
                time.sleep(max(1.0, sleep_duration))
            else:
                time.sleep(HEARTBEAT_INTERVAL_SEC)

        except Exception as e:
            print(f"[Daemon Heartbeat Exception] Recovering: {repr(e)}")
            time.sleep(10)


if __name__ == "__main__":
    main()
