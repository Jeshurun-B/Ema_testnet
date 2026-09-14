r"""
====================================================================================================
ALGORITHM: main.py — Tier-1 In-Memory Engine with Asynchronous Telemetry & Pure Crossover Exits
====================================================================================================
Purpose:
  Institutional 5.5-hour continuous trading engine. Operates with In-Memory State as the primary
  source of truth during runtime (Hot Path) with Supabase serving purely as an asynchronous write-only
  telemetry sink (Cold Path). Hydrates state from Table 1 ONCE at startup, executing all maintenance
  and 15-minute candle pipelines completely in RAM without database read latency. Enforces pure
  crossover exits and authorized self-chaining via GH_PAT.

Key Architectural Invariants:
  1. Zero Database Reads During Steady-State Trading:
     - On boot, hydrates active trades from Supabase ONCE (`telemetry.hydrate_active_trades_from_db()`).
     - Throughout the entire 5.5-hour run, the 10s maintenance loop and 15m candle pipeline execute
       100% in RAM against `active_by_symbol`. Eliminates the dead TCP socket trap and 504 timeouts.
  2. Write-Through Telemetry Sink:
     - Supabase is touched ONLY on write events (inserting an order, recording a fill, or archiving
       a trade closure). Database latency is completely decoupled from trade execution.
  3. Authorized Self-Chaining via GH_PAT:
     - Dispatches successor runner via GitHub Actions REST API at 320 minutes using the authorized
       Personal Access Token (`GH_PAT`), permanently resolving the HTTP 404 dispatch error.
  4. Phase-Locked Sleep Timing (Exact :05.00 Close):
     - When within 15 seconds of candle close, phase-locks sleep to target T+5.0s past the close.
  5. Pure Event-Driven Crossover Exits:
     - Trades exit early IF AND ONLY IF a formal opposite 9/15 EMA crossover is registered on a
       completed candle (Index [-1] vs [-2]).
  6. Dynamic Low-Notional Price Precision:
     - Formats prices dynamically: 4 decimals for <$1.00 (DOGE), 3 decimals for <$10.00 (XRP),
       and 2 decimals for >=$10.00 (BTC, ETH, SOL).

Algorithm Steps:
  Step 1: Module Setup, Dynamic Price Formatter & Output Unbuffering.
  Step 2: Engine Initialization & One-Time Startup Hydration:
          - Pre-load all 48 models into RAM once.
          - Hydrate `active_by_symbol` from Supabase once on boot. Cross-reference Binance positions.
  Step 3: Self-Chaining Dispatcher (Authorized via GH_PAT).
  Step 4: Silent In-Memory Position Maintenance (Every 10s):
          - Inspects fills and timeouts in RAM; writes updates to Supabase as background events.
  Step 5: 15-Minute Pipeline (100% In-Memory State & Pure Crossover Exits):
          - Reads active state from RAM in 1 microsecond.
          - Detects 9/15 EMA crossover on completed candle [-1] vs [-2].
          - On confirmed opposite crossover, liquidates position immediately via Market Close.
          - On valid signal: extracts 25 features, runs RAM inference, evaluates R:R >= 2.0 hurdle,
            and routes quantized limit entry or logs rejection telemetry to Table 2.
  Step 6: Master Phase-Locked Loop (Target: Exact :05.00 Close).
====================================================================================================
"""

# =============================================================================
# STEP 1: Module Setup, Price Formatter & Force Unbuffered Output
# =============================================================================
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
HEARTBEAT_INTERVAL_SEC   = 10   # Silent background tick interval
SETTLEMENT_BUFFER_SEC    = 5.0  # 5.0s settlement buffer for Binance candle aggregation

# Prefer GH_PAT for authorized workflow_dispatch, fallback to GITHUB_TOKEN
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

# ── COLD-START STATE HYDRATION (ONE-TIME READ ON BOOT) ──
print("4. Hydrating In-Memory Active Trade State from Supabase...")
active_by_symbol = telemetry.hydrate_active_trades_from_db()

# Cross-reference with live Binance positions
live_positions = execution.get_active_positions()
for sym, pos_data in live_positions.items():
    if sym not in active_by_symbol:
        print(f"[Startup Reconciliation] Live Binance position detected for {sym} ({pos_data['side'].upper()}). Tracking in RAM.")
        active_by_symbol[sym] = {
            "id": None,
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
# STEP 3: Self-Chaining Dispatcher (Authorized via GH_PAT)
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
# STEP 4: Silent In-Memory Position Maintenance (Zero DB Reads)
# =============================================================================
def run_position_maintenance():
    """
    Monitors in-flight orders silently every 10 seconds.
    HOT-PATH ENFORCEMENT: Operates 100% in RAM against `active_by_symbol`. Zero database reads!
    """
    global active_by_symbol
    
    if not active_by_symbol:
        return

    now_utc = datetime.now(timezone.utc)
    live_positions = execution.get_active_positions()
    symbols_to_remove = []

    for sym, trade in list(active_by_symbol.items()):
        status = trade.get("order_status")

        # 1. Handle PENDING_LIMIT: Check fills & deploy brackets
        if status == "PENDING_LIMIT":
            binance_id = trade.get("binance_order_id")
            if binance_id and "MOCK" not in str(binance_id):
                try:
                    order_info = execution.exchange.fetch_order(binance_id, sym)
                    if order_info.get("status", "").lower() == "closed":
                        actual_fill = float(order_info.get("average") or order_info.get("price") or trade["limit_entry_price"])
                        print(f"[Order Fill Detected] {sym} {trade['direction']} filled at {format_price(actual_fill)}. Deploying resting brackets...")
                        
                        # Update RAM state
                        trade["order_status"] = "FILLED"
                        trade["actual_fill_price"] = actual_fill

                        # Asynchronous Write-Through to Supabase Table 1
                        if trade.get("id"):
                            telemetry.record_order_fill(trade["id"], actual_fill)

                        # Deploy native brackets on Binance
                        qty = float(trade["contract_quantity"])
                        tp_px, _ = execution.quantize_order_params(sym, float(trade["dynamic_tp_price"]), qty)
                        sl_px, _ = execution.quantize_order_params(sym, float(trade["dynamic_sl_price"]), qty)
                        close_side = "sell" if trade["direction"].upper() == "LONG" else "buy"

                        tp_order = execution.exchange.create_order(
                            symbol=sym, type="TAKE_PROFIT_MARKET", side=close_side, amount=qty, params={"stopPrice": tp_px, "reduceOnly": True}
                        )
                        sl_order = execution.exchange.create_order(
                            symbol=sym, type="STOP_MARKET", side=close_side, amount=qty, params={"stopPrice": sl_px, "reduceOnly": True}
                        )
                        trade["binance_tp_id"] = str(tp_order["id"])
                        trade["binance_sl_id"] = str(sl_order["id"])

                        if trade.get("id"):
                            telemetry.record_bracket_order_ids(trade["id"], str(tp_order["id"]), str(sl_order["id"]))
                        print(f"   --> Native Brackets Deployed: TP @ {format_price(tp_px)} | SL @ {format_price(sl_px)}")
                except Exception as e:
                    pass

            # Check 15-minute wall-clock timeout
            created_str = trade.get("created_at")
            order_dt = pd.to_datetime(created_str, utc=True) if created_str else now_utc
            elapsed_min = (now_utc - order_dt).total_seconds() / 60.0

            if elapsed_min >= 15.0:
                print(f"[Timeout Triggered] {sym} limit entry unfilled after {elapsed_min:.1f}m wall-clock time. Cancelling...")
                if binance_id and "MOCK" not in str(binance_id):
                    try:
                        execution.exchange.cancel_order(binance_id, sym)
                    except Exception:
                        pass
                if trade.get("id"):
                    telemetry.record_missed_trade(trade["id"], notes=f"Limit entry expired unfilled after {elapsed_min:.1f}m")
                symbols_to_remove.append(sym)

        # 2. Handle FILLED: Check for bracket executions on Binance
        elif status == "FILLED":
            if sym not in live_positions:
                created_dt = pd.to_datetime(trade["created_at"], utc=True)
                hold_mins = (now_utc - created_dt).total_seconds() / 60.0

                try:
                    recent_trades = execution.exchange.fetch_my_trades(sym, limit=2)
                    last_trade = recent_trades[-1] if recent_trades else {}
                    exit_price = float(last_trade.get("price", trade.get("dynamic_tp_price", trade["limit_entry_price"])))
                    fees_paid = sum(float(t.get("fee", {}).get("cost", 0.0)) for t in recent_trades)
                except Exception:
                    exit_price = float(trade.get("dynamic_tp_price", trade["limit_entry_price"]))
                    fees_paid = float(trade["allocated_cash"]) * 0.0008

                direction = trade["direction"].upper()
                entry_fill = float(trade.get("actual_fill_price") or trade["limit_entry_price"])

                is_win = (exit_price > entry_fill) if direction == "LONG" else (exit_price < entry_fill)
                close_reason = "TP_HIT" if is_win else "SL_HIT"

                gross_ret = (exit_price - entry_fill) / entry_fill if direction == "LONG" else (entry_fill - exit_price) / entry_fill
                realized_pnl = (float(trade["allocated_cash"]) * gross_ret) - fees_paid
                idealized_pnl = float(trade["allocated_cash"]) * (float(trade.get("idealized_tp_pct", 0.5) if is_win else -trade.get("idealized_sl_pct", 0.25)) / 100.0)

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
                    notes=f"Resting bracket {close_reason} executed by Binance matching engine",
                    symbol=sym,
                    direction=direction,
                    entry_price=entry_fill
                )
                symbols_to_remove.append(sym)

    for s in symbols_to_remove:
        if s in active_by_symbol:
            del active_by_symbol[s]


# =============================================================================
# STEP 5: 15-Minute Pipeline (100% In-Memory State & Pure Crossover Exits)
# =============================================================================
def run_candle_close_pipeline():
    """
    Evaluates completed 15m candle close. Reads active state from RAM (0.0001ms).
    Enforces Pure Crossover Exits with asynchronous write-through telemetry.
    """
    global active_by_symbol
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

            # Detect formal 9/15 EMA Crossover on completed candle [-1] vs [-2]
            signal, cross_price, candle_close_utc = detect_crossover(df_15m)

            if not signal:
                continue

            # ── PURE CROSSOVER EXIT ENFORCEMENT ──
            has_ram_trade = symbol in active_by_symbol
            has_live_pos  = symbol in live_binance_positions

            if has_ram_trade or has_live_pos:
                active_record = active_by_symbol.get(symbol)
                pos_dir = ""
                if active_record:
                    pos_dir = active_record["direction"].upper()
                elif has_live_pos:
                    pos_dir = live_binance_positions[symbol]["side"].upper()

                # Opposite Crossover Registered -> IMMEDIATE SIGNAL FLIP CLOSE!
                if signal != pos_dir:
                    print(f"\n[Crossover Reversal Detected] Confirmed {signal} crossover opposing active {pos_dir}! Liquidating immediately...")
                    
                    trade_to_close = active_record or {
                        "id": None,
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
                    print(f"   --> {symbol} already positioned in {pos_dir}. Repeat signal ignored.")
                    continue

            print(f"\n[Crossover Fired] {symbol} -> {signal} at {format_price(cross_price)} (Candle Close: {candle_close_utc})")

            # Extract 25 master indicators & compute dual-engine model predictions in RAM
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
                print(f"       Dynamic TP: {format_price(manifest['dynamic_tp_price'])} | Dynamic SL: {format_price(manifest['dynamic_sl_price'])}")

                # Place order on Binance & write to Table 1
                trade_id = execution.execute_limit_entry(manifest, candle_close_utc)
                print(f"       Order Dispatched! Trade UUID: {trade_id}")

                # Immediate In-Memory State Registration (Hot Path)
                active_by_symbol[symbol] = {
                    "id": trade_id,
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
    """
    Master daemon loop. Evaluates clock first, runs maintenance in RAM, and
    phase-locks sleep to target T+5.0s past every 15-minute candle close.
    """
    global last_evaluated_15m_block
    daemon_start_time = time.time()
    print(f"[Daemon Started] Continuous Silent Heartbeat active at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC.")

    while True:
        try:
            now_dt = datetime.now(timezone.utc)
            elapsed_minutes = (time.time() - daemon_start_time) / 60.0

            # 1. Self-Chaining Lifespan Check at 320 Mins (Authorized via GH_PAT)
            if elapsed_minutes >= MAX_RUN_DURATION_MINUTES:
                print(f"\n[Lifespan Reached] {elapsed_minutes:.1f} / {MAX_RUN_DURATION_MINUTES} Mins elapsed. Handover initiated.")
                success = dispatch_successor_workflow()
                if success:
                    time.sleep(15)
                    break
                else:
                    daemon_start_time += 900

            # 2. Clock Discovery
            current_minute = now_dt.minute
            current_second = now_dt.second
            current_15m_block = (now_dt.year, now_dt.month, now_dt.day, now_dt.hour, current_minute // 15)

            seconds_into_15m = (current_minute % 15) * 60 + current_second
            seconds_until_close = 900 - seconds_into_15m

            # 3. CLOCK-FIRST PRIORITY: Execute candle pipeline at T+5.0s
            is_candle_close_window = (current_minute % 15 == 0) and (current_second >= SETTLEMENT_BUFFER_SEC)

            if is_candle_close_window and (last_evaluated_15m_block != current_15m_block):
                run_candle_close_pipeline()
                last_evaluated_15m_block = current_15m_block

            # 4. Silent In-Memory Maintenance (Hot Path - Zero DB Reads)
            run_position_maintenance()

            # 5. Phase-Locked Sleep: Wake up precisely at T+5.0s past the next candle close
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
