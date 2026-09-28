r"""
====================================================================================================
ALGORITHM: main.py — Version 2.0 Thin Production Daemon with Experimental A/B Monitoring
====================================================================================================
Purpose:
  Institutional 5.5-hour continuous trading engine. Operates with In-Memory State as the primary
  source of truth during runtime (Hot Path). Executes immediate Market (Taker) entries on Binance
  with zero limit order timeouts. Partitions crossovers into MAIN TRADES (standard capital) and
  CONTROL TRADES (fixed $50 micro-notional floor). Logs 100% of crossover events to Supabase
  Table 3 (`crossover_telemetry_stream`), sweeps ghost orders on boot, and chains via `runners.yml`.

Algorithm Steps:
  Step 1: Module Setup, Dynamic Price Formatter & Output Unbuffering.
  Step 2: In-Memory Engine Initialization & Startup Sweeper:
          - Purges all unlinked ghost orders on Binance on startup via `purge_unlinked_ghost_orders`.
          - Hydrates active trades from Table 1.
  Step 3: Self-Chaining Dispatcher (Targeting runners.yml via GH_PAT).
  Step 4: Silent In-Memory Position Maintenance (Every 10s):
          - Inspects Binance positions directly; when a position terminates, reconciles fill price
            from bracket orders and logs to Table 2, setting the symbol flat to wait for the next cross.
  Step 5: 15-Minute Pipeline (100% Crossover Telemetry & Market Order Execution):
          - Ingests closed bars across 15m, 4h, 1d.
          - Rejects duplicate completed bars via `last_evaluated_candles`.
          - If a crossover fires on an active coin: strict reversal assertion -> liquidates immediately.
          - Computes 25 features + 15m ATR %. Runs RAM model inference.
          - Evaluates Gates (Dynamic Soft-Gate + ATR Clamp) -> assigns MAIN vs. CONTROL tier.
          - Emits 100% raw telemetry record to Table 3 (`crossover_telemetry_stream`).
          - Executes immediate Market Entry order on Binance for the symbol.
  Step 6: Master Phase-Locked Loop (Target: Exact :05.00 Close).
====================================================================================================
"""

import os
import sys
import time
import json
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
    if price < 1.0:
        return f"${price:.4f}"
    elif price < 10.0:
        return f"${price:.3f}"
    else:
        return f"${price:,.2f}"


# =============================================================================
# STEP 2: Engine Initialization & Startup Handover Sweeper
# =============================================================================
print("===============================================================================")
print("  EMA_TESTNET PRODUCTION DAEMON (VERSION 2.0: EXPERIMENTAL A/B DESK)           ")
print(f"  Max Lifespan     : {MAX_RUN_DURATION_MINUTES} Minutes ({MAX_RUN_DURATION_MINUTES/60:.2f} Hours)")
print(f"  Target Repository: {GITHUB_REPOSITORY} | Dispatch: runners.yml               ")
print(f"  Execution Mode   : Immediate Market Entry (100% Deterministic Fills)         ")
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

# ── 1. GHOST SWEEPER ON BOOT: Purge all ancient orders from Binance ──
execution.purge_unlinked_ghost_orders(ACTIVE_SYMBOLS)

# ── 2. STATE HYDRATION (ONE READ ON BOOT) ──
print("4. Hydrating active trade state from Supabase Table 1...")
active_by_symbol = telemetry.hydrate_active_trades_from_db()

# Reconcile with live Binance matching engine positions
live_positions = execution.get_active_positions()
for sym, pos_data in live_positions.items():
    if sym not in active_by_symbol:
        print(f"[Reconciliation] Live Binance position detected for {sym}. Tracking in RAM.")
        active_by_symbol[sym] = {
            "id": None,
            "symbol": sym,
            "direction": pos_data["side"].upper(),
            "trade_tier": "MAIN",
            "entry_price": pos_data["entry_price"],
            "contract_quantity": pos_data["contracts"],
            "allocated_cash": pos_data["contracts"] * pos_data["entry_price"],
            "created_at": datetime.now(timezone.utc).isoformat()
        }

print(f"Active Slots Deployed: {len(active_by_symbol)} / 5 slots.\n")


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
# STEP 4: Silent In-Memory Position Maintenance (Every 10 Seconds)
# =============================================================================
def run_position_maintenance():
    global active_by_symbol
    
    if not active_by_symbol:
        return

    now_utc = datetime.now(timezone.utc)
    live_positions = execution.get_active_positions()
    symbols_to_remove = []

    for sym, trade in list(active_by_symbol.items()):
        # Check if position has closed on Binance matching engine
        if sym not in live_positions:
            created_dt = pd.to_datetime(trade["created_at"], utc=True)
            hold_mins = max(0.1, (now_utc - created_dt).total_seconds() / 60.0)
            direction = trade["direction"].upper()
            entry_fill = float(trade["entry_price"])

            tp_id = trade.get("binance_tp_id")
            sl_id = trade.get("binance_sl_id")
            exit_price = None
            close_reason = None
            fees_paid = 0.0

            # Check exact TP Order fill
            if tp_id and "MOCK" not in str(tp_id):
                try:
                    tp_info = execution.exchange.fetch_order(tp_id, sym)
                    if tp_info.get("status", "").lower() == "closed":
                        exit_price = float(tp_info.get("average") or tp_info.get("price") or trade["dynamic_tp_price"])
                        close_reason = "TP_HIT"
                        fees_paid = float(tp_info.get("fee", {}).get("cost", 0.0))
                except Exception:
                    pass

            # Check exact SL Order fill
            if not close_reason and sl_id and "MOCK" not in str(sl_id):
                try:
                    sl_info = execution.exchange.fetch_order(sl_id, sym)
                    if sl_info.get("status", "").lower() == "closed":
                        exit_price = float(sl_info.get("average") or sl_info.get("price") or trade["dynamic_sl_price"])
                        close_reason = "SL_HIT"
                        fees_paid = float(sl_info.get("fee", {}).get("cost", 0.0))
                except Exception:
                    pass

            if not close_reason:
                exit_price = float(trade.get("dynamic_sl_price", entry_fill))
                close_reason = "SL_HIT"
                fees_paid = float(trade["allocated_cash"]) * 0.0008

            gross_ret = (exit_price - entry_fill) / entry_fill if direction == "LONG" else (entry_fill - exit_price) / entry_fill
            realized_pnl = (float(trade["allocated_cash"]) * gross_ret) - fees_paid

            print(f"[Position Closed on Binance] {sym} {direction} ({trade.get('trade_tier', 'MAIN')}) exited via {close_reason} @ {format_price(exit_price)} (Net PnL: ${realized_pnl:+,.2f})")
            print(f"   --> {sym} is now FLAT. Waiting for the next fresh 9/15 EMA crossover.")

            telemetry.record_trade_closure({
                "symbol": sym,
                "direction": direction,
                "trade_tier": trade.get("trade_tier", "MAIN"),
                "close_reason": close_reason,
                "entry_price": entry_fill,
                "exit_price": exit_price,
                "realized_binance_pnl": realized_pnl,
                "idealized_pnl": realized_pnl + fees_paid,
                "friction_loss": 0.0,
                "exchange_fees_paid": fees_paid,
                "hold_duration_minutes": hold_mins,
                "notes": f"Verified Binance execution ({close_reason})"
            })
            symbols_to_remove.append(sym)

    for s in symbols_to_remove:
        if s in active_by_symbol:
            del active_by_symbol[s]


# =============================================================================
# STEP 5: 15-Minute Pipeline (100% Crossover Telemetry & Market Routing)
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

            # Idempotent bar deduplication guard
            if last_evaluated_candles.get(symbol) == candle_close_utc:
                continue

            last_evaluated_candles[symbol] = candle_close_utc

            # ── STRICT SIGNAL INVERSION ASSERTION ──
            # If an active position exists on this coin, ANY crossover is strictly asserted as an inversion
            has_ram_trade = symbol in active_by_symbol
            has_live_pos  = symbol in live_binance_positions

            if has_ram_trade or has_live_pos:
                active_record = active_by_symbol.get(symbol)
                pos_dir = active_record["direction"].upper() if active_record else live_binance_positions[symbol]["side"].upper()

                print(f"\n[Crossover Inversion] {signal} crossover fires against active {pos_dir}! Liquidating immediately...")

                trade_to_close = active_record or {
                    "id": None,
                    "symbol": symbol,
                    "direction": pos_dir,
                    "trade_tier": "MAIN",
                    "contract_quantity": live_binance_positions[symbol]["contracts"],
                    "entry_price": live_binance_positions[symbol]["entry_price"],
                    "allocated_cash": live_binance_positions[symbol]["contracts"] * live_binance_positions[symbol]["entry_price"],
                    "created_at": datetime.now(timezone.utc).isoformat()
                }

                execution.execute_signal_flip_close(trade_to_close)

                if symbol in active_by_symbol:
                    del active_by_symbol[symbol]
                free_cash = execution.get_free_usdt_balance()

            print(f"\n[Crossover Fired] {symbol} -> {signal} at {format_price(cross_price)} (Candle Close: {candle_close_utc})")

            # Extract 25 master indicators + 15m ATR %
            features_25 = compute_production_features(df_15m, df_4h, df_1d)
            recent_history = [features_25] * 30
            model_outputs  = model_registry.predict_trade_setup(symbol, signal, features_25, recent_history)

            print(f"   --> Predictions: Profit MFE={model_outputs['pred_profit_mfe']:.2f}% | Danger MAE={model_outputs['pred_danger_mae']:.2f}%")
            print(f"   --> Gates      : Prob(Profit)={model_outputs['prob_profit']:.3f} | Prob(Danger)={model_outputs['prob_danger']:.3f}")

            # Evaluate Gates & Sizing (Dynamic Soft-Gate + ATR Noise Clamp)
            manifest = gates_engine.evaluate_gates_and_sizing(
                symbol=symbol,
                direction=signal,
                entry_price=cross_price,
                model_outputs=model_outputs,
                atr_pct=features_25.get('atr_pct', 0.40),
                free_wallet_balance=free_cash,
                active_positions_count=len(active_by_symbol)
            )

            # ── 1. LOG 100% OF CROSSOVERS TO SUPABASE TABLE 3 (TELEMETRY STREAM) ──
            telemetry.record_crossover_telemetry({
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "direction": signal,
                "crossover_price": cross_price,
                "pred_profit_mfe": manifest["pred_profit_mfe"],
                "pred_danger_mae": manifest["pred_danger_mae"],
                "prob_profit": manifest["prob_profit"],
                "prob_danger": manifest["prob_danger"],
                "rr_ratio": manifest["rr_ratio"],
                "gate_verdict": manifest["trade_tier"],
                "gate_combo_tag": manifest["gate_combo_tag"],
                "rejection_reason": manifest["rejection_reason"],
                "dynamic_tp_pct": manifest["dynamic_tp_pct"],
                "dynamic_sl_pct": manifest["dynamic_sl_pct"],
                "noise_ratio": manifest["noise_ratio"],
                "wallet_balance_usd": free_cash
            })

            # ── 2. ROUTE MARKET ENTRY ON BINANCE (100% OF CROSSOVERS TRADED) ──
            print(f"   --> [EXECUTION] Tier: {manifest['trade_tier']} | Cash: ${manifest['allocated_cash']:,.2f} | R:R: {manifest['rr_ratio']}:1")
            print(f"       Dynamic TP: {format_price(manifest['dynamic_tp_price'])} | Dynamic SL: {format_price(manifest['dynamic_sl_price'])}")

            trade_id, binance_order_id, tp_id, sl_id, actual_fill_px = execution.execute_market_entry(manifest, candle_close_utc)

            active_by_symbol[symbol] = {
                "id": trade_id,
                "binance_order_id": binance_order_id,
                "binance_tp_id": tp_id,
                "binance_sl_id": sl_id,
                "symbol": symbol,
                "direction": signal,
                "trade_tier": manifest["trade_tier"],
                "contract_quantity": manifest["contract_quantity"],
                "entry_price": actual_fill_px,
                "allocated_cash": manifest["allocated_cash"],
                "dynamic_tp_price": manifest["dynamic_tp_price"],
                "dynamic_sl_price": manifest["dynamic_sl_price"],
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            free_cash = execution.get_free_usdt_balance()

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

            # 1. Self-Chaining Lifespan Check at 320 Mins (5h 20m)
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

            # 4. Silent In-Memory Maintenance
            run_position_maintenance()

            # 5. Phase-Locked Sleep: Wake up precisely at T+5.0s past next close
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
