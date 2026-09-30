r"""
====================================================================================================
ALGORITHM: main.py — Version 3.0 Pure Market-Execution FSM Orchestrator (2-Speed Architecture)
====================================================================================================
Purpose:
  Institutional continuous trading daemon engineered exclusively for GitHub Actions CI/CD runners.
  Enforces the Pure Market-Execution FSM across 5 assets (BTC, ETH, SOL, DOGE, XRP):
    - Zero Conditional Orders: Never deploys resting TP or SL limit orders on the exchange.
    - 2.0-Second Fast Heartbeat: Synthetic barrier vigilance using a single 1-weight Mark Price call.
    - 15-Minute Slow Pipeline: Evaluates closed candles, runs ML inference, and routes market orders.
    - Cold-Start RAM Hydration: Single database read on startup; zero database reads during runtime.
    - Autonomous Self-Chaining: Dispatches successor workflow at minute 320 via GH_PAT (HTTP 204).

Algorithm Steps:
  Step 1: Module Setup, GitHub Actions Output Unbuffering & Environment Ingestion:
          - Extract GH_PAT, GITHUB_REPOSITORY, and Binance credentials.
  Step 2: Engine Initialization & Single-Read Cold-Start Hydration:
          - Initialize Telemetry, RAM Model Registry, Gates Engine, and Execution Gateway.
          - Hydrate the 5-row FSM state from Supabase `asset_state` once into RAM.
          - Reconcile with live Binance matching engine positions.
  Step 3: Self-Chaining Workflow Dispatcher (`dispatch_successor_workflow`):
          - Dispatches `runners.yml` at minute 320 to maintain unbroken continuous execution.
  Step 4: Fast Loop — Synthetic Barrier Vigilance (Every 2.0 Seconds):
          - Fetch real-time Mark Prices for all assets via single 1-weight call.
          - For coins in 'MAIN' or 'CONTROL': Check if Mark Price breaches Target TP or Target SL.
          - If breached: Fire immediate Market Close (`reduceOnly: True`), log terminal receipt,
            and transition state to 'AWAITING' (Flat).
  Step 5: Slow Pipeline — 15-Minute Crossover Evaluation (T+5.0s Buffer):
          - Ingest closed OHLCV across 15m (100 bars), 4h (60 bars), 1d (50 bars).
          - Detect 9/15 EMA crossover on closed candle (Index -1 vs. -2).
          - If crossover fires on active coin: Liquidate immediately (SIGNAL_FLIP).
          - Extract true 15-bar rolling sequence for Funnel GRU inference.
          - Evaluate Risk Gates (Soft-Gate Hurdle + 5-Slot Concurrency Cap).
          - Log 100% of crossover events to Supabase `crossover_telemetry_stream`.
          - If approved: Dispatch Market Entry, set Target TP/SL in RAM, transition state to MAIN/CONTROL.
  Step 6: Master Phase-Locked Loop:
          - Synchronize sleep cycle to maintain the 2.0s fast heartbeat while targeting exact :05.00s.
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
    from src.features import fetch_closed_ohlcv, detect_crossover, extract_rolling_features_history
    from src.models_engine import ProductionModelRegistry
    from src.gates_engine import ProductionGatesEngine
    from src.execution import ExecutionGateway
except ImportError:
    from telemetry import TelemetryEngine
    from features import fetch_closed_ohlcv, detect_crossover, extract_rolling_features_history
    from models_engine import ProductionModelRegistry
    from gates_engine import ProductionGatesEngine
    from execution import ExecutionGateway

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# STEP 1: Environment Ingestion & Output Formatting
# =============================================================================
MAX_RUN_DURATION_MINUTES = 320
FAST_LOOP_INTERVAL_SEC   = 2.0
SETTLEMENT_BUFFER_SEC    = 5.0

GITHUB_TOKEN      = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "").strip()
BINANCE_KEY       = os.environ.get("BINANCE_TESTNET_API_KEY", "").strip()
BINANCE_SECRET    = os.environ.get("BINANCE_TESTNET_API_SECRET", "").strip()
BINANCE_PROXY     = os.environ.get("BINANCE_PROXY_URL", "").strip()

ACTIVE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]


def format_price(price: float) -> str:
    """Formats asset price based on magnitude precision."""
    if price < 1.0:
        return f"${price:.4f}"
    elif price < 10.0:
        return f"${price:.3f}"
    else:
        return f"${price:,.2f}"


# =============================================================================
# STEP 2: Engine Initialization & Single-Read Cold-Start Hydration
# =============================================================================
print("===============================================================================")
print("  EMA_TESTNET PRODUCTION DAEMON (VERSION 3.0: PURE MARKET FSM DESK)            ")
print(f"  Max Lifespan     : {MAX_RUN_DURATION_MINUTES} Minutes ({MAX_RUN_DURATION_MINUTES/60:.2f} Hours)")
print(f"  Target Repository: {GITHUB_REPOSITORY} | Workflow: runners.yml               ")
print(f"  Execution Mode   : Pure Market Entries & Exits (Zero Conditional Orders)     ")
print(f"  Vigilance Cycle  : 2.0s Fast Mark Price Loop (Weight = 1)                    ")
print("===============================================================================\n")

print("1. Initializing Persistence Gateway...")
telemetry = TelemetryEngine()

print("2. Loading Production Models into RAM Singleton...")
model_registry = ProductionModelRegistry()
gates_engine   = ProductionGatesEngine()

print("3. Connecting Pure Market Execution Gateway to Binance Futures Testnet...")
execution = ExecutionGateway(
    api_key=BINANCE_KEY,
    api_secret=BINANCE_SECRET,
    proxy_url=BINANCE_PROXY
)

# ── 1. COLD-START HYDRATION: Single Read from Supabase Table 1 on Boot ──
print("4. Hydrating FSM State from Supabase `asset_state` (One-Time Startup Read)...")
fsm_ram_state = telemetry.hydrate_fsm_state()

# ── 2. RECONCILE WITH PHYSICAL BINANCE POSITIONS ──
live_positions = execution.get_active_positions()
for sym, pos_data in live_positions.items():
    if sym in fsm_ram_state:
        current_state = fsm_ram_state[sym]["state"]
        if current_state == "AWAITING":
            print(f"[Reconciliation] Physical position detected on Binance for {sym} while AWAITING. Tracking in RAM.")
            fsm_ram_state[sym]["state"] = "MAIN"
            fsm_ram_state[sym]["direction"] = pos_data["side"].upper()
            fsm_ram_state[sym]["entry_price"] = pos_data["entry_price"]
            fsm_ram_state[sym]["contract_quantity"] = pos_data["contracts"]
            fsm_ram_state[sym]["allocated_cash"] = pos_data["contracts"] * pos_data["entry_price"]
            # Set default 0.5% SL and 1.5% TP barrier if unrecorded
            dir_mult = 1.0 if pos_data["side"].upper() == "LONG" else -1.0
            fsm_ram_state[sym]["target_tp"] = pos_data["entry_price"] * (1.0 + (dir_mult * 0.015))
            fsm_ram_state[sym]["target_sl"] = pos_data["entry_price"] * (1.0 - (dir_mult * 0.005))
            telemetry.transition_asset_state(fsm_ram_state[sym])

active_count = sum(1 for data in fsm_ram_state.values() if data["state"] in ["MAIN", "CONTROL"])
print(f"FSM State Initialized in RAM: {active_count} / 5 slots active.\n")

last_evaluated_15m_block = None
last_evaluated_candles   = {sym: None for sym in ACTIVE_SYMBOLS}


# =============================================================================
# STEP 3: Self-Chaining Workflow Dispatcher (Targeting runners.yml)
# =============================================================================
def dispatch_successor_workflow() -> bool:
    """Dispatches next runner before GitHub Actions 6-hour execution timeout kills job."""
    if not GITHUB_TOKEN or not GITHUB_REPOSITORY:
        print("[Warning] GITHUB_TOKEN/GH_PAT missing. Relying on scheduled cron triggers.")
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/actions/workflows/runners.yml/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}"
    }
    payload = {"ref": "main"}

    print(f"\n[Self-Chaining] Dispatching successor runner to {GITHUB_REPOSITORY} via runners.yml...")
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
# STEP 4: Fast Loop — Synthetic Barrier Vigilance (Every 2.0 Seconds)
# =============================================================================
def run_fast_barrier_check():
    """
    Queries real-time Mark Prices for all assets in ONE call (Weight = 1).
    Evaluates synthetic barriers for active coins in RAM and fires market exits.
    Zero conditional orders on Binance. Zero database reads.
    """
    global fsm_ram_state

    # 1. Check if any coin is active before querying
    active_symbols = [s for s, d in fsm_ram_state.items() if d["state"] in ["MAIN", "CONTROL"]]
    if not active_symbols:
        return

    # 2. Fetch all Mark Prices via 1-weight public endpoint
    mark_prices = execution.get_all_mark_prices()
    if not mark_prices:
        return

    now_utc = datetime.now(timezone.utc)

    for sym in active_symbols:
        trade = fsm_ram_state[sym]
        current_mark = mark_prices.get(sym)
        if not current_mark or current_mark <= 0.0:
            continue

        direction = trade["direction"].upper()
        target_tp = float(trade["target_tp"])
        target_sl = float(trade["target_sl"])
        entry_px  = float(trade["entry_price"])
        qty       = float(trade["contract_quantity"])
        cash      = float(trade["allocated_cash"])
        tier      = trade.get("trade_tier", "MAIN")

        close_reason = None

        # Barrier Breach Evaluation
        if direction == "LONG":
            if current_mark >= target_tp and target_tp > 0:
                close_reason = "TP_HIT"
            elif current_mark <= target_sl and target_sl > 0:
                close_reason = "SL_HIT"
        elif direction == "SHORT":
            if current_mark <= target_tp and target_tp > 0:
                close_reason = "TP_HIT"
            elif current_mark >= target_sl and target_sl > 0:
                close_reason = "SL_HIT"

        # Execute Immediate Market Exit on Breach
        if close_reason:
            print(f"\n[Synthetic Barrier Breach] {sym} {direction} ({tier}) reached {close_reason} at {format_price(current_mark)}!")
            exit_px, real_pnl, fees = execution.execute_market_close(
                symbol=sym,
                direction=direction,
                quantity=qty,
                entry_price=entry_px,
                allocated_cash=cash,
                close_reason=close_reason
            )

            # Record Terminal Receipt in Supabase Table 2 (and reset asset_state to AWAITING)
            telemetry.record_trade_closure({
                "symbol": sym,
                "direction": direction,
                "trade_tier": tier,
                "close_reason": close_reason,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "realized_binance_pnl": real_pnl,
                "idealized_pnl": real_pnl + fees,
                "friction_loss": 0.0,
                "exchange_fees_paid": fees,
                "hold_duration_minutes": 0.0,
                "opened_at": trade.get("updated_at"),
                "notes": f"Synthetic barrier trigger ({close_reason})"
            })

            # Update In-Memory FSM State to AWAITING (Flat)
            fsm_ram_state[sym]["state"]             = "AWAITING"
            fsm_ram_state[sym]["direction"]         = "NONE"
            fsm_ram_state[sym]["entry_price"]       = 0.0
            fsm_ram_state[sym]["target_tp"]         = 0.0
            fsm_ram_state[sym]["target_sl"]         = 0.0
            fsm_ram_state[sym]["contract_quantity"] = 0.0
            fsm_ram_state[sym]["allocated_cash"]    = 0.0
            fsm_ram_state[sym]["trade_tier"]        = "NONE"
            print(f"   --> {sym} state transitioned to AWAITING. Locked until next fresh crossover.\n")


# =============================================================================
# STEP 5: Slow Pipeline — 15-Minute Crossover Evaluation (T+5.0s Buffer)
# =============================================================================
def run_candle_close_pipeline():
    """
    Evaluates finalized 15m candle closes at T+5.0s.
    Executes Signal Inversions (Reversals) and routes Market Entries.
    """
    global fsm_ram_state, last_evaluated_candles
    t_start = time.perf_counter()
    eval_time_str = datetime.now(timezone.utc).strftime('%H:%M:%S')

    print(f"\n───────────────────────────────────────────────────────────────────────────────")
    print(f"  EVALUATING 15-MINUTE CANDLE CLOSE AT {eval_time_str} UTC")
    print(f"───────────────────────────────────────────────────────────────────────────────")

    free_cash = execution.get_free_usdt_balance()
    active_count = sum(1 for d in fsm_ram_state.values() if d["state"] in ["MAIN", "CONTROL"])
    print(f"Active Slots Deployed: {active_count} / 5 | Free Cash Available: ${free_cash:,.2f} USDT")

    for symbol in ACTIVE_SYMBOLS:
        try:
            # Multi-timeframe ingestion with complete mathematical warmup
            df_15m = fetch_closed_ohlcv(execution.exchange, symbol, '15m', limit=100)
            df_4h  = fetch_closed_ohlcv(execution.exchange, symbol, '4h',  limit=60)
            df_1d  = fetch_closed_ohlcv(execution.exchange, symbol, '1d',  limit=50)

            signal, cross_price, candle_close_utc = detect_crossover(df_15m)

            if not signal:
                continue

            # Idempotent bar deduplication guard
            if last_evaluated_candles.get(symbol) == candle_close_utc:
                continue
            last_evaluated_candles[symbol] = candle_close_utc

            current_trade = fsm_ram_state[symbol]
            current_state = current_trade["state"]
            is_active     = current_state in ["MAIN", "CONTROL"]
            is_reversal   = False

            # ── 1. SIGNAL INVERSION (REVERSAL ON ACTIVE ASSET) ──
            if is_active:
                pos_dir = current_trade["direction"].upper()
                if signal != pos_dir:
                    print(f"\n[Signal Inversion] {signal} crossover fires against active {pos_dir} on {symbol}! Liquidating immediately...")
                    exit_px, real_pnl, fees = execution.execute_market_close(
                        symbol=symbol,
                        direction=pos_dir,
                        quantity=float(current_trade["contract_quantity"]),
                        entry_price=float(current_trade["entry_price"]),
                        allocated_cash=float(current_trade["allocated_cash"]),
                        close_reason="SIGNAL_FLIP"
                    )

                    telemetry.record_trade_closure({
                        "symbol": symbol,
                        "direction": pos_dir,
                        "trade_tier": current_trade.get("trade_tier", "MAIN"),
                        "close_reason": "SIGNAL_FLIP",
                        "entry_price": float(current_trade["entry_price"]),
                        "exit_price": exit_px,
                        "realized_binance_pnl": real_pnl,
                        "idealized_pnl": real_pnl + fees,
                        "friction_loss": 0.0,
                        "exchange_fees_paid": fees,
                        "hold_duration_minutes": 0.0,
                        "opened_at": current_trade.get("updated_at"),
                        "notes": "Closed on confirmed opposite 9/15 EMA crossover"
                    })

                    fsm_ram_state[symbol]["state"] = "AWAITING"
                    free_cash = execution.get_free_usdt_balance()
                    is_reversal = True
                else:
                    # Same direction crossover on an already open trade -> ignore
                    continue

            print(f"\n[Crossover Fired] {symbol} -> {signal} at {format_price(cross_price)} (Bar Close: {candle_close_utc})")

            # ── 2. TRUE ROLLING SEQUENCE EXTRACTION FOR GRU ──
            recent_history = extract_rolling_features_history(df_15m, df_4h, df_1d, seq_len=15)
            features_latest = recent_history[-1]

            # ── 3. REAL-TIME MODEL INFERENCE ──
            model_outputs = model_registry.predict_trade_setup(symbol, signal, features_latest, recent_history)
            print(f"   --> Predictions: Profit MFE={model_outputs['pred_profit_mfe']:.2f}% | Danger MAE={model_outputs['pred_danger_mae']:.2f}%")
            print(f"   --> Gates      : Prob(Profit)={model_outputs['prob_profit']:.3f} | Prob(Danger)={model_outputs['prob_danger']:.3f}")

            # ── 4. EVALUATE GATES & SIZING ──
            active_count = sum(1 for d in fsm_ram_state.values() if d["state"] in ["MAIN", "CONTROL"])
            manifest = gates_engine.evaluate_gates_and_sizing(
                symbol=symbol,
                direction=signal,
                entry_price=cross_price,
                model_outputs=model_outputs,
                atr_pct=features_latest.get('atr_pct', 0.40),
                free_wallet_balance=free_cash,
                active_positions_count=active_count,
                is_reversal=is_reversal
            )

            # ── 5. LOG 100% OF CROSSOVERS TO SUPABASE TABLE 3 ──
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
                "gate_combo_tag": manifest.get("gate_combo_tag", "NONE"),
                "rejection_reason": manifest["rejection_reason"],
                "dynamic_tp_pct": manifest["dynamic_tp_pct"],
                "dynamic_sl_pct": manifest["dynamic_sl_pct"],
                "noise_ratio": manifest.get("noise_ratio", 1.0),
                "wallet_balance_usd": free_cash
            })

            if not manifest.get("approved", True):
                print(f"   --> [REJECTED] {manifest['rejection_reason']}")
                continue

            # ── 6. EXECUTE IMMEDIATE MARKET ENTRY ON BINANCE ──
            print(f"   --> [ENTRY ORDER] Tier: {manifest['trade_tier']} | Cash: ${manifest['allocated_cash']:,.2f} | R:R: {manifest['rr_ratio']}:1")
            print(f"       Target TP: {format_price(manifest['dynamic_tp_price'])} | Target SL: {format_price(manifest['dynamic_sl_price'])}")

            order_id, actual_fill_px, clean_contracts = execution.execute_market_entry(manifest)

            # Re-derive exact barriers from authentic fill price
            tp_pct = manifest["dynamic_tp_pct"] / 100.0
            sl_pct = manifest["dynamic_sl_pct"] / 100.0
            if signal == "LONG":
                target_tp_px = actual_fill_px * (1.0 + tp_pct)
                target_sl_px = actual_fill_px * (1.0 - sl_pct)
            else:
                target_tp_px = actual_fill_px * (1.0 - tp_pct)
                target_sl_px = actual_fill_px * (1.0 + sl_pct)

            # ── 7. PERSIST FSM TRANSITION TO RAM & ASSET_STATE TABLE ──
            new_fsm_state = {
                "symbol": symbol,
                "state": manifest["trade_tier"],
                "direction": signal,
                "entry_price": actual_fill_px,
                "target_tp": target_tp_px,
                "target_sl": target_sl_px,
                "contract_quantity": clean_contracts,
                "allocated_cash": manifest["allocated_cash"],
                "trade_tier": manifest["trade_tier"]
            }

            fsm_ram_state[symbol] = new_fsm_state
            telemetry.transition_asset_state(new_fsm_state)
            free_cash = execution.get_free_usdt_balance()

        except Exception as e:
            print(f"[Pipeline Error] Failed processing {symbol}: {repr(e)}")

    elapsed_pipeline = time.perf_counter() - t_start
    print(f"15-Minute Pipeline Completed in {elapsed_pipeline:.2f}s.")


# =============================================================================
# STEP 6: Master Phase-Locked Loop (Fast Heartbeat + Slow Trigger)
# =============================================================================
def main():
    global last_evaluated_15m_block
    daemon_start_time = time.time()
    print(f"\n[Daemon Active] Two-speed execution loop operational at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC.")

    while True:
        try:
            now_dt = datetime.now(timezone.utc)
            elapsed_minutes = (time.time() - daemon_start_time) / 60.0

            # 1. Handover Lifespan Check at Minute 320 (5 Hours 20 Minutes)
            if elapsed_minutes >= MAX_RUN_DURATION_MINUTES:
                print(f"\n[Lifespan Reached] {elapsed_minutes:.1f} / {MAX_RUN_DURATION_MINUTES} Mins. Handing over to next runner...")
                success = dispatch_successor_workflow()
                if success:
                    time.sleep(10)
                    break
                else:
                    daemon_start_time += 900  # Extend lifespan 15m if dispatch had network hiccup

            # 2. Clock Resolution
            current_minute = now_dt.minute
            current_second = now_dt.second
            current_15m_block = (now_dt.year, now_dt.month, now_dt.day, now_dt.hour, current_minute // 15)

            # 3. FAST LOOP: Synthetic Barrier Vigilance (Every 2.0 Seconds)
            run_fast_barrier_check()

            # 4. SLOW PIPELINE: Execute Candle Pipeline at T+5.0s Past Close
            is_candle_close_window = (current_minute % 15 == 0) and (current_second >= SETTLEMENT_BUFFER_SEC)
            if is_candle_close_window and (last_evaluated_15m_block != current_15m_block):
                run_candle_close_pipeline()
                last_evaluated_15m_block = current_15m_block

            # 5. Fast Heartbeat Sleep
            time.sleep(FAST_LOOP_INTERVAL_SEC)

        except Exception as e:
            print(f"[Daemon Loop Notice] Recovering from error: {repr(e)}")
            time.sleep(FAST_LOOP_INTERVAL_SEC)


if __name__ == "__main__":
    main()
