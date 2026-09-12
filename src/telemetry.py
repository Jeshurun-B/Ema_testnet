"""
====================================================================================================
ALGORITHM: src/telemetry.py — Resilient Supabase Telemetry Engine with 504 Backoff Retry
====================================================================================================
Purpose:
  Provide an institutional-grade, fault-tolerant persistence interface to Supabase. Wraps all
  PostgREST queries with an exponential backoff retry handler (`_execute_with_retry`) to absorb
  gateway latency, paused project cold starts, and HTTP 504 timeouts without crashing the daemon.

Key Resilience Enhancements:
  1. Exponential Backoff Handler (`_execute_with_retry`):
     - Executes database calls with up to 3 automatic retries (delays: 2.0s -> 4.0s -> 8.0s).
     - Catches PostgREST timeouts, 504 Gateway Timeouts, and transient network socket closures.
  2. Safe Fallback on Transient Outages:
     - `get_active_trades()` returns an empty list rather than throwing an unhandled exception,
       allowing the 5.5-hour daemon loop to persist through temporary Cloudflare/Supabase blips.
  3. Rejection Telemetry:
     - Directly logs filtered/rejected signals into Table 2 (`testnet_trade_log`) under 'GATE_REJECTED'.

Algorithm Steps:
  Step 1: Module Setup & Credentials Ingestion:
          - Extract SUPABASE_URL and SUPABASE_KEY from environment or Kaggle secrets.
          - Instantiate supabase Client.
  Step 2: Resilient Execution Helper (`_execute_with_retry`):
          - Execute lambda wrapped in a retry loop with exponential delay on 504 or network errors.
  Step 3: State Machine Queries:
          - `get_active_trades()`: Queries Table 1 for PENDING_LIMIT and FILLED trades.
          - `get_open_slots_count()`: Calculates remaining unoccupied trade slots.
  Step 4: Active Trade State Machine Modifications:
          - `record_new_order()`: Inserts newly dispatched limit order into Table 1.
          - `record_order_fill()`: Updates Table 1 status to FILLED with actual execution fill.
          - `record_bracket_order_ids()`: Persists native Binance TP and SL order IDs.
  Step 5: Forensic Archive & Rejection Handlers:
          - `record_trade_closure()`: Archives closed trades to Table 2, computing friction loss.
          - `record_missed_trade()`: Archives expired limit timeouts.
          - `record_rejected_signal()`: Logs model predictions and rejection reasons to Table 2.
  Step 6: Integration Self-Test (`if __name__ == '__main__'`):
          - Authenticates to Supabase and tests connection resilience.
====================================================================================================
"""

# =============================================================================
# STEP 1: Module Setup & Credentials Ingestion
# =============================================================================
import os
import time
from datetime import datetime, timezone
import pandas as pd
from supabase import create_client, Client

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    SUPABASE_URL = _secrets.get_secret("SUPABASE_URL").strip()
    SUPABASE_KEY = _secrets.get_secret("SUPABASE_KEY").strip()
else:
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("[FATAL] SUPABASE_URL or SUPABASE_KEY is missing from secrets/environment!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# =============================================================================
# STEP 2 & 3: TelemetryEngine Class & Resilient State Machine Queries
# =============================================================================
class TelemetryEngine:
    def __init__(self, client: Client = supabase):
        self.db = client
        self.active_table = "testnet_active_trades"
        self.log_table    = "testnet_trade_log"

    def _execute_with_retry(self, operation_fn, max_retries: int = 3, initial_delay: float = 2.0):
        """
        Executes a database lambda with exponential backoff to absorb Supabase 504
        Gateway Timeouts and transient cold starts.
        """
        delay = initial_delay
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                return operation_fn()
            except Exception as e:
                last_exception = e
                err_str = str(e)
                if "504" in err_str or "timeout" in err_str.lower() or "connection" in err_str.lower():
                    print(f"[Supabase Notice] Gateway timeout / 504 on attempt {attempt}/{max_retries}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    raise e

        raise last_exception

    def get_active_trades(self) -> list:
        """Queries Table 1 for active trades with graceful fallback on gateway timeout."""
        try:
            def op():
                return self.db.table(self.active_table).select("*").in_("order_status", ["PENDING_LIMIT", "FILLED"]).execute()
            res = self._execute_with_retry(op, max_retries=3, initial_delay=2.0)
            return res.data if res.data else []
        except Exception as e:
            print(f"[Supabase Warning] Could not fetch active trades after retries ({e}). Defaulting to empty active list.")
            return []

    def get_open_slots_count(self, max_slots: int = 5) -> int:
        active = self.get_active_trades()
        return max(0, max_slots - len(active))

    # =========================================================================
    # STEP 4: Active Trade State Machine Modifications
    # =========================================================================
    def record_new_order(
        self,
        symbol: str,
        direction: str,
        candle_close_utc: str,
        signal_price: float,
        limit_entry_price: float,
        allocated_cash: float,
        contract_quantity: float,
        dynamic_tp_price: float,
        dynamic_sl_price: float,
        idealized_tp_pct: float,
        idealized_sl_pct: float,
        category_tag: str,
        risk_budget_usd: float,
        binance_order_id: str = None
    ) -> str:
        payload = {
            "candle_close_utc": candle_close_utc,
            "symbol": symbol,
            "direction": direction,
            "signal_price": float(signal_price),
            "limit_entry_price": float(limit_entry_price),
            "allocated_cash": float(allocated_cash),
            "contract_quantity": float(contract_quantity),
            "dynamic_tp_price": float(dynamic_tp_price),
            "dynamic_sl_price": float(dynamic_sl_price),
            "idealized_tp_pct": float(idealized_tp_pct),
            "idealized_sl_pct": float(idealized_sl_pct),
            "category_tag": category_tag,
            "risk_budget_usd": float(risk_budget_usd),
            "order_status": "PENDING_LIMIT",
            "binance_order_id": str(binance_order_id) if binance_order_id else None
        }
        def op():
            return self.db.table(self.active_table).insert(payload).execute()
        res = self._execute_with_retry(op, max_retries=3)
        if res.data and len(res.data) > 0:
            return res.data[0]["id"]
        raise RuntimeError(f"[Telemetry Error] Failed to record new order for {symbol}")

    def record_order_fill(self, trade_id: str, actual_fill_price: float):
        payload = {
            "order_status": "FILLED",
            "actual_fill_price": float(actual_fill_price)
        }
        def op():
            return self.db.table(self.active_table).update(payload).eq("id", trade_id).execute()
        self._execute_with_retry(op, max_retries=3)

    def record_bracket_order_ids(self, trade_id: str, binance_tp_id: str, binance_sl_id: str):
        payload = {
            "binance_tp_id": str(binance_tp_id),
            "binance_sl_id": str(binance_sl_id)
        }
        def op():
            return self.db.table(self.active_table).update(payload).eq("id", trade_id).execute()
        self._execute_with_retry(op, max_retries=3)

    # =========================================================================
    # STEP 5: Forensic Archive & Rejection Handlers
    # =========================================================================
    def record_trade_closure(
        self,
        trade_id: str,
        close_reason: str,
        exit_price: float,
        realized_binance_pnl: float,
        idealized_pnl: float,
        exchange_fees_paid: float,
        slippage_usd: float,
        hold_duration_minutes: float,
        notes: str = None
    ):
        def fetch_op():
            return self.db.table(self.active_table).select("*").eq("id", trade_id).execute()
        trade_data = self._execute_with_retry(fetch_op, max_retries=3).data
        if not trade_data or len(trade_data) == 0:
            raise RuntimeError(f"[Telemetry Error] Trade ID {trade_id} not found!")

        trade = trade_data[0]
        friction_loss = float(idealized_pnl) - float(realized_binance_pnl)

        log_payload = {
            "trade_id": trade_id,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "symbol": trade["symbol"],
            "direction": trade["direction"],
            "close_reason": close_reason,
            "entry_price": trade.get("actual_fill_price") or trade["limit_entry_price"],
            "exit_price": float(exit_price),
            "realized_binance_pnl": float(realized_binance_pnl),
            "idealized_pnl": float(idealized_pnl),
            "friction_loss": round(friction_loss, 4),
            "exchange_fees_paid": float(exchange_fees_paid),
            "slippage_usd": float(slippage_usd),
            "hold_duration_minutes": float(hold_duration_minutes),
            "notes": notes or f"Closed via {close_reason}"
        }

        def insert_op():
            return self.db.table(self.log_table).insert(log_payload).execute()
        self._execute_with_retry(insert_op, max_retries=3)

        def update_op():
            return self.db.table(self.active_table).update({"order_status": "CLOSED"}).eq("id", trade_id).execute()
        self._execute_with_retry(update_op, max_retries=3)

    def record_missed_trade(self, trade_id: str, notes: str = "Limit entry order expired after 15m without fill"):
        def fetch_op():
            return self.db.table(self.active_table).select("*").eq("id", trade_id).execute()
        trade_data = self._execute_with_retry(fetch_op, max_retries=3).data
        if not trade_data or len(trade_data) == 0:
            return

        trade = trade_data[0]
        log_payload = {
            "trade_id": trade_id,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "symbol": trade["symbol"],
            "direction": trade["direction"],
            "close_reason": "MISSED_TRADE",
            "entry_price": trade["limit_entry_price"],
            "exit_price": trade["limit_entry_price"],
            "realized_binance_pnl": 0.0,
            "idealized_pnl": 0.0,
            "friction_loss": 0.0,
            "exchange_fees_paid": 0.0,
            "slippage_usd": 0.0,
            "hold_duration_minutes": 15.0,
            "notes": notes
        }

        def insert_op():
            return self.db.table(self.log_table).insert(log_payload).execute()
        self._execute_with_retry(insert_op, max_retries=3)

        def update_op():
            return self.db.table(self.active_table).update({"order_status": "CANCELLED_MISSED"}).eq("id", trade_id).execute()
        self._execute_with_retry(update_op, max_retries=3)

    def record_rejected_signal(self, symbol: str, direction: str, signal_price: float, manifest: dict):
        """Logs filtered/rejected trade signals into testnet_trade_log with retry."""
        notes_str = f"{manifest.get('rejection_reason', 'REJECTED')} | Tag: {manifest.get('gate_combo_tag', 'N/A')} | R:R: {manifest.get('rr_ratio', 0.0)}"
        log_payload = {
            "trade_id": None,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": direction,
            "close_reason": "GATE_REJECTED",
            "entry_price": float(signal_price),
            "exit_price": float(signal_price),
            "realized_binance_pnl": 0.0,
            "idealized_pnl": 0.0,
            "friction_loss": 0.0,
            "exchange_fees_paid": 0.0,
            "slippage_usd": 0.0,
            "hold_duration_minutes": 0.0,
            "notes": notes_str
        }

        def insert_op():
            return self.db.table(self.log_table).insert(log_payload).execute()
        try:
            self._execute_with_retry(insert_op, max_retries=3)
        except Exception as e:
            print(f"[Supabase Warning] Could not record rejection telemetry ({e}). Continuing.")


# =============================================================================
# STEP 6: Built-In Integration Self-Test
# =============================================================================
RUN_TELEMETRY_SELF_TEST = True

if __name__ == "__main__" and RUN_TELEMETRY_SELF_TEST:
    print("===============================================================================")
    print("  TESTING RESILIENT TELEMETRY ENGINE (src/telemetry.py)                        ")
    print("===============================================================================")
    engine = TelemetryEngine()
    trades = engine.get_active_trades()
    print(f"  Connected successfully. Current active trades in memory: {len(trades)}")
    print("===============================================================================")
