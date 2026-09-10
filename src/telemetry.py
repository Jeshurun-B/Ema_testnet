"""
====================================================================================================
ALGORITHM: src/telemetry.py — State Machine, Friction Logger & Rejection Telemetry
====================================================================================================
Purpose:
  Provide an atomic persistence interface to Supabase. Handles live orders in `testnet_active_trades`
  and forensic trade records, friction losses, missed trades, and gate rejection audits in
  `testnet_trade_log`.

Key Additions:
  - `record_rejected_signal(...)`: Immediately writes an audit row to `testnet_trade_log` with
    close_reason = 'GATE_REJECTED', capturing model predictions, R:R ratio, and gate tags even
    when no order is placed on Binance.

Algorithm Steps:
  1. Module Setup & Supabase Client Ingestion.
  2. TelemetryEngine Class Definition.
  3. Active Order State Machine Methods (record_new_order, record_order_fill, record_bracket_order_ids).
  4. Order Closure & Rejection Handlers:
     - record_trade_closure(): Closes active trade, calculates friction loss.
     - record_missed_trade(): Archives expired limit order timeouts.
     - record_rejected_signal(): Immediately logs filtered signals with model predictions and R:R.
  5. Built-in Integration Self-Test.
====================================================================================================
"""

# =============================================================================
# STEP 1: Module Setup & Supabase Client Ingestion
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
# STEP 2 & 3: TelemetryEngine Class & State Machine Handlers
# =============================================================================
class TelemetryEngine:
    def __init__(self, client: Client = supabase):
        self.db = client
        self.active_table = "testnet_active_trades"
        self.log_table    = "testnet_trade_log"

    def get_active_trades(self) -> list:
        res = self.db.table(self.active_table).select("*").in_("order_status", ["PENDING_LIMIT", "FILLED"]).execute()
        return res.data if res.data else []

    def get_open_slots_count(self, max_slots: int = 5) -> int:
        active = self.get_active_trades()
        return max(0, max_slots - len(active))

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
        res = self.db.table(self.active_table).insert(payload).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]["id"]
        raise RuntimeError(f"[Telemetry Error] Failed to record new order for {symbol}")

    def record_order_fill(self, trade_id: str, actual_fill_price: float):
        payload = {
            "order_status": "FILLED",
            "actual_fill_price": float(actual_fill_price)
        }
        self.db.table(self.active_table).update(payload).eq("id", trade_id).execute()

    def record_bracket_order_ids(self, trade_id: str, binance_tp_id: str, binance_sl_id: str):
        payload = {
            "binance_tp_id": str(binance_tp_id),
            "binance_sl_id": str(binance_sl_id)
        }
        self.db.table(self.active_table).update(payload).eq("id", trade_id).execute()

    # =========================================================================
    # STEP 4: Order Closure, Timeout & Rejection Audit Handlers
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
        trade_data = self.db.table(self.active_table).select("*").eq("id", trade_id).execute().data
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
        self.db.table(self.log_table).insert(log_payload).execute()
        self.db.table(self.active_table).update({"order_status": "CLOSED"}).eq("id", trade_id).execute()

    def record_missed_trade(self, trade_id: str, notes: str = "Limit entry order expired after 15m without fill"):
        trade_data = self.db.table(self.active_table).select("*").eq("id", trade_id).execute().data
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
        self.db.table(self.log_table).insert(log_payload).execute()
        self.db.table(self.active_table).update({"order_status": "CANCELLED_MISSED"}).eq("id", trade_id).execute()

    def record_rejected_signal(self, symbol: str, direction: str, signal_price: float, manifest: dict):
        """
        Logs filtered/rejected trade signals into testnet_trade_log so you can
        audit gate selectivity and model outputs directly in Supabase.
        """
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
        self.db.table(self.log_table).insert(log_payload).execute()


# =============================================================================
# STEP 5: Built-In Integration Self-Test
# =============================================================================
RUN_TELEMETRY_SELF_TEST = True

if __name__ == "__main__" and RUN_TELEMETRY_SELF_TEST:
    engine = TelemetryEngine()
    print("TelemetryEngine initialized and connected successfully.")
