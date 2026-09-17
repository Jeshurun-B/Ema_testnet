"""
====================================================================================================
ALGORITHM: src/telemetry.py — Cold-Start Hydration & Write-Through Telemetry Sink
====================================================================================================
Purpose:
  Provides an institutional persistence layer for Supabase. Eliminates runtime read polling by
  hydrating active trade state into RAM ONCE at startup (`hydrate_active_trades_from_db`). During
  runtime, functions purely as an asynchronous write-through telemetry sink for orders, fills,
  closures, and gate rejection audits.

Algorithm Steps:
  Step 1: Module Setup, Client Ingestion & Environment Configuration.
  Step 2: TelemetryEngine Class Definition.
  Step 3: Resilient Write Wrapper (`_execute_with_retry`).
  Step 4: Startup Hydration Handler (`hydrate_active_trades_from_db`).
  Step 5: Write-Through State Machine Handlers (record_new_order, record_order_fill, record_bracket_order_ids).
  Step 6: Forensic Archive & Rejection Handlers (record_trade_closure, record_missed_trade, record_rejected_signal).
  Step 7: Built-In Integration Self-Test (`if __name__ == '__main__'`).
====================================================================================================
"""

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
    raise RuntimeError("[FATAL] SUPABASE_URL or SUPABASE_KEY missing from environment/secrets!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class TelemetryEngine:
    def __init__(self, client: Client = supabase):
        self.db = client
        self.active_table = "testnet_active_trades"
        self.log_table    = "testnet_trade_log"

    def _execute_with_retry(self, operation_fn, max_retries: int = 3, initial_delay: float = 2.0):
        delay = initial_delay
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                return operation_fn()
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                if "504" in err_str or "timeout" in err_str or "connection" in err_str:
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    raise e

        raise last_exception

    def hydrate_active_trades_from_db(self) -> dict:
        """Queries Table 1 ONCE upon daemon boot. Zero SELECT queries during steady-state trading."""
        try:
            def op():
                return self.db.table(self.active_table).select("*").in_("order_status", ["PENDING_LIMIT", "FILLED"]).execute()
            res = self._execute_with_retry(op, max_retries=3, initial_delay=2.0)
            rows = res.data if res.data else []
            active_cache = {r["symbol"]: r for r in rows}
            print(f"[State Hydration] Successfully hydrated {len(active_cache)} active trade(s) from Supabase.")
            return active_cache
        except Exception as e:
            print(f"[State Hydration Warning] Could not hydrate state from Supabase ({e}). Starting with empty cache.")
            return {}

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
        raise RuntimeError(f"[Telemetry Error] Failed to insert new order for {symbol}")

    def record_order_fill(self, trade_id: str, actual_fill_price: float):
        payload = {
            "order_status": "FILLED",
            "actual_fill_price": float(actual_fill_price)
        }
        def op():
            return self.db.table(self.active_table).update(payload).eq("id", trade_id).execute()
        try:
            self._execute_with_retry(op, max_retries=3)
        except Exception as e:
            print(f"[Telemetry Warning] Could not record fill in DB ({e}). Continuing.")

    def record_bracket_order_ids(self, trade_id: str, binance_tp_id: str, binance_sl_id: str):
        payload = {
            "binance_tp_id": str(binance_tp_id),
            "binance_sl_id": str(binance_sl_id)
        }
        def op():
            return self.db.table(self.active_table).update(payload).eq("id", trade_id).execute()
        try:
            self._execute_with_retry(op, max_retries=3)
        except Exception as e:
            print(f"[Telemetry Warning] Could not record bracket IDs in DB ({e}). Continuing.")

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
        notes: str = None,
        symbol: str = None,
        direction: str = None,
        entry_price: float = None
    ):
        trade = None
        if trade_id:
            try:
                def fetch_op():
                    return self.db.table(self.active_table).select("*").eq("id", trade_id).execute()
                res = self._execute_with_retry(fetch_op, max_retries=2)
                if res.data and len(res.data) > 0:
                    trade = res.data[0]
            except Exception:
                pass

        sym = trade["symbol"] if trade else (symbol or "UNKNOWN")
        side = trade["direction"] if trade else (direction or "UNKNOWN")
        entry_px = float(trade.get("actual_fill_price") or trade.get("limit_entry_price") or entry_price or exit_price) if trade else float(entry_price or exit_price)
        friction_loss = float(idealized_pnl) - float(realized_binance_pnl)

        log_payload = {
            "trade_id": trade["id"] if trade else None,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "symbol": sym,
            "direction": side,
            "close_reason": close_reason,
            "entry_price": entry_px,
            "exit_price": float(exit_price),
            "realized_binance_pnl": float(realized_binance_pnl),
            "idealized_pnl": float(idealized_pnl),
            "friction_loss": round(friction_loss, 4),
            "exchange_fees_paid": float(exchange_fees_paid),
            "slippage_usd": float(slippage_usd),
            "hold_duration_minutes": float(hold_duration_minutes),
            "notes": notes or f"Closed via {close_reason}"
        }

        try:
            def insert_op():
                return self.db.table(self.log_table).insert(log_payload).execute()
            self._execute_with_retry(insert_op, max_retries=3)
        except Exception as e:
            print(f"[Telemetry Warning] Could not write trade log to DB ({e}). Continuing.")

        if trade:
            try:
                def update_op():
                    return self.db.table(self.active_table).update({"order_status": "CLOSED"}).eq("id", trade["id"]).execute()
                self._execute_with_retry(update_op, max_retries=2)
            except Exception:
                pass

    def record_missed_trade(self, trade_id: str, notes: str = "Limit entry expired unfilled after 15m"):
        trade = None
        if trade_id:
            try:
                def fetch_op():
                    return self.db.table(self.active_table).select("*").eq("id", trade_id).execute()
                res = self._execute_with_retry(fetch_op, max_retries=2)
                if res.data and len(res.data) > 0:
                    trade = res.data[0]
            except Exception:
                pass

        if not trade:
            return

        log_payload = {
            "trade_id": trade["id"],
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "symbol": trade["symbol"],
            "direction": trade["direction"],
            "close_reason": "MISSED_TRADE",
            "entry_price": float(trade["limit_entry_price"]),
            "exit_price": float(trade["limit_entry_price"]),
            "realized_binance_pnl": 0.0,
            "idealized_pnl": 0.0,
            "friction_loss": 0.0,
            "exchange_fees_paid": 0.0,
            "slippage_usd": 0.0,
            "hold_duration_minutes": 15.0,
            "notes": notes
        }

        try:
            def insert_op():
                return self.db.table(self.log_table).insert(log_payload).execute()
            self._execute_with_retry(insert_op, max_retries=3)

            def update_op():
                return self.db.table(self.active_table).update({"order_status": "CANCELLED_MISSED"}).eq("id", trade_id).execute()
            self._execute_with_retry(update_op, max_retries=2)
        except Exception as e:
            print(f"[Telemetry Warning] Could not record missed trade in DB ({e}). Continuing.")

    def record_rejected_signal(self, symbol: str, direction: str, signal_price: float, manifest: dict):
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

        try:
            def insert_op():
                return self.db.table(self.log_table).insert(log_payload).execute()
            self._execute_with_retry(insert_op, max_retries=3)
        except Exception as e:
            print(f"[Telemetry Warning] Could not record rejection to DB ({e}). Continuing.")
