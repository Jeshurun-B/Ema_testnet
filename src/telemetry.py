"""
====================================================================================================
ALGORITHM: src/telemetry.py — Direct 3-Table Relational Schema Telemetry Sink
====================================================================================================
Purpose:
  Provides the persistence layer directly matching the clean 3-table Supabase schema:
    1. Table 1 (`testnet_active_trades`): Current in-flight open positions (max 5 rows).
    2. Table 2 (`testnet_trade_log`): Master closed trade execution receipts with trade_tier tags.
    3. Table 3 (`crossover_telemetry_stream`): 100% of crossover events with raw model predictions.
  Eliminates read polling during steady-state trading (hydrates once on boot).

Algorithm Steps:
  Step 1: Module Setup & Supabase Client Ingestion.
  Step 2: Resilient Execution Wrapper (`_execute_with_retry`).
  Step 3: Startup Hydration Handler (`hydrate_active_trades_from_db`).
  Step 4: Table 1 In-Flight State Handlers (`record_active_trade`, `clear_active_trade`).
  Step 5: Table 2 Terminal Trade Logging (`record_trade_closure`).
  Step 6: Table 3 Real-Time Crossover Stream Logging (`record_crossover_telemetry`).
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
    raise RuntimeError("[FATAL] Supabase credentials missing from environment!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class TelemetryEngine:
    def __init__(self, client: Client = supabase):
        self.db = client
        self.table_active = "testnet_active_trades"
        self.table_log    = "testnet_trade_log"
        self.table_stream = "crossover_telemetry_stream"

    def _execute_with_retry(self, operation_fn, max_retries: int = 3, initial_delay: float = 2.0):
        delay = initial_delay
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                return operation_fn()
            except Exception as e:
                last_exception = e
                time.sleep(delay)
                delay *= 2.0

        raise last_exception

    # ── 1. Startup Hydration (One Read on Boot) ──
    def hydrate_active_trades_from_db(self) -> dict:
        try:
            def op():
                return self.db.table(self.table_active).select("*").eq("order_status", "FILLED").execute()
            res = self._execute_with_retry(op, max_retries=3)
            rows = res.data if res.data else []
            active_cache = {r["symbol"]: r for r in rows}
            print(f"[State Hydration] Successfully hydrated {len(active_cache)} active trade(s) from Supabase Table 1.")
            return active_cache
        except Exception as e:
            print(f"[State Hydration Notice] Starting with clean in-memory state: {e}")
            return {}

    # ── 2. Table 1: In-Flight State Handlers ──
    def record_active_trade(self, trade_payload: dict) -> str:
        def op():
            return self.db.table(self.table_active).upsert(trade_payload, on_conflict="symbol").execute()
        try:
            res = self._execute_with_retry(op, max_retries=3)
            if res.data and len(res.data) > 0:
                return str(res.data[0]["id"])
        except Exception as e:
            print(f"[Telemetry Warning] Table 1 upsert notice: {e}")
        return None

    def clear_active_trade(self, symbol: str):
        def op():
            return self.db.table(self.table_active).delete().eq("symbol", symbol).execute()
        try:
            self._execute_with_retry(op, max_retries=3)
        except Exception:
            pass

    # ── 3. Table 2: Terminal Execution Receipts ──
    def record_trade_closure(self, closure_payload: dict):
        # 1. Insert into Table 2
        def insert_op():
            return self.db.table(self.table_log).insert({
                "symbol": closure_payload["symbol"],
                "direction": closure_payload["direction"],
                "trade_tier": closure_payload.get("trade_tier", "MAIN"),
                "close_reason": closure_payload["close_reason"],
                "entry_price": float(closure_payload["entry_price"]),
                "exit_price": float(closure_payload["exit_price"]),
                "realized_binance_pnl": float(closure_payload["realized_binance_pnl"]),
                "idealized_pnl": float(closure_payload.get("idealized_pnl", closure_payload["realized_binance_pnl"])),
                "friction_loss": float(closure_payload.get("friction_loss", 0.0)),
                "exchange_fees_paid": float(closure_payload["exchange_fees_paid"]),
                "hold_duration_minutes": float(closure_payload.get("hold_duration_minutes", 1.0)),
                "notes": closure_payload.get("notes", "Executed on Binance")
            }).execute()

        try:
            self._execute_with_retry(insert_op, max_retries=3)
        except Exception as e:
            print(f"[Telemetry Warning] Table 2 insert notice: {e}")

        # 2. Delete from Table 1 to free the slot
        self.clear_active_trade(closure_payload["symbol"])

    # ── 4. Table 3: 100% Crossover Sensor Stream (Feeds Your Dashboard) ──
    def record_crossover_telemetry(self, telemetry_payload: dict):
        def op():
            return self.db.table(self.table_stream).insert(telemetry_payload).execute()
        try:
            self._execute_with_retry(op, max_retries=2)
        except Exception as e:
            print(f"[Telemetry Stream Notice] Failed to log crossover telemetry: {e}")
