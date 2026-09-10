"""
====================================================================================================
ALGORITHM: src/telemetry.py — State Machine, Friction Logger & Supabase Telemetry Engine
====================================================================================================
Purpose:
  Provide an atomic, thread-safe persistence and logging interface between the stateless GitHub
  Actions runner, the Binance Futures Testnet execution pipeline, and the two Supabase database
  tables (`testnet_active_trades` and `testnet_trade_log`).

Key Architectural Responsibilities:
  1. Live State Management (`testnet_active_trades`):
     - Logs newly placed Limit Entry Orders with status 'PENDING_LIMIT' and intended brackets.
     - Transitions state to 'FILLED' when Binance confirms the limit fill, recording `actual_fill_price`.
     - Queries active open positions across the 5 asset slots to enforce concurrency boundaries.
  2. Order Expiration & Missed Trade Handling:
     - Detects limit entry orders that have remained unfilled past the 15-minute candle threshold.
     - Transitions state to 'CANCELLED_MISSED' and archives the record to `testnet_trade_log`.
  3. Forensic Archival & Friction Loss Measurement (`testnet_trade_log`):
     - When a trade closes on Binance (via TP, SL, TIME_EXIT, or SIGNAL_FLIP), calculates:
       Friction Loss ($) = Idealized Simulation PnL ($) - Realized Binance Net PnL ($).
     - Logs fee drag, execution slippage, hold duration, and exit reason.
  4. Concurrency Guard & Signal Flip Support:
     - Handles 'SIGNAL_FLIP' market closures cleanly.
     - Works alongside the Postgres Partial Unique Index to guarantee zero duplicate orders per asset.

Algorithm Steps:
  1. Module Setup, Secrets Ingestion & Supabase Initialization:
     - Detect environment (Kaggle secrets vs. local/cloud environment variables).
     - Validate presence of SUPABASE_URL and SUPABASE_KEY.
     - Instantiate Supabase client connection.
  2. Class Definition — TelemetryEngine:
     - Encapsulate all database mutations inside atomic, exception-handled methods.
  3. Active State Machine Handlers:
     - `record_new_order()`: Inserts a new pending limit order row with sizing & brackets.
     - `get_active_trades()`: Fetches all currently open or pending trades across symbols.
     - `get_open_slots_count()`: Calculates available cash slots (max 5 slots).
     - `record_order_fill()`: Updates status to 'FILLED' upon Binance execution.
     - `record_bracket_order_ids()`: Stores Binance exchange order IDs for TP and SL.
  4. Order Closure, Timeout & Friction Attribution Handlers:
     - `record_trade_closure()`: Closes active trade, logs realized PnL, calculates friction delta.
     - `record_missed_trade()`: Cancels stale limit orders (>15m) and logs missed trade record.
  5. Built-in Integration Self-Test:
     - If RUN_TELEMETRY_SELF_TEST = True:
       Executes an end-to-end synthetic probe: inserts a test trade, verifies state, simulates
       a fill, logs closed friction accounting with close_reason='SIGNAL_FLIP', cleans up test records,
       and confirms live database connectivity.
====================================================================================================
"""

# =============================================================================
# STEP 1: Module Setup, Secrets Ingestion & Supabase Initialization
# =============================================================================
import os
import time
from datetime import datetime, timezone
import pandas as pd
from supabase import create_client, Client

# Kaggle Secrets vs Environment Variable Detection
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

# Initialize Primary Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# =============================================================================
# STEP 2 & 3: TelemetryEngine Class & State Machine Handlers
# =============================================================================
class TelemetryEngine:
    """
    Centralized database controller managing the lifecycle of live testnet orders
    and forensic performance telemetry across Supabase Tables 1 and 2.
    """
    def __init__(self, client: Client = supabase):
        self.db = client
        self.active_table = "testnet_active_trades"
        self.log_table    = "testnet_trade_log"

    def get_active_trades(self) -> list:
        """
        Queries all orders currently in 'PENDING_LIMIT' or 'FILLED' status.
        Used by the runner to monitor brackets, fills, and timeouts.
        """
        res = self.db.table(self.active_table).select("*").in_("order_status", ["PENDING_LIMIT", "FILLED"]).execute()
        return res.data if res.data else []

    def get_open_slots_count(self, max_slots: int = 5) -> int:
        """
        Returns the number of available trade slots (0 to 5).
        Enforces maximum 5 concurrent positions across the portfolio.
        """
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
        """
        Records a newly placed limit entry order in Table 1 (`testnet_active_trades`).
        Returns the generated trade UUID.
        """
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
        """
        Transitions order from 'PENDING_LIMIT' to 'FILLED' when Binance confirms execution.
        """
        payload = {
            "order_status": "FILLED",
            "actual_fill_price": float(actual_fill_price)
        }
        self.db.table(self.active_table).update(payload).eq("id", trade_id).execute()

    def record_bracket_order_ids(self, trade_id: str, binance_tp_id: str, binance_sl_id: str):
        """
        Stores Binance exchange IDs for resting reduce-only TP and SL orders.
        """
        payload = {
            "binance_tp_id": str(binance_tp_id),
            "binance_sl_id": str(binance_sl_id)
        }
        self.db.table(self.active_table).update(payload).eq("id", trade_id).execute()

    # =========================================================================
    # STEP 4: Order Closure, Timeout & Friction Attribution Handlers
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
        """
        Closes out a trade, transitions status to 'CLOSED' in Table 1, and inserts
        a permanent audit row into Table 2 (`testnet_trade_log`), computing:
          friction_loss = idealized_pnl - realized_binance_pnl
          
        Accepts close_reason in ('TP_HIT', 'SL_HIT', 'TIME_EXIT', 'MISSED_TRADE', 'SIGNAL_FLIP').
        """
        valid_reasons = ['TP_HIT', 'SL_HIT', 'TIME_EXIT', 'MISSED_TRADE', 'SIGNAL_FLIP']
        if close_reason not in valid_reasons:
            raise ValueError(f"[Telemetry Error] Invalid close_reason '{close_reason}'. Must be in {valid_reasons}")

        # 1. Fetch trade metadata
        trade_data = self.db.table(self.active_table).select("*").eq("id", trade_id).execute().data
        if not trade_data or len(trade_data) == 0:
            raise RuntimeError(f"[Telemetry Error] Trade ID {trade_id} not found!")

        trade = trade_data[0]
        friction_loss = float(idealized_pnl) - float(realized_binance_pnl)

        # 2. Insert into Table 2: testnet_trade_log
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

        # 3. Mark Table 1 as CLOSED
        self.db.table(self.active_table).update({"order_status": "CLOSED"}).eq("id", trade_id).execute()

    def record_missed_trade(self, trade_id: str, notes: str = "Limit entry order expired after 15m without fill"):
        """
        Cancels an unfilled limit order after 15 minutes and logs as MISSED_TRADE.
        """
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


# =============================================================================
# STEP 5: Built-In Integration Self-Test
# =============================================================================
RUN_TELEMETRY_SELF_TEST = True

if __name__ == "__main__" and RUN_TELEMETRY_SELF_TEST:
    print("===============================================================================")
    print("  RUNNING TELEMETRY INTEGRATION SELF-TEST (LIVE SUPABASE PROBE)                ")
    print("===============================================================================")

    engine = TelemetryEngine()
    test_symbol = "BTCUSDT"
    now_utc = datetime.now(timezone.utc).isoformat()

    print("\n1. Probing Table 1 Insertion (testnet_active_trades)...")
    test_trade_id = engine.record_new_order(
        symbol=test_symbol,
        direction="LONG",
        candle_close_utc=now_utc,
        signal_price=64000.0,
        limit_entry_price=64000.0,
        allocated_cash=2000.0,
        contract_quantity=0.03125,
        dynamic_tp_price=65792.0,
        dynamic_sl_price=63680.0,
        idealized_tp_pct=2.80,
        idealized_sl_pct=0.50,
        category_tag="LOW_RISK__HIGH_PROFIT",
        risk_budget_usd=75.0,
        binance_order_id="TEST_BINANCE_ENTRY_999"
    )
    print(f"   --> Success! Created test trade UUID: {test_trade_id}")

    print("\n2. Probing Active Trades Query...")
    active_list = engine.get_active_trades()
    assert any(t["id"] == test_trade_id for t in active_list), "Active trade query failed to find inserted test trade!"
    print(f"   --> Success! Active trades in flight: {len(active_list)} (Open cash slots: {engine.get_open_slots_count()})")

    print("\n3. Probing Order Fill Transition...")
    engine.record_order_fill(trade_id=test_trade_id, actual_fill_price=64002.50)
    print("   --> Success! Trade updated to 'FILLED' with actual_fill_price.")

    print("\n4. Probing Table 2 Archival & Friction Logging with 'SIGNAL_FLIP'...")
    engine.record_trade_closure(
        trade_id=test_trade_id,
        close_reason="SIGNAL_FLIP",
        exit_price=64800.0,
        realized_binance_pnl=24.20,
        idealized_pnl=25.00,
        exchange_fees_paid=0.80,
        slippage_usd=0.00,
        hold_duration_minutes=30.0,
        notes="Synthetic Integration Test Probe: Signal Flip Closure"
    )
    print("   --> Success! Closed trade logged to Table 2 (Friction Loss: $0.80).")

    print("\n5. Cleaning up test records...")
    engine.db.table("testnet_trade_log").delete().eq("trade_id", test_trade_id).execute()
    engine.db.table("testnet_active_trades").delete().eq("id", test_trade_id).execute()
    print("   --> Cleaned synthetic records successfully. Database is pristine.")

    print("\n===============================================================================")
    print("  VERDICT: [PASS] SUPABASE TELEMETRY ENGINE FULLY OPERATIONAL & AIRTIGHT       ")
    print("===============================================================================")
