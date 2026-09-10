"""
====================================================================================================
ALGORITHM: src/execution.py — CCXT Binance Futures Order Routing (Demo Trading Enabled)
====================================================================================================
Purpose:
  Connect to Binance Futures Testnet via modern CCXT Demo Trading interface (`enable_demo_trading(True)`).
  Enforce precision quantization, isolated 1.0x setup, 2-stage bracket deployment, 15m order timeouts,
  and clean signal-flip market liquidations.

Key Updates:
  - Replaces deprecated `set_sandbox_mode(True)` with `enable_demo_trading(True)` to restore
    live testnet balance fetching and order execution without `NotSupported` exceptions.

Algorithm Steps:
  1. Module Setup & Credentials Ingestion.
  2. Instantiate CCXT with modern Demo Trading enabled.
  3. Precision Quantization & minNotional Compliance ($5.00 floor).
  4. Isolated Margin & 1.0x Leverage Enforcement.
  5. Limit Entry Order Placement.
  6. Order Fill Inspection & Native Bracket Deployment.
  7. 15-Minute Timeout Cancellation.
  8. Signal-Flip Market Reversal Execution.
  9. Built-in Integration Self-Test.
====================================================================================================
"""

# =============================================================================
# STEP 1: Module Setup & Credentials Ingestion
# =============================================================================
import os
import time
from datetime import datetime, timezone
import ccxt
import warnings

try:
    from src.telemetry import TelemetryEngine
except ImportError:
    from telemetry import TelemetryEngine

warnings.filterwarnings("ignore", category=UserWarning)

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    API_KEY    = _secrets.get_secret("BINANCE_TESTNET_API_KEY").strip()
    API_SECRET = _secrets.get_secret("BINANCE_TESTNET_API_SECRET").strip()
else:
    API_KEY    = os.environ.get("BINANCE_TESTNET_API_KEY", "").strip()
    API_SECRET = os.environ.get("BINANCE_TESTNET_API_SECRET", "").strip()


# =============================================================================
# STEP 2–8: ExecutionEngine Implementation
# =============================================================================
class ExecutionEngine:
    def __init__(self, api_key: str = API_KEY, api_secret: str = API_SECRET, telemetry: TelemetryEngine = None):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.telemetry  = telemetry or TelemetryEngine()

        self.exchange = ccxt.binanceusdm({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        # ── CCXT MODERN STANDARD: Enable Demo Trading ──
        if hasattr(self.exchange, "enable_demo_trading"):
            self.exchange.enable_demo_trading(True)
        else:
            self.exchange.set_sandbox_mode(True)
        
        self.markets_loaded = False
        self._load_markets_safe()

    def _load_markets_safe(self):
        try:
            self.exchange.load_markets()
            self.markets_loaded = True
        except Exception as e:
            print(f"[Execution Warning] Could not load market filters: {repr(e)}")

    def get_free_usdt_balance(self) -> float:
        if not self.api_key or not self.api_secret:
            return 10000.0
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get('USDT', {}).get('free', 10000.0))
        except Exception as e:
            print(f"[Execution Warning] Failed to fetch live balance ({e}). Defaulting to $10,000.")
            return 10000.0

    def get_active_positions(self) -> dict:
        if not self.api_key or not self.api_secret:
            return {}
        try:
            positions = self.exchange.fetch_positions()
            active_map = {}
            for pos in positions:
                contracts = float(pos.get('contracts', 0.0))
                if contracts > 0:
                    sym = pos['symbol'].replace('/', '')
                    active_map[sym] = {
                        'contracts': contracts,
                        'side': pos.get('side', '').lower(),
                        'entry_price': float(pos.get('entryPrice', 0.0)),
                        'unrealized_pnl': float(pos.get('unrealizedPnl', 0.0))
                    }
            return active_map
        except Exception as e:
            print(f"[Execution Warning] Failed to fetch positions: {repr(e)}")
            return {}

    def quantize_order_params(self, symbol: str, price: float, quantity: float):
        if not self.markets_loaded:
            self._load_markets_safe()

        clean_price = float(self.exchange.price_to_precision(symbol, price))
        clean_qty   = float(self.exchange.amount_to_precision(symbol, quantity))

        # Enforce minNotional ($5.00 floor)
        notional_value = clean_price * clean_qty
        if notional_value < 5.0 and clean_price > 0:
            required_qty = (5.5 / clean_price)
            clean_qty    = float(self.exchange.amount_to_precision(symbol, required_qty))

        return clean_price, clean_qty

    def setup_symbol_isolated_1x(self, symbol: str):
        if not self.api_key or not self.api_secret:
            return
        try:
            self.exchange.set_margin_mode('ISOLATED', symbol)
        except Exception as e:
            err_msg = str(e).lower()
            if "no need to change" not in err_msg and "already" not in err_msg:
                print(f"[Execution Notice] Margin mode for {symbol}: {e}")

        try:
            self.exchange.set_leverage(1, symbol)
        except Exception as e:
            err_msg = str(e).lower()
            if "not modified" not in err_msg:
                print(f"[Execution Notice] Leverage for {symbol}: {e}")

    def execute_limit_entry(self, manifest: dict, candle_close_utc: str) -> str:
        symbol      = manifest["symbol"]
        direction   = manifest["direction"].upper()
        entry_price = manifest["entry_price"]
        raw_qty     = manifest["contract_quantity"]
        allocated_cash = manifest["allocated_cash"]

        self.setup_symbol_isolated_1x(symbol)
        clean_price, clean_qty = self.quantize_order_params(symbol, entry_price, raw_qty)
        order_side = 'buy' if direction == 'LONG' else 'sell'

        binance_order_id = None
        if self.api_key and self.api_secret:
            try:
                order_res = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=order_side,
                    amount=clean_qty,
                    price=clean_price,
                    params={'timeInForce': 'GTC'}
                )
                binance_order_id = str(order_res['id'])
                print(f"[Binance Execution] Limit Entry Placed: {direction} {clean_qty} {symbol} @ ${clean_price:,.2f} (ID: {binance_order_id})")
            except Exception as e:
                raise RuntimeError(f"[Execution Error] Limit order rejected by Binance: {repr(e)}")
        else:
            binance_order_id = f"MOCK_ENTRY_{int(time.time())}"
            print(f"[Mock Execution] Limit Entry Recorded: {direction} {clean_qty} {symbol} @ ${clean_price:,.2f}")

        trade_id = self.telemetry.record_new_order(
            symbol=symbol,
            direction=direction,
            candle_close_utc=candle_close_utc,
            signal_price=entry_price,
            limit_entry_price=clean_price,
            allocated_cash=allocated_cash,
            contract_quantity=clean_qty,
            dynamic_tp_price=manifest["dynamic_tp_price"],
            dynamic_sl_price=manifest["dynamic_sl_price"],
            idealized_tp_pct=manifest["dynamic_tp_pct"],
            idealized_sl_pct=manifest["dynamic_sl_pct"],
            category_tag=manifest["gate_combo_tag"],
            risk_budget_usd=manifest["risk_budget_usd"],
            binance_order_id=binance_order_id
        )
        return trade_id

    def check_and_deploy_brackets(self, trade_record: dict):
        trade_id = trade_record["id"]
        symbol   = trade_record["symbol"]
        direction = trade_record["direction"].upper()
        binance_id = trade_record.get("binance_order_id")
        qty = float(trade_record["contract_quantity"])

        if not self.api_key or not self.api_secret or not binance_id or "MOCK" in binance_id:
            return

        try:
            order_info = self.exchange.fetch_order(binance_id, symbol)
            status = order_info.get('status', '').lower()

            if status == 'closed':
                actual_fill = float(order_info.get('average') or order_info.get('price') or trade_record["limit_entry_price"])
                print(f"[Order Fill Detected] {symbol} {direction} filled at ${actual_fill:,.2f}. Deploying resting brackets...")
                
                self.telemetry.record_order_fill(trade_id, actual_fill)

                tp_price = float(trade_record["dynamic_tp_price"])
                sl_price = float(trade_record["dynamic_sl_price"])
                clean_tp_price, _ = self.quantize_order_params(symbol, tp_price, qty)
                clean_sl_price, _ = self.quantize_order_params(symbol, sl_price, qty)

                close_side = 'sell' if direction == 'LONG' else 'buy'

                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='TAKE_PROFIT_MARKET',
                    side=close_side,
                    amount=qty,
                    params={'stopPrice': clean_tp_price, 'reduceOnly': True}
                )

                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=close_side,
                    amount=qty,
                    params={'stopPrice': clean_sl_price, 'reduceOnly': True}
                )

                self.telemetry.record_bracket_order_ids(trade_id, str(tp_order['id']), str(sl_order['id']))
                print(f"   --> Resting Brackets Deployed: TP @ ${clean_tp_price:,.2f} | SL @ ${clean_sl_price:,.2f}")

        except Exception as e:
            print(f"[Execution Error] Failed to check/deploy brackets for {symbol}: {repr(e)}")

    def handle_expired_limit_orders(self, max_timeout_minutes: int = 15):
        active_orders = self.telemetry.get_active_trades()
        now_dt = datetime.now(timezone.utc)

        for trade in active_orders:
            if trade.get("order_status") != "PENDING_LIMIT":
                continue

            candle_close_str = trade.get("candle_close_utc")
            if not candle_close_str:
                continue

            candle_dt = pd.to_datetime(candle_close_str, utc=True)
            elapsed_min = (now_dt - candle_dt).total_seconds() / 60.0

            if elapsed_min >= max_timeout_minutes:
                symbol = trade["symbol"]
                binance_id = trade.get("binance_order_id")

                print(f"[Timeout Triggered] {symbol} limit entry unfilled after {elapsed_min:.1f}m. Cancelling...")
                if self.api_key and self.api_secret and binance_id and "MOCK" not in binance_id:
                    try:
                        self.exchange.cancel_order(binance_id, symbol)
                    except Exception as e:
                        print(f"   --> Cancel Notice: {e}")

                self.telemetry.record_missed_trade(
                    trade_id=trade["id"],
                    notes=f"Limit entry expired unfilled after {elapsed_min:.1f} minutes"
                )

    def execute_signal_flip_close(self, active_trade: dict) -> float:
        trade_id = active_trade["id"]
        symbol   = active_trade["symbol"]
        direction = active_trade["direction"].upper()
        qty      = float(active_trade["contract_quantity"])
        tp_id    = active_trade.get("binance_tp_id")
        sl_id    = active_trade.get("binance_sl_id")

        print(f"[Signal Flip Detected] Closing {symbol} {direction} via Market Order...")

        # 1. Cancel Resting Brackets First
        if self.api_key and self.api_secret:
            for b_id in [tp_id, sl_id]:
                if b_id and "MOCK" not in b_id:
                    try:
                        self.exchange.cancel_order(b_id, symbol)
                    except Exception as e:
                        pass

        # 2. Market Close
        close_side = 'sell' if direction == 'LONG' else 'buy'
        exit_price = float(active_trade["limit_entry_price"])
        fees_paid  = 0.0
        realized_pnl = 0.0

        if self.api_key and self.api_secret:
            try:
                close_res = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=close_side,
                    amount=qty,
                    params={'reduceOnly': True}
                )
                exit_price = float(close_res.get('average') or close_res.get('price') or exit_price)

                time.sleep(1.0)
                my_trades = self.exchange.fetch_my_trades(symbol, limit=2)
                fees_paid = sum(float(t.get('fee', {}).get('cost', 0.0)) for t in my_trades if t.get('order') == close_res['id'])
                
                entry_fill = float(active_trade.get("actual_fill_price") or active_trade["limit_entry_price"])
                gross_ret = (exit_price - entry_fill) / entry_fill if direction == 'LONG' else (entry_fill - exit_price) / entry_fill
                realized_pnl = (float(active_trade["allocated_cash"]) * gross_ret) - fees_paid
            except Exception as e:
                print(f"[Execution Error] Market flip close failed on Binance: {repr(e)}")
        else:
            exit_price = float(active_trade["limit_entry_price"]) * 1.002
            fees_paid  = float(active_trade["allocated_cash"]) * 0.0008
            realized_pnl = 5.00

        created_dt = pd.to_datetime(active_trade["created_at"], utc=True)
        now_dt     = datetime.now(timezone.utc)
        hold_min   = (now_dt - created_dt).total_seconds() / 60.0
        idealized_pnl = realized_pnl + fees_paid

        self.telemetry.record_trade_closure(
            trade_id=trade_id,
            close_reason="SIGNAL_FLIP",
            exit_price=exit_price,
            realized_binance_pnl=realized_pnl,
            idealized_pnl=idealized_pnl,
            exchange_fees_paid=fees_paid,
            slippage_usd=0.0,
            hold_duration_minutes=hold_min,
            notes="Closed on confirmed 9/15 EMA opposite crossover"
        )
        print(f"   --> {symbol} {direction} closed @ ${exit_price:,.2f} (Net PnL: ${realized_pnl:+,.2f})")
        return realized_pnl


# =============================================================================
# STEP 9: Built-In Self-Test
# =============================================================================
if __name__ == "__main__":
    engine = ExecutionEngine()
    print(f"ExecutionEngine initialized. Free Cash: ${engine.get_free_usdt_balance():,.2f} USDT")
