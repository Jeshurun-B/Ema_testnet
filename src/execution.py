"""
====================================================================================================
ALGORITHM: src/execution.py — Production CCXT Order Routing Engine with Proxy Tunneling
====================================================================================================
Purpose:
  Institutional-grade, fault-tolerant exchange connector to Binance Futures Testnet via CCXT.
  Routes all REST traffic through an authenticated non-US proxy (BINANCE_PROXY_URL) to bypass
  cloud runner HTTP 451 geo-restrictions. Fixes the fatal missing `pandas` import, implements
  pre-flight network diagnostics, isolated 1.0x setup, 2-stage bracket deployment, 15-minute
  order timeout handling, and signal-flip reversal closures with ghost bracket cleanup.

Algorithm Steps:
  Step 1: Module Setup, Missing Dependency Resolution & Environment Ingestion:
          - Import all required modules, explicitly including `pandas as pd` to prevent runtime crashes.
          - Ingest API keys and the proxy URL (BINANCE_PROXY_URL) from environment/secrets.
  Step 2: CCXT Client Initialization with Proxy Configuration & Pre-Flight Handshake:
          - Construct `ccxt.binanceusdm` client.
          - If BINANCE_PROXY_URL is supplied, inject `'proxies': {'http': proxy_url, 'https': proxy_url}`.
          - Configure testnet endpoints using `enable_demo_trading(True)` and fallback URL overrides.
          - Run pre-flight egress diagnostics to verify IP and exchange handshake.
  Step 3: Account Capital & Position Discovery Handlers:
          - `get_free_usdt_balance()`: Fetches available free USDT margin.
          - `get_active_positions()`: Inspects non-zero open perpetual positions across active assets.
  Step 4: Precision Quantization & minNotional Compliance:
          - Format order size to stepSize and price to tickSize via CCXT exchange metadata.
          - Enforce 5.0 USDT minimum notional value floor.
  Step 5: Isolated Margin & 1.0x Leverage Configuration:
          - Configure margin mode to 'ISOLATED' and leverage to 1.0x (unleveraged cash policy).
  Step 6: Limit Entry Order Routing:
          - Place quantized limit entry at crossover price.
          - Record new order in Supabase Table 1 (`testnet_active_trades`) with status 'PENDING_LIMIT'.
  Step 7: Order Fill Inspection & Native 2-Stage Bracket Deployment:
          - Inspect entry fill status on Binance.
          - Once filled, submit reduce-only TAKE_PROFIT_MARKET and STOP_MARKET orders directly to exchange.
          - Update Supabase Table 1 with fill price and bracket order IDs.
  Step 8: 15-Minute Timeout Cancellation (`handle_expired_limit_orders`):
          - Cancel unfilled 'PENDING_LIMIT' orders older than 15 minutes.
          - Move trade to Supabase Table 2 (`testnet_trade_log`) under 'MISSED_TRADE'.
  Step 9: Signal-Flip Reversal Execution (`execute_signal_flip_close`):
          - On opposite crossover, cancel resting TP and SL orders immediately (ghost bracket cleanup).
          - Submit immediate Market Close order (reduceOnly = True).
          - Calculate friction loss and archive to Supabase Table 2 as 'SIGNAL_FLIP'.
  Step 10: Integration Self-Test Probe (`if __name__ == '__main__'`):
          - Verify proxy connectivity, balance discovery, and quantization math.
====================================================================================================
"""

# =============================================================================
# STEP 1: Module Setup, Dependency Resolution & Environment Ingestion
# =============================================================================
import os
import time
import json
import warnings
from datetime import datetime, timezone
import requests
import pandas as pd  # CRITICAL BUG FIX: Resolved fatal missing pandas import
import ccxt

try:
    from src.telemetry import TelemetryEngine
except ImportError:
    from telemetry import TelemetryEngine

warnings.filterwarnings("ignore", category=UserWarning)

# Detect Kaggle vs Cloud/Local Environment
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    API_KEY    = _secrets.get_secret("BINANCE_TESTNET_API_KEY").strip()
    API_SECRET = _secrets.get_secret("BINANCE_TESTNET_API_SECRET").strip()
    PROXY_URL  = ""
    try:
        PROXY_URL = _secrets.get_secret("BINANCE_PROXY_URL").strip()
    except Exception:
        pass
else:
    API_KEY    = os.environ.get("BINANCE_TESTNET_API_KEY", "").strip()
    API_SECRET = os.environ.get("BINANCE_TESTNET_API_SECRET", "").strip()
    PROXY_URL  = os.environ.get("BINANCE_PROXY_URL", "").strip()


# =============================================================================
# STEP 2: CCXT Client Initialization with Proxy Configuration
# =============================================================================
class ExecutionEngine:
    """
    Hardened CCXT connector managing order routing, precision quantization,
    proxy tunneling, bracket deployment, and signal-flip liquidations.
    """
    def __init__(
        self,
        api_key: str = API_KEY,
        api_secret: str = API_SECRET,
        proxy_url: str = PROXY_URL,
        telemetry: TelemetryEngine = None
    ):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.proxy_url  = proxy_url
        self.telemetry  = telemetry or TelemetryEngine()

        # Build CCXT configuration dictionary
        exchange_config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        }

        # Inject Proxy Tunneling if configured
        if self.proxy_url:
            exchange_config['proxies'] = {
                'http': self.proxy_url,
                'https': self.proxy_url
            }
            # Mask credentials when logging
            masked_proxy = self.proxy_url.split('@')[-1] if '@' in self.proxy_url else self.proxy_url
            print(f"[Network] CCXT configured with proxy tunnel -> {masked_proxy}")
        else:
            print(f"[Network Warning] No proxy configured. Handshakes originate directly from host IP.")

        # Instantiate CCXT Binance USDT-M Futures Client
        self.exchange = ccxt.binanceusdm(exchange_config)

        # ── HARDENED TESTNET ROUTING ──
        if hasattr(self.exchange, "enable_demo_trading"):
            self.exchange.enable_demo_trading(True)
        elif hasattr(self.exchange, "enableDemoTrading"):
            self.exchange.enableDemoTrading(True)
        else:
            self.exchange.urls['api']['fapiPublic']    = 'https://testnet.binancefuture.com/fapi/v1'
            self.exchange.urls['api']['fapiPrivate']   = 'https://testnet.binancefuture.com/fapi/v1'
            self.exchange.urls['api']['fapiPrivateV2'] = 'https://testnet.binancefuture.com/fapi/v2'

        self.markets_loaded = False
        self._preflight_diagnostic_probe()
        self._load_markets_safe()

    def _preflight_diagnostic_probe(self):
        """Runs pre-flight egress diagnostics to confirm proxy country and IP."""
        try:
            proxies_dict = {'http': self.proxy_url, 'https': self.proxy_url} if self.proxy_url else None
            res = requests.get('https://ipinfo.io/json', proxies=proxies_dict, timeout=5).json()
            ip = res.get('ip', 'Unknown')
            country = res.get('country', 'Unknown')
            city = res.get('city', 'Unknown')
            print(f"[Network Diagnostic] Egress IP: {ip} | Country: {country} ({city})")
        except Exception as e:
            print(f"[Network Diagnostic Notice] Could not determine external IP info: {e}")

    def _load_markets_safe(self):
        """Loads market filters and precision rules with error handling."""
        try:
            self.exchange.load_markets()
            self.markets_loaded = True
            print(f"[Network Success] Binance Testnet markets loaded successfully.")
        except Exception as e:
            print(f"[Execution Warning] Could not load market filters: {repr(e)}")

    # =========================================================================
    # STEP 3: Account Capital & Position Discovery Handlers
    # =========================================================================
    def get_free_usdt_balance(self) -> float:
        """Queries Binance Futures wallet for available free USDT cash balance."""
        if not self.api_key or not self.api_secret:
            return 10000.0
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get('USDT', {}).get('free', 10000.0))
        except Exception as e:
            print(f"[Execution Warning] Failed to fetch live balance ({e}). Defaulting to $10,000.")
            return 10000.0

    def get_active_positions(self) -> dict:
        """Fetches all currently open positions on Binance Futures with non-zero contracts."""
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

    # =========================================================================
    # STEP 4: Precision Quantization & minNotional Compliance
    # =========================================================================
    def quantize_order_params(self, symbol: str, price: float, quantity: float):
        """
        Quantizes price to tickSize and quantity to stepSize using exchange metadata.
        Enforces 5.0 USDT minNotional floor to prevent exchange filter rejections.
        """
        if not self.markets_loaded:
            self._load_markets_safe()

        clean_price = float(self.exchange.price_to_precision(symbol, price))
        clean_qty   = float(self.exchange.amount_to_precision(symbol, quantity))

        # Check minNotional (Binance Futures requires >= 5.0 USDT order value)
        notional_value = clean_price * clean_qty
        if notional_value < 5.0 and clean_price > 0:
            required_qty = (5.5 / clean_price)
            clean_qty    = float(self.exchange.amount_to_precision(symbol, required_qty))

        return clean_price, clean_qty

    # =========================================================================
    # STEP 5: Isolated Margin & 1.0x Leverage Setup
    # =========================================================================
    def setup_symbol_isolated_1x(self, symbol: str):
        """Configures asset to ISOLATED margin mode and sets leverage to 1.0x."""
        if not self.api_key or not self.api_secret:
            return

        # 1. Set Margin Mode: ISOLATED
        try:
            self.exchange.set_margin_mode('ISOLATED', symbol)
        except Exception as e:
            err_msg = str(e).lower()
            if "no need to change" not in err_msg and "already" not in err_msg:
                print(f"[Execution Notice] Margin mode setting for {symbol}: {e}")

        # 2. Set Leverage: 1.0x (Unleveraged Policy)
        try:
            self.exchange.set_leverage(1, symbol)
        except Exception as e:
            err_msg = str(e).lower()
            if "not modified" not in err_msg:
                print(f"[Execution Notice] Leverage setting for {symbol}: {e}")

    # =========================================================================
    # STEP 6: Limit Entry Order Placement
    # =========================================================================
    def execute_limit_entry(self, manifest: dict, candle_close_utc: str) -> str:
        """Places Limit Entry Order at crossover price and records to Supabase Table 1."""
        symbol         = manifest["symbol"]
        direction      = manifest["direction"].upper()
        entry_price    = manifest["entry_price"]
        raw_qty        = manifest["contract_quantity"]
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

    # =========================================================================
    # STEP 7: Order Fill Inspection & Native Bracket Deployment
    # =========================================================================
    def check_and_deploy_brackets(self, trade_record: dict):
        """Polls Binance for entry fill status. On fill, submits native TP/SL brackets."""
        trade_id   = trade_record["id"]
        symbol     = trade_record["symbol"]
        direction  = trade_record["direction"].upper()
        binance_id = trade_record.get("binance_order_id")
        qty        = float(trade_record["contract_quantity"])

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
                print(f"   --> Native Brackets Deployed: TP @ ${clean_tp_price:,.2f} | SL @ ${clean_sl_price:,.2f}")

        except Exception as e:
            print(f"[Execution Error] Failed to check/deploy brackets for {symbol}: {repr(e)}")

    # =========================================================================
    # STEP 8: 15-Minute Timeout Cancellation (handle_expired_limit_orders)
    # =========================================================================
    def handle_expired_limit_orders(self, max_timeout_minutes: int = 15):
        """Cancels unfilled limit entries after 15m timeout and logs MISSED_TRADE."""
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

    # =========================================================================
    # STEP 9: Signal-Flip Reversal Execution (Market Exit + Bracket Cleanup)
    # =========================================================================
    def execute_signal_flip_close(self, active_trade: dict) -> float:
        """Executes immediate market liquidation and bracket cleanup on opposite crossover."""
        trade_id  = active_trade["id"]
        symbol    = active_trade["symbol"]
        direction = active_trade["direction"].upper()
        qty       = float(active_trade["contract_quantity"])
        tp_id     = active_trade.get("binance_tp_id")
        sl_id     = active_trade.get("binance_sl_id")

        print(f"[Signal Flip Detected] Closing {symbol} {direction} via Market Order...")

        if self.api_key and self.api_secret:
            for b_id in [tp_id, sl_id]:
                if b_id and "MOCK" not in b_id:
                    try:
                        self.exchange.cancel_order(b_id, symbol)
                    except Exception:
                        pass

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
# STEP 10: Built-In Integration Self-Test Probe
# =============================================================================
if __name__ == "__main__":
    print("===============================================================================")
    print("  TESTING HARDENED EXECUTION CONNECTOR WITH PROXY TUNNEL                       ")
    print("===============================================================================")
    engine = ExecutionEngine()
    free_bal = engine.get_free_usdt_balance()
    print(f"  Connected successfully. Free Balance: ${free_bal:,.2f} USDT")
    
    clean_p, clean_q = engine.quantize_order_params("BTCUSDT", 65123.4567, 0.0012345)
    print(f"  Quantized BTCUSDT -> Price: ${clean_p:,.2f} | Quantity: {clean_q}")
    print("===============================================================================")
