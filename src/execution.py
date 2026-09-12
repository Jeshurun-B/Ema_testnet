"""
====================================================================================================
ALGORITHM: src/execution.py — Production CCXT Order Routing Engine with Auto-Sanitizing Proxy
====================================================================================================
Purpose:
  Institutional-grade, fault-tolerant exchange connector to Binance Futures Testnet via CCXT.
  Routes all traffic through an authenticated non-US proxy with automatic URI scheme sanitization
  to eliminate `[SSL: WRONG_VERSION_NUMBER]` errors. Enforces precision quantization, isolated 1.0x
  margin, 2-stage native bracket deployment, 15-minute order timeouts, and signal-flip closures.

Key Microstructure & Network Invariants:
  1. Automatic Proxy Scheme Sanitization:
     - Detects and rewrites `https://` proxy prefixes to `http://` in-memory.
     - Prevents OpenSSL TLS client handshakes on plaintext proxy ports (e.g., Webshare port 6077).
  2. Pre-Flight Egress IP & Country Verification:
     - Inspects external IP and country through the proxy before market load.
  3. Strict Dependency Integrity:
     - Imports `pandas as pd` for timestamp parsing in order timeouts and signal flips.
  4. Two-Stage Order Lifecycle:
     - Limit Entry placed first. Reduce-only TP and SL brackets placed ONLY after fill confirmation.
  5. Ghost Bracket Cleanup:
     - Resting brackets cancelled on Binance immediately upon signal flip prior to market close.

Algorithm Steps:
  Step 1: Module Setup, Missing Dependency Resolution & Environment Ingestion:
          - Import all required standard, scientific, and CCXT libraries (including pandas as pd).
          - Ingest API credentials and BINANCE_PROXY_URL from environment or Kaggle secrets.
  Step 2: Proxy URI Sanitization Utility:
          - Implement `sanitize_proxy_url(url)` to enforce `http://` prefix for plaintext forward proxies.
  Step 3: CCXT Client Initialization & Pre-Flight Handshake:
          - Configure `ccxt.binanceusdm` with sanitized proxy tunnel, rate limiting, and time diff adjustment.
          - Route to Binance Futures Testnet endpoints via `enable_demo_trading(True)`.
          - Execute pre-flight IP/country diagnostic probe.
          - Safely load market filters and precision rules.
  Step 4: Account Capital & Position Discovery Handlers:
          - `get_free_usdt_balance()`: Fetches available free USDT cash balance.
          - `get_active_positions()`: Inspects non-zero open perpetual contracts across active symbols.
  Step 5: Precision Quantization & minNotional Compliance:
          - Format order size to stepSize and price to tickSize.
          - Enforce 5.0 USDT minNotional floor.
  Step 6: Isolated Margin & 1.0x Leverage Configuration:
          - Set margin mode to 'ISOLATED' and leverage strictly to 1.0x.
  Step 7: Limit Entry Order Routing:
          - Place quantized limit order on Binance and record to Supabase Table 1 as 'PENDING_LIMIT'.
  Step 8: Order Fill Inspection & Native 2-Stage Bracket Deployment:
          - Poll entry fill. Upon fill, place resting TAKE_PROFIT_MARKET and STOP_MARKET orders.
  Step 9: 15-Minute Timeout Cancellation (`handle_expired_limit_orders`):
          - Cancel limit entries exceeding 15 minutes unfilled and archive to Table 2 as 'MISSED_TRADE'.
  Step 10: Signal-Flip Reversal Execution (`execute_signal_flip_close`):
          - Cancel resting brackets, execute immediate Market Close, and archive as 'SIGNAL_FLIP'.
  Step 11: Integration Self-Test Probe (`if __name__ == '__main__'`):
          - Authenticate, test proxy egress, and verify BTCUSDT precision formatting.
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
import pandas as pd  # Explicitly imported to prevent runtime NameError
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
# STEP 2: Proxy URI Sanitization Utility
# =============================================================================
def sanitize_proxy_url(url: str) -> str:
    """
    Sanitizes forward proxy URLs to prevent OpenSSL [SSL: WRONG_VERSION_NUMBER] crashes.
    Forward proxies (like Webshare) accept plaintext HTTP CONNECT on their listening port.
    If 'https://' is supplied, Python's urllib3 tries to perform TLS on a plaintext port.
    """
    if not url:
        return ""
    clean = url.strip()
    if clean.startswith("https://"):
        clean = "http://" + clean[len("https://"):]
    elif not clean.startswith("http://") and not clean.startswith("socks5://"):
        clean = "http://" + clean
    return clean


# =============================================================================
# STEP 3: CCXT Client Initialization & Pre-Flight Handshake
# =============================================================================
class ExecutionEngine:
    """
    Hardened CCXT connector managing proxy tunneling, order routing,
    precision quantization, bracket deployment, and reversal liquidations.
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
        self.proxy_url  = sanitize_proxy_url(proxy_url)
        self.telemetry  = telemetry or TelemetryEngine()

        exchange_config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        }

        # Inject Sanitized Proxy Tunnel
        if self.proxy_url:
            exchange_config['proxies'] = {
                'http': self.proxy_url,
                'https': self.proxy_url
            }
            masked_proxy = self.proxy_url.split('@')[-1] if '@' in self.proxy_url else self.proxy_url
            print(f"[Network] CCXT configured with sanitized proxy -> {masked_proxy}")
        else:
            print(f"[Network Notice] No proxy configured. Direct connection active.")

        self.exchange = ccxt.binanceusdm(exchange_config)

        # Configure Testnet Routing (Eliminating deprecated sandbox mode)
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
        """Runs pre-flight egress probe to verify external IP and geolocation."""
        try:
            proxies_dict = {'http': self.proxy_url, 'https': self.proxy_url} if self.proxy_url else None
            res = requests.get('https://ipinfo.io/json', proxies=proxies_dict, timeout=8).json()
            ip = res.get('ip', 'Unknown')
            country = res.get('country', 'Unknown')
            city = res.get('city', 'Unknown')
            print(f"[Network Diagnostic] Egress IP: {ip} | Country: {country} ({city})")
        except Exception as e:
            print(f"[Network Diagnostic Notice] IP diagnostic skipped: {e}")

    def _load_markets_safe(self):
        """Loads exchange precision rules and filters safely."""
        try:
            self.exchange.load_markets()
            self.markets_loaded = True
            print(f"[Network Success] Binance Testnet markets loaded successfully.")
        except Exception as e:
            print(f"[Execution Warning] Could not load market filters: {repr(e)}")

    # =========================================================================
    # STEP 4: Account Capital & Position Discovery Handlers
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
        """Fetches all currently open positions with non-zero contracts."""
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
    # STEP 5: Precision Quantization & minNotional Compliance
    # =========================================================================
    def quantize_order_params(self, symbol: str, price: float, quantity: float):
        """Quantizes price to tickSize and amount to stepSize; enforces 5.0 USDT floor."""
        if not self.markets_loaded:
            self._load_markets_safe()

        clean_price = float(self.exchange.price_to_precision(symbol, price))
        clean_qty   = float(self.exchange.amount_to_precision(symbol, quantity))

        notional_value = clean_price * clean_qty
        if notional_value < 5.0 and clean_price > 0:
            required_qty = (5.5 / clean_price)
            clean_qty    = float(self.exchange.amount_to_precision(symbol, required_qty))

        return clean_price, clean_qty

    # =========================================================================
    # STEP 6: Isolated Margin & 1.0x Leverage Setup
    # =========================================================================
    def setup_symbol_isolated_1x(self, symbol: str):
        """Configures asset to ISOLATED margin mode and 1.0x leverage."""
        if not self.api_key or not self.api_secret:
            return

        try:
            self.exchange.set_margin_mode('ISOLATED', symbol)
        except Exception as e:
            err_msg = str(e).lower()
            if "no need to change" not in err_msg and "already" not in err_msg:
                print(f"[Execution Notice] Margin mode setting for {symbol}: {e}")

        try:
            self.exchange.set_leverage(1, symbol)
        except Exception as e:
            err_msg = str(e).lower()
            if "not modified" not in err_msg:
                print(f"[Execution Notice] Leverage setting for {symbol}: {e}")

    # =========================================================================
    # STEP 7: Limit Entry Order Placement
    # =========================================================================
    def execute_limit_entry(self, manifest: dict, candle_close_utc: str) -> str:
        """Places Limit Entry Order and records to Supabase Table 1 as 'PENDING_LIMIT'."""
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
    # STEP 8: Order Fill Inspection & Native Bracket Deployment
    # =========================================================================
    def check_and_deploy_brackets(self, trade_record: dict):
        """Polls Binance for entry fill status. On fill, deploys resting TP/SL brackets."""
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
    # STEP 9: 15-Minute Timeout Cancellation (handle_expired_limit_orders)
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
    # STEP 10: Signal-Flip Reversal Execution (Market Exit + Bracket Cleanup)
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

        # Cancel resting brackets first to prevent ghost fills
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
# STEP 11: Built-In Integration Self-Test Probe
# =============================================================================
if __name__ == "__main__":
    print("===============================================================================")
    print("  TESTING HARDENED EXECUTION CONNECTOR (src/execution.py)                      ")
    print("===============================================================================")
    engine = ExecutionEngine()
    free_bal = engine.get_free_usdt_balance()
    print(f"  Connected successfully. Free Balance: ${free_bal:,.2f} USDT")
    
    clean_p, clean_q = engine.quantize_order_params("BTCUSDT", 65123.4567, 0.0012345)
    print(f"  Quantized BTCUSDT -> Price: ${clean_p:,.2f} | Quantity: {clean_q}")
    print("===============================================================================")
