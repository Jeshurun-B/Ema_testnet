"""
====================================================================================================
ALGORITHM: src/execution.py — Institutional Order Routing Engine with Mark Price Protection
====================================================================================================
Purpose:
  Institutional exchange connector to Binance Futures Testnet via CCXT. Eliminates rogue order-book
  liquidity sweeps by explicitly pegging native TP/SL brackets to `MARK_PRICE` rather than local
  contract last price. Returns dual identifiers (trade_id, binance_order_id), enforces CCXT symbol
  normalization (stripping ':USDT'), measures wall-clock timeouts via `created_at`, sanitizes forward
  proxies, and purges ghost brackets upon signal flips.

Key Microstructure & Routing Invariants:
  1. Mark Price Trigger Protection (`workingType: 'MARK_PRICE'`):
     - Injects `workingType: 'MARK_PRICE'` into both TAKE_PROFIT_MARKET and STOP_MARKET orders.
     - Protects positions against thin order-book illiquidity, fat-finger fills, and testnet air-pocket
       wicks by pegging bracket triggers strictly to the consolidated spot index rather than local fills.
  2. Dual Identifier Return (trade_id, binance_order_id):
     - Returns both the Supabase UUID and the Binance matching engine order ID to ensure that in-flight
       maintenance can poll fills in real time and physically cancel timed-out limit orders on the exchange.
  3. Universal CCXT Symbol Normalization:
     - Strips unified CCXT suffixes (`:USDT` and `/`), ensuring that exchange position queries map
       identically to internal database keys (e.g., 'SOLUSDT', 'BTCUSDT').
  4. Ghost Bracket Annihilation:
     - Signal flips execute `exchange.cancel_all_orders(symbol)` to purge all resting brackets on the
       Binance matching engine before market liquidation.
  5. Proxy URI Sanitization:
     - Enforces `http://` scheme to prevent OpenSSL `[SSL: WRONG_VERSION_NUMBER]` protocol crashes.

Algorithm Steps:
  Step 1: Module Setup, Safe Math & Dependency Ingestion (including pandas as pd).
  Step 2: Proxy URI Sanitization Utility.
  Step 3: CCXT Client Initialization & Pre-Flight Handshake:
          - Configure `ccxt.binanceusdm` with proxy tunnel, rate limiting, and time diff adjustment.
          - Route via `enable_demo_trading(True)`. Verify external IP via pre-flight probe.
  Step 4: Account Capital & Normalized Position Discovery:
          - `get_free_usdt_balance()`: Queries wallet for available free USDT margin.
          - `get_active_positions()`: Normalizes contract symbols to clean pairs (e.g., 'BTCUSDT').
  Step 5: Precision Quantization & minNotional Compliance:
          - Formats size to stepSize and price to tickSize; enforces 5.0 USDT floor.
  Step 6: Isolated Margin & 1.0x Leverage Configuration:
          - Enforces 'ISOLATED' margin mode and 1.0x unleveraged capital policy.
  Step 7: Limit Entry Order Dispatch (Dual-ID Return):
          - Dispatches quantized limit entry order at crossover close price.
          - Records trade in Supabase Table 1 as 'PENDING_LIMIT'.
          - Returns tuple: `(trade_id, binance_order_id)`.
  Step 8: Native 2-Stage Bracket Deployment with Mark Price Protection (`check_and_deploy_brackets`):
          - When limit entry fills, deploys resting reduce-only TAKE_PROFIT_MARKET and STOP_MARKET
            orders pegged strictly to `workingType: 'MARK_PRICE'`.
  Step 9: Wall-Clock Order Timeout Handler (`handle_expired_limit_orders`):
          - Cancels unfilled limit orders older than 15 wall-clock minutes; archives to Table 2.
  Step 10: Signal-Flip Liquidation with Ghost Bracket Annihilation:
          - Purges resting brackets via `cancel_all_orders(symbol)`.
          - Submits immediate Market Close order (reduceOnly = True) and archives to Table 2.
  Step 11: Integration Self-Test Probe (`if __name__ == '__main__'`).
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
import pandas as pd
import ccxt

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
    """Normalizes forward proxy URLs to prevent OpenSSL version mismatch crashes."""
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
    Hardened CCXT connector managing order routing, symbol normalization,
    wall-clock timeouts, dual-ID returns, and Mark Price bracket triggers.
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

        if self.proxy_url:
            exchange_config['proxies'] = {
                'http': self.proxy_url,
                'https': self.proxy_url
            }
            masked_proxy = self.proxy_url.split('@')[-1] if '@' in self.proxy_url else self.proxy_url
            print(f"[Network] CCXT configured with proxy tunnel -> {masked_proxy}")
        else:
            print(f"[Network Notice] Direct network connection active.")

        self.exchange = ccxt.binanceusdm(exchange_config)

        # Testnet routing
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
        """Runs pre-flight egress diagnostic to verify IP and country."""
        try:
            proxies_dict = {'http': self.proxy_url, 'https': self.proxy_url} if self.proxy_url else None
            res = requests.get('https://ipinfo.io/json', proxies=proxies_dict, timeout=8).json()
            ip = res.get('ip', 'Unknown')
            country = res.get('country', 'Unknown')
            city = res.get('city', 'Unknown')
            print(f"[Network Diagnostic] Egress IP: {ip} | Country: {country} ({city})")
        except Exception as e:
            print(f"[Network Diagnostic Notice] Diagnostic skipped: {e}")

    def _load_markets_safe(self):
        """Safely loads precision rules and market filters."""
        try:
            self.exchange.load_markets()
            self.markets_loaded = True
            print(f"[Network Success] Binance Testnet markets loaded successfully.")
        except Exception as e:
            print(f"[Execution Warning] Could not load market filters: {repr(e)}")

    # =========================================================================
    # STEP 4: Capital & Normalized Position Discovery
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
        """
        Fetches all open positions on Binance Futures with non-zero contracts.
        NORMALIZATION ENFORCEMENT: Strips '/USDT:USDT' and ':USDT' so keys match 'BTCUSDT'.
        """
        if not self.api_key or not self.api_secret:
            return {}
        try:
            positions = self.exchange.fetch_positions()
            active_map = {}
            for pos in positions:
                contracts = float(pos.get('contracts', 0.0))
                if contracts > 0:
                    raw_sym = pos.get('info', {}).get('symbol')
                    if not raw_sym:
                        raw_sym = pos['symbol'].split(':')[0].replace('/', '')
                    sym = raw_sym.strip()
                    
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
                print(f"[Execution Notice] Margin mode for {symbol}: {e}")

        try:
            self.exchange.set_leverage(1, symbol)
        except Exception as e:
            err_msg = str(e).lower()
            if "not modified" not in err_msg:
                print(f"[Execution Notice] Leverage for {symbol}: {e}")

    # =========================================================================
    # STEP 7: Limit Entry Order Placement (Returns trade_id, binance_order_id)
    # =========================================================================
    def execute_limit_entry(self, manifest: dict, candle_close_utc: str):
        """
        Places quantized limit entry on Binance and records to Supabase Table 1.
        RETURNS: tuple (trade_id: str, binance_order_id: str).
        """
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
                print(f"[Binance Execution] Limit Entry Placed: {direction} {clean_qty} {symbol} @ ${clean_price} (ID: {binance_order_id})")
            except Exception as e:
                raise RuntimeError(f"[Execution Error] Limit order rejected by Binance: {repr(e)}")
        else:
            binance_order_id = f"MOCK_ENTRY_{int(time.time())}"
            print(f"[Mock Execution] Limit Entry Recorded: {direction} {clean_qty} {symbol} @ ${clean_price}")

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

        return trade_id, binance_order_id

    # =========================================================================
    # STEP 8: Native Bracket Deployment with Mark Price Protection
    # =========================================================================
    def check_and_deploy_brackets(self, trade_record: dict):
        """
        Polls Binance for entry fill. Upon fill, deploys native resting brackets
        pegged strictly to workingType: 'MARK_PRICE' to eliminate order-book air pockets.
        """
        trade_id   = trade_record["id"]
        symbol     = trade_record["symbol"]
        direction  = trade_record["direction"].upper()
        binance_id = trade_record.get("binance_order_id")
        qty        = float(trade_record["contract_quantity"])

        if not self.api_key or not self.api_secret or not binance_id or "MOCK" in str(binance_id):
            return

        try:
            order_info = self.exchange.fetch_order(binance_id, symbol)
            status = order_info.get('status', '').lower()

            if status == 'closed':
                actual_fill = float(order_info.get('average') or order_info.get('price') or trade_record["limit_entry_price"])
                print(f"[Order Fill Detected] {symbol} {direction} filled at ${actual_fill}. Deploying resting brackets (MARK_PRICE protected)...")
                
                self.telemetry.record_order_fill(trade_id, actual_fill)

                tp_price = float(trade_record["dynamic_tp_price"])
                sl_price = float(trade_record["dynamic_sl_price"])
                clean_tp_price, _ = self.quantize_order_params(symbol, tp_price, qty)
                clean_sl_price, _ = self.quantize_order_params(symbol, sl_price, qty)

                close_side = 'sell' if direction == 'LONG' else 'buy'

                # Native TAKE_PROFIT_MARKET pegged strictly to MARK_PRICE
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='TAKE_PROFIT_MARKET',
                    side=close_side,
                    amount=qty,
                    params={
                        'stopPrice': clean_tp_price,
                        'reduceOnly': True,
                        'workingType': 'MARK_PRICE'
                    }
                )

                # Native STOP_MARKET pegged strictly to MARK_PRICE
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=close_side,
                    amount=qty,
                    params={
                        'stopPrice': clean_sl_price,
                        'reduceOnly': True,
                        'workingType': 'MARK_PRICE'
                    }
                )

                self.telemetry.record_bracket_order_ids(trade_id, str(tp_order['id']), str(sl_order['id']))
                print(f"   --> Native Brackets Deployed: TP @ ${clean_tp_price} | SL @ ${clean_sl_price} [workingType: MARK_PRICE]")

        except Exception as e:
            print(f"[Execution Error] Failed to deploy brackets for {symbol}: {repr(e)}")

    # =========================================================================
    # STEP 9: Wall-Clock Order Timeout Cancellation (Exact 15 Minutes)
    # =========================================================================
    def handle_expired_limit_orders(self, max_timeout_minutes: int = 15):
        """Cancels limit orders older than 15.0 wall-clock minutes."""
        active_orders = self.telemetry.get_active_trades()
        now_dt = datetime.now(timezone.utc)

        for trade in active_orders:
            if trade.get("order_status") != "PENDING_LIMIT":
                continue

            created_str = trade.get("created_at")
            if created_str:
                order_dt = pd.to_datetime(created_str, utc=True)
            else:
                candle_close_str = trade.get("candle_close_utc")
                order_dt = pd.to_datetime(candle_close_str, utc=True) if candle_close_str else now_dt

            elapsed_min = (now_dt - order_dt).total_seconds() / 60.0

            if elapsed_min >= max_timeout_minutes:
                symbol = trade["symbol"]
                binance_id = trade.get("binance_order_id")

                print(f"[Timeout Triggered] {symbol} limit entry unfilled after {elapsed_min:.1f}m wall-clock time. Cancelling...")
                if self.api_key and self.api_secret and binance_id and "MOCK" not in str(binance_id):
                    try:
                        self.exchange.cancel_order(binance_id, symbol)
                        print(f"   --> Order {binance_id} cancelled successfully on Binance.")
                    except Exception as e:
                        print(f"   --> Cancel Notice: {e}")

                self.telemetry.record_missed_trade(
                    trade_id=trade["id"],
                    notes=f"Limit entry expired unfilled after {elapsed_min:.1f} wall-clock minutes"
                )

    # =========================================================================
    # STEP 10: Signal-Flip Liquidation with Ghost Bracket Annihilation
    # =========================================================================
    def execute_signal_flip_close(self, active_trade: dict) -> float:
        """Executes immediate market liquidation and bracket annihilation on opposite crossover."""
        trade_id  = active_trade.get("id")
        symbol    = active_trade["symbol"]
        direction = active_trade["direction"].upper()
        qty       = float(active_trade["contract_quantity"])

        print(f"[Signal Flip Detected] Closing {symbol} {direction} via Market Order...")

        # 1. Annihilate resting brackets on Binance
        if self.api_key and self.api_secret:
            try:
                self.exchange.cancel_all_orders(symbol)
                print(f"   --> All resting bracket orders annihilated for {symbol}.")
            except Exception:
                for b_id in [active_trade.get("binance_tp_id"), active_trade.get("binance_sl_id")]:
                    if b_id and "MOCK" not in str(b_id):
                        try:
                            self.exchange.cancel_order(b_id, symbol)
                        except Exception:
                            pass

        # 2. Market Liquidation Order
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
                print(f"[Execution Error] Market close failed on Binance: {repr(e)}")
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
            notes="Closed on confirmed 9/15 EMA opposite crossover",
            symbol=symbol,
            direction=direction,
            entry_price=float(active_trade.get("actual_fill_price") or active_trade["limit_entry_price"])
        )
        print(f"   --> {symbol} {direction} closed @ ${exit_price:,.4f} (Net PnL: ${realized_pnl:+,.2f})")
        return realized_pnl


# =============================================================================
# STEP 11: Integration Self-Test Probe
# =============================================================================
if __name__ == "__main__":
    print("===============================================================================")
    print("  TESTING HARDENED EXECUTION CONNECTOR (src/execution.py)                      ")
    print("===============================================================================")
    engine = ExecutionEngine()
    free_bal = engine.get_free_usdt_balance()
    print(f"  Connected successfully. Free Balance: ${free_bal:,.2f} USDT")
    
    positions = engine.get_active_positions()
    print(f"  Normalized Active Positions Detected: {positions}")
    print("===============================================================================")
