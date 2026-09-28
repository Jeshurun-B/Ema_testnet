"""
====================================================================================================
ALGORITHM: src/execution.py — Production Market Order Routing & Mark Price Execution Gateway
====================================================================================================
Purpose:
  Institutional exchange connector to Binance Futures Testnet via CCXT through the Frankfurt proxy.
  Switches entry execution to immediate Market (Taker) orders, guaranteeing 100% deterministic fills
  and eliminating 15-minute limit timeouts and ghost limit orders on the book. Enforces `MARK_PRICE`
  brackets, sweeps ghost orders on boot, normalizes symbols, and manages signal flips.

Algorithm Steps:
  Step 1: Module Setup, Safe Math & Dependency Ingestion (including pandas as pd).
  Step 2: Proxy URI Sanitization Utility.
  Step 3: CCXT Client Initialization & Pre-Flight Handshake.
  Step 4: Account Capital & Normalized Position Discovery.
  Step 5: The Order Book Sweeper (Purges all unlinked orders across active assets on boot).
  Step 6: Precision Quantization & minNotional Compliance.
  Step 7: Immediate Market Entry Order Dispatch:
          - Places immediate market order on Binance.
          - Deploys native resting brackets pegged strictly to `workingType: MARK_PRICE`.
          - Records active trade in Supabase Table 1 as 'FILLED'.
          - Returns tuple: `(trade_id, binance_order_id, actual_fill_price)`.
  Step 8: Signal-Flip Reversal Liquidation (Ghost Bracket Annihilation + Market Close).
====================================================================================================
"""

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

if not API_KEY or not API_SECRET:
    raise RuntimeError("[FATAL] Binance API credentials missing from environment!")


def sanitize_proxy_url(url: str) -> str:
    if not url:
        return ""
    clean = url.strip()
    if clean.startswith("https://"):
        clean = "http://" + clean[len("https://"):]
    elif not clean.startswith("http://") and not clean.startswith("socks5://"):
        clean = "http://" + clean
    return clean


class ExecutionEngine:
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
            exchange_config['proxies'] = {'http': self.proxy_url, 'https': self.proxy_url}
            masked = self.proxy_url.split('@')[-1] if '@' in self.proxy_url else self.proxy_url
            print(f"[Network] CCXT configured with proxy tunnel -> {masked}")

        self.exchange = ccxt.binanceusdm(exchange_config)

        if hasattr(self.exchange, "enable_demo_trading"):
            self.exchange.enable_demo_trading(True)
        elif hasattr(self.exchange, "enableDemoTrading"):
            self.exchange.enableDemoTrading(True)
        else:
            self.exchange.urls['api']['fapiPublic']    = 'https://testnet.binancefuture.com/fapi/v1'
            self.exchange.urls['api']['fapiPrivate']   = 'https://testnet.binancefuture.com/fapi/v1'
            self.exchange.urls['api']['fapiPrivateV2'] = 'https://testnet.binancefuture.com/fapi/v2'

        self.markets_loaded = False
        self._load_markets_safe()

    def _load_markets_safe(self):
        try:
            self.exchange.load_markets()
            self.markets_loaded = True
            print("[Network Success] Binance Testnet markets loaded successfully.")
        except Exception as e:
            print(f"[Execution Warning] Could not load market filters: {repr(e)}")

    def purge_unlinked_ghost_orders(self, active_symbols: list):
        """Sweeps and purges all open orders on Binance on startup to eliminate ghost orders."""
        print("[Sweeper Probe] Checking for resting ghost orders on Binance matching engine...")
        for sym in active_symbols:
            try:
                open_orders = self.exchange.fetch_open_orders(sym)
                if open_orders:
                    print(f"   --> Found {len(open_orders)} open order(s) for {sym}. Purging all...")
                    self.exchange.cancel_all_orders(sym)
                    print(f"       [PURGED] Successfully purged open orders for {sym}.")
            except Exception as e:
                print(f"   --> Notice sweeping {sym}: {e}")

    def get_free_usdt_balance(self) -> float:
        if not self.api_key or not self.api_secret:
            return 5000.0
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get('USDT', {}).get('free', 5000.0))
        except Exception as e:
            print(f"[Execution Warning] Balance check fallback: {e}")
            return 5000.0

    def get_active_positions(self) -> dict:
        if not self.api_key or not self.api_secret:
            return {}
        try:
            positions = self.exchange.fetch_positions()
            active_map = {}
            for pos in positions:
                contracts = float(pos.get('contracts', 0.0))
                if contracts > 0:
                    raw_sym = pos.get('info', {}).get('symbol') or pos['symbol'].split(':')[0].replace('/', '')
                    sym = raw_sym.strip()
                    active_map[sym] = {
                        'contracts': contracts,
                        'side': pos.get('side', '').lower(),
                        'entry_price': float(pos.get('entryPrice', 0.0)),
                        'unrealized_pnl': float(pos.get('unrealizedPnl', 0.0))
                    }
            return active_map
        except Exception as e:
            print(f"[Execution Warning] Positions check fallback: {e}")
            return {}

    def quantize_order_params(self, symbol: str, price: float, quantity: float):
        if not self.markets_loaded:
            self._load_markets_safe()

        clean_price = float(self.exchange.price_to_precision(symbol, price))
        clean_qty   = float(self.exchange.amount_to_precision(symbol, quantity))

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
            if "-4067" not in err_msg and "no need to change" not in err_msg and "already" not in err_msg:
                print(f"[Execution Notice] Margin mode for {symbol}: {e}")

        try:
            self.exchange.set_leverage(1, symbol)
        except Exception as e:
            err_msg = str(e).lower()
            if "not modified" not in err_msg:
                print(f"[Execution Notice] Leverage for {symbol}: {e}")

    def execute_market_entry(self, manifest: dict, candle_close_utc: str):
        """
        Executes immediate Market (Taker) entry on Binance. Fills instantly,
        deploys native MARK_PRICE resting brackets, and records in Supabase Table 1 as FILLED.
        """
        symbol         = manifest["symbol"]
        direction      = manifest["direction"].upper()
        entry_price    = manifest["entry_price"]
        raw_qty        = manifest["contract_quantity"]
        allocated_cash = manifest["allocated_cash"]
        trade_tier     = manifest["trade_tier"]

        self.setup_symbol_isolated_1x(symbol)
        clean_price, clean_qty = self.quantize_order_params(symbol, entry_price, raw_qty)
        order_side = 'buy' if direction == 'LONG' else 'sell'
        close_side = 'sell' if direction == 'LONG' else 'buy'

        binance_order_id = None
        actual_fill_px   = clean_price

        # 1. Dispatch Immediate Market Order (Taker)
        if self.api_key and self.api_secret:
            try:
                order_res = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=order_side,
                    amount=clean_qty
                )
                binance_order_id = str(order_res['id'])
                actual_fill_px = float(order_res.get('average') or order_res.get('price') or clean_price)
                print(f"[Binance Execution] {trade_tier} Market Entry Filled: {direction} {clean_qty} {symbol} @ ${actual_fill_px} (ID: {binance_order_id})")
            except Exception as e:
                raise RuntimeError(f"[Execution Error] Market order rejected by Binance: {repr(e)}")
        else:
            binance_order_id = f"MOCK_{int(time.time())}"
            print(f"[Mock Execution] {trade_tier} Market Entry: {direction} {clean_qty} {symbol} @ ${clean_price}")

        # 2. Deploy Native MARK_PRICE Brackets Immediately
        tp_id, sl_id = None, None
        if self.api_key and self.api_secret:
            try:
                tp_px = float(manifest["dynamic_tp_price"])
                sl_px = float(manifest["dynamic_sl_price"])
                clean_tp, _ = self.quantize_order_params(symbol, tp_px, clean_qty)
                clean_sl, _ = self.quantize_order_params(symbol, sl_px, clean_qty)

                tp_order = self.exchange.create_order(
                    symbol=symbol, type='TAKE_PROFIT_MARKET', side=close_side, amount=clean_qty,
                    params={'stopPrice': clean_tp, 'reduceOnly': True, 'workingType': 'MARK_PRICE'}
                )
                sl_order = self.exchange.create_order(
                    symbol=symbol, type='STOP_MARKET', side=close_side, amount=clean_qty,
                    params={'stopPrice': clean_sl, 'reduceOnly': True, 'workingType': 'MARK_PRICE'}
                )
                tp_id = str(tp_order['id'])
                sl_id = str(sl_order['id'])
                print(f"   --> Native Brackets Deployed: TP @ ${clean_tp} | SL @ ${clean_sl} [MARK_PRICE]")
            except Exception as e:
                print(f"[Execution Warning] Bracket placement notice: {e}")

        # 3. Record Active Trade in Supabase Table 1 as FILLED
        trade_id = self.telemetry.record_active_trade({
            "symbol": symbol,
            "direction": direction,
            "trade_tier": trade_tier,
            "entry_price": actual_fill_px,
            "contract_quantity": clean_qty,
            "allocated_cash": allocated_cash,
            "dynamic_tp_price": manifest["dynamic_tp_price"],
            "dynamic_sl_price": manifest["dynamic_sl_price"],
            "binance_order_id": binance_order_id,
            "binance_tp_id": tp_id,
            "binance_sl_id": sl_id,
            "order_status": "FILLED"
        })

        return trade_id, binance_order_id, tp_id, sl_id, actual_fill_px

    def execute_signal_flip_close(self, active_trade: dict) -> float:
        """Kills resting brackets and issues an immediate Market Close order."""
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
                pass

        # 2. Market Liquidation Order
        close_side = 'sell' if direction == 'LONG' else 'buy'
        exit_price = float(active_trade["entry_price"])
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
                
                entry_fill = float(active_trade["entry_price"])
                gross_ret = (exit_price - entry_fill) / entry_fill if direction == 'LONG' else (entry_fill - exit_price) / entry_fill
                realized_pnl = (float(active_trade["allocated_cash"]) * gross_ret) - fees_paid
            except Exception as e:
                print(f"[Execution Error] Market flip close failed on Binance: {repr(e)}")
        else:
            exit_price = float(active_trade["entry_price"]) * 1.002
            fees_paid  = float(active_trade["allocated_cash"]) * 0.0008
            realized_pnl = 0.50

        hold_min = max(0.1, (datetime.now(timezone.utc) - pd.to_datetime(active_trade["created_at"], utc=True)).total_seconds() / 60.0)

        self.telemetry.record_trade_closure({
            "id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "trade_tier": active_trade.get("trade_tier", "MAIN"),
            "close_reason": "SIGNAL_FLIP",
            "entry_price": float(active_trade["entry_price"]),
            "exit_price": exit_price,
            "realized_binance_pnl": realized_pnl,
            "idealized_pnl": realized_pnl + fees_paid,
            "friction_loss": 0.0,
            "exchange_fees_paid": fees_paid,
            "hold_duration_minutes": hold_min,
            "notes": "Closed on confirmed opposite 9/15 EMA crossover"
        })
        print(f"   --> {symbol} {direction} closed @ ${exit_price:,.4f} (Net PnL: ${realized_pnl:+,.2f})")
        return realized_pnl
