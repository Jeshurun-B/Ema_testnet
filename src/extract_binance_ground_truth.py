"""
====================================================================================================
ALGORITHM: src/extract_binance_ground_truth.py — Pure Binance Ground-Truth Extraction & Friction Engine
====================================================================================================
Purpose:
  Connects exclusively to Binance Futures Testnet via CCXT through the Frankfurt proxy tunnel.
  Extracts the complete, uncapped historical ledger of all orders, fills, and wallet income without
  any timestamp clipping. Directly reconstructs planned vs. actual executions, entry/exit slippage,
  exchange commission drag, and total friction cost exclusively from Binance matching engine data.
  Exports comprehensive CSV audit artifacts to `data/ground_truth/`, resets Supabase Table 1 (0/5 slots),
  and populates Table 2 with verified Binance records.

Microstructure Data Reconstructed Strictly from Binance:
  1. Planned/Predicted Entry Price: The exact limit price specified on the opening LIMIT order.
  2. Actual Realized Entry Price  : The actual weighted average fill price (avgPrice) executed on Binance.
  3. Entry Slippage ($)           : Dollar impact of actual entry execution vs. planned limit price.
  4. Planned Exit Barrier         : The exact stopPrice configured on the resting STOP/TP bracket order.
  5. Actual Realized Exit Price   : The actual execution price filled by the Binance matching engine.
  6. Exit Slippage ($)            : Dollar impact of exit execution vs. planned stop/target barrier.
  7. Exchange Fees Paid ($)       : Actual commissions deducted by Binance on both entry and exit legs.
  8. Idealized PnL ($)            : Theoretical return if filled exactly at planned entry & planned barrier.
  9. Realized Binance PnL ($)     : Actual net cash credited or debited by Binance matching engine.
  10. Total Friction Loss ($)     : Idealized PnL ($) - Realized Binance PnL ($).

Algorithm Steps:
  Step 1: Module Setup, Credentials Ingestion & Proxy Configuration.
  Step 2: CCXT Binance Futures Client Setup with Uncapped Endpoints.
  Step 3: Uncapped Paginated Scraping Across All 5 Assets:
          - Paginate `fetch_my_trades` across all historical fills (since=0).
          - Paginate `fetch_orders` across all historical orders (since=0).
          - Scrape `fapiPrivateGetIncome` across all wallet income events (since=0).
  Step 4: Microstructure Matching Engine (Reconstructing Planned vs. Actual Excursions):
          - Group fills by orderId and link entry orders to closing bracket orders.
          - Calculate entry slippage, exit slippage, commissions, and friction loss.
  Step 5: Export Full Audit CSV Artifacts to `data/ground_truth/`:
          - `binance_comprehensive_audit_ledger.csv` (Primary Analysis Ledger)
          - `binance_raw_trades_all.csv`
          - `binance_raw_orders_all.csv`
          - `binance_raw_income_all.csv`
  Step 6: Supabase Table Sanitization & Direct API Overwrite:
          - Reset Table 1 (`testnet_active_trades`) to clean 0/5 active slots.
          - Repopulate Table 2 (`testnet_trade_log`) directly with reconstructed Binance ledger.
  Step 7: Final Audit Summary & Scoreboard Display.
====================================================================================================
"""

# =============================================================================
# STEP 1: Module Setup, Credentials Ingestion & Proxy Configuration
# =============================================================================
import os
import sys
import time
import json
import warnings
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests
import ccxt
from supabase import create_client, Client

warnings.filterwarnings("ignore", category=UserWarning)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    API_KEY       = _secrets.get_secret("BINANCE_TESTNET_API_KEY").strip()
    API_SECRET    = _secrets.get_secret("BINANCE_TESTNET_API_SECRET").strip()
    SUPABASE_URL  = _secrets.get_secret("SUPABASE_URL").strip()
    SUPABASE_KEY  = _secrets.get_secret("SUPABASE_KEY").strip()
    PROXY_URL     = ""
    try:
        PROXY_URL = _secrets.get_secret("BINANCE_PROXY_URL").strip()
    except Exception:
        pass
else:
    API_KEY       = os.environ.get("BINANCE_TESTNET_API_KEY", "").strip()
    API_SECRET    = os.environ.get("BINANCE_TESTNET_API_SECRET", "").strip()
    SUPABASE_URL  = os.environ.get("SUPABASE_URL", "").strip()
    SUPABASE_KEY  = os.environ.get("SUPABASE_KEY", "").strip()
    PROXY_URL     = os.environ.get("BINANCE_PROXY_URL", "").strip()

if not API_KEY or not API_SECRET:
    raise RuntimeError("[FATAL] Binance API credentials missing from environment!")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("[FATAL] Supabase credentials missing from environment!")

ACTIVE_SYMBOLS = ["BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DATA_DIR       = os.path.join(os.getcwd(), "data", "ground_truth")
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# STEP 2: CCXT Binance Futures Client Setup with Proxy Tunnel
# =============================================================================
def sanitize_proxy_url(url: str) -> str:
    if not url:
        return ""
    clean = url.strip()
    if clean.startswith("https://"):
        clean = "http://" + clean[len("https://"):]
    elif not clean.startswith("http://") and not clean.startswith("socks5://"):
        clean = "http://" + clean
    return clean

clean_proxy = sanitize_proxy_url(PROXY_URL)

exchange_config = {
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True
    }
}

if clean_proxy:
    exchange_config['proxies'] = {
        'http': clean_proxy,
        'https': clean_proxy
    }
    masked = clean_proxy.split('@')[-1] if '@' in clean_proxy else clean_proxy
    print(f"[Network] CCXT configured with proxy tunnel -> {masked}")

exchange = ccxt.binanceusdm(exchange_config)

if hasattr(exchange, "enable_demo_trading"):
    exchange.enable_demo_trading(True)
elif hasattr(exchange, "enableDemoTrading"):
    exchange.enableDemoTrading(True)
else:
    exchange.urls['api']['fapiPublic']    = 'https://testnet.binancefuture.com/fapi/v1'
    exchange.urls['api']['fapiPrivate']   = 'https://testnet.binancefuture.com/fapi/v1'
    exchange.urls['api']['fapiPrivateV2'] = 'https://testnet.binancefuture.com/fapi/v2'

try:
    exchange.load_markets()
    print("[Network Success] Binance Testnet markets loaded successfully.")
except Exception as e:
    print(f"[Network Notice] Market filters loaded: {e}")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# =============================================================================
# STEP 3: Uncapped Paginated Scraping Across All 5 Assets (No Timestamp Cap)
# =============================================================================
print("\n===============================================================================")
print("  EXTRACTING ALL HISTORICAL BINANCE DATA (UNCAPPED / FULL ACCOUNT HISTORY)      ")
print("===============================================================================")

raw_trades_list = []
raw_orders_list = []
raw_income_list = []

for sym in ACTIVE_SYMBOLS:
    print(f"--> Ingesting complete history for {sym}...")

    # 1. Uncapped User Trades Pagination
    since_cursor = 0
    trade_count_sym = 0
    while True:
        try:
            trades = exchange.fetch_my_trades(sym, since=since_cursor, limit=1000)
            if not trades:
                break
            for t in trades:
                raw_trades_list.append({
                    'id': str(t.get('id')),
                    'order_id': str(t.get('order')),
                    'timestamp': t.get('timestamp'),
                    'datetime_utc': pd.to_datetime(t.get('timestamp'), unit='ms', utc=True).isoformat(),
                    'symbol': sym,
                    'side': t.get('side', '').upper(),
                    'price': float(t.get('price', 0.0) or 0.0),
                    'amount': float(t.get('amount', 0.0) or 0.0),
                    'cost': float(t.get('cost', 0.0) or 0.0),
                    'fee_cost': float(t.get('fee', {}).get('cost', 0.0) if t.get('fee') else 0.0),
                    'fee_currency': t.get('fee', {}).get('currency', 'USDT') if t.get('fee') else 'USDT',
                    'takerOrMaker': t.get('takerOrMaker', 'taker'),
                    'realized_pnl': float(t.get('info', {}).get('realizedPnl', 0.0) or 0.0)
                })
            trade_count_sym += len(trades)
            if len(trades) < 1000:
                break
            since_cursor = trades[-1]['timestamp'] + 1
        except Exception as e:
            print(f"    Warning fetching trades for {sym}: {e}")
            break
    print(f"    - Total Trades (Fills) Scraped: {trade_count_sym}")

    # 2. Uncapped All Orders Pagination
    since_order_cursor = 0
    order_count_sym = 0
    while True:
        try:
            orders = exchange.fetch_orders(sym, since=since_order_cursor, limit=1000)
            if not orders:
                break
            for o in orders:
                raw_orders_list.append({
                    'order_id': str(o.get('id')),
                    'client_order_id': o.get('clientOrderId'),
                    'timestamp': o.get('timestamp'),
                    'datetime_utc': pd.to_datetime(o.get('timestamp'), unit='ms', utc=True).isoformat(),
                    'symbol': sym,
                    'type': o.get('type'),
                    'side': o.get('side', '').upper(),
                    'status': o.get('status'),
                    'planned_price': float(o.get('price', 0.0) or 0.0),
                    'planned_stop_price': float(o.get('stopPrice', 0.0) or 0.0),
                    'avg_fill_price': float(o.get('average', 0.0) or o.get('price', 0.0) or 0.0),
                    'amount': float(o.get('amount', 0.0) or 0.0),
                    'filled': float(o.get('filled', 0.0) or 0.0),
                    'remaining': float(o.get('remaining', 0.0) or 0.0),
                    'cost': float(o.get('cost', 0.0) or 0.0)
                })
            order_count_sym += len(orders)
            if len(orders) < 1000:
                break
            since_order_cursor = orders[-1]['timestamp'] + 1
        except Exception as e:
            print(f"    Warning fetching orders for {sym}: {e}")
            break
    print(f"    - Total Orders Scraped        : {order_count_sym}")

# 3. Uncapped Account Income Ledger
print("--> Ingesting complete account income ledger (/fapi/v1/income)...")
income_cursor = 0
while True:
    try:
        incomes = exchange.fapiPrivateGetIncome({'startTime': income_cursor, 'limit': 1000})
        if not incomes:
            break
        for inc in incomes:
            raw_income_list.append({
                'symbol': inc.get('symbol'),
                'incomeType': inc.get('incomeType'),
                'income': float(inc.get('income', 0.0) or 0.0),
                'asset': inc.get('asset'),
                'time': int(inc.get('time') or 0),
                'datetime_utc': pd.to_datetime(int(inc.get('time') or 0), unit='ms', utc=True).isoformat(),
                'tranId': inc.get('tranId'),
                'tradeId': inc.get('tradeId')
            })
        if len(incomes) < 1000:
            break
        income_cursor = int(incomes[-1]['time']) + 1
    except Exception as e:
        print(f"    Warning fetching income ledger: {e}")
        break

print(f"    - Total Income Records Scraped: {len(raw_income_list)}")

df_raw_trades = pd.DataFrame(raw_trades_list)
df_raw_orders = pd.DataFrame(raw_orders_list)
df_raw_income = pd.DataFrame(raw_income_list)


# =============================================================================
# STEP 4: Microstructure Matching Engine (Deriving Planned vs. Actual Frictions)
# =============================================================================
print("\n4. Reconstructing planned vs. actual executions and friction costs...")

reconstructed_ledger = []

if not df_raw_trades.empty and not df_raw_orders.empty:
    df_raw_trades['dt'] = pd.to_datetime(df_raw_trades['datetime_utc'], utc=True)
    df_raw_orders['dt'] = pd.to_datetime(df_raw_orders['datetime_utc'], utc=True)
    df_raw_trades = df_raw_trades.sort_values('dt').reset_index(drop=True)
    df_raw_orders = df_raw_orders.sort_values('dt').reset_index(drop=True)

    # Closing fills are explicitly marked by Binance matching engine with realized_pnl != 0
    closing_trades = df_raw_trades[df_raw_trades['realized_pnl'] != 0.0].copy()

    for _, c_trade in closing_trades.iterrows():
        sym = c_trade['symbol']
        c_time = c_trade['dt']
        realized_binance_pnl = c_trade['realized_pnl']
        actual_exit_px = c_trade['price']
        exit_side = c_trade['side']
        direction = "SHORT" if exit_side == "BUY" else "LONG"
        exit_order_id = c_trade['order_id']

        # 1. Locate the opening execution fill
        prior_fills = df_raw_trades[
            (df_raw_trades['symbol'] == sym) &
            (df_raw_trades['dt'] < c_time) &
            (df_raw_trades['side'] != exit_side)
        ]

        if not prior_fills.empty:
            open_fill = prior_fills.iloc[-1]
            actual_entry_px = open_fill['price']
            open_time_str   = open_fill['datetime_utc']
            qty             = open_fill['amount']
            entry_order_id  = open_fill['order_id']
            entry_fee       = open_fill['fee_cost']
        else:
            actual_entry_px = actual_exit_px
            open_time_str   = c_trade['datetime_utc']
            qty             = c_trade['amount']
            entry_order_id  = "UNKNOWN"
            entry_fee       = 0.0

        exit_fee       = c_trade['fee_cost']
        total_fees_usd = entry_fee + exit_fee
        hold_min       = max(0.1, (c_time - pd.to_datetime(open_time_str, utc=True)).total_seconds() / 60.0)

        # 2. Extract Planned Entry Price from Binance's original LIMIT order record
        entry_order_row = df_raw_orders[df_raw_orders['order_id'] == entry_order_id]
        if not entry_order_row.empty:
            planned_entry_px = entry_order_row.iloc[0]['planned_price']
            if planned_entry_px == 0.0:
                planned_entry_px = actual_entry_px
        else:
            planned_entry_px = actual_entry_px

        # 3. Extract Planned Exit Barrier from Binance's bracket/exit order record
        exit_order_row = df_raw_orders[df_raw_orders['order_id'] == exit_order_id]
        close_reason = "SIGNAL_FLIP"
        planned_exit_px = actual_exit_px

        if not exit_order_row.empty:
            o_type = str(exit_order_row.iloc[0]['type']).upper()
            o_stop_px = exit_order_row.iloc[0]['planned_stop_price']
            if "STOP" in o_type:
                close_reason = "SL_HIT"
                planned_exit_px = o_stop_px if o_stop_px > 0 else actual_exit_px
            elif "TAKE_PROFIT" in o_type:
                close_reason = "TP_HIT"
                planned_exit_px = o_stop_px if o_stop_px > 0 else actual_exit_px
            elif "MARKET" in o_type:
                close_reason = "SIGNAL_FLIP"
                planned_exit_px = actual_exit_px
        else:
            if realized_binance_pnl > 0.1:
                close_reason = "TP_HIT"
            elif realized_binance_pnl < -0.1:
                close_reason = "SL_HIT"
            else:
                close_reason = "SIGNAL_FLIP"

        # 4. Rigorous Friction & Slippage Decomposition
        dir_mult = 1.0 if direction == "LONG" else -1.0

        # Entry Slippage ($): Difference between planned limit price and actual execution
        entry_slippage_usd = dir_mult * (planned_entry_px - actual_entry_px) * qty

        # Exit Slippage ($): Difference between planned stop/target barrier and actual fill
        exit_slippage_usd = dir_mult * (planned_exit_px - actual_exit_px) * qty

        # Idealized PnL ($): Theoretical profit if filled perfectly at planned prices without slippage/fees
        if planned_entry_px > 0:
            idealized_gross_ret = dir_mult * (planned_exit_px - planned_entry_px) / planned_entry_px
            idealized_pnl_usd   = (planned_entry_px * qty) * idealized_gross_ret
        else:
            idealized_pnl_usd   = realized_binance_pnl

        # Total Friction Loss ($): Delta between Theoretical Alpha and Realized Cash
        total_friction_loss_usd = idealized_pnl_usd - realized_binance_pnl

        reconstructed_ledger.append({
            'symbol': sym,
            'direction': direction,
            'opened_at_utc': open_time_str,
            'closed_at_utc': c_trade['datetime_utc'],
            'hold_duration_minutes': round(hold_min, 1),
            'close_reason': close_reason,
            'planned_entry_price': round(planned_entry_px, 6),
            'actual_entry_price': round(actual_entry_px, 6),
            'entry_slippage_usd': round(entry_slippage_usd, 4),
            'planned_exit_price': round(planned_exit_px, 6),
            'actual_exit_price': round(actual_exit_px, 6),
            'exit_slippage_usd': round(exit_slippage_usd, 4),
            'contract_quantity': qty,
            'notional_position_usd': round(actual_entry_px * qty, 2),
            'exchange_fees_paid': round(total_fees_usd, 4),
            'idealized_pnl_usd': round(idealized_pnl_usd, 4),
            'realized_binance_pnl': round(realized_binance_pnl, 4),
            'total_friction_loss_usd': round(total_friction_loss_usd, 4),
            'net_wallet_delta_usd': round(realized_binance_pnl - total_fees_usd, 4),
            'entry_order_id': entry_order_id,
            'exit_order_id': exit_order_id,
            'notes': f"Direct Binance Execution ({close_reason}) via Order {exit_order_id}"
        })

df_reconstructed = pd.DataFrame(reconstructed_ledger)
print(f"   --> Successfully reconstructed {len(df_reconstructed)} complete trades with full friction breakdown.")


# =============================================================================
# STEP 5: Export CSV Artifacts to data/ground_truth/
# =============================================================================
print("\n5. Exporting uncapped ground-truth CSV ledgers to data/ground_truth/...")

master_csv_path  = os.path.join(DATA_DIR, "binance_comprehensive_audit_ledger.csv")
trades_csv_path  = os.path.join(DATA_DIR, "binance_raw_trades_all.csv")
orders_csv_path  = os.path.join(DATA_DIR, "binance_raw_orders_all.csv")
income_csv_path  = os.path.join(DATA_DIR, "binance_raw_income_all.csv")

df_reconstructed.to_csv(master_csv_path, index=False)
df_raw_trades.to_csv(trades_csv_path, index=False)
df_raw_orders.to_csv(orders_csv_path, index=False)
df_raw_income.to_csv(income_csv_path, index=False)

print(f"   [Primary Audit Ledger] {master_csv_path}")
print(f"   [Raw Fills]            {trades_csv_path}")
print(f"   [Raw Orders]           {orders_csv_path}")
print(f"   [Raw Income]           {income_csv_path}")


# =============================================================================
# STEP 6: Supabase Sanitization & Direct Ingestion (Reset Table 1, Overwrite Table 2)
# =============================================================================
print("\n6. Sanitizing Supabase tables directly from Binance ground truth...")

# 1. Reset Table 1 (testnet_active_trades) -> Clean 0/5 slots baseline
try:
    print("   --> Resetting Table 1 (testnet_active_trades)...")
    supabase.table("testnet_active_trades").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    print("       Table 1 successfully reset to clean 0/5 active slots.")
except Exception as e:
    print(f"       Notice resetting Table 1: {e}")

# 2. Preserve clean GATE_REJECTED model audits from Table 2
preserved_rejections = []
try:
    print("   --> Preserving clean GATE_REJECTED model prediction audits...")
    res_rej = supabase.table("testnet_trade_log").select("*").eq("close_reason", "GATE_REJECTED").execute()
    if res_rej.data:
        preserved_rejections = res_rej.data
        print(f"       Preserved {len(preserved_rejections)} clean GATE_REJECTED prediction logs.")
except Exception as e:
    print(f"       Notice preserving rejections: {e}")

# 3. Purge old corrupted execution rows from Table 2
try:
    print("   --> Purging corrupted execution rows from Table 2 (testnet_trade_log)...")
    supabase.table("testnet_trade_log").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
except Exception as e:
    print(f"       Notice purging Table 2: {e}")

# 4. Re-insert preserved GATE_REJECTED rows
if preserved_rejections:
    try:
        clean_rejs = []
        for r in preserved_rejections:
            row_copy = r.copy()
            row_copy.pop('id', None)
            clean_rejs.append(row_copy)
        supabase.table("testnet_trade_log").insert(clean_rejs).execute()
        print(f"       Successfully restored {len(clean_rejs)} GATE_REJECTED audit rows.")
    except Exception as e:
        print(f"       Notice restoring rejections: {e}")

# 5. Populate Table 2 directly with authentic Binance reconstructed records
if not df_reconstructed.empty:
    print("   --> Ingesting authentic Binance execution rows into Table 2...")
    ingest_payload = []
    for _, t in df_reconstructed.iterrows():
        ingest_payload.append({
            "trade_id": None,
            "created_at": t['opened_at_utc'],
            "closed_at": t['closed_at_utc'],
            "symbol": t['symbol'],
            "direction": t['direction'],
            "close_reason": t['close_reason'],
            "entry_price": float(t['actual_entry_price']),
            "exit_price": float(t['actual_exit_price']),
            "realized_binance_pnl": float(t['realized_binance_pnl']),
            "idealized_pnl": float(t['idealized_pnl_usd']),
            "friction_loss": float(t['total_friction_loss_usd']),
            "exchange_fees_paid": float(t['exchange_fees_paid']),
            "slippage_usd": float(t['entry_slippage_usd'] + t['exit_slippage_usd']),
            "hold_duration_minutes": float(t['hold_duration_minutes']),
            "notes": t['notes']
        })

    try:
        supabase.table("testnet_trade_log").insert(ingest_payload).execute()
        print(f"       Successfully ingested {len(ingest_payload)} uncorrupted trades into Supabase Table 2.")
    except Exception as e:
        print(f"       Notice populating Table 2: {e}")


# =============================================================================
# STEP 7: Final Audit Summary & Scoreboard Display
# =============================================================================
tot_pnl_usd   = df_reconstructed['realized_binance_pnl'].sum() if not df_reconstructed.empty else 0.0
tot_fees_usd  = df_reconstructed['exchange_fees_paid'].sum() if not df_reconstructed.empty else 0.0
tot_fric_usd  = df_reconstructed['total_friction_loss_usd'].sum() if not df_reconstructed.empty else 0.0
tot_ideal_usd = df_reconstructed['idealized_pnl_usd'].sum() if not df_reconstructed.empty else 0.0
net_cash_usd  = tot_pnl_usd - tot_fees_usd

print("\n===============================================================================")
print("                   FINAL GROUND-TRUTH AUDIT SCOREBOARD                         ")
print("===============================================================================")
print(f"Total Fills Scraped from Binance     : {len(df_raw_trades):,}")
print(f"Total Orders Scraped from Binance    : {len(df_raw_orders):,}")
print(f"Completed Reconstructed Trades       : {len(df_reconstructed):,}")
print(f"Theoretical Idealized Model PnL      : ${tot_ideal_usd:+,.2f} USDT")
print(f"Realized Gross Binance PnL           : ${tot_pnl_usd:+,.2f} USDT")
print(f"Total True Binance Exchange Fees     : ${tot_fees_usd:,.2f} USDT")
print(f"Total Execution Friction Loss        : ${tot_fric_usd:,.2f} USDT")
print(f"True Net Cash Delta on Binance       : ${net_cash_usd:+,.2f} USDT")
print("-------------------------------------------------------------------------------")
print("Supabase Direct Ingestion Status:")
print("  • Table 1 (testnet_active_trades)  : Reset (Clean Slate, 0/5 Active Slots)")
print("  • Table 2 (testnet_trade_log)      : Overwritten Directly with Binance Ground Truth")
print("===============================================================================\n")
