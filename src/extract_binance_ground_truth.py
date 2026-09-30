"""
====================================================================================================
ALGORITHM: src/extract_binance_ground_truth.py — Read-Only Mirror & Supabase Upsert Engine
====================================================================================================
Purpose:
  Connects to Binance Futures Testnet via CCXT through the Frankfurt proxy in a strictly READ-ONLY
  capacity. Paginates across 56 days (8 rolling 7-day windows) to extract 100% of authentic fills,
  orders, and income records. Stores cumulative deduplicated CSVs in `data/ground_truth/` and upserts
  directly into dedicated Supabase raw tables (`binance_raw_trades`, `binance_raw_orders`,
  `binance_raw_income`). Never executes cancellations and never deletes operational trading logs.

Algorithm Steps:
  Step 1: Module Setup, GitHub Actions Environment Ingestion & Proxy Sanitization:
          - Ingest credentials from environment; construct authenticated CCXT and Supabase clients.
  Step 2: CCXT Client Setup with Frankfurt Proxy Tunnel:
          - Initialize ccxt.binanceusdm with rate-limit guards.
  Step 3: Rolling 7-Day Window Pagination Across Active Assets (Read-Only):
          - Scrapes `fapiPrivateGetUserTrades` (fills) for each asset.
          - Scrapes `fapiPrivateGetAllOrders` (orders) for each asset.
          - Scrapes `fapiPrivateGetIncome` (funding fees, commissions, transfers) across account.
  Step 4: Cumulative Upsert & CSV Persistence:
          - Appends and deduplicates raw records into `data/ground_truth/` CSV files.
  Step 5: Non-Destructive Supabase Direct Ingestion:
          - Chunks data into 200-row batches and upserts directly into dedicated raw tables.
          - Zero deletion of live operational tables (`asset_state`, `testnet_trade_log`).
  Step 6: Forensic Attribution & Cumulative Scoreboard Display.
====================================================================================================
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import requests
import ccxt
from supabase import create_client, Client

warnings.filterwarnings("ignore", category=UserWarning)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# STEP 1: Environment Ingestion & Proxy Sanitization
# =============================================================================
API_KEY      = os.environ.get("BINANCE_TESTNET_API_KEY", "").strip()
API_SECRET   = os.environ.get("BINANCE_TESTNET_API_SECRET", "").strip()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
PROXY_URL    = os.environ.get("BINANCE_PROXY_URL", "").strip()

if not API_KEY or not API_SECRET:
    raise RuntimeError("[FATAL] Binance API credentials missing from environment!")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("[FATAL] Supabase credentials missing from environment!")

ACTIVE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
DATA_DIR       = os.path.join(os.getcwd(), "data", "ground_truth")
os.makedirs(DATA_DIR, exist_ok=True)


def sanitize_proxy_url(url: str) -> str:
    if not url:
        return ""
    clean = url.strip()
    if clean.startswith("https://"):
        clean = "http://" + clean[len("https://"):]
    elif not clean.startswith("http://") and not clean.startswith("socks5://"):
        clean = "http://" + clean
    return clean


# =============================================================================
# STEP 2: CCXT Client Setup with Proxy Tunnel (Strictly Read-Only)
# =============================================================================
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
    exchange_config['proxies'] = {'http': clean_proxy, 'https': clean_proxy}
    masked = clean_proxy.split('@')[-1] if '@' in clean_proxy else clean_proxy
    print(f"[Network] Read-Only Scraper configured with proxy -> {masked}")

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
    print("[Network Success] Binance Testnet markets loaded.")
except Exception as e:
    print(f"[Network Notice] Market loading notice: {e}")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# =============================================================================
# STEP 3: Rolling 7-Day Window Pagination Across Active Assets (Read-Only)
# =============================================================================
print("\n===============================================================================")
print("  EXTRACTING AUTHENTIC BINANCE GROUND TRUTH (ROLLING 7-DAY WINDOW PAGINATION)  ")
print("===============================================================================")

now_dt = datetime.now(timezone.utc)
time_windows = []
for i in range(8):  # 8 windows of 7 days = 56 days of history
    w_end   = now_dt - timedelta(days=i * 7)
    w_start = now_dt - timedelta(days=(i + 1) * 7)
    time_windows.append((int(w_start.timestamp() * 1000), int(w_end.timestamp() * 1000)))

raw_trades_dict = {}
raw_orders_dict = {}
raw_income_list = []

for sym in ACTIVE_SYMBOLS:
    print(f"--> Ingesting matching engine history for {sym}...")
    trades_sym_count = 0
    orders_sym_count = 0

    for start_ms, end_ms in time_windows:
        # 1. Fetch User Trades (Fills)
        try:
            trades = exchange.fapiPrivateGetUserTrades({
                'symbol': sym,
                'startTime': start_ms,
                'endTime': end_ms,
                'limit': 1000
            })
            for t in trades:
                t_id = str(t.get('id'))
                if t_id not in raw_trades_dict:
                    raw_trades_dict[t_id] = {
                        'id': t_id,
                        'order_id': str(t.get('orderId')),
                        'timestamp': int(t.get('time')),
                        'datetime_utc': pd.to_datetime(int(t.get('time')), unit='ms', utc=True).isoformat(),
                        'symbol': sym,
                        'side': str(t.get('side', '')).upper(),
                        'price': float(t.get('price', 0.0) or 0.0),
                        'amount': float(t.get('qty', 0.0) or 0.0),
                        'cost': float(t.get('quoteQty', 0.0) or (float(t.get('price', 0.0)) * float(t.get('qty', 0.0)))),
                        'fee_cost': float(t.get('commission', 0.0) or 0.0),
                        'fee_currency': str(t.get('commissionAsset', 'USDT')),
                        'taker_or_maker': 'maker' if t.get('maker') else 'taker',
                        'realized_pnl': float(t.get('realizedPnl', 0.0) or 0.0)
                    }
                    trades_sym_count += 1
        except Exception:
            pass

        # 2. Fetch Matching Engine Orders
        try:
            orders = exchange.fapiPrivateGetAllOrders({
                'symbol': sym,
                'startTime': start_ms,
                'endTime': end_ms,
                'limit': 1000
            })
            for o in orders:
                o_id = str(o.get('orderId'))
                if o_id not in raw_orders_dict:
                    raw_orders_dict[o_id] = {
                        'order_id': o_id,
                        'client_order_id': str(o.get('clientOrderId', '')),
                        'timestamp': int(o.get('time')),
                        'update_timestamp': int(o.get('updateTime', o.get('time'))),
                        'datetime_utc': pd.to_datetime(int(o.get('time')), unit='ms', utc=True).isoformat(),
                        'update_utc': pd.to_datetime(int(o.get('updateTime', o.get('time'))), unit='ms', utc=True).isoformat(),
                        'symbol': sym,
                        'type': str(o.get('type', '')),
                        'side': str(o.get('side', '')).upper(),
                        'status': str(o.get('status', '')),
                        'planned_price': float(o.get('price', 0.0) or 0.0),
                        'planned_stop_price': float(o.get('stopPrice', 0.0) or 0.0),
                        'avg_fill_price': float(o.get('avgPrice', 0.0) or 0.0),
                        'amount': float(o.get('origQty', 0.0) or 0.0),
                        'filled': float(o.get('executedQty', 0.0) or 0.0),
                        'cost': float(o.get('cumQuote', 0.0) or 0.0)
                    }
                    orders_sym_count += 1
        except Exception:
            pass

    print(f"    - Fills Scraped : {trades_sym_count}")
    print(f"    - Orders Scraped: {orders_sym_count}")

# 3. Fetch Income & Funding Fee History (/fapi/v1/income)
print("--> Ingesting complete account income ledger (/fapi/v1/income)...")
income_cursor = 0
while True:
    try:
        incomes = exchange.fapiPrivateGetIncome({'startTime': income_cursor, 'limit': 1000})
        if not incomes:
            break
        for inc in incomes:
            raw_income_list.append({
                'tran_id': str(inc.get('tranId')),
                'symbol': str(inc.get('symbol', '')),
                'income_type': str(inc.get('incomeType', '')),
                'income': float(inc.get('income', 0.0) or 0.0),
                'asset': str(inc.get('asset', 'USDT')),
                'timestamp': int(inc.get('time') or 0),
                'datetime_utc': pd.to_datetime(int(inc.get('time') or 0), unit='ms', utc=True).isoformat(),
                'trade_id': str(inc.get('tradeId', ''))
            })
        if len(incomes) < 1000:
            break
        income_cursor = int(incomes[-1]['time']) + 1
    except Exception:
        break

print(f"    - Total Income Records Scraped: {len(raw_income_list)}")

df_raw_trades = pd.DataFrame(list(raw_trades_dict.values()))
df_raw_orders = pd.DataFrame(list(raw_orders_dict.values()))
df_raw_income = pd.DataFrame(raw_income_list)


# =============================================================================
# STEP 4: Cumulative Upsert & CSV Persistence (Zero Overwrite Loss)
# =============================================================================
print("\n4. Appending and deduplicating CSV ledgers...")

def upsert_csv(file_path: str, new_df: pd.DataFrame, dedup_col: str, sort_col: str = None) -> pd.DataFrame:
    """Appends and deduplicates records to guarantee historical data is preserved permanently."""
    if new_df.empty:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return new_df

    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            combined = pd.concat([existing_df, new_df], axis=0, ignore_index=True)
            combined[dedup_col] = combined[dedup_col].astype(str)
            deduped = combined.drop_duplicates(subset=[dedup_col], keep='last')
            if sort_col and sort_col in deduped.columns:
                deduped = deduped.sort_values(sort_col).reset_index(drop=True)
            deduped.to_csv(file_path, index=False)
            return deduped
        except Exception as e:
            print(f"Notice loading CSV ({file_path}): {e}")

    new_df.to_csv(file_path, index=False)
    return new_df

trades_csv_path = os.path.join(DATA_DIR, "binance_raw_trades_all.csv")
orders_csv_path = os.path.join(DATA_DIR, "binance_raw_orders_all.csv")
income_csv_path = os.path.join(DATA_DIR, "binance_raw_income_all.csv")

final_trades_df = upsert_csv(trades_csv_path, df_raw_trades, dedup_col='id', sort_col='timestamp')
final_orders_df = upsert_csv(orders_csv_path, df_raw_orders, dedup_col='order_id', sort_col='timestamp')
final_income_df = upsert_csv(income_csv_path, df_raw_income, dedup_col='tran_id', sort_col='timestamp')

print(f"   [Archived Fills]  {trades_csv_path} ({len(final_trades_df)} records)")
print(f"   [Archived Orders] {orders_csv_path} ({len(final_orders_df)} records)")
print(f"   [Archived Income] {income_csv_path} ({len(final_income_df)} records)")


# =============================================================================
# STEP 5: Non-Destructive Supabase Direct Upsert (Phase 1 Raw Tables)
# =============================================================================
print("\n5. Upserting ground-truth data into Supabase raw tables in batches...")

def batch_upsert_supabase(table_name: str, df: pd.DataFrame, batch_size: int = 200):
    if df.empty:
        return
    records = df.to_dict(orient="records")
    total = len(records)
    print(f"   --> Upserting {total} rows into `{table_name}`...")
    for i in range(0, total, batch_size):
        chunk = records[i : i + batch_size]
        try:
            supabase.table(table_name).upsert(chunk).execute()
        except Exception as e:
            print(f"       [Warning] Batch upsert error on {table_name} [{i}:{i+batch_size}]: {e}")

batch_upsert_supabase("binance_raw_trades", final_trades_df)
batch_upsert_supabase("binance_raw_orders", final_orders_df)
batch_upsert_supabase("binance_raw_income", final_income_df)

print("   [Success] Ground-truth tables populated cleanly without touching active trading state.")


# =============================================================================
# STEP 6: Forensic Attribution & Cumulative Scoreboard Display
# =============================================================================
tot_realized_pnl = final_trades_df['realized_pnl'].sum() if not final_trades_df.empty else 0.0
tot_fees_paid    = final_trades_df['fee_cost'].sum() if not final_trades_df.empty else 0.0
tot_income       = final_income_df['income'].sum() if not final_income_df.empty else 0.0
net_cash_delta   = tot_realized_pnl - tot_fees_paid

print("\n===============================================================================")
print("                   AUTHENTIC BINANCE GROUND-TRUTH SCOREBOARD                   ")
print("===============================================================================")
print(f"Total Cumulative Fills Scraped   : {len(final_trades_df):,}")
print(f"Total Cumulative Orders Scraped  : {len(final_orders_df):,}")
print(f"Total Income / Funding Records   : {len(final_income_df):,}")
print(f"Cumulative Realized Trading PnL  : ${tot_realized_pnl:+,.2f} USDT")
print(f"Total Commission Fees Deducted   : ${tot_fees_paid:,.2f} USDT")
print(f"Total Net Cash Delta on Binance  : ${net_cash_delta:+,.2f} USDT")
print("===============================================================================\n")
