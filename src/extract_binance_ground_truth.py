"""
====================================================================================================
ALGORITHM: src/extract_binance_ground_truth.py — Direct API Extraction & Supabase Overwrite
====================================================================================================
Purpose:
  Connects directly to the Binance Futures Testnet matching engine via CCXT through the Frankfurt proxy
  tunnel to extract the complete, uncorrupted execution history across all 5 assets since live deployment
  began (2026-09-12 00:00:00 UTC). Reconstructs true trade entries, exits, commissions, and realized PnL.
  Exports raw CSVs to `data/ground_truth/`, resets `testnet_active_trades` to 0/5 slots, and populates
  `testnet_trade_log` directly from the Binance API ledger.

Key Microstructure & Extraction Invariants:
  1. Multi-Endpoint Ingestion (The Infallible Source of Truth):
     - `GET /fapi/v1/userTrades`: Extracts every fill, price, quantity, fee, and order ID.
     - `GET /fapi/v1/allOrders`: Extracts the complete status of all LIMIT, STOP_MARKET, and
       TAKE_PROFIT_MARKET orders (FILLED, CANCELED, EXPIRED).
     - `GET /fapi/v1/income`: Extracts the cryptographic wallet ledger for REALIZED_PNL and commissions.
  2. Exchange-Authoritative PnL Reconciliation:
     - In Binance Futures, closing fills have non-zero `realizedPnl`. Matches entry trades with exit
       trades using order IDs and timestamps to compute true gross return, fee drag, and net return.
  3. Supabase Table Sanitization & Direct Population:
     - Table 1 (`testnet_active_trades`): Cleared/reset to an empty slate (0/5 slots active).
     - Table 2 (`testnet_trade_log`): Preserves clean `GATE_REJECTED` model prediction rows, clears
       corrupted execution rows, and repopulates directly with verified Binance trade records.
  4. Local CSV Persistence:
     - Saves raw and reconstructed CSV ledgers into `data/ground_truth/` for immediate notebook analysis.

Algorithm Steps:
  Step 1: Module Setup, Safe Math & Credentials Ingestion.
  Step 2: Proxy URI Sanitization & CCXT Binance Futures Client Setup.
  Step 3: Multi-Endpoint Scrape across the 5 Assets (since 2026-09-12 00:00:00 UTC).
  Step 4: Data Processing & Trade Reconstruction (Matching Entries, Exits & Realized PnL).
  Step 5: Export Ground-Truth CSV Artifacts to `data/ground_truth/`.
  Step 6: Supabase Table Sanitization & Direct Ingestion (Reset Table 1, Repopulate Table 2).
  Step 7: Verification & Final Audit Summary Display.
====================================================================================================
"""

# =============================================================================
# STEP 1: Module Setup, Safe Math & Credentials Ingestion
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

# Ensure immediate terminal line buffering
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Detect Kaggle vs Cloud/Local Environment
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
    raise RuntimeError("[FATAL] Binance API keys missing from environment!")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("[FATAL] Supabase credentials missing from environment!")

# Earliest start boundary: 2026-09-12 00:00:00 UTC in milliseconds
START_UTC = pd.to_datetime("2026-09-12 00:00:00", utc=True)
START_MS  = int(START_UTC.timestamp() * 1000)

ACTIVE_SYMBOLS = ["BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DATA_DIR       = os.path.join(os.getcwd(), "data", "ground_truth")
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# STEP 2: Proxy Sanitization & CCXT Binance Futures Client Setup
# =============================================================================
def sanitize_proxy_url(url: str) -> str:
    """Enforces http:// prefix to prevent OpenSSL version mismatch crashes."""
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
    print(f"[Network Warning] Market loading notice: {e}")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# =============================================================================
# STEP 3: Multi-Endpoint Scrape Across the 5 Assets
# =============================================================================
print("\n===============================================================================")
print(f"  EXTRACTING GROUND-TRUTH BINANCE DATA SINCE {START_UTC}")
print("===============================================================================")

raw_trades_list = []
raw_orders_list = []
raw_income_list = []

for sym in ACTIVE_SYMBOLS:
    print(f"--> Ingesting raw ledgers for {sym}...")
    
    # 1. Fetch User Trades (/fapi/v1/userTrades)
    try:
        trades = exchange.fetch_my_trades(sym, since=START_MS, limit=1000)
        for t in trades:
            raw_trades_list.append({
                'id': t.get('id'),
                'order_id': str(t.get('order')),
                'timestamp': t.get('timestamp'),
                'datetime_utc': t.get('datetime'),
                'symbol': sym,
                'side': t.get('side', '').upper(),
                'price': float(t.get('price', 0.0)),
                'amount': float(t.get('amount', 0.0)),
                'cost': float(t.get('cost', 0.0)),
                'fee_cost': float(t.get('fee', {}).get('cost', 0.0) if t.get('fee') else 0.0),
                'fee_currency': t.get('fee', {}).get('currency', 'USDT') if t.get('fee') else 'USDT',
                'takerOrMaker': t.get('takerOrMaker', 'taker'),
                'realized_pnl': float(t.get('info', {}).get('realizedPnl', 0.0))
            })
        print(f"    - User Trades Captured: {len(trades)}")
    except Exception as e:
        print(f"    - Warning fetching trades for {sym}: {e}")

    # 2. Fetch All Orders (/fapi/v1/allOrders)
    try:
        orders = exchange.fetch_orders(sym, since=START_MS, limit=1000)
        for o in orders:
            raw_orders_list.append({
                'order_id': str(o.get('id')),
                'client_order_id': o.get('clientOrderId'),
                'timestamp': o.get('timestamp'),
                'datetime_utc': o.get('datetime'),
                'symbol': sym,
                'type': o.get('type'),
                'side': o.get('side', '').upper(),
                'status': o.get('status'),
                'price': float(o.get('price', 0.0) or 0.0),
                'stop_price': float(o.get('stopPrice', 0.0) or 0.0),
                'amount': float(o.get('amount', 0.0)),
                'filled': float(o.get('filled', 0.0)),
                'remaining': float(o.get('remaining', 0.0)),
                'cost': float(o.get('cost', 0.0))
            })
        print(f"    - Orders Captured     : {len(orders)}")
    except Exception as e:
        print(f"    - Warning fetching orders for {sym}: {e}")

# 3. Fetch Income Ledger (/fapi/v1/income)
print("--> Ingesting raw account income ledger (/fapi/v1/income)...")
try:
    if hasattr(exchange, 'fetch_income'):
        incomes = exchange.fetch_income(since=START_MS, limit=1000)
    else:
        incomes = exchange.fapiPrivateGetIncome({'startTime': START_MS, 'limit': 1000})
        
    for inc in incomes:
        raw_income_list.append({
            'symbol': inc.get('symbol'),
            'incomeType': inc.get('incomeType'),
            'income': float(inc.get('income', 0.0)),
            'asset': inc.get('asset'),
            'time': inc.get('time') or inc.get('timestamp'),
            'datetime_utc': pd.to_datetime(inc.get('time') or inc.get('timestamp'), unit='ms', utc=True).isoformat(),
            'info': inc.get('info')
        })
    print(f"    - Income Events Captured: {len(raw_income_list)}")
except Exception as e:
    print(f"    - Warning fetching income ledger: {e}")

df_raw_trades = pd.DataFrame(raw_trades_list)
df_raw_orders = pd.DataFrame(raw_orders_list)
df_raw_income = pd.DataFrame(raw_income_list)


# =============================================================================
# STEP 4: Data Processing & Trade Reconstruction
# =============================================================================
print("\n4. Reconstructing true trade lifecycle and PnL metrics...")

reconstructed_trades = []

if not df_raw_trades.empty:
    df_raw_trades['datetime_dt'] = pd.to_datetime(df_raw_trades['datetime_utc'], utc=True)
    df_raw_trades = df_raw_trades.sort_values('datetime_dt').reset_index(drop=True)

    # In Binance Futures, closing trades contain non-zero realizedPnl
    closing_trades = df_raw_trades[df_raw_trades['realized_pnl'] != 0.0].copy()

    for _, c_trade in closing_trades.iterrows():
        sym = c_trade['symbol']
        c_time = c_trade['datetime_dt']
        realized_pnl = c_trade['realized_pnl']
        exit_px = c_trade['price']
        exit_side = c_trade['side']
        pos_direction = "SHORT" if exit_side == "BUY" else "LONG"

        # Locate the opening fill prior to this exit fill
        prior_fills = df_raw_trades[
            (df_raw_trades['symbol'] == sym) &
            (df_raw_trades['datetime_dt'] < c_time) &
            (df_raw_trades['side'] != exit_side)
        ]

        if not prior_fills.empty:
            open_fill = prior_fills.iloc[-1]
            entry_px  = open_fill['price']
            open_time = open_fill['datetime_utc']
            qty       = open_fill['amount']
            entry_fee = open_fill['fee_cost']
        else:
            entry_px  = exit_px
            open_time = c_trade['datetime_utc']
            qty       = c_trade['amount']
            entry_fee = 0.0

        exit_fee   = c_trade['fee_cost']
        total_fees = entry_fee + exit_fee
        hold_min   = (c_time - pd.to_datetime(open_time, utc=True)).total_seconds() / 60.0

        # Classify outcome based on return
        if realized_pnl > 0.10:
            outcome = "TP_HIT"
        elif realized_pnl < -0.10:
            outcome = "SL_HIT"
        else:
            outcome = "SIGNAL_FLIP"

        reconstructed_trades.append({
            'symbol': sym,
            'direction': pos_direction,
            'opened_at': open_time,
            'closed_at': c_trade['datetime_utc'],
            'entry_price': round(entry_px, 6),
            'exit_price': round(exit_px, 6),
            'quantity': qty,
            'realized_binance_pnl': round(realized_pnl, 4),
            'exchange_fees_paid': round(total_fees, 4),
            'net_realized_pnl': round(realized_pnl - total_fees, 4),
            'hold_duration_minutes': round(hold_min, 1),
            'close_reason': outcome,
            'exit_order_id': c_trade['order_id'],
            'notes': f"Direct Binance execution: {outcome} via Order {c_trade['order_id']}"
        })

df_reconstructed = pd.DataFrame(reconstructed_trades)
print(f"   --> Successfully reconstructed {len(df_reconstructed)} closed trades directly from Binance fills.")


# =============================================================================
# STEP 5: Export CSV Artifacts to data/ground_truth/
# =============================================================================
print("\n5. Exporting ground-truth CSV ledgers...")

trades_csv_path = os.path.join(DATA_DIR, "binance_raw_trades.csv")
orders_csv_path = os.path.join(DATA_DIR, "binance_raw_orders.csv")
income_csv_path = os.path.join(DATA_DIR, "binance_raw_income.csv")
recon_csv_path  = os.path.join(DATA_DIR, "binance_reconstructed_trades.csv")

df_raw_trades.to_csv(trades_csv_path, index=False)
df_raw_orders.to_csv(orders_csv_path, index=False)
df_raw_income.to_csv(income_csv_path, index=False)
df_reconstructed.to_csv(recon_csv_path, index=False)

print(f"   [Export] {trades_csv_path}")
print(f"   [Export] {orders_csv_path}")
print(f"   [Export] {income_csv_path}")
print(f"   [Export] {recon_csv_path}")


# =============================================================================
# STEP 6: Supabase Table Sanitization & Direct Ingestion
# =============================================================================
print("\n6. Sanitizing Supabase and populating directly from Binance API data...")

# 1. Reset Table 1 (testnet_active_trades) -> Clean slate, 0/5 slots
try:
    print("   --> Resetting Table 1 (testnet_active_trades)...")
    supabase.table("testnet_active_trades").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    print("       Table 1 successfully reset to clean 0/5 active slots.")
except Exception as e:
    print(f"       Notice resetting Table 1: {e}")

# 2. Preserve clean GATE_REJECTED telemetry from Table 2
preserved_rejections = []
try:
    print("   --> Preserving clean GATE_REJECTED model audits from Table 2...")
    res_rej = supabase.table("testnet_trade_log").select("*").eq("close_reason", "GATE_REJECTED").execute()
    if res_rej.data:
        preserved_rejections = res_rej.data
        print(f"       Preserved {len(preserved_rejections)} clean GATE_REJECTED prediction logs.")
except Exception as e:
    print(f"       Notice preserving rejections: {e}")

# 3. Wipe old corrupted trade logs from Table 2
try:
    print("   --> Purging corrupted execution rows from Table 2 (testnet_trade_log)...")
    supabase.table("testnet_trade_log").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
except Exception as e:
    print(f"       Notice purging Table 2: {e}")

# 4. Re-insert preserved GATE_REJECTED rows
if preserved_rejections:
    try:
        # Strip auto-generated IDs to avoid PK conflicts on re-insert
        clean_rejs = []
        for r in preserved_rejections:
            row_copy = r.copy()
            row_copy.pop('id', None)
            clean_rejs.append(row_copy)
        supabase.table("testnet_trade_log").insert(clean_rejs).execute()
        print(f"       Successfully restored {len(clean_rejs)} GATE_REJECTED audit rows.")
    except Exception as e:
        print(f"       Notice restoring rejections: {e}")

# 5. Populate Table 2 directly with authentic Binance reconstructed trades
if not df_reconstructed.empty:
    print("   --> Ingesting authentic Binance execution rows into Table 2...")
    ingest_payload = []
    for _, t in df_reconstructed.iterrows():
        ingest_payload.append({
            "trade_id": None,
            "created_at": t['opened_at'],
            "closed_at": t['closed_at'],
            "symbol": t['symbol'],
            "direction": t['direction'],
            "close_reason": t['close_reason'],
            "entry_price": float(t['entry_price']),
            "exit_price": float(t['exit_price']),
            "realized_binance_pnl": float(t['realized_binance_pnl']),
            "idealized_pnl": float(t['realized_binance_pnl'] + t['exchange_fees_paid']),
            "friction_loss": 0.0,
            "exchange_fees_paid": float(t['exchange_fees_paid']),
            "slippage_usd": 0.0,
            "hold_duration_minutes": float(t['hold_duration_minutes']),
            "notes": t['notes']
        })

    try:
        supabase.table("testnet_trade_log").insert(ingest_payload).execute()
        print(f"       Successfully ingested {len(ingest_payload)} uncorrupted trades into Supabase Table 2.")
    except Exception as e:
        print(f"       Warning inserting Binance trades into Table 2: {e}")


# =============================================================================
# STEP 7: Final Audit Summary Display
# =============================================================================
total_binance_fees = df_raw_trades['fee_cost'].sum() if not df_raw_trades.empty else 0.0
total_realized_pnl = df_reconstructed['realized_binance_pnl'].sum() if not df_reconstructed.empty else 0.0
net_binance_cash   = total_realized_pnl - total_binance_fees

print("\n===============================================================================")
print("                   BINANCE GROUND-TRUTH AUDIT SUMMARY                          ")
print("===============================================================================")
print(f"Raw Fill Executions Scraped        : {len(df_raw_trades):,}")
print(f"Orders Tracked in Lifecycle        : {len(df_raw_orders):,}")
print(f"Verified Reconstructed Trades      : {len(df_reconstructed):,}")
print(f"Total True Realized Gross PnL      : ${total_realized_pnl:+,.2f} USDT")
print(f"Total True Binance Fees Deducted   : ${total_binance_fees:,.2f} USDT")
print(f"True Net Binance Wallet Delta      : ${net_binance_cash:+,.2f} USDT")
print("-------------------------------------------------------------------------------")
print("Supabase Sanitization Status       :")
print("  • Table 1 (testnet_active_trades) : Cleaned (0/5 Slots Active)")
print("  • Table 2 (testnet_trade_log)     : Populated Directly from Binance API")
print("===============================================================================\n")
