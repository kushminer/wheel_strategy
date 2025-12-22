#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[229]:


from pathlib import Path
from dotenv import dotenv_values, load_dotenv
import sys
import os
import pandas as pd
import databento as db
import pandas_market_calendars as mcal

sys.executable

env_path = Path("/Users/samuelminer/Projects/nissan_options/wheel_strategy/.env")

print("Parsed keys:", dotenv_values(env_path).keys())

load_dotenv()  # loads .env from current working directory

assert os.getenv("DATABENTO_API_KEY"), "DATABENTO_API_KEY still not found"
print("os.getenv:", bool(os.getenv("DATABENTO_API_KEY")))
client = db.Historical()


# ## Configuration
# 
# All configurable parameters for the backtest. Modify this cell to change settings.

# In[230]:


# =============================================================================
# UNIFIED CONFIGURATION
# =============================================================================

CONFIG = {
    # -------------------------------------------------------------------------
    # SYMBOL & TIMING
    # -------------------------------------------------------------------------
    'symbol': 'TSLA',                          # Underlying symbol to backtest
    'timezone': 'America/New_York',

    # Entry date/time for the single-day backtest
    'entry_date': '2023-06-06',                # Date to enter positions
    'entry_time': '15:45',                     # Time to capture option chain snapshot

    # Historical data lookback for technical indicators (e.g., Bollinger Bands)
    'lookback_days': 252 * 2,                  # ~2 years of daily data

    # -------------------------------------------------------------------------
    # OPTION SELECTION CRITERIA
    # -------------------------------------------------------------------------
    'option_type': 'P',                        # 'P' for puts (CSP), 'C' for calls
    'dte_min': 30,                             # Minimum days to expiration
    'dte_max': 45,                             # Maximum days to expiration
    'delta_min': 0.25,                         # Minimum absolute delta
    'delta_max': 0.35,                         # Maximum absolute delta

    # -------------------------------------------------------------------------
    # LIQUIDITY MODEL (regime-aware, penalty-based)
    # -------------------------------------------------------------------------
    # Hard rejection thresholds (truly untradeable)
    'min_bid_hard': 0.10,                      # Hard floor - reject penny options
    'hard_max_spread_pct': 0.20,               # Hard ceiling - reject extreme spreads

    # Base target spread (calm market conditions)
    'base_max_spread_pct': 0.08,               # Target max spread in normal conditions

    # IV regime adjustments (allow wider spreads in high-vol)
    'ivp_high_threshold': 0.70,                # IV percentile threshold for "high vol"
    'ivp_high_max_spread_pct': 0.12,           # Allowed spread when IV is high
    'ivp_extreme_threshold': 0.90,             # IV percentile threshold for "extreme vol"
    'ivp_extreme_max_spread_pct': 0.15,        # Allowed spread when IV is extreme

    # DTE adjustments (short-dated options have wider spreads)
    'short_dte_threshold': 7,                  # DTE below this gets extra allowance
    'short_dte_extra_spread_pct': 0.02,        # Extra spread allowance for short DTE

    # Penalty tiers (execution tax based on spread quality)
    # tight:    spread <= 0.6 * allowed → penalty = 1.0 (no extra slippage)
    # moderate: spread <= allowed       → penalty = 1.15 (15% wider effective spread)
    # wide:     spread <= hard_max      → penalty = 1.35 (35% wider effective spread)
    # ugly:     spread > hard_max       → REJECT (no trade)

    # -------------------------------------------------------------------------
    # EXIT STRATEGY
    # -------------------------------------------------------------------------
    'exit_pct': 0.50,                          # Exit when option decays to X% of premium
                                               # 0.50 = buy back at 50%, keep 50% profit
    'stop_loss_multiplier': 2.0,               # Exit if option price reaches Nx premium
    'max_hold_dte': None,                      # Exit at X DTE if no other trigger (None = disabled)

    # -------------------------------------------------------------------------
    # TRANSACTION COSTS (NEW - will be applied later)
    # -------------------------------------------------------------------------
    'commission_per_contract': 0.65,           # Per contract commission (round trip = 2x)
    'sec_fee_per_contract': 0.01,              # SEC/TAF fees per contract

    # -------------------------------------------------------------------------
    # EXECUTION / FILL ASSUMPTIONS (NEW - will be applied later)
    # -------------------------------------------------------------------------
    'fill_mode': 'mid',                        # 'mid' (current), 'bid' (realistic), 'pessimistic'
    'use_realistic_fills': False,              # When True: sell at bid, buy back at ask

    # -------------------------------------------------------------------------
    # CACHE
    # -------------------------------------------------------------------------
    'cache_dir': '../cache/',
}

# -------------------------------------------------------------------------
# DERIVED VALUES (computed from CONFIG)
# -------------------------------------------------------------------------
SYMBOL = CONFIG['symbol']
TZ = CONFIG['timezone']
CACHE_DIR = CONFIG['cache_dir']
os.makedirs(CACHE_DIR, exist_ok=True)

# Entry timestamp
ENTRY_DATE = pd.Timestamp(CONFIG['entry_date'], tz=TZ)
ENTRY_TIME = pd.Timestamp(f"{CONFIG['entry_date']} {CONFIG['entry_time']}", tz=TZ)

print("=" * 60)
print("BACKTEST CONFIGURATION")
print("=" * 60)
print(f"Symbol:          {SYMBOL}")
print(f"Entry Date:      {ENTRY_DATE.date()}")
print(f"Entry Time:      {CONFIG['entry_time']}")
print(f"Option Type:     {'Cash-Secured Put' if CONFIG['option_type'] == 'P' else 'Covered Call'}")
print(f"DTE Range:       {CONFIG['dte_min']} - {CONFIG['dte_max']} days")
print(f"Delta Range:     {CONFIG['delta_min']} - {CONFIG['delta_max']}")
print(f"Exit Target:     {CONFIG['exit_pct']*100:.0f}% of premium")
print(f"Stop Loss:       {CONFIG['stop_loss_multiplier']}x premium")
print(f"Fill Mode:       {CONFIG['fill_mode']}")
print(f"Realistic Fills: {CONFIG['use_realistic_fills']}")
print(f"Commission:      ${CONFIG['commission_per_contract']}/contract")
print("=" * 60)
print("\nNOTE: Transaction costs and realistic fills are NOT yet applied.")
print("      Run both notebooks to compare baseline vs realistic results.")


# In[ ]:


# =============================================================================
# HELPER FUNCTIONS FOR REALISTIC EXECUTION
# =============================================================================

def get_entry_price(row, fill_mode='realistic', penalty=1.0):
    """
    Calculate entry price when SELLING a put (we receive premium).
    Higher price = better for us.

    Slippage is calculated as a percentage of the bid-ask spread from mid.
    Penalty multiplier widens the effective spread for illiquid options.

    | Scenario    | Formula                              | Interpretation              |
    |-------------|--------------------------------------|-----------------------------|
    | pessimistic | mid - 75% of (spread * penalty)      | Forced/stressed execution   |
    | realistic   | mid - 30% of (spread * penalty)      | Normal retail execution     |
    | optimistic  | mid                                  | Patient, favorable fills    |

    Args:
        row: DataFrame row with bid_px_00, ask_px_00
        fill_mode: 'optimistic', 'realistic', or 'pessimistic'
        penalty: liquidity penalty multiplier (1.0 = no extra slippage)
    """
    bid = row['bid_px_00']
    ask = row['ask_px_00']
    mid = (bid + ask) / 2
    spread = ask - bid

    # Apply liquidity penalty to effective spread
    effective_spread = spread * penalty

    if fill_mode == 'optimistic':
        return mid                              # Best case - get mid (no penalty applied)
    elif fill_mode == 'pessimistic':
        fill = mid - (0.75 * effective_spread)  # Worst case - 75% toward bid
    else:  # realistic
        fill = mid - (0.30 * effective_spread)  # Normal - 30% toward bid

    # Clamp to [bid, ask] to stay realistic
    return max(bid, min(ask, fill))


def get_exit_price(daily_row, fill_mode=CONFIG['fill_mode'], target_price=None, penalty=1.0): # IS THIS RIGHT TO SET AT PENALTY = 1.0? ##
    """
    Calculate exit price when BUYING BACK a put (we pay to close).
    Lower price = better for us.

    For daily OHLCV data, we estimate spread behavior from the day's range.
    Penalty multiplier widens the effective range for illiquid options.

    | Scenario    | Formula                              | Interpretation              |
    |-------------|--------------------------------------|-----------------------------|
    | pessimistic | close + 75% of (range * penalty)     | Forced/stressed execution   |
    | realistic   | close + 30% of (range * penalty)     | Normal retail execution     |
    | optimistic  | close - 25% of (range * penalty)     | Patient, favorable fills    |

    Args:
        daily_row: DataFrame row with close, high, low
        fill_mode: 'optimistic', 'realistic', or 'pessimistic'
        target_price: Optional target price (not currently used but reserved)
        penalty: liquidity penalty multiplier (1.0 = no extra slippage)
    """
    close = daily_row['close']
    high = daily_row['high']
    low = daily_row['low']
    day_range = high - low  # Proxy for intraday spread/volatility

    # Apply liquidity penalty to effective range
    effective_range = day_range * penalty

    if fill_mode == 'optimistic':
        # Patient buyer - gets below close (toward low)
        fill = close - (0.25 * effective_range)
        return max(low, fill)
    elif fill_mode == 'pessimistic':
        # Forced buyer - pays above close (toward high)
        fill = close + (0.75 * effective_range)
        return min(high, fill)
    else:  # realistic
        # Normal execution - slight slippage above close
        fill = close + (0.30 * effective_range)
        return min(high, fill)


def get_transaction_costs(config, is_round_trip=True):
    """
    Calculate total transaction costs per contract.

    Args:
        config: CONFIG dict with commission and fee rates
        is_round_trip: True if both entry and exit, False if entry only (e.g., expired worthless)

    Returns:
        Total fees in dollars per contract
    """
    per_leg = config['commission_per_contract'] + config['sec_fee_per_contract']
    return per_leg * 2 if is_round_trip else per_leg


def compute_allowed_spread(row, config):
    """
    Compute the allowed spread percentage for a single option based on regime.

    Regime factors:
    - IV percentile (high vol → allow wider spreads)
    - DTE (short-dated → allow wider spreads)

    Returns: allowed_spread_pct for this option
    """
    base = config['base_max_spread_pct']

    # IV regime adjustment
    ivp = row.get('ivp', 0.5)  # Default to median if not computed
    if ivp >= config['ivp_extreme_threshold']:
        base = config['ivp_extreme_max_spread_pct']
    elif ivp >= config['ivp_high_threshold']:
        base = config['ivp_high_max_spread_pct']

    # DTE adjustment
    dte = row.get('dte', 30)
    if dte <= config['short_dte_threshold']:
        base += config['short_dte_extra_spread_pct']

    return base


def compute_liquidity_penalty(spread_pct, allowed_spread_pct, hard_max_spread_pct):
    """
    Compute liquidity penalty multiplier based on spread quality.

    Tiers:
    - tight:    spread <= 0.6 * allowed → penalty = 1.0 (no extra slippage)
    - moderate: spread <= allowed       → penalty = 1.15
    - wide:     spread <= hard_max      → penalty = 1.35
    - ugly:     spread > hard_max       → None (reject)

    Returns: (tier_name, penalty_multiplier) or (None, None) if rejected
    """
    if spread_pct > hard_max_spread_pct:
        return 'reject', None

    tight_threshold = 0.6 * allowed_spread_pct

    if spread_pct <= tight_threshold:
        return 'tight', 1.0
    elif spread_pct <= allowed_spread_pct:
        return 'moderate', 1.15
    else:  # spread_pct <= hard_max_spread_pct
        return 'wide', 1.35


def apply_liquidity_model(df, config):
    """
    Apply regime-aware liquidity model with penalty tiers.

    Instead of binary reject, this:
    1. Computes IV percentile (ivp) for regime detection
    2. Computes allowed_spread_pct per option (regime-aware)
    3. Assigns liquidity_tier and liquidity_penalty
    4. Only hard-rejects truly ugly spreads

    Args:
        df: DataFrame with option quotes (needs bid_px_00, ask_px_00, spread_pct, iv, dte)
        config: CONFIG dict with liquidity model settings

    Returns:
        DataFrame with liquidity columns added, ugly spreads removed
    """
    if len(df) == 0:
    return df

    df = df.copy()
    original_count = len(df)

    # Ensure required columns exist
    if 'spread_pct' not in df.columns:
        df['spread'] = df['ask_px_00'] - df['bid_px_00']
        df['spread_pct'] = df['spread'] / df['mid']

    # Step 1: Compute IV percentile (cross-sectional within this snapshot)
    if 'iv' in df.columns:
        df['ivp'] = df['iv'].rank(pct=True)
    else:
        df['ivp'] = 0.5  # Default to median if IV not available

    # Step 2: Compute allowed spread per option
    df['allowed_spread_pct'] = df.apply(
        lambda row: compute_allowed_spread(row, config), axis=1
    )

    # Step 3: Compute liquidity tier and penalty
    def get_tier_and_penalty(row):
        return compute_liquidity_penalty(
            row['spread_pct'], 
            row['allowed_spread_pct'],
            config['hard_max_spread_pct']
        )

    tiers_penalties = df.apply(get_tier_and_penalty, axis=1)
    df['liquidity_tier'] = tiers_penalties.apply(lambda x: x[0])
    df['liquidity_penalty'] = tiers_penalties.apply(lambda x: x[1])

    # Step 4: Hard reject only truly ugly spreads and penny options
    df = df[
        (df['liquidity_tier'] != 'reject') &
        (df['bid_px_00'] >= config['min_bid_hard'])
    ].copy()

    rejected = original_count - len(df)

    # Print diagnostics
    print(f"\n  Liquidity Model Applied:")
    print(f"    Original: {original_count} options")
    print(f"    Hard rejected: {rejected} ({rejected/original_count*100:.1f}%)")
    print(f"    Remaining: {len(df)} options")

    if len(df) > 0:
        tier_counts = df['liquidity_tier'].value_counts()
        print(f"    Tier breakdown: {dict(tier_counts)}")
        print(f"    Avg spread: {df['spread_pct'].mean()*100:.1f}%, Avg allowed: {df['allowed_spread_pct'].mean()*100:.1f}%")
        print(f"    Avg penalty: {df['liquidity_penalty'].mean():.2f}x")
    return df


def calculate_pnl(premium_received, exit_price_paid, fees, cost_basis):
    """
    Calculate P&L metrics for a trade.

    Args:
        premium_received: Premium collected when selling (contract value)
        exit_price_paid: Price paid to close position (contract value), 0 if expired worthless
        fees: Total transaction costs
        cost_basis: Capital at risk (strike * 100 for CSP)

    Returns:
        dict with pnl, pnl_pct, roc
    """
    pnl = premium_received - exit_price_paid - fees
    pnl_pct = (pnl / premium_received) * 100 if premium_received > 0 else 0
    roc = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

    return {
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'roc': roc,
        'fees': fees
    }


# Print summary of fill assumptions
print("=" * 60)
print("FILL ASSUMPTIONS BY SCENARIO")
print("=" * 60)
print(f"{'Scenario':<12} {'Entry (Sell)':<25} {'Exit (Buy Back)':<25}")
print("-" * 60)
print(f"{'Pessimistic':<12} {'Mid - 75% of spread':<25} {'Close + 75% of range':<25}")
print(f"{'Realistic':<12} {'Mid - 30% of spread':<25} {'Close + 30% of range':<25}")
print(f"{'Optimistic':<12} {'Mid (no slippage)':<25} {'Close - 25% of range':<25}")
print("=" * 60)
print(f"\nTransaction costs: ${CONFIG['commission_per_contract'] + CONFIG['sec_fee_per_contract']:.2f}/leg")
print(f"\nLiquidity Model (regime-aware):")
print(f"  Hard reject: bid < ${CONFIG['min_bid_hard']} or spread > {CONFIG['hard_max_spread_pct']*100:.0f}%")
print(f"  Base target spread: {CONFIG['base_max_spread_pct']*100:.0f}%")
print(f"  High IV ({CONFIG['ivp_high_threshold']*100:.0f}%ile): allow {CONFIG['ivp_high_max_spread_pct']*100:.0f}%")
print(f"  Extreme IV ({CONFIG['ivp_extreme_threshold']*100:.0f}%ile): allow {CONFIG['ivp_extreme_max_spread_pct']*100:.0f}%")
print(f"  Short DTE (≤{CONFIG['short_dte_threshold']}d): +{CONFIG['short_dte_extra_spread_pct']*100:.0f}% allowed")


# ### Import Daily Equity Data For a Single Symbol

# In[232]:


# Use CONFIG values (CACHE_DIR, SYMBOL, TZ already defined in CONFIG cell)
dataset = "EQUS.MINI"     # consolidated US equities (best choice)
schema = "ohlcv-1d"       # DAILY bars

# Calculate date range for historical data
end = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)
start = end - pd.Timedelta(days=CONFIG['lookback_days'])

# Generate cache filename
start_str = start.strftime('%Y%m%d')
end_str = end.strftime('%Y%m%d')
cache_file = os.path.join(CACHE_DIR, f"equity_daily_{SYMBOL}_{start_str}_{end_str}.parquet")

# Check cache first
if os.path.exists(cache_file):
    print(f"[CACHE HIT] Loading daily equity data for {SYMBOL} from cache")
    data = pd.read_parquet(cache_file)
    print(f"  Loaded {len(data)} days of data")
else:
    print(f"[API] Fetching daily equity data for {SYMBOL} from {start.date()} to {end.date()}...")
    data = client.timeseries.get_range(
        dataset=dataset,
        symbols=SYMBOL,
        schema=schema,
        stype_in="raw_symbol",
        start=start,
        end=end,
    )
    # Convert to DataFrame and save to cache
    data = data.to_df(tz=TZ)
    data.to_parquet(cache_file)
    print(f"[CACHE SAVE] Saved {len(data)} days to cache")




# In[233]:


# data is already a DataFrame from cache or API fetch
equity_data = data
equity_data.head()


# ### Equity Technical Filter

# In[234]:


import pandas as pd

entry_technical_filter = equity_data.copy().sort_index()

# Bollinger Bands parameters
window = 20
k = 2.0  # 2-sigma Bollinger Bands

# Calculate rolling statistics on close price
roll = entry_technical_filter["close"].rolling(window=window, min_periods=window)
entry_technical_filter["sma20"] = roll.mean()
entry_technical_filter["std20"] = roll.std(ddof=0)

# Calculate Bollinger Bands
entry_technical_filter["bb_upper"] = entry_technical_filter["sma20"] + k * entry_technical_filter["std20"]
entry_technical_filter["bb_lower"] = entry_technical_filter["sma20"] - k * entry_technical_filter["std20"]

# Optional: Bollinger %B (position within bands)
entry_technical_filter["bb_pctb"] = (
    (entry_technical_filter["close"] - entry_technical_filter["bb_lower"]) / 
    (entry_technical_filter["bb_upper"] - entry_technical_filter["bb_lower"])
)

# Optional: Bollinger Bandwidth (width of bands relative to SMA)
entry_technical_filter["bb_bandwidth"] = (
    (entry_technical_filter["bb_upper"] - entry_technical_filter["bb_lower"]) / 
    entry_technical_filter["sma20"]
)

entry_technical_filter.dropna().head()


# ### Equity Technical Filter

# In[235]:


# With BB Filter
df_equity_entry = entry_technical_filter.copy()[['close','sma20','bb_upper']].dropna()
df_equity_entry['bb_entry'] = df_equity_entry['close'] <= df_equity_entry['bb_upper']
df_equity_entry[['bb_entry']].value_counts()
df_equity_entry.head()


# ### Get Options Data For Dates that Pass Technical Filter

# In[236]:


# Options data settings (uses CONFIG values)
dataset = "OPRA.PILLAR"
schema = "cmbp-1"

# Use entry time from CONFIG
start = ENTRY_TIME
end = start + pd.Timedelta(minutes=1)

# Generate cache filename for options data
date_str = start.strftime('%Y%m%d')
time_str = start.strftime('%H%M')
cache_file = os.path.join(CACHE_DIR, f"options_{SYMBOL}_{date_str}_{time_str}.parquet")

# Check cache first
if os.path.exists(cache_file):
    print(f"[CACHE HIT] Loading options data for {SYMBOL} on {start.date()} at {start.time()}")
    df_opts = pd.read_parquet(cache_file)
    print(f"  Loaded {len(df_opts)} option quotes")
else:
    print(f"[API] Fetching options for {SYMBOL} on {start.date()} at {start.time()}...")
    data = client.timeseries.get_range(
        dataset=dataset,
        schema=schema,
        symbols=f"{SYMBOL}.OPT",     # ✅ parent symbology format
        stype_in="parent",           # ✅ parent lookup
        start=start,
        end=end,
    )

    df_opts = data.to_df(tz=TZ).sort_values("ts_event")

    # Save to cache
    df_opts.to_parquet(cache_file)
    print(f"[CACHE SAVE] Saved {len(df_opts)} option quotes to cache")

df_opts.head()


# In[237]:


sym = df_opts["symbol"]

# Split ROOT and OPRA code (e.g. "AAPL" and "240119P00205000")
root_and_code = sym.str.split(expand=True)
df_opts["root"] = root_and_code[0]
code = root_and_code[1]

# Expiration: YYMMDD in positions 0–5
df_opts["expiration"] = pd.to_datetime(code.str[:6], format="%y%m%d")

# Call/Put flag: single char at position 6
df_opts["call_put"] = code.str[6]

# Strike: remaining digits, usually in 1/1000 dollars
# Example: "00205000" -> 205.000
strike_int = code.str[7:].astype("int32")
df_opts["strike"] = strike_int / 1000.0

# Calculate DTE (Days to Expiry)
# Localize expiration to match ts_event timezone, then normalize both to midnight
expiration_tz = df_opts["expiration"].dt.tz_localize(df_opts["ts_event"].dt.tz)
df_opts["dte"] = (expiration_tz - df_opts["ts_event"].dt.normalize()).dt.days
print(f'df shape: {df_opts.shape}')
df_opts.head()



# In[238]:


# Filter options using CONFIG values
df_opts = df_opts[
    (df_opts['dte'] >= CONFIG['dte_min']) & 
    (df_opts['dte'] <= CONFIG['dte_max']) & 
    (df_opts['call_put'] == CONFIG['option_type'])
].sort_values(['dte', 'strike'])
print(f'df shape: {df_opts.shape}')
df_opts.head()


# In[239]:


# Get unique timestamps from your filtered options
unique_timestamps = df_opts.index.unique()

# Use entry time from CONFIG
start_time = ENTRY_TIME
end_time = start_time + pd.Timedelta(minutes=1)

# Generate cache filename for minute equity data
date_str = start_time.strftime('%Y%m%d')
time_str = start_time.strftime('%H%M')
cache_file = os.path.join(CACHE_DIR, f"equity_minute_{SYMBOL}_{date_str}_{time_str}.parquet")

# Check cache first
if os.path.exists(cache_file):
    print(f"[CACHE HIT] Loading minute equity data for {SYMBOL} on {start_time.date()} at {start_time.time()}")
    equity_df = pd.read_parquet(cache_file)
    print(f"  Loaded {len(equity_df)} minute records")
else:
    print(f"[API] Fetching minute equity data for {SYMBOL} on {start_time.date()} at {start_time.time()}...")

    # Fetch OHLCV data for TSLA at the specific timestamp
    equity_data = client.timeseries.get_range(
        dataset='XNAS.ITCH',  # NASDAQ for TSLA
        symbols=[SYMBOL],
        schema='ohlcv-1m',  # 1-minute OHLCV bars
        start=start_time,
        end=end_time,
        stype_in='raw_symbol'
    )

    # Convert to dataframe
    equity_df = equity_data.to_df()
    print(f"[CACHE SAVE] Saved {len(equity_df)} minute records to cache")
    equity_df.to_parquet(cache_file)

print(f"Total: {len(equity_df)} equity records")
equity_df


# In[240]:


import numpy as np
import pandas as pd
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes.greeks.analytical import delta

r = 0.04  # fixed risk-free rate (4% as decimal for py_vollib)

# 0) Keep only rows that actually have a quote (bid/ask)
quotes = df_opts[df_opts["bid_px_00"].notna() & df_opts["ask_px_00"].notna()].copy()

# 1) Compute mid price per tick
quotes["mid"] = (quotes["bid_px_00"] + quotes["ask_px_00"]) / 2
quotes["spread"] = quotes["ask_px_00"] - quotes["bid_px_00"]
quotes["spread_pct"] = quotes["spread"] / quotes["mid"]

# 2) Collapse to ONE row per option contract (snapshot at ~3:45 pm)
chain_snapshot = (
    quotes
    .sort_values("ts_event")   # important: so tail(1) is the latest
    .groupby(["symbol", "expiration", "strike", "call_put"])
    .tail(1)                   # last quote for each contract
    .copy()
)
underlying_price = equity_df["close"].iloc[0]   # 15:45 close
chain_snapshot["underlying_last"] = underlying_price

# Note: Entry price will be calculated AFTER liquidity model applies penalties
# For now, just store mid price - actual entry_price calculated in backtest_candidates
print(f"Fill mode: {CONFIG['fill_mode']}")
print(f"  Mid prices available; entry prices will include liquidity penalty after filtering")


# In[241]:


def compute_iv(row):
    price = row["mid"]
    S     = row["underlying_last"]
    K     = row["strike"]
    t     = row["dte"] / 365.0
    flag  = "p" if row["call_put"] == "P" else "c"

    if not (np.isfinite(price) and np.isfinite(S) and np.isfinite(K) and t > 0):
        return np.nan
    if price <= 0 or S <= 0 or K <= 0:
        return np.nan

    try:
        return implied_volatility(price, S, K, t, r, flag)
    except Exception:
        return np.nan


def compute_delta(row):
    sigma = row["iv"]
    if not np.isfinite(sigma):
        return np.nan

    S    = row["underlying_last"]
    K    = row["strike"]
    t    = row["dte"] / 365.0
    flag = "p" if row["call_put"] == "P" else "c"

    return abs(delta(flag, S, K, t, r, sigma))

chain_snapshot["iv"] = chain_snapshot.apply(compute_iv, axis=1)
chain_snapshot["delta"] = chain_snapshot.apply(compute_delta, axis=1)

chain_snapshot.head()


# In[242]:


chain_snapshot['date'] = chain_snapshot['ts_event'].dt.date

candidates = chain_snapshot[
    (chain_snapshot["call_put"] == CONFIG['option_type'])
    & chain_snapshot["dte"].between(CONFIG['dte_min'], CONFIG['dte_max'])
    & chain_snapshot["delta"].abs().between(CONFIG['delta_min'], CONFIG['delta_max'])
].copy()

# Apply liquidity model (regime-aware, penalty-based)
candidates = apply_liquidity_model(candidates, CONFIG)

candidates[["symbol", "expiration", "strike", "dte", "iv", "delta",'mid']].sort_values(
    ["dte", "strike"]
)
candidates


# In[243]:


backtest_candidates = candidates.copy()

# Calculate entry price WITH liquidity penalty
backtest_candidates['entry_price'] = backtest_candidates.apply(
    lambda row: get_entry_price(row, CONFIG['fill_mode'], row.get('liquidity_penalty', 1.0)), 
    axis=1
)

# Premium and cost basis
backtest_candidates['per_share_premium'] = backtest_candidates['entry_price']
backtest_candidates['premium'] = backtest_candidates['per_share_premium'] * 100
backtest_candidates['cost_basis'] = backtest_candidates['strike'] * 100  # CSP cost basis = strike * 100

# Exit parameters
backtest_candidates['exit_pct'] = CONFIG['exit_pct']
backtest_candidates['exit_price_per_share'] = backtest_candidates['per_share_premium'] * backtest_candidates['exit_pct']

# Keep liquidity info for exit calculations
backtest_candidates = backtest_candidates[[
    'symbol', 'cost_basis', 'premium', 'exit_pct', 'exit_price_per_share',
    'date', 'dte', 'expiration', 'mid', 'strike', 'entry_price',
    'liquidity_tier', 'liquidity_penalty'
]]

# Show summary
print(f"\nBacktest Candidates: {len(backtest_candidates)} options")
print(f"  Avg entry price: ${backtest_candidates['entry_price'].mean():.2f}/share")
print(f"  Avg mid price: ${backtest_candidates['mid'].mean():.2f}/share")
print(f"  Avg slippage: ${(backtest_candidates['mid'] - backtest_candidates['entry_price']).mean():.2f}/share")
print(f"  Liquidity tiers: {dict(backtest_candidates['liquidity_tier'].value_counts())}")

backtest_candidates


# In[244]:


def fetch_daily_prices_for_option(symbol, entry_date, expiration_date, client, config):
    """
    Fetch daily OHLC prices for an option from entry date to expiration.

    Args:
        symbol: Option symbol
        entry_date: Entry date (normalized)
        expiration_date: Expiration date (normalized)
        client: Databento client
        config: Configuration dict

    Returns:
        DataFrame with daily OHLC data
    """
    # Generate cache filename for daily option prices
    entry_str = entry_date.strftime('%Y%m%d')
    exp_str = expiration_date.strftime('%Y%m%d')
    cache_file = os.path.join(CACHE_DIR, f"option_daily_{symbol}_{entry_str}_{exp_str}.parquet")

    # Check cache first
    if os.path.exists(cache_file):
        print(f"    [CACHE HIT] Loading daily prices for {symbol}")
        return pd.read_parquet(cache_file)

    # Cache miss - fetch from API
    print(f"    [API] Fetching daily prices for {symbol} from {entry_date.date()} to {expiration_date.date()}")

    start_daily = entry_date + pd.Timedelta(days=1)  # Day after entry
    end_daily = expiration_date + pd.Timedelta(days=1)  # Include expiration day

    daily_data = client.timeseries.get_range(
        dataset='OPRA.PILLAR',
        schema='ohlcv-1d',
        symbols=symbol,
        stype_in='raw_symbol',
        start=start_daily,
        end=end_daily,
    )

    df_daily = daily_data.to_df(tz=config['timezone'])

    # Save to cache
    df_daily.to_parquet(cache_file)
    print(f"    [CACHE SAVE] Saved {len(df_daily)} days to cache")

    return df_daily


def check_profit_target_hit(df_daily, exit_price_per_share, entry_date):
    """
    Check if the exit price target was hit in the daily price data.

    Args:
        df_daily: DataFrame with daily OHLC data (prices are per-share)
        exit_price_per_share: Target price per share to exit at
        entry_date: Entry date to skip (we can't exit same day we entered)

    Returns:
        tuple: (hit_date, daily_row) if hit, (None, None) if not hit
    """
    for check_date, daily_row in df_daily.iterrows():
        # Skip the entry date - we can't exit on the same day we entered
        check_date_normalized = check_date.tz_localize(None) if hasattr(check_date, 'tz_localize') and check_date.tz else check_date
        if check_date_normalized.date() <= entry_date.date():
            continue

        daily_low = daily_row['low']
        daily_high = daily_row['high']

        # Check if our exit target (per-share) is within the daily range
        if daily_low <= exit_price_per_share <= daily_high:
            return check_date, daily_row

    return None, None


def create_exit_record(symbol, entry_date, expiration_date, premium, exit_pct,
                       exit_price, exit_reason, check_date, daily_row, cost_basis):
    """
    Create an exit record dictionary.

    Args:
        symbol: Option symbol
        entry_date: Entry date
        expiration_date: Expiration date
        premium: Premium received
        exit_pct: Exit percentage (e.g., 0.25 = exit when decays 25%)
        exit_price: Actual exit price
        exit_reason: Reason for exit
        check_date: Date of exit
        daily_row: Daily price data row
        cost_basis: Cost basis (strike * 100)

    Returns:
        dict: Exit record
    """
    return {
        'symbol': symbol,
        'entry_date': entry_date,
        'exit_date': check_date.tz_localize(None) if hasattr(check_date, 'tz_localize') and check_date.tz else check_date,
        'expiration': expiration_date,
        'cost_basis': cost_basis,
        'premium': premium,
        'exit_pct': exit_pct,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'days_held': (check_date.tz_localize(None) - entry_date).days if check_date else None,
        'daily_low': daily_row['low'] if daily_row is not None else None,
        'daily_high': daily_row['high'] if daily_row is not None else None,
    }


def calculate_pnl_metrics(exits_df, config):
    """
    Calculate P&L metrics for exit results.

    Args:
        exits_df: DataFrame with exit records
        config: Configuration dict with fee settings

    Returns:
        DataFrame with P&L metrics added
    """
    if len(exits_df) > 0:
        exits_df = exits_df.copy()

        # Calculate transaction costs based on exit reason
        # Expired worthless = entry fee only (no buyback needed)
        # All other exits = round-trip fees
        exits_df['fees'] = exits_df['exit_reason'].apply(
            lambda reason: get_transaction_costs(config, is_round_trip=(reason != 'expired_worthless'))
        )

        # P&L after fees
        exits_df['exit_pnl'] = exits_df['premium'] - exits_df['exit_price'] - exits_df['fees']
        exits_df['exit_pnl_pct'] = (exits_df['exit_pnl'] / exits_df['premium']) * 100
        exits_df['roc'] = (exits_df['exit_pnl'] / exits_df['cost_basis']) * 100

        # Summary stats
        total_fees = exits_df['fees'].sum()
        print(f"\n  Transaction costs: ${total_fees:.2f} total ({len(exits_df)} trades)")

    return exits_df


def backtest_exit_strategy(backtest_candidates, client, config):
    """
    Backtest exit strategy for wheel options

    Exit conditions:
    1. Profit target: Exit when option price <= premium * (1 - exit_pct)
       - If daily range contains exit_price_target, assume we exited at that exact price

    Args:
        backtest_candidates: DataFrame with options to backtest
        client: Databento client
        config: Configuration dict

    Returns:
        DataFrame with exit results
    """
    exits = []

    for idx, row in backtest_candidates.iterrows():
        symbol = row['symbol']

        # Normalize dates
        entry_date = pd.Timestamp(row['date']).tz_localize(None)
        expiration_date = pd.Timestamp(row['expiration']).tz_localize(None)

        # Entry details - work with per-share prices for comparison, contract prices for P&L
        premium_per_share = row['mid']
        premium = premium_per_share * 100  # Contract premium (100 shares per contract)
        exit_pct = row['exit_pct']  # e.g., 0.25 = exit when option is at 25% of original premium
        exit_price_per_share = premium_per_share * exit_pct  # Per-share exit price (buy back at this price)
        exit_price_contract = exit_price_per_share * 100  # Contract exit price for P&L
        cost_basis = row['strike'] * 100  # Contract cost basis
        liquidity_penalty = row.get('liquidity_penalty', 1.0)  # Liquidity penalty for slippage

        print(f"\nProcessing {symbol}...")
        print(f"  Entry: {entry_date.date()}, Premium: ${premium:.2f} (${premium_per_share:.2f}/share)")
        print(f"  Exit target: ${exit_price_contract:.2f} (${exit_price_per_share:.2f}/share, exit at {exit_pct*100:.0f}% of premium)")

        try:
            # Fetch daily prices
            df_daily = fetch_daily_prices_for_option(symbol, entry_date, expiration_date, client, config)

            # Check for profit target hit (using per-share prices, skipping entry date)
            hit_date, daily_row = check_profit_target_hit(df_daily, exit_price_per_share, entry_date)

            if hit_date:
                # Profit target hit - calculate realistic exit price with slippage
                actual_exit_per_share = get_exit_price(daily_row, config.get('fill_mode', 'realistic'), penalty=liquidity_penalty)
                actual_exit_contract = actual_exit_per_share * 100

                exit_record = create_exit_record(
                    symbol, entry_date, expiration_date, premium, exit_pct,
                    actual_exit_contract, 'profit_target', hit_date, daily_row, cost_basis
                )
                exits.append(exit_record)

                print(f"  ✓ Profit target hit on {hit_date.date()}")
                print(f"    Target: ${exit_price_per_share:.2f}/share, Actual fill: ${actual_exit_per_share:.2f}/share ({config.get('fill_mode', 'realistic')} mode)")
                print(f"    (Daily range: ${daily_row['low']:.2f} - ${daily_row['high']:.2f} per share)")
            else:
                # Option expired worthless - this is a WIN for CSP sellers!
                # Keep 100% of premium
                exit_record = create_exit_record(
                    symbol, entry_date, expiration_date, premium, exit_pct,
                    0.0, 'expired_worthless', expiration_date, None, cost_basis
                )
                exits.append(exit_record)
                print(f"  🎉 Option expired worthless on {expiration_date.date()} - KEEP 100% PREMIUM!")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create results DataFrame and calculate P&L
    exits_df = pd.DataFrame(exits)
    exits_df = calculate_pnl_metrics(exits_df, config)

    return exits_df

# Run backtest (uses CONFIG from top of notebook)
exits_df = backtest_exit_strategy(
    backtest_candidates=backtest_candidates,
    client=client,
    config=CONFIG
)

# Display results
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(f"\nTotal exits: {len(exits_df)}")

if len(exits_df) > 0:
    print(f"\nExit reasons:")
    print(exits_df['exit_reason'].value_counts())
    print(f"\nP&L Summary:")
    print(exits_df[['exit_pnl', 'exit_pnl_pct', 'roc']].describe())

    # Show sample
    print("\nSample exits:")
    print(exits_df[['symbol', 'entry_date', 'exit_date', 'premium', 'exit_price', 
                   'exit_pnl', 'roc', 'exit_reason']].head(10))
else:
    print("\n⚠ No exits recorded - check for errors above")


# In[245]:


exits_df.round(2)


# In[ ]:


100*(exits_df.exit_pnl.sum()/exits_df.cost_basis.sum())


# In[ ]:


# We need to save backtest results with metadata as our strategy evolves
# exists_df should contain option data such as delta at entry, peak delta, maybe other information that would be helpful for analysis

# 
exits_df['daily_adjusted_roc'] = exits_df['exit_pnl']/exits_df['cost_basis']
exits_df['daily_adjusted_roc'].describe()
exits_df['days_held'].describe()
exits_df['exit_reason'].value_counts()


# In[ ]:




