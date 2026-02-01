# Options Chain Data Compression Strategy

## Overview

Raw options chain data from Databento contains **multiple rows per contract** (tick-by-tick quotes), which leads to large file sizes even after filtering. However, for backtesting purposes, you typically only need **one snapshot per contract** at your entry time.

## Important: What Data is Lost?

**Yes, collapsing loses intraday bid/ask movements.** Here's what happens:

### Example Scenario:
Imagine a contract with quotes at different times in your 1-minute snapshot window:
- **15:45:00.100**: bid=$2.00, ask=$2.10, mid=$2.05
- **15:45:00.500**: bid=$2.02, ask=$2.12, mid=$2.07  
- **15:45:00.900**: bid=$2.01, ask=$2.11, mid=$2.06 ← **This is what you keep**

**With collapse (latest quote only):**
- You keep: bid=$2.01, ask=$2.11, mid=$2.06
- You lose: The fact that bid was $2.02 at some point, mid was $2.07

**Impact on `get_entry_price()`:**
- With realistic mode (mid - 30% of spread): Latest quote → $2.06 - 0.30*(2.11-2.01) = **$2.03**
- If you used the 15:45:00.500 quote instead: $2.07 - 0.30*(2.12-2.02) = **$2.04** 
- **Difference: $0.01 per share = $1.00 per contract** (100 shares)

**This difference is usually small** within a 1-minute window, but it's not zero. The question is: **Does this matter for your backtest?**

## Compression Approach

Yes, you can compress options chain data significantly. However, **this involves a trade-off**: you lose intraday bid/ask movements but keep the information needed for **single-entry-time backtesting**.

Compression strategy:

1. **Collapse to one row per contract** (most important - reduces rows dramatically)
   - **Trade-off**: Lose intraday movements, keep latest quote
2. **Keep only essential columns** (reduces file size)
   - **Safe**: Drop metadata that's not needed
3. **Optional: Pre-compute derived fields** (IV, delta) if you want to avoid recomputation

## Expected Compression

- **Before compression**: 55,939 rows (multiple ticks per contract)
- **After compression**: ~644 unique contracts (one row per contract)
- **Compression ratio**: ~87x reduction in rows

The exact number depends on:
- How many unique contracts you have
- How many quotes per contract in the time window

## Implementation

### Step 1: Collapse to One Row Per Contract

```python
# Filter to rows with valid quotes
quotes = df_opts[df_opts["bid_px_00"].notna() & df_opts["ask_px_00"].notna()].copy()

# Compute mid price
quotes["mid"] = (quotes["bid_px_00"] + quotes["ask_px_00"]) / 2

# Collapse to ONE row per contract (latest quote snapshot)
chain_snapshot = (
    quotes
    .sort_values("ts_event")   # Important: so tail(1) is the latest
    .groupby(["symbol", "expiration", "strike", "call_put"])
    .tail(1)                   # Last quote for each contract
    .copy()
)
```

### Step 2: Keep Only Essential Columns

For backtesting, you only need:

**Required for identification:**
- `symbol` (full option symbol)
- `expiration` 
- `strike`
- `call_put`
- `dte` (days to expiration)

**Required for pricing:**
- `bid_px_00` (or just compute mid from stored bid/ask)
- `ask_px_00`
- `mid` (can be computed: (bid + ask) / 2)

**Optional but recommended:**
- `underlying_last` (underlying price at snapshot time)
- `ts_event` (or just date - for record keeping)
- `iv` (implied volatility - can be recomputed)
- `delta` (can be recomputed)

**Columns you can DROP:**
- `rtype`, `publisher_id`, `instrument_id` (metadata)
- `action`, `side`, `price`, `size` (tick-level data, redundant after collapse)
- `flags`, `ts_in_delta` (metadata)
- `bid_sz_00`, `ask_sz_00`, `bid_pb_00`, `ask_pb_00` (size/priority - not needed for backtesting)
- `root` (can be derived from symbol if needed)
- `ts_recv` (use ts_event or date instead)

### Step 3: Minimal Storage Schema

```python
# After collapse, keep only what's needed for backtesting
compressed_chain = chain_snapshot[[
    'symbol',
    'expiration', 
    'strike',
    'call_put',
    'dte',
    'bid_px_00',
    'ask_px_00',
    'mid',
    'underlying_last',
    'date'  # from ts_event.dt.date
]].copy()

# Optional: Pre-compute IV and delta to avoid recomputation
# (depends on whether you want to store or recompute)
compressed_chain['iv'] = chain_snapshot['iv']
compressed_chain['delta'] = chain_snapshot['delta']
```

### Step 4: Save Compressed Data

```python
# Save compressed version
compressed_chain.to_parquet(cache_file_compressed)

print(f"Original rows: {len(df_opts):,}")
print(f"Compressed rows: {len(compressed_chain):,}")
print(f"Compression ratio: {len(df_opts) / len(compressed_chain):.1f}x")
```

## Even More Aggressive Compression (If Needed)

If you want to minimize storage further:

1. **Store only mid price** (drop bid/ask if you don't need spread analysis):
   ```python
   compressed_chain = chain_snapshot[['symbol', 'expiration', 'strike', 'call_put', 
                                      'dte', 'mid', 'underlying_last', 'date']].copy()
   ```

2. **Use categorical dtypes** for string columns:
   ```python
   compressed_chain['call_put'] = compressed_chain['call_put'].astype('category')
   compressed_chain['symbol'] = compressed_chain['symbol'].astype('category')
   ```

3. **Use appropriate numeric dtypes** (float32 instead of float64 if precision allows):
   ```python
   compressed_chain['mid'] = compressed_chain['mid'].astype('float32')
   compressed_chain['strike'] = compressed_chain['strike'].astype('float32')
   ```

## Trade-offs

### ⚠️ What You LOSE by Collapsing (Important!):

When you collapse to one row per contract (taking the latest quote), you **lose**:

1. **Intraday bid/ask movements**: Multiple quotes throughout the snapshot window
   - You can't see how bid/ask changed during the day
   - You can't analyze spread variations
   - You can't compute volume-weighted average price (VWAP)

2. **Entry price uncertainty**: Different quotes might give different entry prices
   - If you're using `get_entry_price()` with realistic/pessimistic modes, different quotes = different fills
   - You lose the ability to model "what if I entered at a different time in the window?"

3. **Spread analysis over time**: Can't see if spreads widen/narrow during the day
   - Important for liquidity analysis
   - Important for understanding execution timing

### ✅ What You KEEP:

1. **Latest bid/ask at snapshot time**: The most recent quote per contract
2. **Single entry price calculation**: One entry price per contract (assuming you enter at the snapshot time)
3. **All contract metadata**: Symbol, expiration, strike, DTE, etc.

### ✅ Safe to Compress (No Information Loss for Simple Backtesting):
- If you're using a **single entry time** (e.g., "I enter at 3:45pm sharp")
- If you don't need **intraday price movements** for that contract
- If you accept that entry price is based on the **snapshot at entry time**
- Dropping metadata columns (rtype, publisher_id, etc.) ✅
- Dropping tick-level fields (action, side, size) ✅

### ⚠️ Consider Alternative Approaches:

**Option 1: Keep aggregated statistics** (compromise approach)
```python
# Instead of just latest, keep min/max/mean bid/ask
chain_agg = quotes.groupby(["symbol", "expiration", "strike", "call_put"]).agg({
    'bid_px_00': ['last', 'min', 'max', 'mean'],  # latest, min, max, mean
    'ask_px_00': ['last', 'min', 'max', 'mean'],
    'mid': ['last', 'mean'],
    'dte': 'first',  # same for all quotes of same contract
}).reset_index()
```

**Option 2: Keep multiple snapshots at key times** (if you want intraday analysis)
```python
# Keep quotes at specific times (e.g., 9:30am, 12pm, 3:45pm)
key_times = ['09:30', '12:00', '15:45']
# Filter and collapse per time slot
```

**Option 3: Store VWAP if volume data available** (more accurate entry price)
```python
# Volume-weighted average price (if you have bid_sz/ask_sz)
# More realistic entry price than simple mid
```

### ❌ Don't Compress If You Need:
- **Intraday price movements** for the same contract
- **Spread analysis over time** (how spreads change during the day)
- **Multiple entry scenarios** (entering at different times)
- **Bid/ask size for volume analysis**
- **Trade-by-trade execution analysis**

## Example: Full Compression Function

```python
def compress_options_chain(df_opts, underlying_price=None, keep_greeks=True):
    """
    Compress options chain data to one row per contract.
    
    Parameters:
    -----------
    df_opts : DataFrame
        Raw options chain data with multiple rows per contract
    underlying_price : float, optional
        Underlying price at snapshot time
    keep_greeks : bool
        Whether to compute and keep IV/delta
    
    Returns:
    --------
    DataFrame with one row per contract
    """
    # 1. Filter to rows with quotes
    quotes = df_opts[df_opts["bid_px_00"].notna() & df_opts["ask_px_00"].notna()].copy()
    
    # 2. Compute mid price
    quotes["mid"] = (quotes["bid_px_00"] + quotes["ask_px_00"]) / 2
    
    # 3. Collapse to one row per contract (latest quote)
    chain_snapshot = (
        quotes
        .sort_values("ts_event")
        .groupby(["symbol", "expiration", "strike", "call_put"])
        .tail(1)
        .copy()
    )
    
    # 4. Add underlying price
    if underlying_price is not None:
        chain_snapshot["underlying_last"] = underlying_price
    
    # 5. Compute Greeks if requested
    if keep_greeks and underlying_price is not None:
        from py_vollib.black_scholes.implied_volatility import implied_volatility
        from py_vollib.black_scholes.greeks.analytical import delta
        
        r = 0.04
        def compute_iv(row):
            # ... IV computation code ...
            pass
        
        def compute_delta(row):
            # ... Delta computation code ...
            pass
        
        chain_snapshot["iv"] = chain_snapshot.apply(compute_iv, axis=1)
        chain_snapshot["delta"] = chain_snapshot.apply(compute_delta, axis=1)
    
    # 6. Keep only essential columns
    essential_cols = [
        'symbol', 'expiration', 'strike', 'call_put', 'dte',
        'bid_px_00', 'ask_px_00', 'mid', 'underlying_last'
    ]
    
    if keep_greeks:
        essential_cols.extend(['iv', 'delta'])
    
    # Add date from ts_event
    if 'ts_event' in chain_snapshot.columns:
        chain_snapshot['date'] = chain_snapshot['ts_event'].dt.date
        essential_cols.append('date')
    
    compressed = chain_snapshot[essential_cols].copy()
    
    return compressed


# Alternative: Aggregated Statistics Approach (preserves more information)
def compress_options_chain_with_stats(df_opts, underlying_price=None):
    """
    Compress options chain data but keep aggregated statistics.
    Preserves min/max/mean bid/ask to show intraday range.
    
    Returns:
    --------
    DataFrame with one row per contract, but with aggregated stats
    """
    quotes = df_opts[df_opts["bid_px_00"].notna() & df_opts["ask_px_00"].notna()].copy()
    quotes["mid"] = (quotes["bid_px_00"] + quotes["ask_px_00"]) / 2
    quotes["spread"] = quotes["ask_px_00"] - quotes["bid_px_00"]
    
    # Sort by time so 'last' gives latest quote
    quotes = quotes.sort_values('ts_event')
    
    # Aggregate: keep latest + min/max/mean
    agg_dict = {
        'bid_px_00': ['last', 'min', 'max', 'mean'],
        'ask_px_00': ['last', 'min', 'max', 'mean'],
        'mid': ['last', 'min', 'max', 'mean'],
        'spread': ['last', 'min', 'max', 'mean'],
        'dte': 'first',  # Same for all quotes of same contract
    }
    
    # Add underlying_last if provided
    if underlying_price is not None:
        quotes['underlying_last'] = underlying_price
        agg_dict['underlying_last'] = 'first'
    
    # Group and aggregate by contract (symbol, expiration, strike, call_put)
    chain_agg = quotes.groupby(['symbol', 'expiration', 'strike', 'call_put']).agg(agg_dict)
    
    # Flatten multi-level columns
    chain_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in chain_agg.columns.values]
    chain_agg = chain_agg.reset_index()
    
    # Rename for clarity (latest becomes the primary value)
    rename_map = {
        'bid_px_00_last': 'bid_px_00',
        'bid_px_00_min': 'bid_px_00_min',
        'bid_px_00_max': 'bid_px_00_max',
        'bid_px_00_mean': 'bid_px_00_mean',
        'ask_px_00_last': 'ask_px_00',
        'ask_px_00_min': 'ask_px_00_min',
        'ask_px_00_max': 'ask_px_00_max',
        'ask_px_00_mean': 'ask_px_00_mean',
        'mid_last': 'mid',
        'mid_min': 'mid_min',
        'mid_max': 'mid_max',
        'mid_mean': 'mid_mean',
        'spread_last': 'spread',
        'spread_min': 'spread_min',
        'spread_max': 'spread_max',
        'spread_mean': 'spread_mean',
    }
    chain_agg = chain_agg.rename(columns=rename_map)
    
    return chain_agg
```

## Recommendation

**For your use case (wheel strategy backtesting):**

### If you use a single entry time (e.g., 3:45pm):
1. ✅ **Collapse to one row per contract** - This is the biggest win (87x reduction)
2. ✅ **Keep bid/ask** - Needed for `get_entry_price()` with realistic/pessimistic modes
3. ✅ **Keep IV/delta** - Useful for filtering and analysis
4. ✅ **Drop all metadata columns** - Not needed
5. ✅ **Use appropriate dtypes** - Save space with categorical and float32

**Rationale**: Since your backtest uses `get_entry_price(row, fill_mode, penalty)` which takes a single bid/ask snapshot, collapsing is appropriate. You're modeling "I enter at this snapshot time" - which is realistic for a backtest.

### If you want to preserve intraday information:

**Compromise approach - Keep aggregated stats:**
```python
# Keep latest bid/ask + min/max/mean for analysis
chain_agg = quotes.groupby(["symbol", "expiration", "strike", "call_put"]).agg({
    'bid_px_00': ['last', 'min', 'max'],  # latest + range
    'ask_px_00': ['last', 'min', 'max'],
    'mid': ['last', 'mean'],
    'dte': 'first',
    # ... other fields
})
```

This gives you:
- Latest quote for entry price (same as before)
- Min/max to see price range during the window
- Still much smaller than full tick data

**File size impact:**
- Full collapse: ~87x reduction (55,939 → ~644 rows)
- Aggregated stats: ~30-40x reduction (55,939 → ~1,500-2,000 rows, but with more columns)

### Bottom Line:

For **simple backtesting** where you enter at a specific time: **Collapsing is fine** ✅
- You keep the bid/ask at entry time
- Your `get_entry_price()` function works with a single snapshot
- You lose intraday movements, but that's usually acceptable for backtests

For **advanced analysis** requiring intraday movements: **Keep aggregated stats or multiple snapshots**

