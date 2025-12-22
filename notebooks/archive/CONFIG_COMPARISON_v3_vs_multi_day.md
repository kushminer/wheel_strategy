# Configuration Comparison: v3 vs multi_day_backtest

## Key Differences Found

| Setting | multi_day_backtest | wheel_multi_symbol_backtest_v3 | Status |
|---------|-------------------|--------------------------------|--------|
| **Ticker(s)** | Single: `'ticker': 'TSLA'` | Multiple: `'tickers': ['TSLA', 'AAPL']` | ⚠️ Different (v3 is multi-ticker) |
| **min_dte** | 30 | 30 | ✅ Same |
| **max_dte** | 45 | 45 | ✅ Same |
| **min_delta** | 0.25 | 0.25 | ✅ Same |
| **max_delta** | 0.35 | 0.35 | ✅ Same |
| **option_type** | 'P' | 'P' | ✅ Same |
| **profit_target_pct** | 0.50 | 0.50 | ✅ Same |
| **exit_dte** | 21 | 0 | ❌ **DIFFERENT** |
| **use_trading_days_for_dte_filter** | Not set (defaults to True) | False | ❌ **DIFFERENT** |
| **use_trading_days_for_exit** | Not set (defaults to True) | False | ❌ **DIFFERENT** |
| **Technical Filter** | ✅ Enabled by default | ❌ Disabled by default | ❌ **DIFFERENT** |
| **Transaction Costs** | ❌ Not implemented | ✅ Enabled (commission: 0.65) | ✅ Expected difference |
| **Slippage** | ❌ Not implemented | ✅ Enabled (spread_percentage) | ✅ Expected difference |
| **Liquidity Filters** | ❌ Not implemented | ✅ Enabled | ✅ Expected difference |
| **Exit Fallback** | ❌ Not implemented | ✅ Implemented | ✅ Expected difference |

## Critical Mismatches

### 1. ❌ exit_dte: 21 vs 0
- **multi_day**: Exits 21 days before expiration
- **v3**: Holds to expiration (0 DTE)
- **Impact**: Completely different risk profiles

### 2. ❌ DTE Calculation Method
- **multi_day**: Uses trading days (default behavior)
- **v3**: Uses calendar days (`use_trading_days_for_dte_filter: False`)
- **Impact**: Different option selection, different actual DTE

### 3. ❌ Technical Filter
- **multi_day**: Enabled by default (BB/SMA filter)
- **v3**: Disabled by default
- **Impact**: v3 will enter on ALL dates, multi_day only on dips

## Recommendation

To make v3 consistent with multi_day (except for cost/liquidity implementation):

```python
# In v3 CONFIG, change these:
'exit_dte': 21,  # Match multi_day (not 0)
'use_trading_days_for_dte_filter': True,  # Match multi_day default
'use_trading_days_for_exit': True,  # Match multi_day default
'technical_filter_enabled': True,  # Match multi_day default
```

The only differences should be:
- v3 uses multiple tickers (`'tickers'` vs `'ticker'`)
- v3 has costs, slippage, liquidity filters (advanced features)
