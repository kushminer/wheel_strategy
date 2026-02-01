# Comparison: wheel_multi_symbol_backtest_v3 vs wheel_multi_day_backtest

## Critical Differences

| Feature | wheel_multi_day_backtest | wheel_multi_symbol_backtest_v3 | Issue? |
|---------|--------------------------|--------------------------------|--------|
| **Ticker(s)** | Single (`'ticker': 'TSLA'`) | Multiple (`'tickers': ['TSLA', 'AAPL']`) | ❌ Inconsistent purpose |
| **Header** | "Multi-Day Wheel Strategy" | "Multi-Symbol Wheel Strategy v2" | ❌ Wrong title (should be v3) |
| **Transaction Costs** | ❌ No | ✅ Yes (IBKR commissions) | Different feature set |
| **Slippage** | ❌ No | ✅ Yes (bid-ask spread model) | Different feature set |
| **Liquidity Filters** | ❌ No | ✅ Yes (spread %, size) | Different feature set |
| **Exit Fallback** | ❌ No | ✅ Yes (daily close → expiry logic) | Different feature set |
| **Technical Filter** | ✅ Yes (BB/SMA) | ✅ Yes (BB/SMA) | ✅ Same |
| **DTE Range** | 30-45 days (monthly) | 5-10 days (weekly) | ❌ **COMPLETELY DIFFERENT** |
| **Exit DTE** | 21 days before expiry | 0 days (hold to expiration) | ❌ **COMPLETELY DIFFERENT** |
| **Delta Range** | 0.25-0.35 | 0.00-0.15 | ❌ **COMPLETELY DIFFERENT** |

## The Core Problem

**v3 is NOT an evolution of multi_day_backtest!**

- **multi_day_backtest** = Simple, single-ticker, monthly options, exit at 21 DTE
- **v3** = Complex, multi-ticker, weekly options, hold to expiration, with costs/slippage

These are **completely different strategies**:
- multi_day = Conservative monthly put selling (30-45 DTE, 0.25-0.35 delta, exit early)
- v3 = Aggressive weekly put selling (5-10 DTE, 0.00-0.15 delta, hold to expiry)

## What Should v3 Be?

Based on the filename pattern:
- `wheel_multi_day_backtest` = Single ticker, multiple days, simple
- `wheel_multi_symbol_backtest_v2` = Multiple tickers, advanced features
- `wheel_multi_symbol_backtest_v3` = Should be an **evolution of v2**, not a different strategy

## Recommendation

**Option 1: Rename v3 to match its actual purpose**
- Rename to `wheel_weekly_strategy_backtest.ipynb` or similar
- Update header to reflect weekly options strategy

**Option 2: Make v3 actually be v3**
- Copy the correct configuration from multi_day_backtest
- Keep the advanced features (costs, slippage, etc.)
- Use the same DTE/delta/exit settings as multi_day

**Option 3: Keep both strategies**
- Rename v3 to reflect weekly strategy
- Create a proper v3 that extends v2 with incremental improvements
