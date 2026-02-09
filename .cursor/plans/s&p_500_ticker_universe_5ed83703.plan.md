---
name: S&P 500 ticker universe
overview: Replace the hardcoded 3-ticker universe with a function that fetches all ~503 S&P 500 constituents from Wikipedia at startup, with a cached fallback. Update the Phase 3 notebook to match.
todos:
  - id: create-sp500-module
    content: Create `csp/data/sp500.py` with `fetch_sp500_tickers()` that scrapes Wikipedia and caches results
    status: completed
  - id: update-config
    content: Update `_parse_ticker_universe()` in `csp/config.py` to use `fetch_sp500_tickers()` as default
    status: in_progress
  - id: add-lxml-dep
    content: Add `lxml>=5.0` to `requirements.txt` for `pd.read_html()`
    status: pending
  - id: update-notebook
    content: Update `3_csp_strategy_phase3.ipynb` StrategyConfig cell to use S&P 500 list as default universe
    status: pending
isProject: false
---

# S&P 500 Ticker Universe

## Approach

Add a helper function `fetch_sp500_tickers()` that scrapes the [Wikipedia S&P 500 list](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) using `pandas.read_html()`. This requires no new dependency (pandas is already installed). The function caches results in memory so repeated calls during a session don't re-fetch.

The `TICKER_UNIVERSE` env var override still works -- if set, it takes priority over the Wikipedia fetch, letting Cloud Run deployments pin a custom list.

## Files to change

### 1. New file: `live_trading/csp/data/sp500.py`

A small utility module:

- `fetch_sp500_tickers() -> List[str]` -- scrapes Wikipedia, returns sorted list of ~503 symbols
- Handles BRK.B -> BRK-B style conversion (Alpaca uses hyphens)
- Caches result in a module-level variable so it's fetched once per process
- On failure (network error), logs a warning and returns a hardcoded fallback of ~10 large-cap tickers

```python
import logging
import pandas as pd

_cache: list | None = None

def fetch_sp500_tickers() -> list[str]:
    global _cache
    if _cache is not None:
        return _cache
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = sorted(df["Symbol"].str.replace(".", "-", regex=False).tolist())
        _cache = tickers
        return tickers
    except Exception:
        logging.getLogger(__name__).warning("Failed to fetch S&P 500 list; using fallback")
        return ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT"]
```

### 2. Update: `[live_trading/csp/config.py](live_trading/csp/config.py)`

- Change `_parse_ticker_universe()` to call `fetch_sp500_tickers()` as the default instead of returning `["AAPL", "MSFT", "GOOG"]`
- The env var `TICKER_UNIVERSE` still overrides if set

```python
def _parse_ticker_universe() -> List[str]:
    val = os.getenv("TICKER_UNIVERSE")
    if val:
        return [s.strip() for s in val.split(",") if s.strip()]
    from csp.data.sp500 import fetch_sp500_tickers
    return fetch_sp500_tickers()
```

### 3. Update: `[live_trading/requirements.txt](live_trading/requirements.txt)`

- `lxml` is needed by `pd.read_html()` -- add `lxml>=5.0` (or `html5lib` as alternative)

### 4. Update: `[live_trading/3_csp_strategy_phase3.ipynb](live_trading/3_csp_strategy_phase3.ipynb)`

- In the `StrategyConfig` cell, change the `ticker_universe` default from the 3-ticker list to call the same Wikipedia scrape function (defined inline or imported)
- The notebook's config cell currently has:
  ```python
  ticker_universe: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOG'])
  ```
  Replace with a function that fetches the S&P 500 list

## Considerations

- **Scanning 503 symbols takes time.** The equity filter will quickly reject most symbols (SMA/RSI checks are fast since it's just math on cached price history). The options chain fetch only runs for symbols passing the equity filter. This is the existing behavior -- no architecture change needed.
- **Alpaca rate limits:** `StockBarsRequest` supports batching multiple symbols in one call, which `EquityDataFetcher.get_close_history()` already does. 503 symbols in one bars request is fine.
- `**num_tickers` stays at 10** -- this controls max concurrent positions, not the scan universe size. No change needed.

