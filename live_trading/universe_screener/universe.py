"""Ticker universe builders — S&P 500, DJIA, and NASDAQ-100 from Wikipedia."""

import requests
import pandas as pd
from io import StringIO

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_sp500_tickers() -> list:
    """Fetch current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=_HEADERS)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[0]

    tickers = []
    for sym in table["Symbol"]:
        sym = sym.strip()
        if "." in sym:
            continue
        tickers.append(sym)
    return sorted(tickers)


def get_djia_tickers() -> list:
    """Fetch current DJIA (Dow 30) constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    resp = requests.get(url, headers=_HEADERS)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[2]  # components table

    tickers = []
    for sym in table["Symbol"]:
        sym = sym.strip()
        if "." in sym:
            continue
        tickers.append(sym)
    return sorted(tickers)


def get_nasdaq100_tickers() -> list:
    """Fetch current NASDAQ-100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    resp = requests.get(url, headers=_HEADERS)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[4]  # components table

    tickers = []
    for sym in table["Ticker"]:
        sym = sym.strip()
        if "." in sym:
            continue
        tickers.append(sym)
    return sorted(tickers)


def get_combined_universe(verbose=False) -> list:
    """Build deduplicated universe from S&P 500 + DJIA + NASDAQ-100."""
    sp500 = get_sp500_tickers()
    djia = get_djia_tickers()
    nasdaq = get_nasdaq100_tickers()

    combined = sorted(set(sp500) | set(djia) | set(nasdaq))

    if verbose:
        print(f"Universe: S&P={len(sp500)} | DJIA={len(djia)} | "
              f"NASDAQ-100={len(nasdaq)} | Combined={len(combined)}")
    return combined
