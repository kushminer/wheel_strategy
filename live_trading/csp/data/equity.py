"""Equity price data fetcher via Alpaca."""

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class EquityDataFetcher:
    """Fetches equity price data from Alpaca."""

    def __init__(self, alpaca_manager):
        self.client = alpaca_manager.data_client

    def get_close_history(
        self,
        symbols: List[str],
        days: int = 60,
    ) -> Dict[str, pd.Series]:
        start_date = datetime.now() - timedelta(days=int(days * 1.5))

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            feed="iex",
        )

        bars = self.client.get_stock_bars(request)

        result = {}
        for symbol in symbols:
            if symbol in bars.data:
                symbol_bars = bars.data[symbol]
                closes = pd.Series(
                    [bar.close for bar in symbol_bars],
                    index=[bar.timestamp for bar in symbol_bars],
                    name=symbol,
                )
                result[symbol] = closes.tail(days)

        return result

    def get_current_price(self, symbol: str, price_lookback_days: int = 5) -> float:
        history = self.get_close_history([symbol], days=price_lookback_days)
        if symbol in history and len(history[symbol]) > 0:
            return float(history[symbol].iloc[-1])
        raise ValueError(f"No price data for {symbol}")

    def get_current_prices(self, symbols: List[str], price_lookback_days: int = 5) -> Dict[str, float]:
        history = self.get_close_history(symbols, days=price_lookback_days)
        return {
            symbol: float(prices.iloc[-1])
            for symbol, prices in history.items()
            if len(prices) > 0
        }
