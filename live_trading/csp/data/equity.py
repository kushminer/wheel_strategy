"""Equity price data fetcher from Alpaca."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List

import pandas as pd

if TYPE_CHECKING:
    from csp.clients import AlpacaClientManager

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)


class EquityDataFetcher:
    """
    Fetches equity price data from Alpaca.
    Provides historical bars and current prices.
    """

    def __init__(self, alpaca_manager: "AlpacaClientManager") -> None:
        self.client = alpaca_manager.data_client

    def get_close_history(
        self,
        symbols: List[str],
        days: int = 60,
    ) -> Dict[str, pd.Series]:
        """
        Get closing price history for multiple symbols.

        Returns:
            Dict mapping symbol -> pd.Series of close prices
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
        )

        bars = self.client.get_stock_bars(request)

        result: Dict[str, pd.Series] = {}
        for symbol in symbols:
            if symbol in bars.data:
                symbol_bars = bars.data[symbol]
                closes = pd.Series(
                    [bar.close for bar in symbol_bars],
                    index=[bar.timestamp for bar in symbol_bars],
                    name=symbol,
                )
                result[symbol] = closes.tail(days)
            else:
                logger.warning("No data for symbol %s", symbol)

        return result

    def get_current_price(self, symbol: str) -> float:
        """Get the most recent price for a symbol."""
        history = self.get_close_history([symbol], days=5)
        if symbol in history and len(history[symbol]) > 0:
            return float(history[symbol].iloc[-1])
        raise ValueError(f"No price data for {symbol}")

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols efficiently."""
        history = self.get_close_history(symbols, days=5)
        return {
            symbol: float(prices.iloc[-1])
            for symbol, prices in history.items()
            if len(prices) > 0
        }
