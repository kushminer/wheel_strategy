"""Alpaca API client manager."""

import os
import logging

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

logger = logging.getLogger(__name__)


class AlpacaClientManager:
    """
    Manages Alpaca API clients for data and trading.
    Handles authentication and provides unified access.
    """

    def __init__(self, paper: bool = True) -> None:
        """
        Initialize Alpaca clients.

        Args:
            paper: If True, use paper trading credentials
        """
        self.paper = paper

        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca credentials not found. Set environment variables:\n"
                "  ALPACA_API_KEY and ALPACA_SECRET_KEY"
            )

        self._data_client = None
        self._trading_client = None

    @property
    def data_client(self) -> StockHistoricalDataClient:
        """Lazy initialization of stock data client."""
        if self._data_client is None:
            self._data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
        return self._data_client

    @property
    def trading_client(self) -> TradingClient:
        """Lazy initialization of trading client."""
        if self._trading_client is None:
            self._trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper,
            )
        return self._trading_client

    def get_account_info(self) -> dict:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "status": account.status,
            "trading_blocked": account.trading_blocked,
            "options_trading_level": getattr(account, "options_trading_level", None),
        }
