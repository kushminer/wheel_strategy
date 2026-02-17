"""Alpaca API client management."""

import os
import re
from datetime import date
from typing import Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient


class AlpacaClientManager:
    """Manages Alpaca API clients for data and trading."""

    def __init__(self, paper: bool = True):
        self.paper = paper

        if paper:
            self.api_key = os.getenv('ALPACA_API_KEY')
            self.secret_key = os.getenv('ALPACA_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca credentials not found. Set environment variables:\n"
                "  ALPACA_API_KEY and ALPACA_SECRET_KEY"
            )

        self._data_client = None
        self._trading_client = None

    @property
    def data_client(self) -> StockHistoricalDataClient:
        if self._data_client is None:
            self._data_client = StockHistoricalDataClient(
                api_key=self.api_key, secret_key=self.secret_key
            )
        return self._data_client

    @property
    def trading_client(self) -> TradingClient:
        if self._trading_client is None:
            self._trading_client = TradingClient(
                api_key=self.api_key, secret_key=self.secret_key, paper=self.paper
            )
        return self._trading_client

    @staticmethod
    def parse_strike_from_symbol(symbol: str) -> float:
        """Parse strike price from OCC format option symbol."""
        match = re.search(r'[PC](\d+)$', symbol)
        if match:
            return int(match.group(1)) / 1000.0
        return 0.0

    @staticmethod
    def parse_expiration_from_symbol(symbol: str) -> Optional[date]:
        """Parse expiration date from OCC format option symbol."""
        match = re.search(r'(\d{6})[PC]', symbol)
        if match:
            d = match.group(1)
            return date(2000 + int(d[:2]), int(d[2:4]), int(d[4:6]))
        return None

    def get_account_info(self) -> dict:
        account = self.trading_client.get_account()
        return {
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'status': account.status,
            'trading_blocked': account.trading_blocked,
            'options_trading_level': getattr(account, 'options_trading_level', None),
        }

    def get_short_collateral(self) -> float:
        total = 0.0
        try:
            positions = self.trading_client.get_all_positions()
            for pos in positions:
                qty = float(pos.qty)
                side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                if side == 'short' or qty < 0:
                    strike = self.parse_strike_from_symbol(pos.symbol)
                    total += abs(qty) * strike * 100
        except Exception:
            pass
        return total

    def compute_available_capital(self) -> float:
        account_info = self.get_account_info()
        collateral = self.get_short_collateral()
        return account_info['cash'] - collateral
