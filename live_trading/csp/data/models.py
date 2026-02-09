"""Data model dataclasses for the CSP strategy."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


@dataclass
class OptionContract:
    """
    Represents a single option contract with relevant data.
    """

    symbol: str
    underlying: str
    contract_type: str
    strike: float
    expiration: object  # date
    dte: int
    bid: float
    ask: float
    mid: float
    stock_price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None

    @property
    def premium(self) -> float:
        """Premium received when selling (use bid price)."""
        return self.bid

    @property
    def premium_per_day(self) -> float:
        """Daily premium decay if held to expiration."""
        if self.dte <= 0:
            return 0.0
        return self.premium / self.dte

    @property
    def collateral_required(self) -> float:
        """Cash required to secure 1 contract."""
        return self.strike * 100

    @property
    def cost_basis(self) -> float:
        """Cost basis = stock price * 100 (exposure value)."""
        return self.stock_price * 100

    @property
    def daily_return_on_collateral(self) -> float:
        """Daily yield as % of collateral (strike-based)."""
        if self.strike <= 0 or self.dte <= 0:
            return 0.0
        return self.premium_per_day / self.strike

    @property
    def daily_return_on_cost_basis(self) -> float:
        """Daily yield as % of cost basis (stock price-based)."""
        if self.stock_price <= 0 or self.dte <= 0:
            return 0.0
        return self.premium_per_day / self.stock_price

    @property
    def delta_abs(self) -> Optional[float]:
        """Absolute value of delta for filtering."""
        return abs(self.delta) if self.delta else None


@dataclass
class MarketSnapshot:
    """
    Complete market state at a point in time.
    Used by strategy logic to make decisions.
    """

    timestamp: datetime
    vix_current: float
    vix_open: float
    deployable_cash: float
    equity_prices: Dict[str, float]
    equity_history: Dict[str, pd.Series]
