"""Data layer models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import pandas as pd


@dataclass
class MarketSnapshot:
    """Complete market state at a point in time."""
    timestamp: datetime
    vix_current: float
    vix_open: float
    deployable_cash: float
    equity_prices: Dict[str, float]
    equity_history: Dict[str, pd.Series]
