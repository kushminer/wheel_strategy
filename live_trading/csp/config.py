"""CSP strategy configuration with environment variable overrides."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


def _parse_ticker_universe() -> List[str]:
    """Parse TICKER_UNIVERSE from env as comma-separated list."""
    val = os.getenv("TICKER_UNIVERSE")
    if val:
        return [s.strip() for s in val.split(",") if s.strip()]
    return ["AAPL", "MSFT", "GOOG"]


def _parse_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _parse_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _parse_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).lower() in ("true", "1", "yes")


@dataclass
class StrategyConfig:
    """
    Central configuration for the CSP strategy.
    All parameters in one place for easy tuning.
    Environment variables can override defaults (e.g. PAPER_TRADING, POLL_INTERVAL_SECONDS).
    """

    # ==================== UNIVERSE & CAPITAL ====================
    ticker_universe: List[str] = field(default_factory=_parse_ticker_universe)
    num_tickers: int = 10
    starting_cash: float = 50_000

    # ==================== VIX REGIME RULES ====================
    vix_deployment_rules: Dict[Tuple[float, float], float] = field(
        default_factory=lambda: {
            (0, 15): 1.0,
            (15, 20): 0.8,
            (20, 25): 0.2,
            (25, float("inf")): 0.0,
        }
    )

    # ==================== EQUITY FILTER PARAMS ====================
    sma_periods: List[int] = field(default_factory=lambda: [8, 20, 50])
    rsi_period: int = 14
    rsi_lower: int = 30
    rsi_upper: int = 70
    bb_period: int = 50
    bb_std: float = 1.0
    sma_trend_lookback: int = 3
    history_days: int = 60

    # ==================== OPTIONS FILTER PARAMS ====================
    min_daily_return: float = 0.15
    max_strike_pct: float = 0.98
    min_strike_pct: float = 0.85
    delta_min: float = 0.15
    delta_max: float = 0.40
    max_dte: int = 10
    min_dte: int = 1

    # ==================== RISK MANAGEMENT ====================
    delta_stop_multiplier: float = 2.0
    stock_drop_stop_pct: float = 0.05
    vix_spike_multiplier: float = 1.15
    early_exit_buffer: float = 0.15

    # ==================== OPERATIONAL ====================
    poll_interval_seconds: int = 60
    paper_trading: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        self.ticker_universe = _parse_ticker_universe()
        self.num_tickers = _parse_int("NUM_TICKERS", self.num_tickers)
        self.starting_cash = _parse_float("STARTING_CASH", self.starting_cash)
        self.poll_interval_seconds = _parse_int("POLL_INTERVAL_SECONDS", self.poll_interval_seconds)
        self.paper_trading = _parse_bool("PAPER_TRADING", self.paper_trading)
        log_val = os.getenv("LOG_LEVEL")
        if log_val:
            self.log_level = log_val.upper()

    def get_deployment_multiplier(self, vix: float) -> float:
        """Get cash deployment multiplier based on current VIX."""
        for (lower, upper), multiplier in self.vix_deployment_rules.items():
            if lower <= vix < upper:
                return multiplier
        return 0.0

    def get_deployable_cash(self, vix: float) -> float:
        """Calculate deployable cash based on VIX regime."""
        return self.starting_cash * self.get_deployment_multiplier(vix)
