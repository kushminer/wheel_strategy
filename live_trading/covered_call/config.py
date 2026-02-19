"""Configuration for the Covered Call (wheel) strategy."""

from dataclasses import dataclass


@dataclass
class CoveredCallConfig:
    """Configuration for covered call management.

    Strike selection, DTE range, and termination conditions.
    Exit/risk params are inherited from StrategyConfig.
    """

    enabled: bool = False

    # ==================== DTE RANGE ====================
    cc_min_dte: int = 1
    cc_max_dte: int = 6

    # ==================== STRIKE SELECTION MODE ====================
    cc_strike_mode: str = "delta"
    # "delta"            -> strike nearest to cc_strike_delta (e.g., 0.30)
    # "min_delta"        -> strike nearest but delta >= cc_strike_delta (floor)
    # "pct_change"       -> strike nearest to stock_price * (1 + cc_strike_pct)
    # "min_pct_change"   -> strike nearest but >= stock_price * (1 + cc_strike_pct)
    # "min_daily_return" -> strike with daily return on collateral closest to
    #                       but >= cc_min_daily_return_pct

    cc_strike_delta: float = 0.30       # used by "delta" and "min_delta" modes
    cc_strike_pct: float = 0.02         # used by "pct_change" and "min_pct_change" (2% OTM)
    cc_min_daily_return_pct: float = 0.0015  # used by "min_daily_return" mode (0.15%/day)

    # ==================== TERMINATION MODE ====================
    cc_exit_mode: str = "strike_recovery"
    # "strike_recovery"  -> stop when selected CC strike >= cost_basis
    # "premium_recovery" -> stop when total CC premiums >= unrealized loss at assignment
