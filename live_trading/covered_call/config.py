"""Configuration for the Covered Call (wheel) strategy."""

from dataclasses import dataclass


@dataclass
class CoveredCallConfig:
    """Configuration for covered call management.

    Strike selection, DTE range, and chain fetch window.
    Exit/risk params (early_exit_buffer_pct, stops) are inherited
    from StrategyConfig.  CC terminates only via expiration,
    assignment, or early-exit — no pre-entry termination check.
    """

    enabled: bool = False # Enable covered call management (default: disabled)

    # ==================== DTE RANGE ====================
    cc_min_dte: int = 1
    cc_max_dte: int = 10

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

    # ==================== CHAIN FETCH RANGE ====================
    # API search window — how wide a net to cast when fetching call contracts.
    # Only OTM/ATM calls make sense for covered calls.
    cc_min_strike_pct: float = 1.01     # fetch calls with strike >= stock_price * this (1.00 = ATM floor)
    cc_max_strike_pct: float = 1.15     # fetch calls with strike <= stock_price * this (1.15 = 15% OTM ceiling)

    # ==================== DIAGNOSTICS ====================
    cc_verbose: bool = False  # print detailed diagnostics when no contracts found
