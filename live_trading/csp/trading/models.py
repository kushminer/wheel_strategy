"""Core model classes for the CSP trading system."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class ExitReason(Enum):
    """Reason for exiting a position."""
    DELTA_STOP = "delta_doubled"
    DELTA_ABSOLUTE = "delta_exceeded_absolute"
    STOCK_DROP = "stock_dropped_5pct"
    VIX_SPIKE = "vix_spiked_15pct"
    EARLY_EXIT = "premium_captured_early"
    EXPIRY = "expired_worthless"
    ASSIGNED = "assigned"
    MANUAL = "manual_close"
    DATA_UNAVAILABLE = "data_unavailable"


@dataclass
class RiskCheckResult:
    """Result of a risk check on a position."""
    should_exit: bool
    exit_reason: Optional[ExitReason]
    details: str
    current_values: Dict[str, float]


@dataclass
class OrderResult:
    """Result of an order submission."""
    success: bool
    order_id: Optional[str]
    message: str
    order_details: Optional[dict] = None
