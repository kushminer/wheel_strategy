"""Trading layer models - position status, exit reason, active position, order result."""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Optional


class PositionStatus(Enum):
    """Status of an active position."""

    PENDING = "pending"  # Order placed, not yet filled
    ACTIVE = "active"  # Position is open
    CLOSING = "closing"  # Close order placed
    CLOSED_STOP_LOSS = "closed_stop_loss"
    CLOSED_EARLY_EXIT = "closed_early_exit"
    CLOSED_EXPIRY = "closed_expiry"
    CLOSED_MANUAL = "closed_manual"


class ExitReason(Enum):
    """Reason for exiting a position."""

    DELTA_STOP = "delta_doubled"
    STOCK_DROP = "stock_dropped_5pct"
    VIX_SPIKE = "vix_spiked_15pct"
    EARLY_EXIT = "premium_captured_early"
    EXPIRY = "expired_worthless"
    ASSIGNED = "assigned"
    MANUAL = "manual_close"


@dataclass
class ActivePosition:
    """
    Represents an active CSP position with all tracking data.
    """

    # Identification
    position_id: str
    symbol: str  # Underlying symbol
    option_symbol: str  # OCC option symbol

    # Entry data
    entry_date: datetime
    entry_stock_price: float
    entry_delta: float
    entry_premium: float  # Per share premium received
    entry_vix: float
    entry_iv: float

    # Contract details
    strike: float
    expiration: date
    dte_at_entry: int
    quantity: int  # Number of contracts (negative for short)

    # Current state
    status: PositionStatus = PositionStatus.ACTIVE

    # Exit data (populated when closed)
    exit_date: Optional[datetime] = None
    exit_premium: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    exit_details: Optional[str] = None

    # Order tracking
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None

    @property
    def collateral_required(self) -> float:
        """Cash required to secure this position."""
        return self.strike * 100 * abs(self.quantity)

    @property
    def total_premium_received(self) -> float:
        """Total premium received at entry."""
        return self.entry_premium * 100 * abs(self.quantity)

    @property
    def current_dte(self) -> int:
        """Current days to expiration."""
        return (self.expiration - date.today()).days

    @property
    def days_held(self) -> int:
        """Number of days position has been held."""
        end = self.exit_date or datetime.now()
        return (end - self.entry_date).days

    @property
    def is_open(self) -> bool:
        """Whether position is still open."""
        return self.status in (PositionStatus.ACTIVE, PositionStatus.PENDING)

    def calculate_pnl(self, exit_premium: float) -> float:
        """
        Calculate P&L for closing at given premium.
        For short puts: profit = entry_premium - exit_premium
        """
        return (self.entry_premium - exit_premium) * 100 * abs(self.quantity)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "option_symbol": self.option_symbol,
            "entry_date": self.entry_date.isoformat(),
            "entry_stock_price": self.entry_stock_price,
            "entry_delta": self.entry_delta,
            "entry_premium": self.entry_premium,
            "entry_vix": self.entry_vix,
            "entry_iv": self.entry_iv,
            "strike": self.strike,
            "expiration": self.expiration.isoformat(),
            "dte_at_entry": self.dte_at_entry,
            "quantity": self.quantity,
            "status": self.status.value,
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "exit_premium": self.exit_premium,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "exit_details": self.exit_details,
            "entry_order_id": self.entry_order_id,
            "exit_order_id": self.exit_order_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivePosition":
        """Deserialize from dictionary."""
        return cls(
            position_id=data["position_id"],
            symbol=data["symbol"],
            option_symbol=data["option_symbol"],
            entry_date=datetime.fromisoformat(data["entry_date"]),
            entry_stock_price=data["entry_stock_price"],
            entry_delta=data["entry_delta"],
            entry_premium=data["entry_premium"],
            entry_vix=data["entry_vix"],
            entry_iv=data["entry_iv"],
            strike=data["strike"],
            expiration=date.fromisoformat(data["expiration"]),
            dte_at_entry=data["dte_at_entry"],
            quantity=data["quantity"],
            status=PositionStatus(data["status"]),
            exit_date=(
                datetime.fromisoformat(data["exit_date"]) if data.get("exit_date") else None
            ),
            exit_premium=data.get("exit_premium"),
            exit_reason=(
                ExitReason(data["exit_reason"]) if data.get("exit_reason") else None
            ),
            exit_details=data.get("exit_details"),
            entry_order_id=data.get("entry_order_id"),
            exit_order_id=data.get("exit_order_id"),
        )


@dataclass
class OrderResult:
    """Result of an order submission."""

    success: bool
    order_id: Optional[str]
    message: str
    order_details: Optional[Dict[str, Any]] = None


@dataclass
class RiskCheckResult:
    """Result of a risk check on a position."""

    should_exit: bool
    exit_reason: Optional[ExitReason]
    details: str
    current_values: Dict[str, Any]
