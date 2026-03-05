"""Core model classes for the CSP trading system."""

from dataclasses import dataclass
from datetime import date, datetime
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


@dataclass
class PositionProxy:
    """Typed position representation for RiskManager compatibility.

    Replaces ad-hoc SimpleNamespace construction in both CSP and CC strategies.
    Provides factory classmethods for building from different data sources.
    """

    symbol: str                      # underlying ticker (e.g., "AAPL")
    option_symbol: str               # OCC format (e.g., "AAPL260320P00220000")
    quantity: int                    # signed (-1 for short put, 1 for short call qty)
    strike: float
    expiration: date
    entry_delta: float = 0.0
    entry_iv: float = 0.0
    entry_vix: float = 0.0
    entry_stock_price: float = 0.0
    entry_premium: float = 0.0
    entry_daily_return: float = 0.0
    dte_at_entry: int = 0
    entry_order_id: str = ""
    current_dte: int = 0
    days_held: int = 0
    collateral_required: float = 0.0

    def calculate_pnl(self, exit_premium: float) -> float:
        """Calculate P&L from entry premium to exit premium (per-share * contracts * 100)."""
        return (self.entry_premium - exit_premium) * abs(self.quantity) * 100

    @classmethod
    def from_alpaca_and_metadata(cls, alpaca_pos, meta: dict) -> "PositionProxy":
        """Factory for CSP: build from Alpaca position + StrategyMetadataStore entry."""
        from csp.clients import AlpacaClientManager

        strike = AlpacaClientManager.parse_strike_from_symbol(alpaca_pos.symbol)
        expiration = AlpacaClientManager.parse_expiration_from_symbol(alpaca_pos.symbol) or date.today()
        entry_date_str = meta.get('entry_date', datetime.now().isoformat())
        qty = int(float(alpaca_pos.qty))

        return cls(
            symbol=meta.get('underlying', ''),
            option_symbol=alpaca_pos.symbol,
            quantity=qty,
            strike=strike,
            expiration=expiration,
            entry_delta=meta.get('entry_delta', 0),
            entry_iv=meta.get('entry_iv', 0),
            entry_vix=meta.get('entry_vix', 0),
            entry_stock_price=meta.get('entry_stock_price', 0),
            entry_premium=meta.get('entry_premium', 0),
            entry_daily_return=meta.get('entry_daily_return', 0),
            dte_at_entry=meta.get('dte_at_entry', 0),
            entry_order_id=meta.get('entry_order_id', ''),
            current_dte=(expiration - date.today()).days,
            days_held=(date.today() - datetime.fromisoformat(entry_date_str).date()).days,
            collateral_required=abs(qty) * strike * 100,
        )

    @classmethod
    def from_cc_store(cls, underlying: str, cc_data: dict, pos_data: dict) -> "PositionProxy":
        """Factory for CC: build from WheelPositionStore data."""
        entry_premium = cc_data.get("entry_premium", 0)
        entry_date_str = cc_data.get("entry_date", datetime.now().isoformat())
        strike = cc_data.get("strike", 0)
        expiration_str = cc_data.get("expiration")
        expiration = date.fromisoformat(expiration_str) if expiration_str else date.today()

        return cls(
            symbol=underlying,
            option_symbol=cc_data.get("option_symbol", ""),
            quantity=cc_data.get("quantity", 1),
            strike=strike,
            expiration=expiration,
            entry_delta=0,
            entry_stock_price=pos_data.get("cost_basis", 0),
            entry_premium=entry_premium,
            entry_vix=0,
            entry_daily_return=0,
            current_dte=(expiration - date.today()).days,
            days_held=max((datetime.now() - datetime.fromisoformat(entry_date_str)).days, 0),
            collateral_required=0,
        )
