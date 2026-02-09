"""Portfolio manager for CSP positions."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from csp.trading.models import ActivePosition, ExitReason, PositionStatus

if TYPE_CHECKING:
    from csp.config import StrategyConfig

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Manages the portfolio of active positions.
    Handles position tracking, persistence, and portfolio-level metrics.
    """

    def __init__(
        self,
        config: "StrategyConfig",
        persistence_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.positions: Dict[str, ActivePosition] = {}
        self.closed_positions: List[ActivePosition] = []
        self.persistence_path = persistence_path
        self._position_counter = 0

        if persistence_path and os.path.exists(persistence_path):
            self._load_state()

    def _generate_position_id(self) -> str:
        """Generate unique position ID."""
        self._position_counter += 1
        return f"POS_{datetime.now().strftime('%Y%m%d')}_{self._position_counter:04d}"

    def add_position(self, position: ActivePosition) -> str:
        """
        Add a new position to the portfolio.

        Returns:
            Position ID
        """
        if not position.position_id:
            position.position_id = self._generate_position_id()

        self.positions[position.position_id] = position
        self._save_state()
        return position.position_id

    def close_position(
        self,
        position_id: str,
        exit_premium: float,
        exit_reason: ExitReason,
        exit_details: str = "",
    ) -> Optional[ActivePosition]:
        """
        Close a position and move to closed list.

        Returns:
            The closed position, or None if not found
        """
        if position_id not in self.positions:
            return None

        position = self.positions[position_id]
        position.exit_date = datetime.now()
        position.exit_premium = exit_premium
        position.exit_reason = exit_reason
        position.exit_details = exit_details

        status_map = {
            ExitReason.DELTA_STOP: PositionStatus.CLOSED_STOP_LOSS,
            ExitReason.STOCK_DROP: PositionStatus.CLOSED_STOP_LOSS,
            ExitReason.VIX_SPIKE: PositionStatus.CLOSED_STOP_LOSS,
            ExitReason.EARLY_EXIT: PositionStatus.CLOSED_EARLY_EXIT,
            ExitReason.EXPIRY: PositionStatus.CLOSED_EXPIRY,
            ExitReason.MANUAL: PositionStatus.CLOSED_MANUAL,
        }
        position.status = status_map.get(exit_reason, PositionStatus.CLOSED_MANUAL)

        self.closed_positions.append(position)
        del self.positions[position_id]

        self._save_state()
        return position

    def get_position(self, position_id: str) -> Optional[ActivePosition]:
        """Get position by ID."""
        return self.positions.get(position_id)

    def get_position_by_symbol(self, symbol: str) -> Optional[ActivePosition]:
        """Get active position for underlying symbol."""
        for pos in self.positions.values():
            if pos.symbol == symbol and pos.is_open:
                return pos
        return None

    def get_active_positions(self) -> List[ActivePosition]:
        """Get all active positions."""
        return [p for p in self.positions.values() if p.is_open]

    @property
    def active_count(self) -> int:
        """Number of active positions."""
        return len(self.get_active_positions())

    @property
    def total_collateral(self) -> float:
        """Total collateral locked in active positions."""
        return sum(p.collateral_required for p in self.get_active_positions())

    @property
    def active_symbols(self) -> List[str]:
        """List of underlying symbols with active positions."""
        return [p.symbol for p in self.get_active_positions()]

    def get_available_cash(self, deployable_cash: float) -> float:
        """
        Calculate available cash for new positions.

        Args:
            deployable_cash: Total cash allowed to deploy (based on VIX)

        Returns:
            Cash available for new positions
        """
        return max(0.0, deployable_cash - self.total_collateral)

    def can_add_position(
        self,
        collateral_needed: float,
        deployable_cash: float,
    ) -> bool:
        """
        Check if we can add a new position.

        Args:
            collateral_needed: Collateral for new position
            deployable_cash: Total deployable cash

        Returns:
            True if position can be added
        """
        if self.active_count >= self.config.num_tickers:
            return False
        if self.get_available_cash(deployable_cash) < collateral_needed:
            return False
        return True

    def _save_state(self) -> None:
        """Persist current state to file."""
        if not self.persistence_path:
            return

        state = {
            "positions": {pid: p.to_dict() for pid, p in self.positions.items()},
            "closed_positions": [p.to_dict() for p in self.closed_positions],
            "position_counter": self._position_counter,
        }

        try:
            with open(self.persistence_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save portfolio state: %s", e)

    def _load_state(self) -> None:
        """Load state from file."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return

        try:
            with open(self.persistence_path) as f:
                state = json.load(f)

            self.positions = {
                pid: ActivePosition.from_dict(data)
                for pid, data in state.get("positions", {}).items()
            }
            self.closed_positions = [
                ActivePosition.from_dict(data)
                for data in state.get("closed_positions", [])
            ]
            self._position_counter = state.get("position_counter", 0)
        except Exception as e:
            logger.warning("Failed to load portfolio state: %s", e)

    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary statistics."""
        active = self.get_active_positions()

        return {
            "active_positions": len(active),
            "total_collateral": self.total_collateral,
            "total_premium_received": sum(p.total_premium_received for p in active),
            "symbols": self.active_symbols,
            "closed_count": len(self.closed_positions),
            "closed_pnl": sum(
                p.calculate_pnl(p.exit_premium)
                for p in self.closed_positions
                if p.exit_premium is not None
            ),
        }
