"""Wheel position store — tracks stock positions through the CC lifecycle."""

import json
from datetime import datetime
from typing import Dict, List, Optional

from csp.storage import LocalStorage, StorageBackend


class WheelPositionStore:
    """Tracks stock positions being managed via covered calls.

    Keyed by underlying symbol (e.g. "AAPL").
    Persisted as JSON via the shared storage backend.
    """

    def __init__(self, path: str = "wheel_positions.json", backend: StorageBackend = None):
        self.path = path
        self._backend = backend or LocalStorage()
        self.positions: Dict[str, dict] = {}
        self._load()

    # -- Write operations --------------------------------------------------

    def add_position(
        self,
        underlying: str,
        *,
        shares: int,
        cost_basis: float,
        source: str,                   # "csp_assignment" or "alpaca_position"
        csp_entry_premium: Optional[float] = None,
    ):
        """Record a new stock position for CC management."""
        self.positions[underlying] = {
            "underlying": underlying,
            "shares": shares,
            "cost_basis": cost_basis,
            "source": source,
            "detected_date": datetime.now().isoformat(),
            "csp_entry_premium": csp_entry_premium,
            "total_cc_premiums": 0.0,
            "cc_rounds": 0,
            "current_cc": None,
            "cc_history": [],
            "status": "awaiting_cc_entry",
        }
        self._save()

    def update(self, underlying: str, **fields):
        """Update fields on an existing wheel position."""
        pos = self.positions.get(underlying)
        if pos is None:
            print(f"  Warning: no wheel position for {underlying}")
            return
        pos.update(fields)
        self._save()

    def record_cc_entry(
        self,
        underlying: str,
        *,
        option_symbol: str,
        strike: float,
        expiration: str,
        entry_premium: float,
        quantity: int,
        order_id: str,
    ):
        """Record a new covered call entry on a wheel position."""
        pos = self.positions.get(underlying)
        if pos is None:
            return
        pos["current_cc"] = {
            "option_symbol": option_symbol,
            "strike": strike,
            "expiration": expiration,
            "entry_premium": entry_premium,
            "quantity": quantity,
            "order_id": order_id,
            "entry_date": datetime.now().isoformat(),
        }
        pos["status"] = "cc_active"
        self._save()

    def record_cc_exit(
        self,
        underlying: str,
        *,
        exit_reason: str,
        exit_premium: float = 0.0,
    ):
        """Record a covered call exit (expired, closed, or assigned)."""
        pos = self.positions.get(underlying)
        if pos is None:
            return
        cc = pos.get("current_cc")
        if cc:
            premium_earned = cc["entry_premium"] - exit_premium
            cc["exit_date"] = datetime.now().isoformat()
            cc["exit_reason"] = exit_reason
            cc["exit_premium"] = exit_premium
            cc["premium_earned"] = premium_earned
            pos["cc_history"].append(cc)
            pos["total_cc_premiums"] += premium_earned
            pos["cc_rounds"] += 1
        pos["current_cc"] = None
        pos["status"] = "awaiting_cc_entry"
        self._save()

    def terminate(self, underlying: str, reason: str = ""):
        """Mark a wheel position as terminated (shares sold or called away)."""
        pos = self.positions.get(underlying)
        if pos is None:
            return
        pos["status"] = "terminated"
        pos["termination_date"] = datetime.now().isoformat()
        pos["termination_reason"] = reason
        self._save()

    # -- Read operations ---------------------------------------------------

    def get(self, underlying: str) -> Optional[dict]:
        return self.positions.get(underlying)

    def get_active(self) -> Dict[str, dict]:
        """Return positions that are not terminated."""
        return {
            sym: pos for sym, pos in self.positions.items()
            if pos.get("status") != "terminated"
        }

    def get_by_status(self, status: str) -> Dict[str, dict]:
        return {
            sym: pos for sym, pos in self.positions.items()
            if pos.get("status") == status
        }

    # -- Persistence -------------------------------------------------------

    def _save(self):
        self._backend.write(self.path, json.dumps(self.positions, indent=2))

    def _load(self):
        if not self._backend.exists(self.path):
            return
        try:
            self.positions = json.loads(self._backend.read(self.path))
        except (json.JSONDecodeError, Exception):
            self.positions = {}
