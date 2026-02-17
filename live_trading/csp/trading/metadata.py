"""Strategy metadata store — tracks entry context Alpaca doesn't store."""

import json
from datetime import datetime
from typing import Dict, Optional

from csp.storage import LocalStorage, StorageBackend


class StrategyMetadataStore:
    """Tracks strategy-specific metadata that Alpaca doesn't store.

    Alpaca is source of truth for positions and orders.
    This store records the strategy context (entry greeks, exit reasons,
    order attempt diagnostics) that Alpaca has no concept of.

    Keyed by option_symbol (OCC format — globally unique per contract).
    """

    def __init__(self, path: str = "strategy_metadata.json", backend: StorageBackend = None):
        self.path = path
        self._backend = backend or LocalStorage()
        self.entries: Dict[str, dict] = {}   # option_symbol -> metadata
        self._load()

    # ── Write operations ────────────────────────────────────────

    def record_entry(
        self,
        option_symbol: str,
        *,
        underlying: str,
        strike: float,
        expiration: str,          # ISO date string
        entry_delta: float,
        entry_iv: float,
        entry_vix: float,
        entry_stock_price: float,
        entry_premium: float,
        entry_daily_return: float,
        dte_at_entry: int,
        quantity: int,
        entry_order_id: str,
    ):
        """Record strategy metadata when we enter a position."""
        self.entries[option_symbol] = {
            "underlying": underlying,
            "option_symbol": option_symbol,
            "strike": strike,
            "expiration": expiration,
            "entry_date": datetime.now().isoformat(),
            "entry_delta": entry_delta,
            "entry_iv": entry_iv,
            "entry_vix": entry_vix,
            "entry_stock_price": entry_stock_price,
            "entry_premium": entry_premium,
            "entry_daily_return": entry_daily_return,
            "dte_at_entry": dte_at_entry,
            "quantity": quantity,
            "entry_order_id": entry_order_id,
            # Exit fields — filled when we close
            "exit_date": None,
            "exit_reason": None,
            "exit_details": None,
            "exit_order_id": None,
        }
        self._save()

    def record_exit(
        self,
        option_symbol: str,
        *,
        exit_reason: str,
        exit_details: str = "",
        exit_order_id: str = "",
    ):
        """Record why we exited. Alpaca knows we bought-to-close; this records WHY."""
        meta = self.entries.get(option_symbol)
        if meta is None:
            print(f"  Warning: no metadata for {option_symbol}, recording exit anyway")
            meta = {"option_symbol": option_symbol}
            self.entries[option_symbol] = meta
        meta["exit_date"] = datetime.now().isoformat()
        meta["exit_reason"] = exit_reason
        meta["exit_details"] = exit_details
        meta["exit_order_id"] = exit_order_id
        self._save()

    # ── Read operations ─────────────────────────────────────────

    def get(self, option_symbol: str) -> Optional[dict]:
        """Get strategy metadata for a position. Returns None if unknown."""
        return self.entries.get(option_symbol)

    def get_active(self) -> Dict[str, dict]:
        """Entries that haven't been marked exited yet."""
        return {
            sym: meta for sym, meta in self.entries.items()
            if meta.get("exit_date") is None
        }

    def has_symbol(self, underlying: str) -> bool:
        """Check if we have an active entry for this underlying."""
        return any(
            m.get("underlying") == underlying and m.get("exit_date") is None
            for m in self.entries.values()
        )

    # ── Persistence ─────────────────────────────────────────────

    def _save(self):
        self._backend.write(self.path, json.dumps(self.entries, indent=2))

    def _load(self):
        if not self._backend.exists(self.path):
            return
        try:
            self.entries = json.loads(self._backend.read(self.path))
        except (json.JSONDecodeError, Exception):
            self.entries = {}
