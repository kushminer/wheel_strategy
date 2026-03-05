"""Configuration for the equity screener."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


_CONFIGS_DIR = Path(__file__).parent / "configs"


@dataclass
class EquityScreenerConfig:
    """Equity screener configuration. Loaded from named JSON profiles.

    Contains technical indicator params, filter toggles, and event settings.
    Position sizing is deliberately excluded — that's a strategy concern.
    """

    # ==================== IDENTITY ====================
    name: str = "default"

    # ==================== UNIVERSE SOURCE ====================
    universe_source: str = "screened_universe.json"

    # ==================== TECHNICAL INDICATOR PARAMS ====================
    sma_periods: List[int] = field(default_factory=lambda: [8, 20, 50])
    rsi_period: int = 14
    rsi_lower: int = 30
    rsi_upper: int = 70
    bb_period: int = 20
    bb_std: float = 1.0
    sma_bb_period: int = 20
    sma_trend_lookback: int = 3
    history_days: int = 60

    # ==================== FILTER TOGGLES ====================
    enable_sma8_check: bool = True
    enable_sma20_check: bool = True
    enable_sma50_check: bool = True
    enable_bb_upper_check: bool = False
    enable_band_check: bool = True
    enable_sma50_trend_check: bool = True
    enable_rsi_check: bool = True

    # ==================== PRICE FILTER ====================
    share_price_max: Optional[float] = None  # e.g. 500.0 — exclude stocks above this

    # ==================== EVENT TOGGLES ====================
    trade_during_earnings: bool = False
    trade_during_dividends: bool = False
    trade_during_fomc: bool = False
    max_dte: int = 10  # event lookahead window (days)

    @classmethod
    def from_json(cls, path: str) -> "EquityScreenerConfig":
        """Load config from a JSON file path."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_name(cls, name: str) -> "EquityScreenerConfig":
        """Load a named config profile from the configs/ directory."""
        path = _CONFIGS_DIR / f"{name}.json"
        if not path.exists():
            available = [p.stem for p in _CONFIGS_DIR.glob("*.json")]
            raise FileNotFoundError(
                f"Config '{name}' not found at {path}. "
                f"Available: {available}"
            )
        config = cls.from_json(str(path))
        config.name = name
        return config

    def to_dict(self) -> dict:
        """Serialize to dict (for JSON output metadata)."""
        return asdict(self)
