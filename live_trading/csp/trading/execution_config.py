"""Focused config dataclasses for execution and risk management.

Extracted from StrategyConfig so both CSP and CC can use the same
stepped order executor and risk manager without depending on the
full strategy config.
"""

from dataclasses import dataclass


@dataclass
class ExecutionConfig:
    """Stepped order execution parameters."""

    # Entry
    entry_order_type: str = "stepped"       # "market" or "stepped"
    entry_start_price: str = "mid"          # "mid" or "bid"
    entry_step_pct: float = 0.25            # each step reduces by this fraction of spread
    entry_max_steps: int = 4                # max price reductions (total attempts = max_steps + 1)
    entry_step_interval: int = 3            # seconds between steps
    entry_refetch_snapshot: bool = True

    # Exit
    exit_start_price: str = "mid"           # "mid" or "ask"
    exit_step_pct: float = 0.25             # each step increases by this fraction of spread
    exit_max_steps: int = 4                 # max price increases
    exit_step_interval: int = 3             # seconds between steps
    exit_refetch_snapshot: bool = True

    @classmethod
    def from_strategy_config(cls, cfg) -> "ExecutionConfig":
        """Build ExecutionConfig from a StrategyConfig (adapter)."""
        return cls(
            entry_order_type=cfg.entry_order_type,
            entry_start_price=cfg.entry_start_price,
            entry_step_pct=cfg.entry_step_pct,
            entry_max_steps=cfg.entry_max_steps,
            entry_step_interval=cfg.entry_step_interval,
            entry_refetch_snapshot=cfg.entry_refetch_snapshot,
            exit_start_price=cfg.exit_start_price,
            exit_step_pct=cfg.exit_step_pct,
            exit_max_steps=cfg.exit_max_steps,
            exit_step_interval=cfg.exit_step_interval,
            exit_refetch_snapshot=cfg.exit_refetch_snapshot,
        )


@dataclass
class RiskConfig:
    """Risk management parameters (stops and early exit)."""

    # Stop thresholds
    delta_stop_multiplier: float = 2.0
    delta_absolute_stop: float = 0.40
    stock_drop_stop_pct: float = 0.05
    vix_spike_multiplier: float = 1.15

    # Early exit
    early_exit_buffer_pct: float = 1.00
    early_exit_return_source: str = "entry"     # "entry" or "config"
    min_daily_return: float = 0.0015

    # Toggles
    enable_delta_stop: bool = False
    enable_delta_absolute_stop: bool = True
    enable_stock_drop_stop: bool = False
    enable_vix_spike_stop: bool = True
    enable_early_exit: bool = True

    @classmethod
    def from_strategy_config(cls, cfg) -> "RiskConfig":
        """Build RiskConfig from a StrategyConfig (adapter)."""
        return cls(
            delta_stop_multiplier=cfg.delta_stop_multiplier,
            delta_absolute_stop=cfg.delta_absolute_stop,
            stock_drop_stop_pct=cfg.stock_drop_stop_pct,
            vix_spike_multiplier=cfg.vix_spike_multiplier,
            early_exit_buffer_pct=cfg.early_exit_buffer_pct,
            early_exit_return_source=cfg.early_exit_return_source,
            min_daily_return=cfg.min_daily_return,
            enable_delta_stop=cfg.enable_delta_stop,
            enable_delta_absolute_stop=cfg.enable_delta_absolute_stop,
            enable_stock_drop_stop=cfg.enable_stock_drop_stop,
            enable_vix_spike_stop=cfg.enable_vix_spike_stop,
            enable_early_exit=cfg.enable_early_exit,
        )
