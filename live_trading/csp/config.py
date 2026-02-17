"""Central configuration for the CSP strategy."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from io import StringIO


def get_sp500_tickers() -> list:
    """Fetch current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[0]

    tickers = []
    for sym in table['Symbol']:
        sym = sym.strip()
        if '.' in sym:
            continue
        tickers.append(sym)

    return sorted(tickers)


@dataclass
class StrategyConfig:
    """Central configuration for the CSP strategy. All parameters in one place."""

    # ==================== UNIVERSE & CAPITAL ====================
    ticker_universe: List[str] = field(default_factory=get_sp500_tickers)
    num_tickers: float = np.inf  # Use np.inf for "unlimited"
    starting_cash: float = 1000000

    # ==================== VIX REGIME RULES ====================
    vix_deployment_rules: Dict[Tuple[float, float], float] = field(default_factory=lambda: {
        (0, 12): 0.0,
        (12, 15): 0.2,
        (15, 18): 0.8,
        (18, 21): 0.9,
        (21, float('inf')): 1.0,
    })

    # ==================== EQUITY FILTER PARAMS ====================
    sma_periods: List[int] = field(default_factory=lambda: [8, 20, 50])
    rsi_period: int = 14
    rsi_lower: int = 30
    rsi_upper: int = 70
    bb_period: int = 20
    bb_std: float = 1.0
    sma_bb_period: int = 20
    sma_trend_lookback: int = 3
    history_days: int = 60
    max_position_pct: float = 0.10

    # ==================== CONTRACT SELECTION & SIZING ====================
    contract_rank_mode: str = "lowest_strike_price"
    max_contracts_per_ticker: int = 5

    # ==================== OPTIONS FILTER PARAMS ====================
    min_daily_return: float = 0.0015
    max_strike_sma_period: int = 20
    max_strike_mode: str = "sma"
    min_strike_pct: float = 0.50
    max_strike_pct: float = 0.90
    delta_min: float = 0.00
    delta_max: float = 0.40
    max_dte: int = 10
    min_dte: int = 1
    max_candidates_per_symbol: int = 20
    max_candidates_total: int = 1000
    min_volume: int = 0
    min_open_interest: int = 0
    max_spread_pct: float = 1.0

    # ==================== ENTRY ORDER PARAMS ====================
    entry_start_price: str = "mid"
    entry_step_interval: int = 30
    entry_step_pct: float = 0.20
    entry_max_steps: int = 4
    entry_refetch_snapshot: bool = True

    # ==================== RISK MANAGEMENT ====================
    delta_stop_multiplier: float = 2.0
    delta_absolute_stop: float = 0.40
    stock_drop_stop_pct: float = 0.05
    vix_spike_multiplier: float = 1.15

    # ==================== EXIT ORDER PARAMS ====================
    stop_loss_order_type: str = "market"
    stop_exit_start_offset_pct: float = 0.50
    stop_exit_step_pct: float = 0.20
    stop_exit_step_interval: int = 5
    stop_exit_max_steps: int = 3
    exit_start_price: str = "mid"
    exit_step_interval: int = 30
    exit_step_pct: float = 0.20
    exit_max_steps: int = 4
    exit_refetch_snapshot: bool = True
    close_before_expiry_days: int = 0
    exit_on_missing_delta: bool = False
    early_exit_return_source: str = "entry"
    early_exit_buffer_pct: float = 1.00

    # ==================== OPERATIONAL ====================
    poll_interval_seconds: int = 60
    paper_trading: bool = True
    price_lookback_days: int = 60
    liquidate_all: bool = False

    # ==================== STORAGE ====================
    storage_backend: str = "local"              # "local" or "gcs"
    gcs_bucket_name: Optional[str] = None       # Required when storage_backend="gcs"
    gcs_prefix: str = ""                        # e.g. "prod" or "paper"

    # ==================== FILTER TOGGLES ====================
    enable_sma8_check: bool = True
    enable_sma20_check: bool = True
    enable_sma50_check: bool = True
    enable_bb_upper_check: bool = False
    enable_band_check: bool = True
    enable_sma50_trend_check: bool = True
    enable_rsi_check: bool = True
    enable_position_size_check: bool = True
    enable_premium_check: bool = True
    enable_delta_check: bool = True
    enable_dte_check: bool = True
    enable_volume_check: bool = True
    enable_open_interest_check: bool = True
    enable_spread_check: bool = True
    trade_during_earnings: bool = False
    trade_during_dividends: bool = False
    trade_during_fomc: bool = False
    enable_delta_stop: bool = False
    enable_delta_absolute_stop: bool = True
    enable_stock_drop_stop: bool = False
    enable_vix_spike_stop: bool = True
    enable_early_exit: bool = True

    def get_deployment_multiplier(self, vix: float) -> float:
        """Get cash deployment multiplier based on current VIX."""
        for (lower, upper), multiplier in self.vix_deployment_rules.items():
            if lower <= vix < upper:
                return multiplier
        return 0.0

    def get_deployable_cash(self, vix: float) -> float:
        """Calculate deployable cash based on VIX regime."""
        return self.starting_cash * self.get_deployment_multiplier(vix)
