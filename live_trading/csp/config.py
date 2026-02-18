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
    sma_periods: List[int] = field(default_factory=lambda: [8, 20, 50]) # Simple Moving Average (SMA) periods
    rsi_period: int = 14 # Relative Strength Index (RSI) period
    rsi_lower: int = 30 # Relative Strength Index (RSI) lower bound
    rsi_upper: int = 70 # Relative Strength Index (RSI) upper bound
    bb_period: int = 20 # Bollinger Bands (BB) period
    bb_std: float = 1.0 # Bollinger Bands (BB) standard deviation
    sma_bb_period: int = 20 # Simple Moving Average (SMA) period for Bollinger Bands (BB)
    sma_trend_lookback: int = 3 # Days to confirm SMA(50) uptrend
    history_days: int = 60 # Days of price history to fetch
    max_position_pct: float = 0.10 # Max % of portfolio per ticker for position sizing & qty calc

    # ==================== CONTRACT SELECTION & SIZING ====================
    contract_rank_mode: str = "lowest_strike_price" # "daily_return_per_delta" | "days_since_strike" | "daily_return_on_collateral" | "lowest_strike_price" | "lowest_delta"
    universe_rank_mode: str = "lowest_delta" # Cross-ticker: "daily_return_per_delta" | "days_since_strike" | "daily_return_on_collateral" | "lowest_strike_price" | "lowest_delta"
    max_contracts_per_ticker: int = 10 # Hard cap on contracts per ticker

    # ==================== OPTIONS FILTER PARAMS ====================
    min_daily_return: float = 0.0015 # Minimum daily return (premium/dte/strike)    
    max_strike_sma_period: int = 20 # Simple Moving Average (SMA) period for strike ceiling
    max_strike_mode: str = "sma" # "sma" or "pct"
    min_strike_pct: float = 0.50 # Minimum strike percentage of stock price
    max_strike_pct: float = 0.90 # Maximum strike percentage of stock price
    delta_min: float = 0.00 # Minimum delta
    delta_max: float = 0.40 # Maximum delta
    max_dte: int = 10 # Maximum days to expiry
    min_dte: int = 1 # Minimum days to expiry
    max_candidates_per_symbol: int = 20 # Maximum candidates per symbol
    max_candidates_total: int = 1000 # Maximum candidates total
    min_volume: int = 0 # Minimum volume
    min_open_interest: int = 0 # Minimum open interest
    max_spread_pct: float = 1.0 # Maximum spread percentage

    # ==================== ENTRY ORDER PARAMS ====================
    entry_order_type: str = "stepped"  # "market" or "stepped"
    entry_start_price: str = "mid"  # "mid" or "bid" (only used when stepped)
    entry_step_interval: int = 3  # seconds between steps (only used when stepped)
    entry_step_pct: float = 0.25  # each step reduces by this fraction of the spread
    entry_max_steps: int = 4  # max price reductions (total attempts = max_steps + 1)
    entry_refetch_snapshot: bool = True

    # ==================== RISK MANAGEMENT ====================
    delta_stop_multiplier: float = 2.0 # Exit if delta >= 2x entry
    delta_absolute_stop: float = 0.40 # Exit if delta >= 0.40
    stock_drop_stop_pct: float = 0.05 # Exit if stock drops 5%
    vix_spike_multiplier: float = 1.15 # Exit if VIX >= 1.15x entry

    # ==================== EARLY EXIT ORDER PARAMS ====================
    # Used for EARLY_EXIT and EXPIRY — stepped limit with market fallback.
    # Stop-loss exits (delta, vix spike, stock drop) always use market orders.
    exit_start_price: str = "mid"           # "mid" or "ask"
    exit_step_interval: int = 3             # seconds between steps
    exit_step_pct: float = 0.25             # each step increases by this fraction of the spread
    exit_max_steps: int = 4               # max price increases
    exit_refetch_snapshot: bool = True
    close_before_expiry_days: int = 0       # close N days before expiry
    exit_on_missing_delta: bool = False      # treat missing delta as stop
    early_exit_return_source: str = "entry"  # "entry" or "config"
    early_exit_buffer_pct: float = 1.00      # exit when captured >= expected + expected * buffer_pct

    # ==================== OPERATIONAL ====================
    poll_interval_seconds: int = 60 # seconds between polling the market
    paper_trading: bool = True # Safety first!
    price_lookback_days: int = 60
    liquidate_all: bool = False # If True, cancel all orders and close all positions before scanning
    max_concurrent_options_fetches: int = 5  # Semaphore cap for parallel API calls

    # ==================== STORAGE ====================
    storage_backend: str = "local"              # "local" or "gcs"
    gcs_bucket_name: Optional[str] = None       # Required when storage_backend="gcs"
    gcs_prefix: str = ""                        # e.g. "prod" or "paper" (for GCS storage)  (not used yet)

    # ==================== FILTER TOGGLES ====================
    enable_sma8_check: bool = True # Simple Moving Average (SMA) check
    enable_sma20_check: bool = True # Simple Moving Average (SMA) check
    enable_sma50_check: bool = True # Simple Moving Average (SMA) check
    enable_bb_upper_check: bool = False # Bollinger Bands (BB) check
    enable_band_check: bool = True # Band check
    enable_sma50_trend_check: bool = True
    enable_rsi_check: bool = True # Relative Strength Index (RSI) check
    enable_position_size_check: bool = True # Position size check
    enable_premium_check: bool = True # Premium check
    enable_delta_check: bool = True # Delta check
    enable_dte_check: bool = True # Days to Expiry (DTE) check
    enable_volume_check: bool = True
    enable_open_interest_check: bool = True # Open Interest check
    enable_spread_check: bool = True # Spread check 
    trade_during_earnings: bool = False # Trade during earnings
    trade_during_dividends: bool = False # Trade during dividends
    trade_during_fomc: bool = False # Trade during FOMC
    enable_delta_stop: bool = False # Delta stop
    enable_delta_absolute_stop: bool = True # Delta absolute stop
    enable_stock_drop_stop: bool = False # Stock drop stop
    enable_vix_spike_stop: bool = True # VIX spike stop
    enable_early_exit: bool = True # Early exit

    def get_deployment_multiplier(self, vix: float) -> float:
        """Get cash deployment multiplier based on current VIX."""
        for (lower, upper), multiplier in self.vix_deployment_rules.items():
            if lower <= vix < upper:
                return multiplier
        return 0.0

    def get_deployable_cash(self, vix: float) -> float:
        """Calculate deployable cash based on VIX regime."""
        return self.starting_cash * self.get_deployment_multiplier(vix)
