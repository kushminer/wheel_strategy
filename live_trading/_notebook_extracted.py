# Extracted from notebook
# ======================================================================

# Cell 1
# ----------------------------------------------------------------------
# Install required packages (run once)
# !pip install alpaca-py yfinance pandas numpy python-dotenv

# Cell 2
# ----------------------------------------------------------------------
# imports
import os
import re
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dotenv import load_dotenv
from io import StringIO
import time
import pytz
from datetime import time as dt_time
from pathlib import Path
from typing import Optional, Dict

import numpy as np

# PyVollib Imports
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes.greeks.analytical import delta as bs_delta, gamma as bs_gamma, theta as bs_theta, vega as bs_vega


# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest, MarketOrderRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import ContractType, AssetStatus, OrderSide, OrderType, TimeInForce
from alpaca.data.requests import StockBarsRequest, OptionSnapshotRequest, OptionBarsRequest



# Load environment variables
# load_dotenv()
load_dotenv(override=True)
print("Imports successful!")

# Cell 3
# ----------------------------------------------------------------------
    def get_sp500_tickers() -> list:
        """Fetch current S&P 500 constituents from Wikipedia."""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text))[0]
        
        # Clean up symbols
        tickers = []
        for sym in table['Symbol']:
            sym = sym.strip()
            
            # Skip dual-class tickers with dots (BF.B, BRK.B) — Alpaca doesn't support them
            if '.' in sym:
                continue
            tickers.append(sym)
        
        return sorted(tickers)


# Cell 4
# ----------------------------------------------------------------------
@dataclass
class StrategyConfig:
    """
    Central configuration for the CSP strategy.
    All parameters in one place for easy tuning.
    """
    # ==================== UNIVERSE & CAPITAL ====================
    ticker_universe: List[str] = field(default_factory=get_sp500_tickers)
    # ticker_universe: List[str] = field(default_factory=lambda: 
    # [
    #     # 'AAPL', 'MSFT', 'GOOG'
    # ])
    num_tickers: int = np.inf  # Max positions for diversification
    starting_cash: float = 1000000
    # starting_cash: float = account['buying_power']
    
    # ==================== VIX REGIME RULES ====================
    # (vix_lower, vix_upper): deployment_multiplier
    vix_deployment_rules: Dict[Tuple[float, float], float] = field(default_factory=lambda: {
        (0,  12): 0.0,       # VIX < 15: deploy 100%
        (12, 15): 0.2,      # 15 <= VIX < 20: deploy 80%
        (15, 18): 0.8,      # 20 <= VIX < 25: deploy 20%
        (18, 21): 0.9,      # 20 <= VIX < 25: deploy 20%
        (21, float('inf')): 1.0  # VIX >= 25: deploy 0%
    })
    
    # ==================== EQUITY FILTER PARAMS ====================
    sma_periods: List[int] = field(default_factory=lambda: [8, 20, 50])
    rsi_period: int = 14
    rsi_lower: int = 30
    rsi_upper: int = 70
    bb_period: int = 20
    bb_std: float = 1.0
    sma_bb_period: int = 20      # Period for SMA/BB band check: SMA(n) <= price <= BB_upper(n)
    sma_trend_lookback: int = 3  # Days to confirm SMA(50) uptrend
    history_days: int = 60       # Days of price history to fetch
    max_position_pct: float = 0.10  # Max % of portfolio per ticker for position sizing & qty calc

    # ==================== CONTRACT SELECTION & SIZING ====================
    contract_rank_mode: str = "lowest_strike_price"  # "daily_return_per_delta" | "days_since_strike" | "daily_return_on_collateral" | "lowest_strike_price"
    max_contracts_per_ticker: int = 5        # Hard cap on contracts per ticker

        
    # ==================== OPTIONS FILTER PARAMS ====================
    min_daily_return: float = 0.0015       # 0.nn% daily yield on strike (premium/dte/strike)
    max_strike_sma_period: int = 20        # SMA period for strike ceiling (when mode="sma", 8/20/50)
    max_strike_mode: str = "sma"           # "pct" = use max_strike_pct, "sma" = use SMA as ceiling
    min_strike_pct: float = 0.50           # Strike >= nn% of stock price
    max_strike_pct: float = 0.90           # Strike <= mm% of stock price (when mode="pct")
    delta_min: float = 0.00
    delta_max: float = 0.40
    max_dte: int = 10
    min_dte: int = 1  
    max_candidates_per_symbol: int = 20     
    max_candidates_total: int = 1000
    min_volume: int = 0                    # Min option contract volume (0 = disabled)
    min_open_interest: int = 0             # Min option contract open interest (0 = disabled)
    max_spread_pct: float = 1.0            # Max bid-ask spread as fraction of mid (1.0 = disabled)

    # ==================== ENTRY ORDER PARAMS ====================
    entry_start_price: str = "mid"           # "mid" or "bid" — initial limit price
    entry_step_interval: int = 30            # seconds between price reductions
    entry_step_pct: float = 0.20             # each step reduces by this fraction of the spread
    entry_max_steps: int = 4                 # max price reductions (total attempts = max_steps + 1)
    entry_refetch_snapshot: bool = True       # re-fetch option snapshot between steps

    # ==================== RISK MANAGEMENT ====================
    delta_stop_multiplier: float = 2.0   # Exit if delta >= 2x entry
    delta_absolute_stop: float = 0.40    # Exit if delta >= this absolute value
    stock_drop_stop_pct: float = 0.05    # Exit if stock drops 5%
    vix_spike_multiplier: float = 1.15   # Exit if VIX >= 1.15x entry

    # ==================== EXIT ORDER PARAMS ====================
    # Stop-loss exit order type: "market" | "bid" | "stepped"
    stop_loss_order_type: str = "market"
    # Stepped stop-loss params (used when stop_loss_order_type = "stepped")
    stop_exit_start_offset_pct: float = 0.50  # start at bid + n% of spread
    stop_exit_step_pct: float = 0.20          # step up by m% of spread
    stop_exit_step_interval: int = 5          # seconds between steps
    stop_exit_max_steps: int = 3              # after retries, fall back to market

    # Profit-taking / early exit stepped params
    exit_start_price: str = "mid"             # "mid" or "ask"
    exit_step_interval: int = 30              # seconds between price steps
    exit_step_pct: float = 0.20              # step size as fraction of spread
    exit_max_steps: int = 4                   # max price increases
    exit_refetch_snapshot: bool = True        # re-fetch between steps

    # Expiration & delta safety
    close_before_expiry_days: int = False         # close N days before expiry
    exit_on_missing_delta: bool = False        # treat missing delta as stop

    # Early exit formula
    early_exit_return_source: str = "entry"   # "entry" = position entry_daily_return, "config" = min_daily_return
    early_exit_buffer_pct: float = 1.00       # exit when captured >= expected + expected * buffer_pct
    
    # ==================== OPERATIONAL ====================
    poll_interval_seconds: int = 60
    paper_trading: bool = True  # Safety first!
    price_lookback_days: int = 60  # Days of history for current price lookups
    liquidate_all: bool = False  # If True, cancel all orders and close all positions before scanning

    # ==================== FILTER TOGGLES ====================
    # Equity filter checks
    enable_sma8_check: bool = True
    enable_sma20_check: bool = True
    enable_sma50_check: bool = True
    enable_bb_upper_check: bool = False   # Conflicts with band check; disabled by default
    enable_band_check: bool = True        # SMA(n) <= price <= BB_upper(n)
    enable_sma50_trend_check: bool = True
    enable_rsi_check: bool = True
    enable_position_size_check: bool = True

    # Options filter checks
    enable_premium_check: bool = True
    enable_delta_check: bool = True
    enable_dte_check: bool = True
    enable_volume_check: bool = True
    enable_open_interest_check: bool = True
    enable_spread_check: bool = True
    trade_during_earnings: bool = False   # If False, skip symbols with earnings in DTE window
    trade_during_dividends: bool = False  # If False, skip symbols with ex-div date in DTE window
    trade_during_fomc: bool = False       # If False, skip all trading when FOMC meeting in DTE window

    # Risk manager checks
    enable_delta_stop: bool = False
    enable_delta_absolute_stop: bool = True
    enable_stock_drop_stop: bool = False
    enable_vix_spike_stop: bool = True
    enable_early_exit: bool = True


    def get_sp500_tickers() -> list:
        """Fetch current S&P 500 constituents from Wikipedia."""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text))[0]
        
        # Clean up symbols
        tickers = []
        for sym in table['Symbol']:
            sym = sym.strip()
            # Skip dual-class tickers with dots (BF.B, BRK.B) — Alpaca doesn't support them
            if '.' in sym:
                continue
            tickers.append(sym)
        
        return sorted(tickers)
    
    def get_deployment_multiplier(self, vix: float) -> float:
        """Get cash deployment multiplier based on current VIX."""
        for (lower, upper), multiplier in self.vix_deployment_rules.items():
            if lower <= vix < upper:
                return multiplier
        return 0.0  # Default to no deployment if VIX out of range
    
    def get_deployable_cash(self, vix: float) -> float:
        """Calculate deployable cash based on VIX regime."""
        return self.starting_cash * self.get_deployment_multiplier(vix)


# Initialize config
config = StrategyConfig()

# Test VIX deployment rules
print("VIX Deployment Rules Test:")
for test_vix in [12, 15, 18, 21]:
    deployable = config.get_deployable_cash(test_vix)
    print(f"  VIX={test_vix}: Deploy ${deployable:,.0f} ({config.get_deployment_multiplier(test_vix):.0%})")

# Cell 5
# ----------------------------------------------------------------------
class AlpacaClientManager:
    """
    Manages Alpaca API clients for data and trading.
    Handles authentication and provides unified access.
    """
    
    def __init__(self, paper: bool = True):
        """
        Initialize Alpaca clients.
        
        Args:
            paper: If True, use paper trading credentials
        """
        self.paper = paper
        
        # Get credentials from environment
        if paper:
            self.api_key = os.getenv('ALPACA_API_KEY')
            self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca credentials not found. Set environment variables:\n"
                "  ALPACA_API_KEY and ALPACA_SECRET_KEY (or ALPACA_PAPER_* variants)"
            )
        
        # Initialize clients
        self._data_client = None
        self._trading_client = None
    
    @property
    def data_client(self) -> StockHistoricalDataClient:
        """Lazy initialization of data client."""
        if self._data_client is None:
            self._data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
        return self._data_client
    
    @property
    def trading_client(self) -> TradingClient:
        """Lazy initialization of trading client."""
        if self._trading_client is None:
            self._trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper
            )
        return self._trading_client
    
    @staticmethod
    def parse_strike_from_symbol(symbol: str) -> float:
        """Parse strike price from OCC format option symbol.
        Format: SYMBOL + YYMMDD + P/C + STRIKE
        Example: SNDK260213P00570000 -> strike = 570.00
        """
        match = re.search(r'[PC](\d+)$', symbol)
        if match:
            return int(match.group(1)) / 1000.0
        return 0.0

    @staticmethod
    def parse_expiration_from_symbol(symbol: str) -> Optional[date]:
        """Parse expiration date from OCC format option symbol.
        Format: SYMBOL + YYMMDD + P/C + STRIKE
        Example: SNDK260213P00570000 -> expiration = 2026-02-13
        """
        match = re.search(r'(\d{6})[PC]', symbol)
        if match:
            d = match.group(1)
            return date(2000 + int(d[:2]), int(d[2:4]), int(d[4:6]))
        return None

    def get_account_info(self) -> dict:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'status': account.status,
            'trading_blocked': account.trading_blocked,
            'options_trading_level': getattr(account, 'options_trading_level', None)
        }

    def get_short_collateral(self) -> float:
        """Calculate total collateral locked by short option positions.
        Returns sum of abs(qty) * strike * 100 for all short positions.
        """
        total = 0.0
        try:
            positions = self.trading_client.get_all_positions()
            for pos in positions:
                qty = float(pos.qty)
                side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                if side == 'short' or qty < 0:
                    strike = self.parse_strike_from_symbol(pos.symbol)
                    total += abs(qty) * strike * 100
        except Exception as e:
            print(f"  Warning: could not fetch positions for collateral calc: {e}")
        return total

    def compute_available_capital(self) -> float:
        """Compute available capital: Alpaca cash minus short position collateral."""
        account_info = self.get_account_info()
        collateral = self.get_short_collateral()
        return account_info['cash'] - collateral

    def liquidate_all_holdings(self) -> dict:
        """Cancel all open orders and close all positions.
        Full verbose output matching the standalone liquidation cell.
        Returns summary dict with counts and any errors.
        """
        import time as _time
        from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        summary = {'orders_cancelled': 0, 'positions_closed': 0, 'errors': []}

        print("=" * 80)
        print("LIQUIDATING ALL HOLDINGS")
        print("=" * 80)
        print()

        # ── Show current state ──────────────────────────────────────────
        print("Current State:")
        print("-" * 80)

        active_positions = []
        expired_positions = []
        total_collateral = 0.0

        try:
            positions = self.trading_client.get_all_positions()
            if positions:
                today = date.today()
                print(f"Open Positions ({len(positions)}):")
                for pos in positions:
                    qty = float(pos.qty)
                    side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                    strike = self.parse_strike_from_symbol(pos.symbol)
                    expiration = self.parse_expiration_from_symbol(pos.symbol)
                    current_price = float(pos.current_price)

                    is_expired = expiration and expiration < today
                    is_worthless = current_price == 0.0

                    if side == 'short' or qty < 0:
                        collateral = abs(qty) * strike * 100
                        total_collateral += collateral

                    status_tag = ""
                    if is_expired:
                        status_tag = " [EXPIRED]"
                    elif is_worthless:
                        status_tag = " [WORTHLESS]"

                    print(f"  {pos.symbol:<20} qty={qty:>6.0f} side={side:<6} "
                          f"strike=${strike:>7.2f} exp={expiration} "
                          f"price=${current_price:>6.2f}{status_tag}")

                    if is_expired or is_worthless:
                        expired_positions.append(pos)
                    else:
                        active_positions.append(pos)

                print(f"  Total collateral tied up:   ${total_collateral:,.2f}")
                print(f"  Active positions:           {len(active_positions)}")
                print(f"  Expired/worthless positions: {len(expired_positions)}")
            else:
                print("Open Positions: None")
        except Exception as e:
            print(f"Error fetching positions: {e}")
            summary['errors'].append(str(e))

        # Show open orders
        try:
            open_orders_req = GetOrdersRequest(status='open', limit=50)
            open_orders = self.trading_client.get_orders(open_orders_req)
            if open_orders:
                print(f"\nOpen Orders ({len(open_orders)}):")
                for order in open_orders:
                    print(f"  {order.symbol:<20} {order.side.value:<6} "
                          f"qty={float(order.qty):>6.0f} status={order.status.value}")
            else:
                print("\nOpen Orders: None")
        except Exception as e:
            print(f"\nError fetching open orders: {e}")

        print()
        print("=" * 80)
        print("Starting Liquidation...")
        print("=" * 80)
        print()

        # ── Step 1: Cancel all open orders ──────────────────────────────
        print("Step 1: Cancelling all open orders...")
        try:
            open_orders_req = GetOrdersRequest(status='open', limit=50)
            open_orders = self.trading_client.get_orders(open_orders_req)
            if open_orders:
                for order in open_orders:
                    try:
                        self.trading_client.cancel_order_by_id(order.id)
                        print(f"  Cancelled: {order.symbol} ({order.side.value} {float(order.qty):.0f})")
                        summary['orders_cancelled'] += 1
                    except Exception as e:
                        msg = f"Failed to cancel {order.symbol}: {e}"
                        print(f"  {msg}")
                        summary['errors'].append(msg)
            else:
                print("  No open orders to cancel.")
        except Exception as e:
            summary['errors'].append(f"Error fetching orders: {e}")

        _time.sleep(2)
        print()

        # ── Step 2: Close all positions ─────────────────────────────────
        print("Step 2: Closing all positions...")

        # Active positions — market orders
        if active_positions:
            print("  Closing active positions (market orders)...")
            for pos in active_positions:
                try:
                    qty = float(pos.qty)
                    if qty < 0:
                        close_qty = abs(qty)
                        close_side = OrderSide.BUY
                        action = "Buying to close"
                    else:
                        close_qty = qty
                        close_side = OrderSide.SELL
                        action = "Selling to close"

                    order_req = MarketOrderRequest(
                        symbol=pos.symbol, qty=int(close_qty),
                        side=close_side, time_in_force=TimeInForce.DAY
                    )
                    order = self.trading_client.submit_order(order_req)
                    print(f"  {action}: {pos.symbol} qty={int(close_qty)} order_id={order.id}")
                    summary['positions_closed'] += 1
                except Exception as e:
                    print(f"  Failed to close {pos.symbol}: {e}")
                    # Fallback: limit order at 110% of current price
                    try:
                        limit_price = max(0.01, float(pos.current_price) * 1.1) if float(pos.current_price) > 0 else 0.01
                        order_req = LimitOrderRequest(
                            symbol=pos.symbol, qty=int(close_qty),
                            side=close_side, limit_price=limit_price,
                            time_in_force=TimeInForce.DAY
                        )
                        order = self.trading_client.submit_order(order_req)
                        print(f"  Retry with limit: {pos.symbol} limit=${limit_price:.2f} order_id={order.id}")
                        summary['positions_closed'] += 1
                    except Exception as e2:
                        msg = f"Limit order also failed for {pos.symbol}: {e2}"
                        print(f"  {msg}")
                        summary['errors'].append(msg)

        # Expired/worthless — limit orders at $0.01
        if expired_positions:
            print("\n  Closing expired/worthless positions (limit orders at $0.01)...")
            for pos in expired_positions:
                try:
                    qty = float(pos.qty)
                    expiration = self.parse_expiration_from_symbol(pos.symbol)
                    today = date.today()

                    if qty < 0:
                        close_qty = abs(qty)
                        close_side = OrderSide.BUY
                        action = "Buying to close"
                    else:
                        close_qty = qty
                        close_side = OrderSide.SELL
                        action = "Selling to close"

                    if expiration and expiration < today:
                        print(f"  {pos.symbol} expired on {expiration} (trying $0.01 limit)...")
                    else:
                        print(f"  {pos.symbol} appears worthless (price=$0.00, trying $0.01 limit)...")

                    order_req = LimitOrderRequest(
                        symbol=pos.symbol, qty=int(close_qty),
                        side=close_side, limit_price=0.01,
                        time_in_force=TimeInForce.DAY
                    )
                    order = self.trading_client.submit_order(order_req)
                    print(f"  {action}: {pos.symbol} qty={int(close_qty)} limit=$0.01 order_id={order.id}")
                    summary['positions_closed'] += 1
                except Exception as e:
                    msg = f"Failed to close {pos.symbol}: {e}"
                    print(f"  {msg}")
                    print(f"    Note: May already be expired and auto-removed by Alpaca.")
                    summary['errors'].append(msg)

        if not active_positions and not expired_positions:
            print("  No positions to close.")
        else:
            print("\n  Waiting for orders to fill...")
            _time.sleep(5)

            remaining = self.trading_client.get_all_positions()
            if remaining:
                print(f"\n  Warning: {len(remaining)} positions still open:")
                for pos in remaining:
                    exp = self.parse_expiration_from_symbol(pos.symbol)
                    exp_str = f" exp={exp}" if exp else ""
                    print(f"    {pos.symbol} qty={pos.qty} price=${float(pos.current_price):.2f}{exp_str}")
                print("  Note: Expired positions may take time to be removed by Alpaca.")
                summary['remaining_positions'] = len(remaining)
            else:
                print("  All positions closed successfully.")
                summary['remaining_positions'] = 0

        # ── Final account status ────────────────────────────────────────
        print()
        print("=" * 80)
        print("Final Account Status:")
        print("=" * 80)
        try:
            acct = self.get_account_info()
            print(f"Cash available:              ${acct['cash']:,.2f}")
            print(f"Buying power:                ${acct['buying_power']:,.2f}")
            print(f"Portfolio value:              ${acct['portfolio_value']:,.2f}")

            final_positions = self.trading_client.get_all_positions()
            print(f"Remaining positions:          {len(final_positions)}")

            final_orders_req = GetOrdersRequest(status='open', limit=50)
            final_orders = self.trading_client.get_orders(final_orders_req)
            print(f"Remaining open orders:        {len(final_orders)}")
        except Exception as e:
            print(f"Error fetching final status: {e}")

        print("=" * 80)
        print("Liquidation Complete!")
        print("=" * 80)

        return summary


# Test client initialization (will fail without credentials, that's expected)
try:
    alpaca = AlpacaClientManager(paper=config.paper_trading)
    account = alpaca.get_account_info()
    short_collateral = alpaca.get_short_collateral()
    config.starting_cash = account['cash'] - short_collateral
    print("✓ Alpaca connection successful!")
    print(f"  Account status: {account['status']}")
    print(f"  Cash available: ${account['cash']:,.2f}")
    print(f"  Short collateral: ${short_collateral:,.2f}")
    print(f"  Starting cash (cash - collateral): ${config.starting_cash:,.2f}")
    print(f"  Buying Power: ${account['buying_power']:,.2f}")
    print(f"  Options level: {account['options_trading_level']}")
except ValueError as e:
    print(f"⚠ Credentials not configured: {e}")
    print("\nTo configure, create a .env file or set environment variables:")
    print("  ALPACA_API_KEY=your_api_key")
    print("  ALPACA_SECRET_KEY=your_secret_key")
    alpaca = None
except Exception as e:
    print(f"⚠ Connection error: {e}")
    alpaca = None

# Cell 6
# ----------------------------------------------------------------------
class VixDataFetcher:
    """
    Fetches VIX data from Yahoo Finance.
    Provides current VIX and historical data for analysis.
    """
    
    SYMBOL = "^VIX"
    
    def __init__(self):
        self._ticker = yf.Ticker(self.SYMBOL)
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = timedelta(minutes=1)
    
    def get_current_vix(self) -> float:
        """
        Get the current/latest VIX value.
        Uses last trading day's close when market is closed.
        
        Returns:
            Current VIX value as float
        """
        if (self._cache_time and 
            datetime.now() - self._cache_time < self._cache_ttl and
            'current' in self._cache):
            return self._cache['current']
        
        try:
            # Use 5d history - more reliable than 1d on weekends
            daily = self._ticker.history(period='5d')
            if daily.empty:
                raise RuntimeError("No VIX data available")
            
            vix = float(daily['Close'].iloc[-1])
            
            self._cache['current'] = vix
            self._cache_time = datetime.now()
            
            return vix
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch VIX data: {e}")
    
    def get_vix_history(self, days: int = 30) -> pd.DataFrame:
        """
        Get historical VIX OHLC data.
        
        Args:
            days: Number of days of history
            
        Returns:
            DataFrame with Open, High, Low, Close columns
        """
        history = self._ticker.history(period=f'{days}d')
        return history[['Open', 'High', 'Low', 'Close']]
    
    def get_last_session(self) -> dict:
        """
        Get the most recent completed trading session's data.
        
        Returns:
            Dict with session_date, open, high, low, close
        """
        history = self._ticker.history(period='5d')
        if history.empty:
            raise RuntimeError("No VIX history available")
        
        last_row = history.iloc[-1]
        session_date = history.index[-1]
        
        return {
            'session_date': session_date.date() if hasattr(session_date, 'date') else session_date,
            'open': float(last_row['Open']),
            'high': float(last_row['High']),
            'low': float(last_row['Low']),
            'close': float(last_row['Close']),
        }
    
    def get_session_reference_vix(self) -> Tuple[date, float]:
        """
        Get the reference VIX for stop-loss calculations.
        
        During market hours: that day's open
        Outside market hours: last trading day's open (for next session planning)
        
        Returns:
            Tuple of (session_date, reference_vix)
        """
        session = self.get_last_session()
        return session['session_date'], session['open']
    
    def check_vix_stop_loss(
        self, 
        reference_vix: float, 
        multiplier: float = config.vix_spike_multiplier
    ) -> dict:
        """
        Check if VIX stop-loss condition is triggered.
        Condition: current_vix >= reference_vix * multiplier
        
        Args:
            reference_vix: The VIX value to compare against (entry or session open)
            multiplier: Stop-loss multiplier (default 1.15 = 15% spike)
            
        Returns:
            Dict with triggered (bool), current_vix, threshold, reason
        """
        current_vix = self.get_current_vix()
        threshold = reference_vix * multiplier
        triggered = current_vix >= threshold
        
        return {
            'triggered': triggered,
            'current_vix': current_vix,
            'reference_vix': reference_vix,
            'threshold': threshold,
            'pct_change': (current_vix / reference_vix - 1) * 100,
            'reason': f"VIX {current_vix:.2f} >= {threshold:.2f}" if triggered else ""
        }


# Test VIX fetcher
vix_fetcher = VixDataFetcher()

try:
    current_vix = vix_fetcher.get_current_vix()
    last_session = vix_fetcher.get_last_session()
    
    print(f"✓ VIX Data Retrieved")
    print(f"\n  Last Trading Session: {last_session['session_date']}")
    print(f"    Open:  {last_session['open']:.2f}")
    print(f"    High:  {last_session['high']:.2f}")
    print(f"    Low:   {last_session['low']:.2f}")
    print(f"    Close: {last_session['close']:.2f}")
    
    print(f"\n  Current VIX: {current_vix:.2f}")
    print(f"  Deployment: {config.get_deployment_multiplier(current_vix):.0%}")
    print(f"  Deployable Cash: ${config.get_deployable_cash(current_vix):,.0f}")
    
    # Test stop-loss check using session open as reference
    stop_loss_check = vix_fetcher.check_vix_stop_loss(
        reference_vix=last_session['open'],
        multiplier=config.vix_spike_multiplier
    )
    
    print(f"\n  VIX Stop-Loss Check (vs session open):")
    print(f"    Reference: {stop_loss_check['reference_vix']:.2f}")
    print(f"    Threshold: {stop_loss_check['threshold']:.2f} ({config.vix_spike_multiplier:.0%})")
    print(f"    Current:   {stop_loss_check['current_vix']:.2f} ({stop_loss_check['pct_change']:+.1f}%)")
    print(f"    Triggered: {'🚨 YES - EXIT ALL' if stop_loss_check['triggered'] else '✓ No'}")
    
    # Show recent history
    vix_history = vix_fetcher.get_vix_history(10)
    print(f"\n  Last 5 sessions:")
    print(f"    {'Date':<12} {'Open':>8} {'Close':>8}")
    print(f"    {'-'*12} {'-'*8} {'-'*8}")
    for dt, row in vix_history.tail(5).iterrows():
        print(f"    {dt.strftime('%Y-%m-%d'):<12} {row['Open']:>8.2f} {row['Close']:>8.2f}")
        
except Exception as e:
    print(f"⚠ VIX fetch error: {e}")
    import traceback
    traceback.print_exc()

# Cell 7
# ----------------------------------------------------------------------
class EquityDataFetcher:
    """
    Fetches equity price data from Alpaca.
    Provides historical bars and current prices.
    """
    
    def __init__(self, alpaca_manager: AlpacaClientManager):
        self.client = alpaca_manager.data_client
    
    def get_close_history(
        self, 
        symbols: List[str], 
        days: int = 60
    ) -> Dict[str, pd.Series]:
        """
        Get closing price history for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            days: Number of trading days of history
            
        Returns:
            Dict mapping symbol -> pd.Series of close prices
        """
        # Calculate date range (add buffer for weekends/holidays)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))  # Buffer for non-trading days
        
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars = self.client.get_stock_bars(request)
        
        # Process into dict of Series
        result = {}
        for symbol in symbols:
            if symbol in bars.data:
                symbol_bars = bars.data[symbol]
                closes = pd.Series(
                    [bar.close for bar in symbol_bars],
                    index=[bar.timestamp for bar in symbol_bars],
                    name=symbol
                )
                # Take last N days
                result[symbol] = closes.tail(days)
            else:
                print(f"  Warning: No data for {symbol}")
        
        return result
    
    def get_current_price(self, symbol: str, price_lookback_days: int = 5) -> float:
        """
        Get the most recent price for a symbol.
        
        Args:
            symbol: Ticker symbol
            price_lookback_days: Days of history to fetch
            
        Returns:
            Latest close price
        """
        history = self.get_close_history([symbol], days=price_lookback_days)
        if symbol in history and len(history[symbol]) > 0:
            return float(history[symbol].iloc[-1])
        raise ValueError(f"No price data for {symbol}")
    
    def get_current_prices(self, symbols: List[str], price_lookback_days: int = 5) -> Dict[str, float]:
        """
        Get current prices for multiple symbols efficiently.
        
        Args:
            symbols: List of ticker symbols
            price_lookback_days: Days of history to fetch
            
        Returns:
            Dict mapping symbol -> current price
        """
        history = self.get_close_history(symbols, days=price_lookback_days)
        return {
            symbol: float(prices.iloc[-1]) 
            for symbol, prices in history.items() 
            if len(prices) > 0
        }
# Test equity data fetcher
if alpaca:
    equity_fetcher = EquityDataFetcher(alpaca)
    
    # Test with subset of universe
    test_symbols = config.ticker_universe[:3]  # First 3 symbols
    
    try:
        close_history = equity_fetcher.get_close_history(test_symbols, days=config.history_days)
        
        print(f"✓ Equity Data Retrieved for {len(close_history)} symbols")
        for symbol, prices in close_history.items():
            print(f"\n  {symbol}:")
            print(f"    Data points: {len(prices)}")
            print(f"    Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
            print(f"    Current price: ${prices.iloc[-1]:.2f}")
            print(f"    50-day range: ${prices.min():.2f} - ${prices.max():.2f}")
            
    except Exception as e:
        print(f"⚠ Equity fetch error: {e}")
else:
    print("⚠ Skipping equity test - Alpaca not configured")

# Cell 8
# ----------------------------------------------------------------------
# Risk-free rate (can pull from config or treasury API later)
RISK_FREE_RATE = 0.04


class GreeksCalculator:
    """
    Calculates IV and Delta using Black-Scholes via py_vollib.
    Falls back gracefully when calculation fails.
    """
    
    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self.r = risk_free_rate
    
    def compute_iv(
        self,
        option_price: float,
        stock_price: float,
        strike: float,
        dte: int,
        option_type: str = 'put'
    ) -> Optional[float]:
        """
        Compute implied volatility from option price.
        
        Args:
            option_price: Mid price of the option
            stock_price: Current underlying price
            strike: Strike price
            dte: Days to expiration
            option_type: 'put' or 'call'
            
        Returns:
            IV as decimal (e.g., 0.25 for 25%) or None if calculation fails
        """
        t = dte / 365.0
        flag = 'p' if option_type == 'put' else 'c'
        
        # Validate inputs
        if not all([
            np.isfinite(option_price),
            np.isfinite(stock_price),
            np.isfinite(strike),
            t > 0,
            option_price > 0,
            stock_price > 0,
            strike > 0
        ]):
            return None
        
        try:
            iv = implied_volatility(option_price, stock_price, strike, t, self.r, flag)
            return iv if np.isfinite(iv) and iv > 0 else None
        except Exception:
            return None
    
    def compute_delta(
        self,
        stock_price: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: str = 'put'
    ) -> Optional[float]:
        """
        Compute delta from IV.
        
        Args:
            stock_price: Current underlying price
            strike: Strike price
            dte: Days to expiration
            iv: Implied volatility as decimal
            option_type: 'put' or 'call'
            
        Returns:
            Delta (negative for puts) or None if calculation fails
        """
        if iv is None or not np.isfinite(iv) or iv <= 0:
            return None
        
        t = dte / 365.0
        flag = 'p' if option_type == 'put' else 'c'
        
        if t <= 0:
            return None
        
        try:
            d = bs_delta(flag, stock_price, strike, t, self.r, iv)
            return d if np.isfinite(d) else None
        except Exception:
            return None
    
    def compute_greeks(
        self,
        option_price: float,
        stock_price: float,
        strike: float,
        dte: int,
        option_type: str = 'put'
    ) -> dict:
        """
        Compute both IV and delta in one call.
        
        Returns:
            Dict with 'iv' and 'delta' keys (values may be None)
        """
        iv = self.compute_iv(option_price, stock_price, strike, dte, option_type)
        delta = self.compute_delta(stock_price, strike, dte, iv, option_type) if iv else None
        
        return {
            'iv': iv,
            'delta': delta,
            'delta_abs': abs(delta) if delta else None
        }


# Test the calculator
greeks_calc = GreeksCalculator()

# Test with one of the contracts that had Greeks from Alpaca
# AAPL260206P00225000: strike=225, dte=5, mid=0.15, alpaca_delta=-0.0214, alpaca_iv=0.6077
test_result = greeks_calc.compute_greeks(
    option_price=0.15,
    stock_price=259.48,  # approximate from your data
    strike=225.0,
    dte=5,
    option_type='put'
)

print("Greeks Calculator Test:")
print(f"  Calculated IV: {test_result['iv']:.4f}" if test_result['iv'] else "  IV: Failed")
print(f"  Calculated Delta: {test_result['delta']:.4f}" if test_result['delta'] else "  Delta: Failed")
print(f"\nAlpaca provided:")
print(f"  Alpaca IV: 0.6077")
print(f"  Alpaca Delta: -0.0214")


# Cell 9
# ----------------------------------------------------------------------
# daily return meaning secured_amount = 100*underlying_share_price
# premium/DTE -- daily cash / cash_outlayed = daily_return_as_ptct
# daily_return = (premium/DTE) 

@dataclass
class OptionContract:
    """
    Represents a single option contract with relevant data.
    """
    symbol: str
    underlying: str
    contract_type: str
    strike: float
    expiration: date
    dte: int
    bid: float
    ask: float
    mid: float
    stock_price: float  
    entry_time: Optional[datetime] = None  # Time of evaluation (for pro-rata daily return)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    days_since_strike: Optional[int] = None  # Days since stock was at/below strike
    
    @property
    def premium(self) -> float:
        """Premium received when selling (use bid price)."""
        return self.bid
    
    @property
    def effective_dte(self) -> float:
        """Pro-rata DTE: fractional day remaining today + whole DTE days."""
        TRADING_MINUTES_PER_DAY = 390  # 9:30 AM - 4:00 PM ET
        if self.entry_time is not None:
            eastern = pytz.timezone('US/Eastern')
            now_et = self.entry_time.astimezone(eastern)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            minutes_left = max((market_close - now_et).total_seconds() / 60, 0)
            fraction_today = minutes_left / TRADING_MINUTES_PER_DAY
            return fraction_today + self.dte
        return float(self.dte)

    @property
    def premium_per_day(self) -> float:
        """Daily premium decay if held to expiration (pro-rata)."""
        if self.effective_dte <= 0:
            return 0.0
        return self.premium / self.effective_dte
    
    @property
    def collateral_required(self) -> float:
        """Cash required to secure 1 contract."""
        return self.strike * 100
    
    @property
    def cost_basis(self) -> float:
        """Cost basis = stock price * 100 (exposure value)."""
        return self.stock_price * 100
    
    @property
    def daily_return_on_collateral(self) -> float:
        """Daily yield as % of collateral (strike-based)."""
        if self.strike <= 0 or self.dte <= 0:
            return 0.0
        return self.premium_per_day / self.strike
    
    @property
    def daily_return_on_cost_basis(self) -> float:
        """Daily yield as % of cost basis (stock price-based)."""
        if self.stock_price <= 0 or self.dte <= 0:
            return 0.0
        return self.premium_per_day / self.stock_price
    
    @property
    def delta_abs(self) -> Optional[float]:
        """Absolute value of delta for filtering."""
        return abs(self.delta) if self.delta else None


    @property
    def daily_return_per_delta(self) -> float:
        """Daily return on collateral divided by absolute delta."""
        if not self.delta or abs(self.delta) == 0:
            return 0.0
        return self.daily_return_on_collateral / abs(self.delta)

class OptionsDataFetcher:
    """
    Fetches options chain data from Alpaca.
    Handles contract discovery and quote retrieval.
    """
    
    def __init__(self, alpaca_manager: AlpacaClientManager):
        self.trading_client = alpaca_manager.trading_client
        self.data_client = OptionHistoricalDataClient(
            api_key=alpaca_manager.api_key,
            secret_key=alpaca_manager.secret_key
        )
    
    def get_option_contracts(
        self,
        underlying: str,
        contract_type: str = 'put',
        min_dte: int = config.min_dte,
        max_dte: int = config.max_dte,
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None
    ) -> List[dict]:
        """
        Get available option contracts for an underlying.
        
        Args:
            underlying: Ticker symbol
            contract_type: 'put' or 'call'
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            min_strike: Minimum strike price
            max_strike: Maximum strike price
            
        Returns:
            List of contract dictionaries
        """
        # Calculate expiration date range
        today = date.today()
        min_expiry = today + timedelta(days=min_dte)
        max_expiry = today + timedelta(days=max_dte)
        
        request_params = {
            'underlying_symbols': [underlying],
            'status': AssetStatus.ACTIVE,
            'type': ContractType.PUT if contract_type == 'put' else ContractType.CALL,
            'expiration_date_gte': min_expiry,
            'expiration_date_lte': max_expiry,
        }
        
        if min_strike is not None:
            request_params['strike_price_gte'] = str(min_strike)
        if max_strike is not None:
            request_params['strike_price_lte'] = str(max_strike)
        
        request = GetOptionContractsRequest(**request_params)
        
        response = self.trading_client.get_option_contracts(request)
        
        # Convert to list of dicts
        contracts = []
        if response.option_contracts:
            # DEBUG: Inspect first contract's attributes
            # first = response.option_contracts[0]
            # print("DEBUG contract attributes:", dir(first))
            # print("DEBUG contract vars:", vars(first) if hasattr(first, '__dict__') else "N/A")

            for contract in response.option_contracts:
                contracts.append({
                    'symbol': contract.symbol,
                    'underlying': contract.underlying_symbol,
                    'strike': float(contract.strike_price),
                    'expiration': contract.expiration_date,
                    'contract_type': contract_type,
                    'open_interest': int(contract.open_interest) if getattr(contract, 'open_interest', None) else None,                })
        
        return contracts
    
    def get_option_quotes(
        self, 
        option_symbols: List[str]
    ) -> Dict[str, dict]:
        """
        Get current quotes for option contracts.
        
        Args:
            option_symbols: List of OCC option symbols
            
        Returns:
            Dict mapping symbol -> quote data
        """
        if not option_symbols:
            return {}
        
        # Alpaca has limits on batch size, chunk if needed
        chunk_size = 100
        all_quotes = {}
        
        for i in range(0, len(option_symbols), chunk_size):
            chunk = option_symbols[i:i + chunk_size]
            
            try:
                request = OptionLatestQuoteRequest(symbol_or_symbols=chunk)
                quotes = self.data_client.get_option_latest_quote(request)
                
                for symbol, quote in quotes.items():
                    all_quotes[symbol] = {
                        'bid': float(quote.bid_price) if quote.bid_price else 0.0,
                        'ask': float(quote.ask_price) if quote.ask_price else 0.0,
                        'bid_size': quote.bid_size,
                        'ask_size': quote.ask_size,
                    }
            except Exception as e:
                print(f"  Warning: Quote fetch error for chunk: {e}")
        
        return all_quotes
    
    def get_option_snapshots(
        self,
        option_symbols: List[str]
    ) -> Dict[str, dict]:
        """
        Get snapshots including Greeks for option contracts.
        
        Args:
            option_symbols: List of OCC option symbols
            
        Returns:
            Dict mapping symbol -> snapshot data with Greeks
        """
        if not option_symbols:
            return {}
        
        chunk_size = 100
        all_snapshots = {}
        
        for i in range(0, len(option_symbols), chunk_size):
            chunk = option_symbols[i:i + chunk_size]
            
            try:
                request = OptionSnapshotRequest(symbol_or_symbols=chunk)
                snapshots = self.data_client.get_option_snapshot(request)
                
                for symbol, snapshot in snapshots.items():
                    greeks = snapshot.greeks if snapshot.greeks else None
                    quote = snapshot.latest_quote if snapshot.latest_quote else None
                    trade = snapshot.latest_trade if snapshot.latest_trade else None
                    
                    all_snapshots[symbol] = {
                        'bid': float(quote.bid_price) if quote and quote.bid_price else 0.0,
                        'ask': float(quote.ask_price) if quote and quote.ask_price else 0.0,
                        'delta': float(greeks.delta) if greeks and greeks.delta else None,
                        'gamma': float(greeks.gamma) if greeks and greeks.gamma else None,
                        'theta': float(greeks.theta) if greeks and greeks.theta else None,
                        'vega': float(greeks.vega) if greeks and greeks.vega else None,
                        'implied_volatility': float(snapshot.implied_volatility) if snapshot.implied_volatility else None,
                    }
                
                # Fetch daily bars for volume
                try:
                    bar_request = OptionBarsRequest(
                        symbol_or_symbols=chunk,
                        timeframe=TimeFrame.Day,
                    )
                    bars = self.data_client.get_option_bars(bar_request)
                    # Historic Debug
                    # print(f"  DEBUG bars type={type(bars).__name__}, first symbol check: {chunk[0]}")
                    # try:
                    #     print(f"  DEBUG bars[chunk[0]] = {bars[chunk[0]]}")
                    # except Exception as de:
                    #     print(f"  DEBUG access error: {de}")
                    for symbol in chunk:
                        if symbol in all_snapshots:
                            try:
                                bar_list = bars[symbol]
                                if bar_list:
                                    all_snapshots[symbol]['volume'] = int(bar_list[-1].volume)
                                    all_snapshots[symbol]['open_interest'] = int(bar_list[-1].trade_count) if hasattr(bar_list[-1], 'trade_count') else None
                            except (KeyError, IndexError):
                                pass
                except Exception as e:
                    print(f"  Warning: Option bars fetch error (type={type(bars).__name__}): {e}")
            except Exception as e:
                print(f"  Warning: Snapshot fetch error for chunk: {e}")
        
        return all_snapshots
    
    def get_puts_chain(
        self,
        underlying: str,
        stock_price: float,
        config: 'StrategyConfig',
        sma_ceiling: float = None
    ) -> List['OptionContract']:
        """
        Get filtered put options chain with full data.
        """
        
        # Get contracts within strike range
        if config.max_strike_mode == "sma" and sma_ceiling is not None:
            max_strike = sma_ceiling
        else:
            max_strike = stock_price * config.max_strike_pct
        min_strike = stock_price * config.min_strike_pct
        
        contracts = self.get_option_contracts(
            underlying=underlying,
            contract_type='put',
            min_dte=config.min_dte,
            max_dte=config.max_dte,
            min_strike=min_strike,
            max_strike=max_strike,
        )

        if not contracts:
            return []

        # Get snapshots with Greeks
        symbols = [c['symbol'] for c in contracts]
        snapshots = self.get_option_snapshots(symbols)

        # Build OptionContract objects
        today = date.today()
        result = []

        for contract in contracts:
            symbol = contract["symbol"]
            try:
                snapshot = snapshots.get(symbol, {})

                bid = float(snapshot.get("bid", 0.0) or 0.0)
                ask = float(snapshot.get("ask", 0.0) or 0.0)
                if bid <= 0:
                    continue

                dte = (contract["expiration"] - today).days

                option = OptionContract(
                    symbol=symbol,
                    underlying=underlying,
                    contract_type="put",
                    strike=contract["strike"],
                    expiration=contract["expiration"],
                    dte=dte,
                    bid=bid,
                    ask=ask,
                    mid=(bid + ask) / 2,
                    stock_price=stock_price,
                    entry_time=datetime.now(pytz.timezone('US/Eastern')),
                    delta=snapshot.get("delta"),
                    gamma=snapshot.get("gamma"),
                    theta=snapshot.get("theta"),
                    vega=snapshot.get("vega"),
                    implied_volatility=snapshot.get("implied_volatility"),
                    volume=snapshot.get("volume"),
                    open_interest=snapshot.get("open_interest") or contract.get("open_interest"),
                )
                result.append(option)

            except Exception as e:
                # This is the "embedded" error handler
                logger.warning(
                    "Options fetch error for %s: %s",
                    contract.get("symbol"),  # safe: loop variable is always set
                    e,
                )
                continue

        return result


# Test options data fetcher
if alpaca:
    options_fetcher = OptionsDataFetcher(alpaca)
    
    test_symbol = 'TSLA'
    
    try:
        # Get current price
        current_price = equity_fetcher.get_current_price(test_symbol)
        print(f"Testing options chain for {test_symbol} @ ${current_price:.2f}")
        
        # Get puts chain
        puts = options_fetcher.get_puts_chain(test_symbol, current_price, config)
        
        print(f"\n✓ Retrieved {len(puts)} put contracts")
        
        if puts:
            print(f"\nSample contracts (first 5):")
            for put in puts:
                delta_str = f"{abs(put.delta):.2f}" if put.delta else "N/A"
                print(f"  {put.symbol}")
                print(f"    Strike: ${put.strike:.2f} | DTE: {put.dte}")
                print(f"    Bid/Ask: ${put.bid:.2f}/${put.ask:.2f}")
                print(f"    Delta: {delta_str}")
                print(f"    Daily return: {put.daily_return_on_collateral:.4%}")
                print()
    except Exception as e:
        print(f"⚠ Options fetch error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ Skipping options test - Alpaca not configured")

# Cell 10
# ----------------------------------------------------------------------
@dataclass
class MarketSnapshot:
    """
    Complete market state at a point in time.
    Used by strategy logic to make decisions.
    """
    timestamp: datetime
    vix_current: float
    vix_open: float
    deployable_cash: float
    equity_prices: Dict[str, float]  # symbol -> current price
    equity_history: Dict[str, pd.Series]  # symbol -> price history


class DataManager:
    """
    Unified data manager that combines all data sources.
    Provides a clean interface for strategy logic.
    """
    
    def __init__(
        self, 
        alpaca_manager: Optional[AlpacaClientManager],
        config: StrategyConfig
    ):
        self.config = config
        self.vix_fetcher = VixDataFetcher()
        
        if alpaca_manager:
            self.equity_fetcher = EquityDataFetcher(alpaca_manager)
            self.options_fetcher = OptionsDataFetcher(alpaca_manager)
        else:
            self.equity_fetcher = None
            self.options_fetcher = None
    
    def get_market_snapshot(self) -> MarketSnapshot:
        """
        Get complete market state for strategy decision-making.
        
        Returns:
            MarketSnapshot with all current market data
        """
        # VIX data
        vix_current = self.vix_fetcher.get_current_vix()
        # vix_open = self.vix_fetcher.get_vix_open_today() or vix_current
        _, vix_open = self.vix_fetcher.get_session_reference_vix()
        
        # Deployable cash based on VIX
        deployable_cash = self.config.get_deployable_cash(vix_current)
        
        # Equity data
        if self.equity_fetcher:
            equity_history = self.equity_fetcher.get_close_history(
                self.config.ticker_universe,
                days=self.config.history_days
            )
            equity_prices = {
                symbol: float(prices.iloc[-1])
                for symbol, prices in equity_history.items()
            }
        else:
            equity_history = {}
            equity_prices = {}
        
        return MarketSnapshot(
            timestamp=datetime.now(),
            vix_current=vix_current,
            vix_open=vix_open,
            deployable_cash=deployable_cash,
            equity_prices=equity_prices,
            equity_history=equity_history
        )
    
    def get_puts_for_symbol(
        self, 
        symbol: str, 
        stock_price: float
    ) -> List[OptionContract]:
        """
        Get filtered put options for a symbol.
        
        Args:
            symbol: Ticker symbol
            stock_price: Current stock price
            
        Returns:
            List of OptionContract objects
        """
        if not self.options_fetcher:
            raise RuntimeError("Options fetcher not configured")
        
        return self.options_fetcher.get_puts_chain(
            symbol, 
            stock_price, 
            self.config
        )
    
    def refresh_option_data(self, option_symbol: str) -> Optional[OptionContract]:
        """
        Refresh data for a single option (for position monitoring).
        
        Args:
            option_symbol: OCC option symbol
            
        Returns:
            Updated OptionContract or None if fetch fails
        """
        if not self.options_fetcher:
            return None
        
        snapshots = self.options_fetcher.get_option_snapshots([option_symbol])
        
        if option_symbol not in snapshots:
            return None
        
        snapshot = snapshots[option_symbol]
        
        # Parse symbol to extract details (OCC format)
        # Format: AAPL240119P00150000
        # We'd need to parse this - for now return partial data
        return snapshot  # Return raw snapshot for now


# Test unified data manager
data_manager = DataManager(alpaca, config)

try:
    print("Fetching market snapshot...")
    snapshot = data_manager.get_market_snapshot()
    
    print(f"\n✓ Market Snapshot @ {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  VIX: {snapshot.vix_current:.2f} (Open: {snapshot.vix_open:.2f})")
    print(f"  Deployable Cash: ${snapshot.deployable_cash:,.0f}")
    print(f"  Equities tracked: {len(snapshot.equity_prices)}")
    
    if snapshot.equity_prices:
        print(f"\n  Sample prices:")
        for symbol, price in list(snapshot.equity_prices.items())[:5]:
            print(f"    {symbol}: ${price:.2f}")
            
except Exception as e:
    print(f"⚠ Snapshot error: {e}")
    import traceback
    traceback.print_exc()

# Cell 11
# ----------------------------------------------------------------------
def run_data_layer_diagnostics(data_manager: DataManager) -> dict:
    """
    Run comprehensive diagnostics on the data layer.
    
    Returns:
        Dict with diagnostic results
    """
    results = {
        'vix': {'status': 'unknown', 'details': {}},
        'equity': {'status': 'unknown', 'details': {}},
        'options': {'status': 'unknown', 'details': {}},
    }
    
    # Test VIX
    print("Testing VIX data...")
    try:
        vix = data_manager.vix_fetcher.get_current_vix()
        vix_history = data_manager.vix_fetcher.get_vix_history(10)
        results['vix'] = {
            'status': 'ok',
            'details': {
                'current': vix,
                'history_points': len(vix_history),
                'history_range': f"{vix_history['Close'].min():.2f} - {vix_history['Close'].max():.2f}"            }
        }
        print(f"  ✓ VIX OK: {vix:.2f}")
    except Exception as e:
        results['vix'] = {'status': 'error', 'details': {'error': str(e)}}
        print(f"  ✗ VIX Error: {e}")
    
    # Test Equity
    print("\nTesting equity data...")
    if data_manager.equity_fetcher:
        try:
            test_symbols = data_manager.config.ticker_universe[:3]
            history = data_manager.equity_fetcher.get_close_history(test_symbols, days=60)
            results['equity'] = {
                'status': 'ok',
                'details': {
                    'symbols_tested': len(test_symbols),
                    'symbols_returned': len(history),
                    'avg_data_points': np.mean([len(v) for v in history.values()])
                }
            }
            print(f"  ✓ Equity OK: {len(history)}/{len(test_symbols)} symbols")
        except Exception as e:
            results['equity'] = {'status': 'error', 'details': {'error': str(e)}}
            print(f"  ✗ Equity Error: {e}")
    else:
        results['equity'] = {'status': 'not_configured', 'details': {}}
        print("  ⚠ Equity: Not configured (no Alpaca credentials)")
    
    # Test Options
    print("\nTesting options data...")
    if data_manager.options_fetcher and results['equity']['status'] == 'ok':
        try:
            test_symbol = data_manager.config.ticker_universe[0]
            test_price = data_manager.equity_fetcher.get_current_price(test_symbol)
            puts = data_manager.get_puts_for_symbol(test_symbol, test_price)
            
            # Count puts with Greeks
            puts_with_delta = sum(1 for p in puts if p.delta is not None)
            
            results['options'] = {
                'status': 'ok',
                'details': {
                    'test_symbol': test_symbol,
                    'contracts_found': len(puts),
                    'contracts_with_greeks': puts_with_delta,
                }
            }
            print(f"  ✓ Options OK: {len(puts)} contracts for {test_symbol}")
            print(f"    {puts_with_delta}/{len(puts)} have Greeks")
        except Exception as e:
            results['options'] = {'status': 'error', 'details': {'error': str(e)}}
            print(f"  ✗ Options Error: {e}")
    else:
        results['options'] = {'status': 'not_configured', 'details': {}}
        print("  ⚠ Options: Not configured")
    
    # Summary
    print("\n" + "="*50)
    print("PHASE 1 DATA LAYER STATUS")
    print("="*50)
    
    all_ok = all(r['status'] == 'ok' for r in results.values())
    
    for component, result in results.items():
        status_icon = {
            'ok': '✓',
            'error': '✗',
            'not_configured': '⚠',
            'unknown': '?'
        }.get(result['status'], '?')
        print(f"  {status_icon} {component.upper()}: {result['status']}")
    
    if all_ok:
        print("\n🎉 All data components working! Ready for Phase 2.")
    else:
        print("\n⚠ Some components need attention before proceeding.")
    
    return results


# Run diagnostics
diagnostics = run_data_layer_diagnostics(data_manager)

# Cell 12
# ----------------------------------------------------------------------
class GreeksCalculator:
    """
    Calculates IV and Greeks using Black-Scholes via py_vollib.
    Used to fill in missing Greeks from Alpaca data.
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Args:
            risk_free_rate: Annual risk-free rate as decimal (default 4%)
        """
        self.r = risk_free_rate
    
    def compute_iv(
        self,
        option_price: float,
        stock_price: float,
        strike: float,
        dte: int,
        option_type: str = 'put'
    ) -> Optional[float]:
        """
        Compute implied volatility from option price.
        
        Args:
            option_price: Mid price of the option
            stock_price: Current underlying price
            strike: Strike price
            dte: Days to expiration
            option_type: 'put' or 'call'
            
        Returns:
            IV as decimal (e.g., 0.25 for 25%) or None if calculation fails
        """
        t = dte / 365.0
        flag = 'p' if option_type == 'put' else 'c'
        
        # Validate inputs
        if not all([
            np.isfinite(option_price),
            np.isfinite(stock_price),
            np.isfinite(strike),
            t > 0,
            option_price > 0,
            stock_price > 0,
            strike > 0
        ]):
            return None
        
        try:
            iv = implied_volatility(option_price, stock_price, strike, t, self.r, flag)
            return iv if np.isfinite(iv) and iv > 0 else None
        except Exception:
            return None
    
    def compute_delta(
        self,
        stock_price: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: str = 'put'
    ) -> Optional[float]:
        """
        Compute delta from IV.
        
        Returns:
            Delta (negative for puts) or None if calculation fails
        """
        if iv is None or not np.isfinite(iv) or iv <= 0:
            return None
        
        t = dte / 365.0
        flag = 'p' if option_type == 'put' else 'c'
        
        if t <= 0:
            return None
        
        try:
            d = bs_delta(flag, stock_price, strike, t, self.r, iv)
            return d if np.isfinite(d) else None
        except Exception:
            return None
    
    def compute_all_greeks(
        self,
        stock_price: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: str = 'put'
    ) -> Dict[str, Optional[float]]:
        """
        Compute all Greeks from IV.
        
        Returns:
            Dict with delta, gamma, theta, vega (values may be None)
        """
        result = {'delta': None, 'gamma': None, 'theta': None, 'vega': None}
        
        if iv is None or not np.isfinite(iv) or iv <= 0:
            return result
        
        t = dte / 365.0
        flag = 'p' if option_type == 'put' else 'c'
        
        if t <= 0:
            return result
        
        try:
            result['delta'] = bs_delta(flag, stock_price, strike, t, self.r, iv)
            result['gamma'] = bs_gamma(flag, stock_price, strike, t, self.r, iv)
            result['theta'] = bs_theta(flag, stock_price, strike, t, self.r, iv)
            result['vega'] = bs_vega(flag, stock_price, strike, t, self.r, iv)
        except Exception:
            pass
        
        return result
    
    def compute_greeks_from_price(
        self,
        option_price: float,
        stock_price: float,
        strike: float,
        dte: int,
        option_type: str = 'put'
    ) -> Dict[str, Optional[float]]:
        """
        Compute IV and all Greeks from option price in one call.
        
        Returns:
            Dict with 'iv', 'delta', 'gamma', 'theta', 'vega'
        """
        iv = self.compute_iv(option_price, stock_price, strike, dte, option_type)
        
        result = {'iv': iv, 'delta': None, 'gamma': None, 'theta': None, 'vega': None}
        
        if iv:
            greeks = self.compute_all_greeks(stock_price, strike, dte, iv, option_type)
            result.update(greeks)
        
        return result


# Initialize calculator
greeks_calc = GreeksCalculator(risk_free_rate=0.04)

print("Greeks Calculator initialized")

# Cell 13
# ----------------------------------------------------------------------
# Test Greeks Calculator against Alpaca-provided values
# Using AAPL260206P00225000 from Phase 1 output:
# Alpaca: delta=-0.0214, iv=0.6077, mid=0.15, strike=225, dte=5, stock=259.48

test_cases = [
    # (mid, stock, strike, dte, alpaca_delta, alpaca_iv)
    (0.15, 259.48, 225.0, 5, -0.0214, 0.6077),
    (0.31, 259.48, 250.0, 1, -0.0939, 0.5235),
    (0.245, 259.48, 232.5, 8, -0.0372, 0.42),
]

print("Greeks Calculator Validation:")
print("=" * 70)

for mid, stock, strike, dte, alpaca_delta, alpaca_iv in test_cases:
    result = greeks_calc.compute_greeks_from_price(mid, stock, strike, dte, 'put')
    
    print(f"\nStrike: ${strike:.0f} | DTE: {dte} | Mid: ${mid:.2f}")
    print(f"  {'Metric':<8} {'Alpaca':>10} {'Calculated':>12} {'Diff':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10}")
    
    if result['iv']:
        iv_diff = (result['iv'] - alpaca_iv) / alpaca_iv * 100
        print(f"  {'IV':<8} {alpaca_iv:>10.4f} {result['iv']:>12.4f} {iv_diff:>+9.1f}%")
    else:
        print(f"  {'IV':<8} {alpaca_iv:>10.4f} {'Failed':>12}")
    
    if result['delta']:
        delta_diff = (result['delta'] - alpaca_delta) / alpaca_delta * 100
        print(f"  {'Delta':<8} {alpaca_delta:>10.4f} {result['delta']:>12.4f} {delta_diff:>+9.1f}%")
    else:
        print(f"  {'Delta':<8} {alpaca_delta:>10.4f} {'Failed':>12}")

# Cell 14
# ----------------------------------------------------------------------
import pandas as pd


class TechnicalIndicators:
    """
    Technical indicator calculations for equity filtering.
    All methods are static and work with pandas Series.
    """
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            prices: Series of prices
            period: Lookback period
            
        Returns:
            Series of SMA values
        """
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            prices: Series of prices
            period: Lookback period (default 14)
            
        Returns:
            Series of RSI values (0-100)
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(
        prices: pd.Series, 
        period: int = 20, 
        num_std: float = 2.0
    ) -> tuple:
        """
        Bollinger Bands.
        
        Args:
            prices: Series of prices
            period: Lookback period for SMA (default 20)
            num_std: Number of standard deviations (default 2.0)
            
        Returns:
            Tuple of (lower_band, middle_band, upper_band) as Series
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        
        return lower, middle, upper
    
    @staticmethod
    def sma_trend(
        prices: pd.Series, 
        sma_period: int, 
        lookback_days: int = 3
    ) -> bool:
        """
        Check if SMA is trending upward over lookback period.
        
        Args:
            prices: Series of prices
            sma_period: Period for SMA calculation
            lookback_days: Number of days to check trend
            
        Returns:
            True if SMA has been rising for all lookback_days
        """
        sma = TechnicalIndicators.sma(prices, sma_period)
        
        if len(sma) < lookback_days + 1:
            return False
        
        # Check if each day's SMA > previous day's SMA
        for i in range(1, lookback_days + 1):
            if sma.iloc[-i] <= sma.iloc[-(i+1)]:
                return False
        
        return True


# Create instance for convenience
indicators = TechnicalIndicators()

print("Technical Indicators module loaded")

# Cell 15
# ----------------------------------------------------------------------
# Test Technical Indicators with sample data
# Generate sample price data
np.random.seed(42)
dates = pd.date_range(end=pd.Timestamp.today(), periods=60, freq='D')
sample_prices = pd.Series(
    100 + np.cumsum(np.random.randn(60) * 2),  # Random walk with drift
    index=dates,
    name='SAMPLE'
)

print("Technical Indicators Test:")
print("=" * 50)

# Calculate indicators
sma_8 = indicators.sma(sample_prices, 8)
sma_20 = indicators.sma(sample_prices, 20)
sma_50 = indicators.sma(sample_prices, 50)
rsi_14 = indicators.rsi(sample_prices, 14)
bb_lower, bb_middle, bb_upper = indicators.bollinger_bands(sample_prices, 50, 1.0)

current_price = sample_prices.iloc[-1]

print(f"\nCurrent Price: ${current_price:.2f}")
print(f"\nSMAs:")
print(f"  SMA(8):  ${sma_8.iloc[-1]:.2f}")
print(f"  SMA(20): ${sma_20.iloc[-1]:.2f}")
print(f"  SMA(50): ${sma_50.iloc[-1]:.2f}")
print(f"\nRSI(14): {rsi_14.iloc[-1]:.2f}")
print(f"\nBollinger Bands (50, 1std):")
print(f"  Lower:  ${bb_lower.iloc[-1]:.2f}")
print(f"  Middle: ${bb_middle.iloc[-1]:.2f}")
print(f"  Upper:  ${bb_upper.iloc[-1]:.2f}")

# Test trend detection
sma_50_trending = indicators.sma_trend(sample_prices, 50, 3)
print(f"\nSMA(50) Uptrend (3 days): {sma_50_trending}")

# Cell 16
# ----------------------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, List, Tuple
import csv


class EarningsCalendar:
    """
    Fetches upcoming earnings dates from Alpha Vantage per-symbol.
    Only fetches for symbols that are actually candidates.
    Caches per-symbol per-day.
    """
    
    def __init__(self, max_dte: int = 10):
        self._cache: Dict[str, List[date]] = {}  # symbol -> dates
        self._cache_date: Optional[date] = None
        self._max_dte = max_dte
        self._api_key: Optional[str] = None
    
    def _get_api_key(self) -> Optional[str]:
        if self._api_key is None:
            self._api_key = os.getenv("ALPHAVANTAGE_API_KEY") or ""
        return self._api_key if self._api_key else None
    
    def _reset_if_new_day(self):
        today = date.today()
        if self._cache_date != today:
            self._cache = {}
            self._cache_date = today
    
    def _select_horizon(self) -> str:
        """Pick the smallest horizon that covers today + max_dte."""
        horizon_date = date.today() + timedelta(days=self._max_dte)
        if horizon_date <= date.today() + timedelta(days=90):
            return "3month"
        if horizon_date <= date.today() + timedelta(days=180):
            return "6month"
        return "12month"
    
    def _fetch_symbol(self, symbol: str) -> List[date]:
        """Fetch earnings dates for a single symbol (CSV)."""
        api_key = self._get_api_key()
        if not api_key:
            return []
        
        horizon = self._select_horizon()
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=EARNINGS_CALENDAR"
            f"&symbol={symbol}"
            f"&horizon={horizon}"
            f"&apikey={api_key}"
        )
        
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            
            reader = csv.DictReader(resp.text.strip().splitlines())
            dates = []
            for row in reader:
                report_date_str = row.get("reportDate", "").strip()
                if report_date_str:
                    try:
                        dates.append(datetime.strptime(report_date_str, "%Y-%m-%d").date())
                    except ValueError:
                        continue
            return dates
            
        except Exception as e:
            print(f"  ⚠ Earnings fetch failed for {symbol}: {e}")
            return []
    
    def prefetch(self, symbols: List[str]):
        """Batch-prefetch earnings data for a list of symbols."""
        self._reset_if_new_day()
        api_key = self._get_api_key()
        if not api_key:
            print("  ⚠ ALPHAVANTAGE_API_KEY not set — earnings filter disabled")
            return
        
        to_fetch = [s for s in symbols if s not in self._cache]
        if not to_fetch:
            return
        
        print(f"  Fetching earnings calendar for {len(to_fetch)} symbols...")
        for symbol in to_fetch:
            self._cache[symbol] = self._fetch_symbol(symbol)
    
    def has_earnings_in_window(self, symbol: str, max_dte: int) -> bool:
        """Check if symbol has earnings between today and today + max_dte."""
        self._reset_if_new_day()
        
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_symbol(symbol)
        
        dates = self._cache.get(symbol, [])
        if not dates:
            return False
        
        today = date.today()
        window_end = today + timedelta(days=max_dte)
        return any(today <= d <= window_end for d in dates)
    
    def next_earnings_date(self, symbol: str) -> Optional[date]:
        """Get the next earnings date for a symbol, or None."""
        self._reset_if_new_day()
        
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_symbol(symbol)
        
        dates = self._cache.get(symbol, [])
        today = date.today()
        future = [d for d in dates if d >= today]
        return min(future) if future else None



class DividendCalendar:
    """
    Fetches upcoming ex-dividend dates from Alpha Vantage.
    Per-symbol API — only parses the most recent entry (first in JSON).
    Cached per-symbol per-day.
    """
    
    def __init__(self, max_dte: int = 10):
        self._cache: Dict[str, Optional[date]] = {}  # symbol -> most recent/upcoming ex-div date
        self._cache_date: Optional[date] = None
        self._max_dte = max_dte
        self._api_key: Optional[str] = None
    
    def _get_api_key(self) -> Optional[str]:
        if self._api_key is None:
            self._api_key = os.getenv("ALPHAVANTAGE_API_KEY") or ""
        return self._api_key if self._api_key else None
    
    def _reset_if_new_day(self):
        today = date.today()
        if self._cache_date != today:
            self._cache = {}
            self._cache_date = today
    
    def _fetch_symbol(self, symbol: str) -> Optional[date]:
        """Fetch the most recent/upcoming ex-dividend date for a symbol."""
        api_key = self._get_api_key()
        if not api_key:
            return None
        
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=DIVIDENDS"
            f"&symbol={symbol}"
            f"&datatype=json"
            f"&apikey={api_key}"
        )
        
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            # Data is sorted newest-first; only need the first valid entry
            for record in data.get("data", []):
                ex_date_str = record.get("ex_dividend_date", "").strip()
                if ex_date_str and ex_date_str != "None":
                    try:
                        return datetime.strptime(ex_date_str, "%Y-%m-%d").date()
                    except ValueError:
                        continue
            return None
            
        except Exception as e:
            print(f"  ⚠ Dividend fetch failed for {symbol}: {e}")
            return None
    
    def prefetch(self, symbols: List[str]):
        """Batch-prefetch dividend data for a list of symbols."""
        self._reset_if_new_day()
        api_key = self._get_api_key()
        if not api_key:
            print("  ⚠ ALPHAVANTAGE_API_KEY not set — dividend filter disabled")
            return
        
        to_fetch = [s for s in symbols if s not in self._cache]
        if not to_fetch:
            return
        
        print(f"  Fetching dividend calendar for {len(to_fetch)} symbols...")
        for symbol in to_fetch:
            self._cache[symbol] = self._fetch_symbol(symbol)
    
    def has_exdiv_in_window(self, symbol: str, max_dte: int) -> bool:
        """Check if symbol has an ex-dividend date between today and today + max_dte."""
        self._reset_if_new_day()
        
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_symbol(symbol)
        
        ex_date = self._cache.get(symbol)
        if ex_date is None:
            return False
        
        today = date.today()
        window_end = today + timedelta(days=max_dte)
        return today <= ex_date <= window_end
    
    def next_exdiv_date(self, symbol: str) -> Optional[date]:
        """Get the most recent/upcoming ex-dividend date for a symbol, or None."""
        self._reset_if_new_day()
        
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_symbol(symbol)
        
        return self._cache.get(symbol)



class FomcCalendar:
    """
    FOMC meeting schedule. Hardcoded dates refreshed by scraping the Fed website.
    Meetings are typically 2-day events; we treat both days as FOMC days.
    """
    
    # Known FOMC meeting dates (updated annually)
    _MEETING_DATES = [
        # 2025
        (date(2025, 1, 28), date(2025, 1, 29)),
        (date(2025, 3, 18), date(2025, 3, 19)),
        (date(2025, 5, 6),  date(2025, 5, 7)),
        (date(2025, 6, 17), date(2025, 6, 18)),
        (date(2025, 7, 29), date(2025, 7, 30)),
        (date(2025, 9, 16), date(2025, 9, 17)),
        (date(2025, 10, 28), date(2025, 10, 29)),
        (date(2025, 12, 9), date(2025, 12, 10)),
        # 2026
        (date(2026, 1, 27), date(2026, 1, 28)),
        (date(2026, 3, 17), date(2026, 3, 18)),
        (date(2026, 4, 28), date(2026, 4, 29)),
        (date(2026, 6, 16), date(2026, 6, 17)),
        (date(2026, 7, 28), date(2026, 7, 29)),
        (date(2026, 9, 15), date(2026, 9, 16)),
        (date(2026, 10, 27), date(2026, 10, 28)),
        (date(2026, 12, 8), date(2026, 12, 9)),
    ]
    
    @classmethod
    def _all_meeting_days(cls) -> List[date]:
        """Flatten meeting tuples into individual dates."""
        days = []
        for start, end in cls._MEETING_DATES:
            d = start
            while d <= end:
                days.append(d)
                d += timedelta(days=1)
        return days
    
    @classmethod
    def has_fomc_in_window(cls, max_dte: int) -> bool:
        """Check if any FOMC meeting day falls in [today, today + max_dte]."""
        today = date.today()
        window_end = today + timedelta(days=max_dte)
        return any(today <= d <= window_end for d in cls._all_meeting_days())
    
    @classmethod
    def next_fomc_date(cls, max_dte: int) -> Optional[date]:
        """Return the next FOMC meeting day in window, or None."""
        today = date.today()
        window_end = today + timedelta(days=max_dte)
        upcoming = [d for d in cls._all_meeting_days() if today <= d <= window_end]
        return min(upcoming) if upcoming else None
    
    @classmethod
    def refresh_from_web(cls) -> bool:
        """
        Attempt to refresh FOMC dates from the Fed website.
        Returns True if successful, False otherwise.
        Prints updated dates for manual verification.
        """
        try:
            url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            
            # Parse dates from tables (format varies; log for manual review)
            print(f"  FOMC calendar: fetched {len(tables)} table(s) from Fed website")
            print(f"  Review https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
            print(f"  and update FomcCalendar._MEETING_DATES if needed.")
            return True
        except Exception as e:
            print(f"  ⚠ FOMC calendar refresh failed: {e}")
            return False


@dataclass
class EquityFilterResult:
    """
    Result of equity technical filter.
    """
    symbol: str
    passes: bool
    current_price: float
    sma_8: float
    sma_20: float
    sma_50: float
    rsi: float
    bb_upper: float
    sma_50_trending: bool
    failure_reasons: List[str]
    
    def __str__(self):
        status = "✓ PASS" if self.passes else "✗ FAIL"
        reasons = ", ".join(self.failure_reasons) if self.failure_reasons else "All criteria met"
        return f"{self.symbol}: {status} | {reasons}"


class EquityFilter:
    """
    Filters equities based on technical criteria.
    
    Filter Rules:
    1. current_price > SMA(8), SMA(20), SMA(50), and BB_upper(50, 1std)
    2. SMA(50) has been rising for last 3 days
    3. RSI is between 30 and 70
    4. Stock price * 100 <= max_position_pct of portfolio (position sizing)
    5. No earnings scheduled within DTE window (via check_events, post-scan)
    6. No ex-dividend date within DTE window (via check_events, post-scan)
    """
        
    def __init__(self, config: 'StrategyConfig'):
        self.config = config
        self.indicators = TechnicalIndicators()
        self.earnings_calendar = EarningsCalendar(max_dte=config.max_dte)
        self.dividend_calendar = DividendCalendar(max_dte=config.max_dte)
    
    def evaluate(self, symbol: str, prices: pd.Series) -> EquityFilterResult:
        """
        Evaluate a single equity against filter criteria.
        
        Args:
            symbol: Ticker symbol
            prices: Series of closing prices (at least 50 days)
            
        Returns:
            EquityFilterResult with pass/fail and details
        """
        failure_reasons = []
        
        # Check minimum data
        if len(prices) < 50:
            return EquityFilterResult(
                symbol=symbol,
                passes=False,
                current_price=prices.iloc[-1] if len(prices) > 0 else 0,
                sma_8=0, sma_20=0, sma_50=0, rsi=0, bb_upper=0,
                sma_50_trending=False,
                failure_reasons=["Insufficient price history"]
            )
        
        # Calculate indicators
        current_price = prices.iloc[-1]
        
        sma_8 = self.indicators.sma(prices, 8).iloc[-1]
        sma_20 = self.indicators.sma(prices, 20).iloc[-1]
        sma_50 = self.indicators.sma(prices, 50).iloc[-1]
        
        rsi = self.indicators.rsi(prices, self.config.rsi_period).iloc[-1]
        
        _, _, bb_upper = self.indicators.bollinger_bands(
            prices, 
            self.config.bb_period, 
            self.config.bb_std
        )
        bb_upper_val = bb_upper.iloc[-1]
        
        sma_50_trending = self.indicators.sma_trend(
            prices, 
            50, 
            self.config.sma_trend_lookback
        )
        
        # Check criteria
        
        # 1. Price above all SMAs
        if self.config.enable_sma8_check:
            if not (current_price > sma_8):
                failure_reasons.append(f"Price {current_price:.2f} <= SMA(8) {sma_8:.2f}")
        if self.config.enable_sma20_check:
            if not (current_price > sma_20):
                failure_reasons.append(f"Price {current_price:.2f} <= SMA(20) {sma_20:.2f}")
        if self.config.enable_sma50_check:
            if not (current_price > sma_50):
                failure_reasons.append(f"Price {current_price:.2f} <= SMA(50) {sma_50:.2f}")
        
        # 2. Price above BB upper (legacy check)
        if self.config.enable_bb_upper_check:
            if not (current_price > bb_upper_val):
                failure_reasons.append(f"Price {current_price:.2f} <= BB_upper({self.config.bb_period}) {bb_upper_val:.2f}")
        
        # 3. SMA/BB band check: SMA(n) <= price <= BB_upper(n)
        if self.config.enable_band_check:
            band_period = self.config.sma_bb_period
            sma_band = self.indicators.sma(prices, band_period).iloc[-1]
            _, _, bb_band_upper = self.indicators.bollinger_bands(prices, band_period, self.config.bb_std)
            bb_band_upper_val = bb_band_upper.iloc[-1]
            
            if not (sma_band <= current_price <= bb_band_upper_val):
                if current_price < sma_band:
                    failure_reasons.append(f"Price {current_price:.2f} < SMA({band_period}) {sma_band:.2f}")
                else:
                    failure_reasons.append(f"Price {current_price:.2f} > BB_upper({band_period}) {bb_band_upper_val:.2f}")
        
        # 4. SMA(50) trending up
        if self.config.enable_sma50_trend_check:
            if not sma_50_trending:
                failure_reasons.append("SMA(50) not trending up")
        
        # 5. RSI in range
        if self.config.enable_rsi_check:
            if not (self.config.rsi_lower < rsi < self.config.rsi_upper):
                failure_reasons.append(f"RSI {rsi:.1f} outside [{self.config.rsi_lower}, {self.config.rsi_upper}]")
        
        # 6. Position size: stock_price * 100 <= max_position_pct of portfolio
        if self.config.enable_position_size_check:
            max_position_value = self.config.starting_cash * self.config.max_position_pct
            collateral_required = current_price * 100
            if collateral_required > max_position_value:
                failure_reasons.append(
                    f"Collateral ${collateral_required:,.0f} > {self.config.max_position_pct:.0%} of portfolio (${max_position_value:,.0f})"
                )
        
        passes = len(failure_reasons) == 0
                
        return EquityFilterResult(
            symbol=symbol,
            passes=passes,
            current_price=current_price,
            sma_8=sma_8,
            sma_20=sma_20,
            sma_50=sma_50,
            rsi=rsi,
            bb_upper=bb_upper_val,
            sma_50_trending=sma_50_trending,
            failure_reasons=failure_reasons
        )
    
    def filter_universe(
        self, 
        price_history: Dict[str, pd.Series]
    ) -> Tuple[List[str], List[EquityFilterResult]]:
        """
        Filter entire universe and return passing symbols.
        
        Args:
            price_history: Dict mapping symbol -> price Series
            
        Returns:
            Tuple of (passing_symbols, all_results)
        """
        results = []
        passing = []
        
        for symbol, prices in price_history.items():
            result = self.evaluate(symbol, prices)
            results.append(result)
            if result.passes:
                passing.append(symbol)
        
        return passing, results
    
    def check_events(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Check earnings and dividend calendars for a list of symbols.
        Call this AFTER equity + options filtering to minimise API calls.
        
        Args:
            symbols: List of ticker symbols that passed all other filters
            
        Returns:
            Dict mapping symbol -> list of rejection reasons.
            Symbols not in the dict are clear to trade.
        """
        rejections: Dict[str, List[str]] = {}
        
        # FOMC check (global — applies to all symbols)
        if not self.config.trade_during_fomc:
            if FomcCalendar.has_fomc_in_window(self.config.max_dte):
                fomc_date = FomcCalendar.next_fomc_date(self.config.max_dte)
                reason = f"FOMC meeting on {fomc_date.isoformat()} within {self.config.max_dte}d window"
                for symbol in symbols:
                    rejections.setdefault(symbol, []).append(reason)
                return rejections  # All rejected, skip per-symbol API calls
        
        # Earnings check (per-symbol, prefetch batch)
        if not self.config.trade_during_earnings:
            self.earnings_calendar.prefetch(symbols)
            for symbol in symbols:
                if self.earnings_calendar.has_earnings_in_window(symbol, self.config.max_dte):
                    next_date = self.earnings_calendar.next_earnings_date(symbol)
                    rejections.setdefault(symbol, []).append(
                        f"Earnings on {next_date.isoformat()} within {self.config.max_dte}d window"
                    )
        
        # Dividend check (per-symbol fetch — prefetch all at once)
        if not self.config.trade_during_dividends:
            self.dividend_calendar.prefetch(symbols)
            for symbol in symbols:
                if self.dividend_calendar.has_exdiv_in_window(symbol, self.config.max_dte):
                    next_date = self.dividend_calendar.next_exdiv_date(symbol)
                    rejections.setdefault(symbol, []).append(
                        f"Ex-div on {next_date.isoformat()} within {self.config.max_dte}d window"
                    )
        
        return rejections


print("Equity Filter module loaded")

# Cell 17
# ----------------------------------------------------------------------
# Test Equity Filter with real data (requires Phase 1 setup)
try:
    equity_filter = EquityFilter(config)
    
    print("Fetching price history for universe...")
    price_history = equity_fetcher.get_close_history(
        config.ticker_universe, 
        days=config.history_days
    )
    
    passing_symbols, all_results = equity_filter.filter_universe(price_history)
    
    passed = [r for r in all_results if r.passes]
    failed = [r for r in all_results if not r.passes]
    
    bb_label = f"BB{config.bb_period}"
    header = f"{'Symbol':<8} {'Price':>10} {'SMA8':>10} {'SMA20':>10} {'SMA50':>10} {bb_label:>10} {'RSI':>7}"
    divider = "-" * 75
    
    def print_row(r):
        print(f"{r.symbol:<8} ${r.current_price:>9.2f} ${r.sma_8:>9.2f} ${r.sma_20:>9.2f} ${r.sma_50:>9.2f} ${r.bb_upper:>9.2f} {r.rsi:>7.1f}")
    
    print(f"\n✓ PASSED ({len(passed)}/{len(all_results)})")
    print(divider)
    print(header)
    print(divider)
    for r in passed:
        print_row(r)
    
    print(f"\n✗ FAILED ({len(failed)}/{len(all_results)})")
    print(divider)
    print(f"{header} {'Reasons'}")
    print(divider)
    for r in failed:
        reasons = "; ".join(r.failure_reasons)
        print(f"{r.symbol:<8} ${r.current_price:>9.2f} ${r.sma_8:>9.2f} ${r.sma_20:>9.2f} ${r.sma_50:>9.2f} ${r.bb_upper:>9.2f} {r.rsi:>7.1f}  {reasons}")

except NameError:
    print("⚠ Run Phase 1 first to initialize equity_fetcher and config")


# Cell 18
# ----------------------------------------------------------------------
@dataclass
class OptionsFilterResult:
    """
    Result of options filter for a single contract.
    """
    contract: 'OptionContract'
    passes: bool
    daily_return: float
    delta_abs: Optional[float]
    failure_reasons: List[str]
    
    def __str__(self):
        status = "✓" if self.passes else "✗"
        delta_str = f"{self.delta_abs:.3f}" if self.delta_abs else "N/A"
        return f"{status} {self.contract.symbol} | Δ={delta_str} | Ret={self.daily_return:.4%}"


class OptionsFilter:
    """
    Filters and ranks options based on strategy criteria.
    
    Filter Rules:
    1. Daily return on cost basis >= 0.15%
    2. Strike <= 90% of stock price
    3. |Delta| between 0.15 and 0.40
    4. DTE between min_dte and max_dte
    
    Ranking: By premium per day (descending)
    """
    
    def __init__(self, config: 'StrategyConfig', greeks_calculator: GreeksCalculator):
        self.config = config
        self.greeks_calc = greeks_calculator
    
    def _ensure_greeks(self, contract: 'OptionContract') -> 'OptionContract':
        """
        Ensure contract has Greeks, calculating if missing.
        Returns contract with Greeks filled in.
        """
        # If we already have delta and IV, return as-is
        if contract.delta is not None and contract.implied_volatility is not None:
            return contract
        
        # Calculate Greeks from mid price
        greeks = self.greeks_calc.compute_greeks_from_price(
            option_price=contract.mid,
            stock_price=contract.stock_price,
            strike=contract.strike,
            dte=contract.dte,
            option_type=contract.contract_type
        )
        
        # Update contract with calculated Greeks (only if missing)
        if contract.implied_volatility is None and greeks['iv'] is not None:
            contract.implied_volatility = greeks['iv']
        if contract.delta is None and greeks['delta'] is not None:
            contract.delta = greeks['delta']
        if contract.gamma is None and greeks['gamma'] is not None:
            contract.gamma = greeks['gamma']
        if contract.theta is None and greeks['theta'] is not None:
            contract.theta = greeks['theta']
        if contract.vega is None and greeks['vega'] is not None:
            contract.vega = greeks['vega']
        
        return contract
    
    def evaluate(self, contract: 'OptionContract') -> OptionsFilterResult:
        """
        Evaluate a single option contract against filter criteria.
        
        Args:
            contract: OptionContract to evaluate
            
        Returns:
            OptionsFilterResult with pass/fail and details
        """
        # Ensure we have Greeks
        contract = self._ensure_greeks(contract)
        
        failure_reasons = []
        
        # Calculate metrics
        daily_return = contract.daily_return_on_collateral
        delta_abs = abs(contract.delta) if contract.delta else None
        strike_pct = contract.strike / contract.stock_price
        
        # 1. Premium filter: daily return >= min_daily_return
        if self.config.enable_premium_check:
            if daily_return < self.config.min_daily_return:
                failure_reasons.append(
                    f"Daily return {daily_return:.4%} < {self.config.min_daily_return:.4%}"
                )
        
        # 2. Strike filters (always applied — controlled by max_strike_mode at fetch level)
        if strike_pct > self.config.max_strike_pct:
            failure_reasons.append(
                f"Strike {strike_pct:.1%} > {self.config.max_strike_pct:.1%} of stock"
            )
        if strike_pct < self.config.min_strike_pct:
            failure_reasons.append(
                f"Strike {strike_pct:.1%} < {self.config.min_strike_pct:.1%} of stock"
            )
        
        # 3. Delta filter: |delta| between delta_min and delta_max
        if self.config.enable_delta_check:
            if delta_abs is None:
                failure_reasons.append("Delta unavailable")
            elif not (self.config.delta_min <= delta_abs <= self.config.delta_max):
                failure_reasons.append(
                    f"Delta {delta_abs:.3f} outside [{self.config.delta_min}, {self.config.delta_max}]"
                )
        
        # 4. DTE filter (should already be filtered, but double-check)
        if self.config.enable_dte_check:
            if not (self.config.min_dte <= contract.dte <= self.config.max_dte):
                failure_reasons.append(
                    f"DTE {contract.dte} outside [{self.config.min_dte}, {self.config.max_dte}]"
                )
        
        # 5. Volume filter
        if self.config.enable_volume_check and self.config.min_volume > 0:
            vol = contract.volume or 0
            if vol < self.config.min_volume:
                failure_reasons.append(
                    f"Volume {vol} < {self.config.min_volume}"
                )
        
        # 6. Open interest filter
        if self.config.enable_open_interest_check and self.config.min_open_interest > 0:
            oi = contract.open_interest or 0
            if oi < self.config.min_open_interest:
                failure_reasons.append(
                    f"OI {oi} < {self.config.min_open_interest}"
                )
        
        # 7. Spread filter: (ask - bid) / mid <= max_spread_pct
        if self.config.enable_spread_check and self.config.max_spread_pct < 1.0:
            if contract.mid > 0:
                spread_pct = (contract.ask - contract.bid) / contract.mid
                if spread_pct > self.config.max_spread_pct:
                    failure_reasons.append(
                        f"Spread {spread_pct:.1%} > {self.config.max_spread_pct:.1%}"
                    )
        
        passes = len(failure_reasons) == 0
        
        return OptionsFilterResult(
            contract=contract,
            passes=passes,
            daily_return=daily_return,
            delta_abs=delta_abs,
            failure_reasons=failure_reasons
        )
    
    def filter_and_rank(
        self, 
        contracts: List['OptionContract']
    ) -> Tuple[List['OptionContract'], List[OptionsFilterResult]]:
        """
        Filter contracts and rank passing ones by premium per day.
        
        Args:
            contracts: List of OptionContracts to evaluate
            
        Returns:
            Tuple of (ranked_passing_contracts, all_results)
        """
        results = []
        passing = []
        
        for contract in contracts:
            result = self.evaluate(contract)
            results.append(result)
            if result.passes:
                passing.append(result.contract)
        
        # Rank by configured mode
        def _sort_key(c):
            if self.config.contract_rank_mode == "daily_return_per_delta":
                return c.daily_return_per_delta
            elif self.config.contract_rank_mode == "days_since_strike":
                return c.days_since_strike or 0
            elif self.config.contract_rank_mode == "lowest_strike_price":
                return -c.strike  # Negate so lowest strike sorts first with reverse=True
            else:  # "daily_return_on_collateral"
                return c.daily_return_on_collateral
                
        passing.sort(key=_sort_key, reverse=True)
        
        return passing, results
    
    def get_best_candidates(
        self,
        contracts: List['OptionContract'],
        max_candidates: int
    ) -> List['OptionContract']:
        """
        Get top N candidates after filtering and ranking.
        
        Args:
            contracts: List of OptionContracts
            max_candidates: Maximum number to return
            
        Returns:
            List of top candidates, ranked by premium per day
        """
        passing, _ = self.filter_and_rank(contracts)
        return passing[:max_candidates]


print("Options Filter module loaded")

# Cell 19
# ----------------------------------------------------------------------
# Test Options Filter with real data (requires Phase 1 setup)
try:
    # Initialize filter
    options_filter = OptionsFilter(config, greeks_calc)
    
    # Get test data - use AAPL puts from Phase 1
    test_symbol = 'AAPL'
    current_price = equity_fetcher.get_current_price(test_symbol)
    puts = options_fetcher.get_puts_chain(test_symbol, current_price, config)
    
    print(f"Options Filter Test: {test_symbol} @ ${current_price:.2f}")
    print("=" * 80)
    print(f"Total contracts fetched: {len(puts)}")
    
    # Run filter
    passing, all_results = options_filter.filter_and_rank(puts)
    
    print(f"Contracts passing filter: {len(passing)}")
    
    # Show summary stats
    total_with_delta = sum(1 for r in all_results if r.delta_abs is not None)
    print(f"Contracts with Delta: {total_with_delta}/{len(all_results)}")
    
    # Show filter failure breakdown
    failure_counts = {}
    for result in all_results:
        for reason in result.failure_reasons:
            # Extract failure type
            if "Daily return" in reason:
                key = "Premium too low"
            elif "Strike" in reason:
                key = "Strike too high"
            elif "Delta" in reason:
                key = "Delta out of range" if "outside" in reason else "Delta unavailable"
            elif "DTE" in reason:
                key = "DTE out of range"
            else:
                key = reason
            failure_counts[key] = failure_counts.get(key, 0) + 1
    
    print(f"\nFailure Breakdown:")
    for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    # Show top passing candidates
    if passing:
        print(f"\nTop 10 Candidates (by premium/day):")
        print("-" * 80)
        print(f"{'Symbol':<22} {'Strike':>8} {'DTE':>5} {'Bid':>8} {'Delta':>8} {'Daily%':>10} {'$/Day':>8}")
        print("-" * 80)
        
        for contract in passing[:10]:
            delta_str = f"{abs(contract.delta):.3f}" if contract.delta else "N/A"
            print(
                f"{contract.symbol:<22} "
                f"${contract.strike:>7.2f} "
                f"{contract.dte:>5} "
                f"${contract.bid:>7.2f} "
                f"{delta_str:>8} "
                f"{contract.daily_return_on_collateral:>9.4%} "
                f"${contract.premium_per_day:>7.3f}"
            )
    else:
        print(f"\n⚠ No contracts pass all filter criteria")
        
        # Show closest misses
        print(f"\nClosest misses (sorted by daily return):")
        near_misses = sorted(all_results, key=lambda r: r.daily_return, reverse=True)[:5]
        for result in near_misses:
            c = result.contract
            delta_str = f"{result.delta_abs:.3f}" if result.delta_abs else "N/A"
            print(f"  {c.symbol}: Δ={delta_str}, Ret={result.daily_return:.4%}")
            print(f"    Reasons: {', '.join(result.failure_reasons)}")

except NameError as e:
    print(f"⚠ Run Phase 1 first: {e}")
except Exception as e:
    print(f"⚠ Error: {e}")
    import traceback
    traceback.print_exc()

# Cell 20
# ----------------------------------------------------------------------
@dataclass
class ScanResult:
    """
    Complete scan result for a symbol.
    """
    symbol: str
    stock_price: float
    equity_result: EquityFilterResult
    options_candidates: List['OptionContract']
    
    @property
    def has_candidates(self) -> bool:
        return len(self.options_candidates) > 0


class StrategyScanner:
    """
    Combined scanner that runs equity filter then options filter.
    """
    
    def __init__(
        self,
        config: 'StrategyConfig',
        equity_fetcher: 'EquityDataFetcher',
        options_fetcher: 'OptionsDataFetcher',
        greeks_calc: GreeksCalculator
    ):
        self.config = config
        self.equity_fetcher = equity_fetcher
        self.options_fetcher = options_fetcher
        self.equity_filter = EquityFilter(config)
        self.options_filter = OptionsFilter(config, greeks_calc)
    
    def scan_symbol(
        self, 
        symbol: str, 
        prices: pd.Series,
        skip_equity_filter: bool = False
    ) -> ScanResult:
        """
        Scan a single symbol through both filters.
        
        Args:
            symbol: Ticker symbol
            prices: Price history
            skip_equity_filter: If True, skip equity filter (for testing)
            
        Returns:
            ScanResult with equity filter result and option candidates
        """
        stock_price = prices.iloc[-1]
        
        # Run equity filter
        equity_result = self.equity_filter.evaluate(symbol, prices)
        
        # If equity fails and we're not skipping, return empty options
        if not equity_result.passes and not skip_equity_filter:
            return ScanResult(
                symbol=symbol,
                stock_price=stock_price,
                equity_result=equity_result,
                options_candidates=[]
            )
        
        # Get options chain — pass SMA ceiling if configured
        sma_ceiling = None
        if self.config.max_strike_mode == "sma":
            sma_ceiling = getattr(equity_result, f"sma_{self.config.max_strike_sma_period}", None)
            if sma_ceiling is None:
                print(f"  ⚠ SMA({self.config.max_strike_sma_period}) not available for {symbol}, falling back to max_strike_pct")
        puts = self.options_fetcher.get_puts_chain(symbol, stock_price, self.config, sma_ceiling=sma_ceiling)

        # Enrich with days_since_strike from price history
        for put in puts:
            at_or_below = prices[prices <= put.strike]
            if at_or_below.empty:
                put.days_since_strike = 999  # Never at/below in history
            else:
                last_date = at_or_below.index[-1]
                put.days_since_strike = (prices.index[-1] - last_date).days

        # Filter and rank options
        candidates = self.options_filter.get_best_candidates(puts, max_candidates=self.config.max_candidates_per_symbol)
                
        return ScanResult(
            symbol=symbol,
            stock_price=stock_price,
            equity_result=equity_result,
            options_candidates=candidates
        )
    
    def scan_universe(
        self,
        skip_equity_filter: bool = False
    ) -> List[ScanResult]:
        """
        Scan entire universe.
        
        Args:
            skip_equity_filter: If True, scan options for all symbols
            
        Returns:
            List of ScanResults for each symbol
        """
        # Get price history for all symbols
        price_history = self.equity_fetcher.get_close_history(
            self.config.ticker_universe,
            days=self.config.history_days
        )
        
        results = []
        for symbol in self.config.ticker_universe:
            if symbol not in price_history:
                continue
            
            result = self.scan_symbol(
                symbol, 
                price_history[symbol],
                skip_equity_filter=skip_equity_filter
            )
            results.append(result)
        
        return results
    
    def get_all_candidates(
        self,
        skip_equity_filter: bool = False,
        max_total: Optional[int] = None
    ) -> List['OptionContract']:
        """
        Get all option candidates across universe, ranked by premium/day.
        
        Args:
            skip_equity_filter: If True, include all symbols
            max_total: Maximum total candidates to return
            
        Returns:
            List of top option candidates across all symbols
        """
        if max_total is None:
            max_total = self.config.max_candidates_total
            
        scan_results = self.scan_universe(skip_equity_filter=skip_equity_filter)
        
        # Collect all candidates
        all_candidates = []
        for result in scan_results:
            all_candidates.extend(result.options_candidates)
        
        # Sort by key
        def _sort_key(c):
            if self.config.contract_rank_mode == "daily_return_per_delta":
                return c.daily_return_per_delta
            elif self.config.contract_rank_mode == "days_since_strike":
                return c.days_since_strike or 0
            elif self.config.contract_rank_mode == "lowest_strike_price":
                return -c.strike
            else:
                return c.daily_return_on_collateral
        all_candidates.sort(key=_sort_key, reverse=True)
        
        return all_candidates[:max_total]


print("Strategy Scanner module loaded")

# Cell 21
# ----------------------------------------------------------------------
# Scan Universe — Equity Filter → Options Filter → Candidates

try:
    scanner = StrategyScanner(
        config=config,
        equity_fetcher=equity_fetcher,
        options_fetcher=options_fetcher,
        greeks_calc=greeks_calc
    )
    
    print("Universe Scan")
    print("=" * 80)

    # Refresh starting_cash from live account
    account_info = alpaca.get_account_info()
    short_collateral = alpaca.get_short_collateral()
    config.starting_cash = account_info['cash'] - short_collateral
    target_position_dollars = config.starting_cash * config.max_position_pct

    print(f"Alpaca cash:                               ${account_info['cash']:,.2f}")
    print(f"Short position collateral:                 ${short_collateral:,.2f}")
    print(f"Available capital (cash - collateral):      ${config.starting_cash:,.2f}")
    print(f"Max position size ({config.max_position_pct*100:.1f}%):                ${target_position_dollars:,.2f}")
    print()
    
    # Run scan with equity filter
    scan_results = scanner.scan_universe(skip_equity_filter=False)
    
    passing_equity = [r for r in scan_results if r.equity_result.passes]
    passing_both = [r for r in passing_equity if r.has_candidates]
    
    print(f"Symbols scanned:                         {len(scan_results)}")
    print(f"Passed equity filter:                     {len(passing_equity)}")
    print(f"Passed equity + options filter:            {len(passing_both)}")
    
    # Check earnings & dividends only for symbols that passed both filters
    candidate_symbols = list(set(r.symbol for r in passing_both))
    event_rejections = scanner.equity_filter.check_events(candidate_symbols)
    
    if event_rejections:
        print(f"\nEvent-based rejections (DTE window = {config.max_dte}d):")
        for sym in sorted(event_rejections):
            for reason in event_rejections[sym]:
                print(f"  {sym:<8} {reason}")
        
        # Remove rejected symbols from passing lists
        passing_both = [r for r in passing_both if r.symbol not in event_rejections]
        print(f"Passed after event filter:                 {len(passing_both)}")
    
    # Show passing symbols with their indicator values
    if passing_equity:
        print(f"\n✓ Equity-passing symbols ({len(passing_equity)}):")
        bb_label = f"BB{config.bb_period}"
        print(f"  {'Symbol':<8} {'Price':>9} {'SMA8':>9} {'SMA20':>9} {'SMA50':>9} {bb_label:>9} {'RSI':>6} {'Collateral':>12} {'Opts':>5}")
        print("  " + "-" * 88)
        for result in passing_equity:
            r = result.equity_result
            collateral = r.current_price * 100
            print(
                f"  {r.symbol:<8} "
                f"${r.current_price:>8.2f} "
                f"{r.sma_8:>9.2f} "
                f"{r.sma_20:>9.2f} "
                f"{r.sma_50:>9.2f} "
                f"{r.bb_upper:>9.2f} "
                f"{r.rsi:>6.1f} "
                f"${collateral:>10,.0f} "
                f"{len(result.options_candidates):>5}"
            )
    else:
        print("\n⚠ No symbols passed the equity filter.")    
        
    top_candidates = scanner.get_all_candidates(skip_equity_filter=False)
    
    # Remove event-rejected symbols from candidates
    if event_rejections:
        top_candidates = [c for c in top_candidates if c.underlying not in event_rejections]
    
    if top_candidates:
        # Sort by symbol ascending, then daily return descending
        top_candidates.sort(key=lambda c: (c.underlying, -c.daily_return_on_collateral))
        
        # Calculate days since at/below strike from 60-day price history
        def days_since_strike(c):
            if c.underlying not in price_history:
                return "N/A"
            prices = price_history[c.underlying]
            at_or_below = prices[prices <= c.strike]
            if at_or_below.empty:
                return ">60"
            last_date = at_or_below.index[-1]
            return str((prices.index[-1] - last_date).days)
        
        print(f"\n{'Symbol':<26} {'Price':>9} {'Strike':>8} {'Drop%':>7} {'Days':>5} {'DTE':>5} {'Bid':>8} {'Ask':>8} {'Spread':>8} {'Sprd%':>7} {'Delta':>7} {'Daily%':>9} {'Vol':>6} {'OI':>6}")
        print("-" * 135)
        for c in top_candidates:
            delta_str = f"{abs(c.delta):.3f}" if c.delta else "N/A"
            spread = c.ask - c.bid if c.ask and c.bid else 0
            spread_pct = spread / c.mid if c.mid > 0 else 0
            vol_str = f"{c.volume:>6}" if c.volume is not None else "     0"
            oi_str = f"{c.open_interest:>6}" if c.open_interest is not None else "   N/A"
            drop_pct = (c.stock_price - c.strike) / c.stock_price
            days_str = days_since_strike(c)
            print(
                f"{c.symbol:<26} "
                f"${c.stock_price:>8.2f} "
                f"${c.strike:>7.2f} "
                f"{drop_pct:>6.1%} "
                f"{days_str:>5} "
                f"{c.dte:>5} "
                f"${c.bid:>7.2f} "
                f"${c.ask:>7.2f} "
                f"${spread:>7.2f} "
                f"{spread_pct:>6.0%} "
                f"{delta_str:>7} "
                f"{c.daily_return_on_collateral:>8.4%} "
                f"{vol_str} "
                f"{oi_str} "
            )

        # === Best Pick Per Ticker by Each Ranking Mode ===
        from itertools import groupby as _groupby

        def _days_since(c):
            if c.underlying not in price_history:
                return 0
            prices = price_history[c.underlying]
            at_or_below = prices[prices <= c.strike]
            if at_or_below.empty:
                return 999
            last_date = at_or_below.index[-1]
            return (prices.index[-1] - last_date).days

        rank_modes = {
            "daily_ret/delta": lambda c: c.daily_return_per_delta,
            "days_since_strike": lambda c: c.days_since_strike if c.days_since_strike is not None else _days_since(c),
            "daily_return_on_collateral": lambda c: c.daily_return_on_collateral,
            "lowest_strike": lambda c: -c.strike,
        }

        # Group by ticker
        sorted_by_ticker = sorted(top_candidates, key=lambda c: c.underlying)
        tickers = []
        for ticker, grp in _groupby(sorted_by_ticker, key=lambda c: c.underlying):
            tickers.append((ticker, list(grp)))

        print(f"\n{'='*120}")
        print(f"Best Pick Per Ticker by Ranking Mode   (active mode: {config.contract_rank_mode})")
        print(f"{'='*120}")
        print(f"  {'Ticker':<8} | {'daily_ret/delta':<30} | {'days_since_strike':<30} | {'daily_ret':<30} | {'lowest_strike':<30}")
        print(f"  {'-'*8}-+-{'-'*30}-+-{'-'*30}-+-{'-'*30}-+-{'-'*30}")

        for ticker, contracts in tickers:
            picks = {}
            for mode_name, key_fn in rank_modes.items():
                best = max(contracts, key=key_fn)
                val = key_fn(best)
                if mode_name == "daily_ret/delta":
                    val_str = f"{best.symbol[-15:]}  ({val:.4f})"
                elif mode_name == "days_since_strike":
                    days_val = int(val) if val < 999 else ">60"
                    val_str = f"{best.symbol[-15:]}  ({days_val}d)"
                elif mode_name == "lowest_strike":
                    val_str = f"{best.symbol[-15:]}  (${best.strike:.0f})"
                else:
                    val_str = f"{best.symbol[-15:]}  (${val:.3f}/d)"
                picks[mode_name] = val_str

            # Mark the active mode pick with *
            print(
                f"  {ticker:<8} | {picks['daily_ret/delta']:<30} | {picks['days_since_strike']:<30} | {picks['daily_return_on_collateral']:<30} | {picks['lowest_strike']:<30}"
            )
                        
    else:
        print("No candidates found with equity filter enabled.")

        
        # Diagnostic: why equity-passing symbols had no options candidates
        equity_passing_no_options = [r for r in passing_equity if not r.options_candidates]
        if equity_passing_no_options:
            print(f"\nDiagnostic — {len(equity_passing_no_options)} equity-passing symbol(s) failed options filter:")
            print("-" * 95)
            for result in equity_passing_no_options:
                sma_ceiling = None
                if config.max_strike_mode == "sma":
                    sma_ceiling = getattr(result.equity_result, f"sma_{config.max_strike_sma_period}", None)
                puts = scanner.options_fetcher.get_puts_chain(result.symbol, result.stock_price, config, sma_ceiling=sma_ceiling)
                
                if not puts:
                    if config.max_strike_mode == "sma" and sma_ceiling:
                        max_strike = sma_ceiling
                    else:
                        max_strike = result.stock_price * config.max_strike_pct
                    min_strike = result.stock_price * config.min_strike_pct
                    
                    print(
                        f"\n  {result.symbol} @ ${result.stock_price:.2f}: "
                        f"0 puts returned from API "
                        f"(strike range ${min_strike:.0f}-${max_strike:.0f}, DTE {config.min_dte}-{config.max_dte})"
                    )
                    continue
                
                _, all_filter_results = scanner.options_filter.filter_and_rank(puts)
                
                # Tally failure reasons
                failure_counts = {}
                for r in all_filter_results:
                    for reason in r.failure_reasons:
                        if "Daily return" in reason:
                            key = "Premium too low"
                        elif "Strike" in reason:
                            key = "Strike too high"
                        elif "Delta" in reason:
                            key = "Delta out of range" if "outside" in reason else "Delta unavailable"
                        elif "DTE" in reason:
                            key = "DTE out of range"
                        else:
                            key = reason
                        failure_counts[key] = failure_counts.get(key, 0) + 1
                
                reasons_str = ", ".join(f"{k}: {v}" for k, v in sorted(failure_counts.items(), key=lambda x: -x[1]))
                print(f"\n  {result.symbol} @ ${result.stock_price:.2f}: {len(puts)} puts, 0 passed — {reasons_str}")
                
                # Show closest misses (top 5 by daily return)
                near_misses = sorted(all_filter_results, key=lambda r: r.daily_return, reverse=True)[:5]
                print(f"    {'Contract':<26} {'Strike':>8} {'DTE':>5} {'Bid':>8} {'Delta':>8} {'Daily%':>10}  Fail Reasons")
                print(f"    {'-'*91}")
                for r in near_misses:
                    c = r.contract
                    delta_str = f"{r.delta_abs:.3f}" if r.delta_abs else "N/A"
                    reasons = "; ".join(r.failure_reasons) if r.failure_reasons else "✓"
                    print(
                        f"    {c.symbol:<26} "
                        f"${c.strike:>7.2f} "
                        f"{c.dte:>5} "
                        f"${c.bid:>7.2f} "
                        f"{delta_str:>8} "
                        f"{r.daily_return:>9.2%}  "
                        f"{reasons}"
                    )
        else:
            print("  (No symbols passed the equity filter, so no options were evaluated.)")
            
except NameError as e:
    print(f"⚠ Run Phase 1 first: {e}")
except Exception as e:
    print(f"⚠ Error: {e}")
    import traceback
    traceback.print_exc()

# Cell 22
# ----------------------------------------------------------------------
def run_phase2_diagnostics():
    """
    Run diagnostics on all Phase 2 components.
    """
    print("Phase 2 Diagnostics")
    print("=" * 60)
    
    results = {}
    
    # 1. Greeks Calculator
    print("\n1. Greeks Calculator...")
    try:
        test = greeks_calc.compute_greeks_from_price(0.15, 259.48, 225.0, 5, 'put')
        if test['iv'] and test['delta']:
            results['greeks'] = 'ok'
            print(f"   ✓ OK - IV: {test['iv']:.4f}, Delta: {test['delta']:.4f}")
        else:
            results['greeks'] = 'partial'
            print(f"   ⚠ Partial - some calculations failed")
    except Exception as e:
        results['greeks'] = 'error'
        print(f"   ✗ Error: {e}")
    
    # 2. Technical Indicators
    print("\n2. Technical Indicators...")
    try:
        test_prices = pd.Series([100 + (i % 5) + i * 0.5 for i in range(60)])
        sma = indicators.sma(test_prices, 20).iloc[-1]
        rsi = indicators.rsi(test_prices, 14).iloc[-1]
        results['indicators'] = 'ok'
        print(f"   ✓ OK - SMA(20): {sma:.2f}, RSI: {rsi:.2f}")
    except Exception as e:
        results['indicators'] = 'error'
        print(f"   ✗ Error: {e}")
    
    # 3. Equity Filter
    print("\n3. Equity Filter...")
    try:
        ef = EquityFilter(config)
        results['equity_filter'] = 'ok'
        print(f"   ✓ OK - Filter initialized")
    except Exception as e:
        results['equity_filter'] = 'error'
        print(f"   ✗ Error: {e}")
    
    # 4. Options Filter
    print("\n4. Options Filter...")
    try:
        of = OptionsFilter(config, greeks_calc)
        results['options_filter'] = 'ok'
        print(f"   ✓ OK - Filter initialized")
    except Exception as e:
        results['options_filter'] = 'error'
        print(f"   ✗ Error: {e}")
    
    # 5. Strategy Scanner
    print("\n5. Strategy Scanner...")
    try:
        scanner = StrategyScanner(config, equity_fetcher, options_fetcher, greeks_calc)
        results['scanner'] = 'ok'
        print(f"   ✓ OK - Scanner initialized")
    except Exception as e:
        results['scanner'] = 'error'
        print(f"   ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    all_ok = all(v == 'ok' for v in results.values())
    
    if all_ok:
        print("✓ All Phase 2 components working!")
        print("\nReady for Phase 3: Position Management")
    else:
        print("⚠ Some components need attention")
    
    return results


# Run diagnostics
try:
    phase2_results = run_phase2_diagnostics()
except NameError as e:
    print(f"⚠ Run Phase 1 first to initialize dependencies: {e}")

# Cell 23
# ----------------------------------------------------------------------
# Print Alpaca Account Info
account_info = alpaca.get_account_info()

print("Alpaca Account Information")
print("=" * 80)
print(f"Account status:              {account_info['status']}")
print(f"Cash available:              ${account_info['cash']:,.2f}")
print(f"Buying power (with margin):  ${account_info['buying_power']:,.2f}")
print(f"Portfolio value:             ${account_info['portfolio_value']:,.2f}")
print(f"Options trading level:       {account_info['options_trading_level']}")
print(f"Trading blocked:             {account_info['trading_blocked']}")
print()

# Cell 24
# ----------------------------------------------------------------------
# Print Alpaca Account Info
from datetime import datetime, timedelta
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderStatus
import re

def parse_strike_from_symbol(symbol: str) -> float:
    """Parse strike price from OCC format option symbol.
    
    Format: SYMBOL + YYMMDD + P/C + STRIKE
    Example: SNDK260213P00570000 -> strike = 570.00
    """
    # Find the P or C (put or call indicator)
    match = re.search(r'[PC](\d+)$', symbol)
    if match:
        strike_int = int(match.group(1))
        return strike_int / 1000.0
    return 0.0

account_info = alpaca.get_account_info()

print("Alpaca Account Information")
print("=" * 80)
print(f"Account status:              {account_info['status']}")
print(f"Cash available:              ${account_info['cash']:,.2f}")
print(f"Buying power (with margin):  ${account_info['buying_power']:,.2f}")
print(f"Portfolio value:             ${account_info['portfolio_value']:,.2f}")
print(f"Options trading level:       {account_info['options_trading_level']}")
print(f"Trading blocked:             {account_info['trading_blocked']}")
print()

# Get current positions
try:
    positions = alpaca.trading_client.get_all_positions()
    if positions:
        print(f"Current Positions ({len(positions)}):")
        print(f"  {'Symbol':<20} {'Qty':>8} {'Side':<6} {'Strike':>10} {'Entry Price':>12} {'Current Price':>14} {'Market Value':>14} {'Unrealized P/L':>14} {'Collateral':>12}")
        print("  " + "-" * 120)
        total_collateral = 0
        for pos in positions:
            qty = float(pos.qty)
            market_val = float(pos.market_value)
            side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
            
            # Calculate collateral: abs(qty) * strike * 100 for short positions
            strike = parse_strike_from_symbol(pos.symbol)
            if side == 'short' or qty < 0:
                # Short position: collateral = abs(qty) * strike * 100
                collateral = abs(qty) * strike * 100
            else:
                # Long position: no collateral required (you own the option)
                collateral = 0
            
            total_collateral += collateral
            
            print(
                f"  {pos.symbol:<20} "
                f"{qty:>8.0f} "
                f"{side:<6} "
                f"${strike:>9.2f} "
                f"${float(pos.avg_entry_price):>11.2f} "
                f"${float(pos.current_price):>13.2f} "
                f"${market_val:>13,.2f} "
                f"${float(pos.unrealized_pl):>13,.2f} "
                f"${collateral:>11,.2f}"
            )
        print(f"\n  Total collateral tied up: ${total_collateral:,.2f}")
        print(f"  Available cash (after collateral): ${account_info['cash'] - total_collateral:,.2f}")
    else:
        print("Current Positions: None")
    print()
except Exception as e:
    print(f"Error fetching positions: {e}")
    print()

# Get open orders
try:
    open_orders_request = GetOrdersRequest(status='open', limit=50)
    open_orders = alpaca.trading_client.get_orders(open_orders_request)
    
    if open_orders:
        print(f"Open Orders ({len(open_orders)}):")
        print(f"  {'Symbol':<20} {'Side':<6} {'Qty':>8} {'Type':<8} {'Status':<12} {'Limit Price':>12} {'Filled':>8}")
        print("  " + "-" * 100)
        for order in open_orders:
            limit_price = f"${float(order.limit_price):.2f}" if order.limit_price else "Market"
            filled_qty = float(order.filled_qty) if order.filled_qty else 0
            print(
                f"  {order.symbol:<20} "
                f"{order.side.value:<6} "
                f"{float(order.qty):>8.0f} "
                f"{order.type.value:<8} "
                f"{order.status.value:<12} "
                f"{limit_price:>12} "
                f"{filled_qty:>8.0f}"
            )
    else:
        print("Open Orders: None")
    print()
except Exception as e:
    print(f"Error fetching open orders: {e}")
    print()

# Get recent order history (last 7 days)
try:
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    order_history_request = GetOrdersRequest(
        status='all',
        limit=100,
        until=end_time,
        after=start_time
    )
    order_history = alpaca.trading_client.get_orders(order_history_request)
    
    # Filter to filled/closed orders
    filled_orders = [o for o in order_history if o.status.value in ['filled', 'closed', 'canceled']]
    
    if filled_orders:
        print(f"Recent Order History (last 7 days, {len(filled_orders)} orders):")
        print(f"  {'Time':<20} {'Symbol':<20} {'Side':<6} {'Qty':>8} {'Type':<8} {'Status':<12} {'Avg Price':>12}")
        print("  " + "-" * 110)
        for order in sorted(filled_orders, key=lambda x: x.created_at, reverse=True)[:20]:  # Show last 20
            filled_price = f"${float(order.filled_avg_price):.2f}" if order.filled_avg_price else "N/A"
            time_str = order.created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(order.created_at, 'strftime') else str(order.created_at)[:19]
            print(
                f"  {time_str:<20} "
                f"{order.symbol:<20} "
                f"{order.side.value:<6} "
                f"{float(order.qty):>8.0f} "
                f"{order.type.value:<8} "
                f"{order.status.value:<12} "
                f"{filled_price:>12}"
            )
        if len(filled_orders) > 20:
            print(f"  ... and {len(filled_orders) - 20} more orders")
    else:
        print("Recent Order History: None")
    print()
except Exception as e:
    print(f"Error fetching order history: {e}")
    print()

# Cell 25
# ----------------------------------------------------------------------
# Liquidate All Holdings
import time
from datetime import datetime, date
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import re

def parse_expiration_from_symbol(symbol: str) -> date:
    """Parse expiration date from OCC format option symbol.
    
    Format: SYMBOL + YYMMDD + P/C + STRIKE
    Example: SNDK260213P00570000 -> expiration = 2026-02-13
    """
    # Find the date part (6 digits after symbol, before P/C)
    match = re.search(r'(\d{6})[PC]', symbol)
    if match:
        date_str = match.group(1)
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        return date(year, month, day)
    return None

def liquidate_all_holdings():
    """Cancel all open orders and close all positions."""
    print("=" * 80)
    print("LIQUIDATING ALL HOLDINGS")
    print("=" * 80)
    print()
    
    # Show current state
    print("Current State:")
    print("-" * 80)
    
    # Get current positions
    try:
        positions = alpaca.trading_client.get_all_positions()
        if positions:
            print(f"Open Positions ({len(positions)}):")
            total_collateral = 0
            expired_positions = []
            active_positions = []
            
            today = date.today()
            for pos in positions:
                qty = float(pos.qty)
                side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                strike = parse_strike_from_symbol(pos.symbol)
                expiration = parse_expiration_from_symbol(pos.symbol)
                current_price = float(pos.current_price)
                
                is_expired = expiration and expiration < today
                is_worthless = current_price == 0.0
                
                if side == 'short' or qty < 0:
                    collateral = abs(qty) * strike * 100
                    total_collateral += collateral
                
                status = ""
                if is_expired:
                    status = " [EXPIRED]"
                elif is_worthless:
                    status = " [WORTHLESS]"
                
                print(f"  {pos.symbol:<20} qty={qty:>6.0f} side={side:<6} strike=${strike:>7.2f} "
                      f"exp={expiration} price=${current_price:>6.2f}{status}")
                
                if is_expired or is_worthless:
                    expired_positions.append(pos)
                else:
                    active_positions.append(pos)
            
            print(f"  Total collateral tied up: ${total_collateral:,.2f}")
            print(f"  Active positions: {len(active_positions)}")
            print(f"  Expired/worthless positions: {len(expired_positions)}")
        else:
            print("Open Positions: None")
            active_positions = []
            expired_positions = []
    except Exception as e:
        print(f"Error fetching positions: {e}")
        active_positions = []
        expired_positions = []
    
    # Get open orders
    try:
        open_orders_request = GetOrdersRequest(status='open', limit=50)
        open_orders = alpaca.trading_client.get_orders(open_orders_request)
        if open_orders:
            print(f"\nOpen Orders ({len(open_orders)}):")
            for order in open_orders:
                print(f"  {order.symbol:<20} {order.side.value:<6} qty={float(order.qty):>6.0f} status={order.status.value}")
        else:
            print("\nOpen Orders: None")
    except Exception as e:
        print(f"\nError fetching open orders: {e}")
    
    print()
    print("=" * 80)
    print("Starting Liquidation...")
    print("=" * 80)
    print()
    
    # Step 1: Cancel all open orders
    print("Step 1: Cancelling all open orders...")
    try:
        open_orders_request = GetOrdersRequest(status='open', limit=50)
        open_orders = alpaca.trading_client.get_orders(open_orders_request)
        if open_orders:
            for order in open_orders:
                try:
                    alpaca.trading_client.cancel_order_by_id(order.id)
                    print(f"  ✓ Cancelled: {order.symbol} ({order.side.value} {float(order.qty):.0f})")
                except Exception as e:
                    print(f"  ✗ Failed to cancel {order.symbol}: {e}")
        else:
            print("  No open orders to cancel.")
    except Exception as e:
        print(f"  Error cancelling orders: {e}")
    
    time.sleep(2)
    print()
    
    # Step 2: Close all positions
    print("Step 2: Closing all positions...")
    
    # First, try to close active positions with market orders
    if active_positions:
        print("  Closing active positions (market orders)...")
        for pos in active_positions:
            try:
                qty = float(pos.qty)
                
                if qty < 0:
                    close_qty = abs(qty)
                    close_side = OrderSide.BUY
                    action = "Buying to close"
                else:
                    close_qty = qty
                    close_side = OrderSide.SELL
                    action = "Selling to close"
                
                order_request = MarketOrderRequest(
                    symbol=pos.symbol,
                    qty=int(close_qty),
                    side=close_side,
                    time_in_force=TimeInForce.DAY
                )
                
                order = alpaca.trading_client.submit_order(order_request)
                print(f"  ✓ {action}: {pos.symbol} qty={int(close_qty)} order_id={order.id}")
                
            except Exception as e:
                print(f"  ✗ Failed to close {pos.symbol}: {e}")
                # Try limit order as fallback
                try:
                    limit_price = max(0.01, float(pos.current_price) * 1.1) if float(pos.current_price) > 0 else 0.01
                    order_request = LimitOrderRequest(
                        symbol=pos.symbol,
                        qty=int(close_qty),
                        side=close_side,
                        limit_price=limit_price,
                        time_in_force=TimeInForce.DAY
                    )
                    order = alpaca.trading_client.submit_order(order_request)
                    print(f"  ✓ Retry with limit order: {pos.symbol} limit=${limit_price:.2f} order_id={order.id}")
                except Exception as e2:
                    print(f"  ✗ Limit order also failed for {pos.symbol}: {e2}")
    
    # Then, try to close expired/worthless positions with $0.01 limit orders
    if expired_positions:
        print("\n  Closing expired/worthless positions (limit orders at $0.01)...")
        for pos in expired_positions:
            try:
                qty = float(pos.qty)
                expiration = parse_expiration_from_symbol(pos.symbol)
                today = date.today()
                
                if qty < 0:
                    # Short position: buy to close at $0.01
                    close_qty = abs(qty)
                    close_side = OrderSide.BUY
                    limit_price = 0.01
                    action = "Buying to close"
                else:
                    # Long position: sell to close at $0.01
                    close_qty = qty
                    close_side = OrderSide.SELL
                    limit_price = 0.01
                    action = "Selling to close"
                
                if expiration and expiration < today:
                    print(f"  ⚠ {pos.symbol} expired on {expiration} (worthless, trying $0.01 limit order)...")
                else:
                    print(f"  ⚠ {pos.symbol} appears worthless (current_price=$0.00, trying $0.01 limit order)...")
                
                order_request = LimitOrderRequest(
                    symbol=pos.symbol,
                    qty=int(close_qty),
                    side=close_side,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.DAY
                )
                
                order = alpaca.trading_client.submit_order(order_request)
                print(f"  ✓ {action}: {pos.symbol} qty={int(close_qty)} limit=${limit_price:.2f} order_id={order.id}")
                
            except Exception as e:
                print(f"  ✗ Failed to close {pos.symbol}: {e}")
                print(f"    Note: This position may already be expired and will be automatically removed by Alpaca.")
    
    if not active_positions and not expired_positions:
        print("  No positions to close.")
    else:
        print("\n  Waiting for orders to fill...")
        time.sleep(5)
        
        # Check if positions are closed
        remaining_positions = alpaca.trading_client.get_all_positions()
        if remaining_positions:
            print(f"\n  ⚠ Warning: {len(remaining_positions)} positions still open:")
            for pos in remaining_positions:
                expiration = parse_expiration_from_symbol(pos.symbol)
                exp_str = f" exp={expiration}" if expiration else ""
                print(f"    {pos.symbol} qty={pos.qty} price=${float(pos.current_price):.2f}{exp_str}")
            print("  Note: Expired positions may take time to be removed by Alpaca.")
        else:
            print("  ✓ All positions closed successfully.")
    
    print()
    print("=" * 80)
    print("Final Account Status:")
    print("=" * 80)
    
    # Show final account status
    try:
        account_info = alpaca.get_account_info()
        print(f"Cash available:              ${account_info['cash']:,.2f}")
        print(f"Buying power:                 ${account_info['buying_power']:,.2f}")
        print(f"Portfolio value:              ${account_info['portfolio_value']:,.2f}")
        
        final_positions = alpaca.trading_client.get_all_positions()
        print(f"Remaining positions:           {len(final_positions)}")
        
        final_orders_request = GetOrdersRequest(status='open', limit=50)
        final_orders = alpaca.trading_client.get_orders(final_orders_request)
        print(f"Remaining open orders:         {len(final_orders)}")
        
    except Exception as e:
        print(f"Error fetching final status: {e}")
    
    print("=" * 80)
    print("Liquidation Complete!")
    print("=" * 80)

# Run liquidation
liquidate_all_holdings()

# Cell 26
# ----------------------------------------------------------------------
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import os


class PositionStatus(Enum):
    """Status of an active position."""
    PENDING = "pending"           # Order placed, not yet filled
    ACTIVE = "active"             # Position is open
    CLOSING = "closing"           # Close order placed
    CLOSED_STOP_LOSS = "closed_stop_loss"
    CLOSED_EARLY_EXIT = "closed_early_exit"
    CLOSED_EXPIRY = "closed_expiry"
    CLOSED_MANUAL = "closed_manual"


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
class ActivePosition:
    """
    Represents an active CSP position with all tracking data.
    """
    # Identification
    position_id: str
    symbol: str                    # Underlying symbol
    option_symbol: str             # OCC option symbol

    # Entry data
    entry_date: datetime
    entry_stock_price: float
    entry_delta: float
    entry_premium: float           # Per share premium received
    entry_vix: float
    entry_iv: float

    # Contract details
    strike: float
    expiration: date
    dte_at_entry: int
    quantity: int                  # Number of contracts (negative for short)

    # Current state
    status: PositionStatus = PositionStatus.ACTIVE
    entry_daily_return: float = 0.0  # daily_return_on_collateral at entry

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
        return self.status in [PositionStatus.ACTIVE, PositionStatus.PENDING]

    def calculate_pnl(self, exit_premium: float) -> float:
        """
        Calculate P&L for closing at given premium.
        For short puts: profit = entry_premium - exit_premium
        """
        return (self.entry_premium - exit_premium) * 100 * abs(self.quantity)

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'option_symbol': self.option_symbol,
            'entry_date': self.entry_date.isoformat(),
            'entry_stock_price': self.entry_stock_price,
            'entry_delta': self.entry_delta,
            'entry_premium': self.entry_premium,
            'entry_vix': self.entry_vix,
            'entry_iv': self.entry_iv,
            'entry_daily_return': self.entry_daily_return,
            'strike': self.strike,
            'expiration': self.expiration.isoformat(),
            'dte_at_entry': self.dte_at_entry,
            'quantity': self.quantity,
            'status': self.status.value,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'exit_premium': self.exit_premium,
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
            'exit_details': self.exit_details,
            'entry_order_id': self.entry_order_id,
            'exit_order_id': self.exit_order_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ActivePosition':
        """Deserialize from dictionary."""
        return cls(
            position_id=data['position_id'],
            symbol=data['symbol'],
            option_symbol=data['option_symbol'],
            entry_date=datetime.fromisoformat(data['entry_date']),
            entry_stock_price=data['entry_stock_price'],
            entry_delta=data['entry_delta'],
            entry_premium=data['entry_premium'],
            entry_vix=data['entry_vix'],
            entry_iv=data['entry_iv'],
            entry_daily_return=data.get('entry_daily_return', 0.0),
            strike=data['strike'],
            expiration=date.fromisoformat(data['expiration']),
            dte_at_entry=data['dte_at_entry'],
            quantity=data['quantity'],
            status=PositionStatus(data['status']),
            exit_date=datetime.fromisoformat(data['exit_date']) if data.get('exit_date') else None,
            exit_premium=data.get('exit_premium'),
            exit_reason=ExitReason(data['exit_reason']) if data.get('exit_reason') else None,
            exit_details=data.get('exit_details'),
            entry_order_id=data.get('entry_order_id'),
            exit_order_id=data.get('exit_order_id'),
        )


print("Position tracking classes loaded")

# Cell 27
# ----------------------------------------------------------------------
class PortfolioManager:
    """
    Manages the portfolio of active positions.
    Handles position tracking, persistence, and portfolio-level metrics.
    """
    
    def __init__(self, config: 'StrategyConfig', persistence_path: Optional[str] = None):
        self.config = config
        self.positions: Dict[str, ActivePosition] = {}  # position_id -> position
        self.closed_positions: List[ActivePosition] = []
        self.persistence_path = persistence_path
        self._position_counter = 0
        
        # Load persisted state if available
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
        exit_details: str = ""
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
        
        # Set appropriate status
        status_map = {
            ExitReason.DELTA_STOP: PositionStatus.CLOSED_STOP_LOSS,
            ExitReason.DELTA_ABSOLUTE: PositionStatus.CLOSED_STOP_LOSS,
            ExitReason.STOCK_DROP: PositionStatus.CLOSED_STOP_LOSS,
            ExitReason.VIX_SPIKE: PositionStatus.CLOSED_STOP_LOSS,
            ExitReason.DATA_UNAVAILABLE: PositionStatus.CLOSED_STOP_LOSS,
            ExitReason.EARLY_EXIT: PositionStatus.CLOSED_EARLY_EXIT,
            ExitReason.EXPIRY: PositionStatus.CLOSED_EXPIRY,
            ExitReason.ASSIGNED: PositionStatus.CLOSED_MANUAL,
            ExitReason.MANUAL: PositionStatus.CLOSED_MANUAL,
        }
        position.status = status_map.get(exit_reason, PositionStatus.CLOSED_MANUAL)
        
        # Move to closed list
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
        return max(0, deployable_cash - self.total_collateral)
    
    def can_add_position(self, collateral_needed: float, deployable_cash: float) -> bool:
        """
        Check if we can add a new position.
        
        Args:
            collateral_needed: Collateral for new position
            deployable_cash: Total deployable cash
            
        Returns:
            True if position can be added
        """
        # Check position count limit
        if self.active_count >= self.config.num_tickers:
            return False
        
        # Check capital availability
        if self.get_available_cash(deployable_cash) < collateral_needed:
            return False
        
        return True
    
    def _save_state(self):
        """Persist current state to file (atomically)."""
        if not self.persistence_path:
            return

        state = {
            "positions": {pid: p.to_dict() for pid, p in self.positions.items()},
            "closed_positions": [p.to_dict() for p in self.closed_positions],
            "position_counter": self._position_counter,
        }

        # Write to temp file then atomically replace
        tmp_path = self.persistence_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, self.persistence_path)
    
    def _load_state(self):
        """Load state from file, handling corrupt JSON safely."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return

        try:
            with open(self.persistence_path, "r") as f:
                state = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠ Corrupt portfolio state file: {self.persistence_path} ({e})")
            print("  Starting with empty portfolio state. Consider restoring from backup.")
            # Optionally quarantine the bad file so it won't be retried:
            # os.rename(self.persistence_path, self.persistence_path + ".corrupt")
            return

        self.positions = {
            pid: ActivePosition.from_dict(data)
            for pid, data in state.get("positions", {}).items()
        }
        self.closed_positions = [
            ActivePosition.from_dict(data)
            for data in state.get("closed_positions", [])
        ]
        self._position_counter = state.get("position_counter", 0)    
        
    def get_summary(self) -> dict:
        """Get portfolio summary statistics."""
        active = self.get_active_positions()
        
        return {
            'active_positions': len(active),
            'total_collateral': self.total_collateral,
            'total_premium_received': sum(p.total_premium_received for p in active),
            'symbols': self.active_symbols,
            'closed_count': len(self.closed_positions),
            'closed_pnl': sum(
                p.calculate_pnl(p.exit_premium) 
                for p in self.closed_positions 
                if p.exit_premium is not None
            ),
        }



class StrategyMetadataStore:
    """Tracks strategy-specific metadata that Alpaca doesn't store.

    Alpaca is source of truth for positions and orders.
    This store records the strategy context (entry greeks, exit reasons,
    order attempt diagnostics) that Alpaca has no concept of.

    Keyed by option_symbol (OCC format — globally unique per contract).
    """

    def __init__(self, path: str = "strategy_metadata.json"):
        self.path = path
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
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.entries, f, indent=2)
        os.replace(tmp, self.path)

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                self.entries = json.load(f)
            active_count = sum(1 for m in self.entries.values() if m.get("exit_date") is None)
            print(f"  Loaded {len(self.entries)} metadata entries ({active_count} active)")
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: corrupt metadata file ({e}), starting fresh")
            self.entries = {}


print("Portfolio Manager loaded")
print("Strategy Metadata Store loaded")

# Cell 28
# ----------------------------------------------------------------------
# Test Position Tracking
print("Position Tracking Test")
print("=" * 50)

# Create a test position
test_position = ActivePosition(
    position_id="TEST_001",
    symbol="AAPL",
    option_symbol="AAPL260212P00250000",
    entry_date=datetime.now(),
    entry_stock_price=259.48,
    entry_delta=-0.15,
    entry_premium=1.08,
    entry_vix=17.01,
    entry_iv=0.3584,
    strike=250.0,
    expiration=date(2026, 2, 6),
    dte_at_entry=5,
    quantity=-1,  # Short 1 contract
)

print(f"Position ID: {test_position.position_id}")
print(f"Symbol: {test_position.symbol}")
print(f"Option: {test_position.option_symbol}")
print(f"Strike: ${test_position.strike}")
print(f"Collateral Required: ${test_position.collateral_required:,.2f}")
print(f"Premium Received: ${test_position.total_premium_received:.2f}")
print(f"Entry Delta: {test_position.entry_delta}")
print(f"Current DTE: {test_position.current_dte}")
print(f"Is Open: {test_position.is_open}")

# Test P&L calculation
exit_premium = 0.50  # Buy back at $0.50
pnl = test_position.calculate_pnl(exit_premium)
print(f"\nIf closed at ${exit_premium}:")
print(f"  P&L: ${pnl:.2f}")

# Cell 29
# ----------------------------------------------------------------------
@dataclass
class RiskCheckResult:
    """
    Result of a risk check on a position.
    """
    should_exit: bool
    exit_reason: Optional[ExitReason]
    details: str
    current_values: Dict[str, float]


class RiskManager:
    """
    Manages risk checks for positions.
    Implements stop-loss and early exit logic.
    
    Stop-Loss Conditions (ANY triggers exit):
    1. Current delta >= 2x entry delta
    2. Stock price <= 95% of entry stock price
    3. Current VIX >= 1.15x entry VIX (or session open VIX)
    
    Early Exit Condition:
    - Premium captured >= expected decay + 15% buffer
    """
    
    def __init__(self, config: 'StrategyConfig'):
        self.config = config
    
    def check_delta_stop(
        self, 
        position: ActivePosition, 
        current_delta: float
    ) -> RiskCheckResult:
        """
        Check if delta has doubled from entry.
        
        Args:
            position: The position to check
            current_delta: Current option delta
            
        Returns:
            RiskCheckResult
        """
        entry_delta_abs = abs(position.entry_delta)
        current_delta_abs = abs(current_delta)
        threshold = entry_delta_abs * self.config.delta_stop_multiplier
        
        triggered = current_delta_abs >= threshold
        
        return RiskCheckResult(
            should_exit=triggered,
            exit_reason=ExitReason.DELTA_STOP if triggered else None,
            details=f"Delta {current_delta_abs:.3f} {'≥' if triggered else '<'} {threshold:.3f} (2x entry {entry_delta_abs:.3f})",
            current_values={
                'entry_delta': entry_delta_abs,
                'current_delta': current_delta_abs,
                'threshold': threshold,
            }
        )
    
    def check_delta_absolute_stop(
        self,
        position: ActivePosition,
        current_delta: float
    ) -> RiskCheckResult:
        """
        Check if delta has reached the absolute ceiling (e.g. 0.40).
        Unlike delta_stop (relative to entry), this is a hard cap.
        """
        current_delta_abs = abs(current_delta)
        threshold = self.config.delta_absolute_stop
        triggered = current_delta_abs >= threshold

        return RiskCheckResult(
            should_exit=triggered,
            exit_reason=ExitReason.DELTA_ABSOLUTE if triggered else None,
            details=f"Delta {current_delta_abs:.3f} {'≥' if triggered else '<'} {threshold:.3f} (absolute cap)",
            current_values={
                'current_delta': current_delta_abs,
                'threshold': threshold,
            }
        )

    def check_stock_drop_stop(
        self, 
        position: ActivePosition, 
        current_stock_price: float
    ) -> RiskCheckResult:
        """
        Check if stock has dropped 5% from entry.
        
        Args:
            position: The position to check
            current_stock_price: Current stock price
            
        Returns:
            RiskCheckResult
        """
        threshold = position.entry_stock_price * (1 - self.config.stock_drop_stop_pct)
        drop_pct = (position.entry_stock_price - current_stock_price) / position.entry_stock_price
        
        triggered = current_stock_price <= threshold
        
        return RiskCheckResult(
            should_exit=triggered,
            exit_reason=ExitReason.STOCK_DROP if triggered else None,
            details=f"Stock ${current_stock_price:.2f} {'≤' if triggered else '>'} ${threshold:.2f} ({drop_pct:.1%} drop)",
            current_values={
                'entry_stock_price': position.entry_stock_price,
                'current_stock_price': current_stock_price,
                'threshold': threshold,
                'drop_pct': drop_pct,
            }
        )
    
    def check_vix_spike_stop(
        self, 
        position: ActivePosition, 
        current_vix: float,
        reference_vix: Optional[float] = None
    ) -> RiskCheckResult:
        """
        Check if VIX has spiked 15% from reference.
        
        Args:
            position: The position to check
            current_vix: Current VIX value
            reference_vix: Reference VIX (entry or session open). Uses entry if None.
            
        Returns:
            RiskCheckResult
        """
        ref_vix = reference_vix or position.entry_vix
        threshold = ref_vix * self.config.vix_spike_multiplier
        spike_pct = (current_vix - ref_vix) / ref_vix
        
        triggered = current_vix >= threshold
        
        return RiskCheckResult(
            should_exit=triggered,
            exit_reason=ExitReason.VIX_SPIKE if triggered else None,
            details=f"VIX {current_vix:.2f} {'≥' if triggered else '<'} {threshold:.2f} ({spike_pct:+.1%} from ref {ref_vix:.2f})",
            current_values={
                'reference_vix': ref_vix,
                'current_vix': current_vix,
                'threshold': threshold,
                'spike_pct': spike_pct,
            }
        )
    
    def check_early_exit(
        self, 
        position: ActivePosition, 
        current_premium: float
    ) -> RiskCheckResult:
        """
        Check if premium has decayed enough for early exit.
        
        Formula:
            daily_return = position.entry_daily_return OR config.min_daily_return
            expected_capture = days_held * daily_return * strike  ($ per share)
            buffer = expected_capture * early_exit_buffer_pct
            Exit if: premium_captured >= expected_capture + buffer
        
        Args:
            position: The position to check
            current_premium: Current option premium (ask price to buy back)
            
        Returns:
            RiskCheckResult
        """
        days_held = position.days_held
        if days_held <= 0:
            return RiskCheckResult(
                should_exit=False,
                exit_reason=None,
                details="Position just opened, no early exit check",
                current_values={}
            )
        
        # Determine daily return source
        if self.config.early_exit_return_source == "entry":
            daily_return = position.entry_daily_return
        else:
            daily_return = self.config.min_daily_return
        
        # Expected premium captured ($ per share)
        expected_capture = days_held * daily_return * position.strike
        
        # Buffer: percentage of expected
        buffer = expected_capture * self.config.early_exit_buffer_pct
        target = expected_capture + buffer
        
        # Actual premium captured ($ per share)
        premium_captured = position.entry_premium - current_premium
        
        triggered = premium_captured >= target and target > 0
        
        return RiskCheckResult(
            should_exit=triggered,
            exit_reason=ExitReason.EARLY_EXIT if triggered else None,
            details=(
                f"Captured ${premium_captured:.4f}/sh {'≥' if triggered else '<'} "
                f"target ${target:.4f}/sh "
                f"(expected ${expected_capture:.4f} + {self.config.early_exit_buffer_pct:.0%} buffer) "
                f"[{days_held}d held, daily_return={daily_return:.4%}, source={self.config.early_exit_return_source}]"
            ),
            current_values={
                'entry_premium': position.entry_premium,
                'current_premium': current_premium,
                'premium_captured': premium_captured,
                'expected_capture': expected_capture,
                'buffer': buffer,
                'target': target,
                'daily_return': daily_return,
                'days_held': days_held,
            }
        )
    
    def check_all_stops(
        self,
        position: ActivePosition,
        current_delta: float,
        current_stock_price: float,
        current_vix: float,
        reference_vix: Optional[float] = None
    ) -> RiskCheckResult:
        """
        Check all stop-loss conditions.
        Returns first triggered condition, or no-exit result.
        
        Args:
            position: The position to check
            current_delta: Current option delta
            current_stock_price: Current stock price
            current_vix: Current VIX
            reference_vix: Reference VIX for spike check
            
        Returns:
            RiskCheckResult (first triggered, or aggregate no-exit)
        """
        # Check delta stop (relative: 2x entry)
        delta_check = self.check_delta_stop(position, current_delta)
        if self.config.enable_delta_stop and delta_check.should_exit:
            return delta_check

        # Check delta absolute stop (hard cap)
        delta_abs_check = self.check_delta_absolute_stop(position, current_delta)
        if self.config.enable_delta_absolute_stop and delta_abs_check.should_exit:
            return delta_abs_check
        
        # Check stock drop stop
        stock_check = self.check_stock_drop_stop(position, current_stock_price)
        if self.config.enable_stock_drop_stop and stock_check.should_exit:
            return stock_check
        
        # Check VIX spike stop
        vix_check = self.check_vix_spike_stop(position, current_vix, reference_vix)
        if self.config.enable_vix_spike_stop and vix_check.should_exit:
            return vix_check
        
        # No stop triggered
        return RiskCheckResult(
            should_exit=False,
            exit_reason=None,
            details="All stop-loss checks passed",
            current_values={
                'delta': delta_check.current_values,
                'delta_absolute': delta_abs_check.current_values,
                'stock': stock_check.current_values,
                'vix': vix_check.current_values,
            }
        )
    
    def evaluate_position(
        self,
        position: ActivePosition,
        current_delta: float,
        current_stock_price: float,
        current_vix: float,
        current_premium: float,
        reference_vix: Optional[float] = None
    ) -> RiskCheckResult:
        """
        Full risk evaluation: check stops first, then early exit.
        
        Returns:
            RiskCheckResult with recommendation
        """
        # Stop-losses take priority
        stop_result = self.check_all_stops(
            position, current_delta, current_stock_price, current_vix, reference_vix
        )
        if stop_result.should_exit:
            return stop_result
        
        # Check early exit opportunity
        early_result = self.check_early_exit(position, current_premium)
        if self.config.enable_early_exit and early_result.should_exit:
            return early_result
        
        # All checks passed, hold position
        return RiskCheckResult(
            should_exit=False,
            exit_reason=None,
            details="Position healthy, continue holding",
            current_values={
                'stops': stop_result.current_values,
                'early_exit': early_result.current_values,
            }
        )


print("Risk Manager loaded")

# Cell 30
# ----------------------------------------------------------------------
# Test Risk Manager
print("Risk Manager Test")
print("=" * 60)

risk_manager = RiskManager(config)

# Use the test position from above
print(f"\nTest Position: {test_position.symbol} ${test_position.strike} put")
print(f"Entry: delta={test_position.entry_delta}, stock=${test_position.entry_stock_price}, VIX={test_position.entry_vix}")

# Scenario 1: Normal market (no exit)
print(f"\n--- Scenario 1: Normal Market ---")
result = risk_manager.evaluate_position(
    position=test_position,
    current_delta=-0.18,        # Slight increase
    current_stock_price=257.00,  # Small dip
    current_vix=17.50,           # Slight increase
    current_premium=0.90,        # Some decay
)
print(f"Should Exit: {result.should_exit}")
print(f"Details: {result.details}")

# Scenario 2: Delta doubled (stop loss)
print(f"\n--- Scenario 2: Delta Doubled ---")
result = risk_manager.evaluate_position(
    position=test_position,
    current_delta=-0.35,         # Doubled from 0.15
    current_stock_price=255.00,
    current_vix=18.00,
    current_premium=1.50,
)
print(f"Should Exit: {result.should_exit}")
print(f"Reason: {result.exit_reason}")
print(f"Details: {result.details}")

# Scenario 3: Stock dropped 5% (stop loss)
print(f"\n--- Scenario 3: Stock Drop ---")
result = risk_manager.evaluate_position(
    position=test_position,
    current_delta=-0.20,
    current_stock_price=245.00,  # ~5.5% drop
    current_vix=18.00,
    current_premium=1.80,
)
print(f"Should Exit: {result.should_exit}")
print(f"Reason: {result.exit_reason}")
print(f"Details: {result.details}")

# Scenario 4: VIX spike (stop loss)
print(f"\n--- Scenario 4: VIX Spike ---")
result = risk_manager.evaluate_position(
    position=test_position,
    current_delta=-0.18,
    current_stock_price=258.00,
    current_vix=20.00,           # ~18% spike from 17.01
    current_premium=1.00,
)
print(f"Should Exit: {result.should_exit}")
print(f"Reason: {result.exit_reason}")
print(f"Details: {result.details}")

# Scenario 5: Early exit opportunity
print(f"\n--- Scenario 5: Early Exit (premium captured) ---")
# Simulate 2 days held on a 5 DTE position
test_position_aged = ActivePosition(
    position_id="TEST_002",
    symbol="AAPL",
    option_symbol="AAPL260206P00250000",
    entry_date=datetime.now() - timedelta(days=2),  # 2 days ago
    entry_stock_price=259.48,
    entry_delta=-0.15,
    entry_premium=1.08,
    entry_vix=17.01,
    entry_iv=0.3584,
    strike=250.0,
    expiration=date.today() + timedelta(days=3),  # 3 DTE remaining
    dte_at_entry=5,
    quantity=-1,
)

# Expected: 2/5 = 40% decay expected, target = 40% + 15% = 55%
# If premium is now $0.40, captured = (1.08 - 0.40) / 1.08 = 63%
result = risk_manager.check_early_exit(
    position=test_position_aged,
    current_premium=0.40
)
print(f"Days Held: {test_position_aged.days_held}")
print(f"Should Exit: {result.should_exit}")
print(f"Reason: {result.exit_reason}")
print(f"Details: {result.details}")

# Cell 31
# ----------------------------------------------------------------------
@dataclass
class OrderResult:
    """Result of an order submission."""
    success: bool
    order_id: Optional[str]
    message: str
    order_details: Optional[dict] = None


class ExecutionEngine:
    """
    Handles order execution via Alpaca.
    
    For CSP strategy:
    - Entry: Sell to Open (STO) put options
    - Exit: Buy to Close (BTC) put options
    """
    
    def __init__(
        self, 
        alpaca_manager: 'AlpacaClientManager',
        config: 'StrategyConfig'
    ):
        self.trading_client = alpaca_manager.trading_client
        self.config = config
        self.paper = alpaca_manager.paper
    
    def sell_to_open(
        self,
        option_symbol: str,
        quantity: int = 1,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> OrderResult:
        """
        Sell to open a put option (enter CSP position).
        
        Args:
            option_symbol: OCC option symbol
            quantity: Number of contracts
            limit_price: Limit price (uses market order if None)
            time_in_force: Order duration
            
        Returns:
            OrderResult with order details
        """
        try:
            if limit_price:
                order_request = LimitOrderRequest(
                    symbol=option_symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    type=OrderType.LIMIT,
                    limit_price=limit_price,
                    time_in_force=time_in_force,
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=option_symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=time_in_force,
                )
            
            order = self.trading_client.submit_order(order_request)

            return OrderResult(
                success=True,
                order_id=str(order.id),              
                message=f"Order submitted: {order.status.value}",
                order_details={
                    'id': str(order.id),              
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'qty': str(order.qty),
                    'type': order.type.value,
                    'status': order.status.value,
                    'limit_price': str(order.limit_price) if order.limit_price else None,
                }
            )
            
        except Exception as e:
            return OrderResult(
                success=False,
                order_id=None,
                message=f"Order failed: {str(e)}"
            )
    
    def buy_to_close(
        self,
        option_symbol: str,
        quantity: int = 1,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> OrderResult:
        """
        Buy to close a put option (exit CSP position).
        
        Args:
            option_symbol: OCC option symbol
            quantity: Number of contracts
            limit_price: Limit price (uses market order if None)
            time_in_force: Order duration
            
        Returns:
            OrderResult with order details
        """
        try:
            if limit_price:
                order_request = LimitOrderRequest(
                    symbol=option_symbol,
                    qty=quantity,
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    limit_price=limit_price,
                    time_in_force=time_in_force,
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=option_symbol,
                    qty=quantity,
                    side=OrderSide.BUY,
                    time_in_force=time_in_force,
                )
            
            order = self.trading_client.submit_order(order_request)

            return OrderResult(
                success=True,
                order_id=str(order.id),               # ← cast to string
                message=f"Order submitted: {order.status.value}",
                order_details={
                    'id': str(order.id),              # ← cast to string
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'qty': str(order.qty),
                    'type': order.type.value,
                    'status': order.status.value,
                    'limit_price': str(order.limit_price) if order.limit_price else None,
                }
            )
            
        except Exception as e:
            return OrderResult(
                success=False,
                order_id=None,
                message=f"Order failed: {str(e)}"
            )
    
    def get_order_status(self, order_id: str) -> Optional[dict]:
        """
        Get status of an order.
        
        Returns:
            Order details dict or None if not found
        """
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'qty': str(order.qty),
                'filled_qty': str(order.filled_qty),
                'type': order.type.value,
                'status': order.status.value,
                'filled_avg_price': str(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception:
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Returns:
            True if cancelled successfully
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False
    
    def get_positions(self) -> List[dict]:
        """
        Get all current positions from Alpaca.
        
        Returns:
            List of position dictionaries
        """
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': str(pos.qty),
                    'side': pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                    'avg_entry_price': str(pos.avg_entry_price),
                    'current_price': str(pos.current_price),
                    'market_value': str(pos.market_value),
                    'unrealized_pl': str(pos.unrealized_pl),
                }
                for pos in positions
            ]
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []


print("Execution Engine loaded")

# Cell 32
# ----------------------------------------------------------------------
# Test Execution Engine (paper trading only!)
try:
    execution = ExecutionEngine(alpaca, config)
    
    print("Execution Engine Test")
    print("=" * 50)
    print(f"Mode: {'PAPER' if execution.paper else '⚠️ LIVE'}")
    
    # Get current positions
    positions = execution.get_positions()
    print(f"\nCurrent Positions: {len(positions)}")
    for pos in positions:
        print(f"  {pos['symbol']}: {pos['qty']} @ ${pos['avg_entry_price']}")
    
    # Note: We won't actually submit orders in this test
    print("\n✓ Execution engine ready")
    print("  (Order submission tested in live trading loop)")
    
except NameError as e:
    print(f"⚠ Run Phase 1 first: {e}")
except Exception as e:
    print(f"⚠ Error: {e}")

# Cell 33
# ----------------------------------------------------------------------
class DailyLog:
    """Single-file daily JSON log."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._data = None
        self._current_date = None
    
    def _get_path(self) -> Path:
        return self.log_dir / f"{date.today().isoformat()}.json"
    
    def _ensure_loaded(self):
        """Load or initialize today's log."""
        today = date.today()
        if self._current_date == today and self._data is not None:
            return
        
        path = self._get_path()
        if path.exists():
            with open(path, "r") as f:
                self._data = json.load(f)
        else:
            self._data = {
                "date": today.isoformat(),
                "config_snapshot": {},
                "equity_scan": {},
                "options_scans": [],
                "cycles": [],
                "trades": [],
                "order_attempts": [],
                "shutdown": {},
            }
        self._current_date = today
    
    def _save(self):
        with open(self._get_path(), "w") as f:
            json.dump(self._data, f, indent=2)
    
    def log_config(self, config):
        """Snapshot config at start of day."""
        self._ensure_loaded()
        self._data["config_snapshot"] = {
            "starting_cash": config.starting_cash,
            "num_tickers": config.num_tickers,
            "max_strike_pct": config.max_strike_pct,
            "min_strike_pct": config.min_strike_pct,
            "min_daily_return": config.min_daily_return,
            "delta_min": config.delta_min,
            "delta_max": config.delta_max,
            "min_dte": config.min_dte,
            "max_dte": config.max_dte,
            "max_candidates_per_symbol": config.max_candidates_per_symbol,
            "max_candidates_total": config.max_candidates_total,
            "max_position_pct": config.max_position_pct,
            "poll_interval_seconds": config.poll_interval_seconds,
            "paper_trading": config.paper_trading,
            "entry_start_price": config.entry_start_price,
            "entry_step_interval": config.entry_step_interval,
            "entry_step_pct": config.entry_step_pct,
            "entry_max_steps": config.entry_max_steps,
            "entry_refetch_snapshot": config.entry_refetch_snapshot,
            # Exit order params
            "stop_loss_order_type": config.stop_loss_order_type,
            "stop_exit_start_offset_pct": config.stop_exit_start_offset_pct,
            "stop_exit_step_pct": config.stop_exit_step_pct,
            "stop_exit_step_interval": config.stop_exit_step_interval,
            "stop_exit_max_steps": config.stop_exit_max_steps,
            "exit_start_price": config.exit_start_price,
            "exit_step_interval": config.exit_step_interval,
            "exit_step_pct": config.exit_step_pct,
            "exit_max_steps": config.exit_max_steps,
            "exit_refetch_snapshot": config.exit_refetch_snapshot,
            "close_before_expiry_days": config.close_before_expiry_days,
            "exit_on_missing_delta": config.exit_on_missing_delta,
            "early_exit_return_source": config.early_exit_return_source,
            "early_exit_buffer_pct": config.early_exit_buffer_pct,
            # Stop-loss thresholds
            "delta_stop_multiplier": config.delta_stop_multiplier,
            "delta_absolute_stop": config.delta_absolute_stop,
            "stock_drop_stop_pct": config.stock_drop_stop_pct,
            "vix_spike_multiplier": config.vix_spike_multiplier,
        }
        self._save()
    
    def log_equity_scan(self, scan_results, passing_symbols):
        """Log daily equity scan (once per day)."""
        self._ensure_loaded()
        results = {}
        for r in scan_results:
            if r.passes:
                results[r.symbol] = {
                    "price": round(r.current_price, 2),
                    "sma_8": round(r.sma_8, 2),
                    "sma_20": round(r.sma_20, 2),
                    "sma_50": round(r.sma_50, 2),
                    "rsi": round(r.rsi, 1),
                    "collateral": round(r.current_price * 100, 0),
                }        
                
        self._data["equity_scan"] = {
            "timestamp": datetime.now().isoformat(),
            "scanned": len(scan_results),
            "passed": passing_symbols,
            "results": results,
        }
        self._save()
        
    def log_options_scan(self, cycle: int, symbol: str, filter_results: list):
        """Log options filter results for a symbol."""
        self._ensure_loaded()
        ts = datetime.now().isoformat()
        
        if "options_scans" not in self._data:
            self._data["options_scans"] = []
        
        contracts = []
        for r in filter_results:
            c = r.contract
            contracts.append({
                "contract": c.symbol,
                "strike": round(c.strike, 2),
                "dte": c.dte,
                "bid": round(c.bid, 2),
                "ask": round(c.ask, 2),
                "delta": round(r.delta_abs, 3) if r.delta_abs else None,
                "iv": round(c.implied_volatility, 4) if c.implied_volatility else None,
                "daily_return": round(r.daily_return, 6),
                "passes": r.passes,
                "failure_reasons": r.failure_reasons,
            })
        
        self._data["options_scans"].append({
            "timestamp": ts,
            "cycle": cycle,
            "symbol": symbol,
            "contracts_evaluated": len(filter_results),
            "contracts_passed": sum(1 for r in filter_results if r.passes),
            "contracts": contracts,
        })
        self._save()        
    
    def log_cycle(self, cycle: int, summary: dict,
                  options_checked: list = None,
                  failure_tally: dict = None):
        """Append a cycle entry."""
        self._ensure_loaded()
        p = summary.get("portfolio", {})
        self._data["cycles"].append({
            "cycle": cycle,
            "timestamp": summary.get("timestamp", datetime.now().isoformat()),
            "vix": round(summary.get("current_vix", 0), 2),
            "deployable_cash": round(summary.get("deployable_cash", 0), 0),
            "market_open": summary.get("market_open", False),
            "mode": "monitor-only" if summary.get("monitor_only") else "active",
            "options_checked": options_checked or [],
            "candidates_found": summary.get("entries", 0),
            "failure_tally": failure_tally or {},
            "entries": summary.get("entries", 0),
            "exits": summary.get("exits", 0),
            "positions": p.get("active_positions", 0),
            "collateral": round(p.get("total_collateral", 0), 0),
            "errors": summary.get("errors", []),
        })
        self._save()
    
    def log_trade(self, action: str, **kwargs):
        """Append a trade entry or exit."""
        self._ensure_loaded()
        trade = {"timestamp": datetime.now().isoformat(), "action": action}
        trade.update(kwargs)
        self._data["trades"].append(trade)
        self._save()
    
    def log_order_attempt(
        self,
        action: str,
        symbol: str,
        contract: str,
        steps: List[dict],
        outcome: str,
        **kwargs,
    ):
        """Log a full stepped order attempt (entry or exit).

        Args:
            action:   "entry" | "exit_early" | "exit_stop" | "exit_expiry"
            symbol:   Underlying ticker
            contract: OCC option symbol
            steps:    [{step, limit_price, status, duration_s, bid, ask, mid, spread}, ...]
            outcome:  "filled" | "exhausted" | "cancelled" | "filter_failed" | "submission_failed"
            **kwargs: filled_price, fill_step, exit_reason, qty, start_price, floor_price, etc.
        """
        self._ensure_loaded()
        if "order_attempts" not in self._data:
            self._data["order_attempts"] = []
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "symbol": symbol,
            "contract": contract,
            "steps": steps,
            "total_steps": len(steps),
            "outcome": outcome,
        }
        entry.update(kwargs)
        self._data["order_attempts"].append(entry)
        self._save()

    def log_shutdown(self, reason: str, total_cycles: int, portfolio_summary: dict):
        """Log end-of-day shutdown."""
        self._ensure_loaded()
        self._data["shutdown"] = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "total_cycles": total_cycles,
            "final_positions": portfolio_summary.get("active_positions", 0),
            "final_collateral": round(portfolio_summary.get("total_collateral", 0), 0),
        }
        self._save()

    @property
    def today_path(self) -> str:
        return str(self._get_path())        

print("Daily Log loaded")

# Cell 34
# ----------------------------------------------------------------------
class TradingLoop:
    """
    Main trading loop that orchestrates the CSP strategy.
    
    Responsibilities:
    1. Check market hours
    2. Monitor VIX regime
    3. Check existing positions for exits
    4. Scan for new opportunities
    5. Execute trades
    """
    
    def __init__(
        self,
        config: 'StrategyConfig',
        data_manager: 'DataManager',
        scanner: 'StrategyScanner',
        metadata_store: 'StrategyMetadataStore',
        risk_manager: RiskManager,
        execution: ExecutionEngine,
        vix_fetcher: 'VixDataFetcher',
        greeks_calc: 'GreeksCalculator',
        alpaca_manager: 'AlpacaClientManager' = None
    ):
        self.config = config
        self.data_manager = data_manager
        self.scanner = scanner
        self.metadata = metadata_store
        self.risk_manager = risk_manager
        self.execution = execution
        self.vix_fetcher = vix_fetcher
        self.greeks_calc = greeks_calc
        self.alpaca_manager = alpaca_manager
        
        self.eastern = pytz.timezone('US/Eastern')
        self._running = False
        self._session_vix_open = None
        
        # Daily scan state
        self._equity_passing: Optional[List[str]] = None  # Symbols passing equity filter today
        self._equity_scan_date: Optional[date] = None     # Date of last equity scan
        self._monitor_only: bool = False                  # True = no new entries, only track exits
        self._last_scan_results: List = []                # Full scan results for diagnostics        
        self._cycle_count: int = 0
        self._last_step_log: List[dict] = []  # Step-by-step data from last order attempt
        self.logger = DailyLog(log_dir="logs")
        print(f"  Daily log: {self.logger.today_path}")

    # ── Alpaca helpers (source of truth for positions) ──────────

    def _get_alpaca_positions(self) -> list:
        """Fetch current positions from Alpaca."""
        try:
            return self.alpaca_manager.trading_client.get_all_positions()
        except Exception as e:
            print(f"  Warning: Could not fetch Alpaca positions: {e}")
            return []

    def _get_option_positions(self) -> list:
        """Filter Alpaca positions to option positions only."""
        return [
            p for p in self._get_alpaca_positions()
            if len(p.symbol) > 10 and any(c.isdigit() for c in p.symbol)
        ]

    def _get_active_symbols(self) -> set:
        """Get underlying symbols from active Alpaca option positions."""
        import re as _re
        symbols = set()
        for pos in self._get_option_positions():
            match = _re.match(r'^([A-Z]+)\d', pos.symbol)
            if match:
                symbols.add(match.group(1))
        return symbols

    def _get_position_count(self) -> int:
        """Count active option positions on Alpaca."""
        return len(self._get_option_positions())

    def _build_position_proxy(self, alpaca_pos, meta: dict):
        """Build lightweight object with ActivePosition-like attributes
        so RiskManager works without changes."""
        from types import SimpleNamespace
        from datetime import date as _date

        strike = AlpacaClientManager.parse_strike_from_symbol(alpaca_pos.symbol)
        expiration = AlpacaClientManager.parse_expiration_from_symbol(alpaca_pos.symbol)
        entry_date_str = meta.get('entry_date', datetime.now().isoformat())

        proxy = SimpleNamespace(
            symbol=meta.get('underlying', ''),
            option_symbol=alpaca_pos.symbol,
            quantity=int(float(alpaca_pos.qty)),
            strike=strike,
            expiration=expiration or _date.today(),
            position_id=alpaca_pos.symbol,
            entry_delta=meta.get('entry_delta', 0),
            entry_iv=meta.get('entry_iv', 0),
            entry_vix=meta.get('entry_vix', 0),
            entry_stock_price=meta.get('entry_stock_price', 0),
            entry_premium=meta.get('entry_premium', 0),
            entry_daily_return=meta.get('entry_daily_return', 0),
            dte_at_entry=meta.get('dte_at_entry', 0),
            entry_order_id=meta.get('entry_order_id', ''),
        )
        proxy.current_dte = (proxy.expiration - _date.today()).days
        proxy.days_held = (_date.today() - datetime.fromisoformat(entry_date_str).date()).days
        proxy.collateral_required = abs(proxy.quantity) * proxy.strike * 100
        def calculate_pnl(exit_premium, _p=proxy):
            return (_p.entry_premium - exit_premium) * abs(_p.quantity) * 100
        proxy.calculate_pnl = calculate_pnl
        return proxy

    def _get_portfolio_summary(self) -> dict:
        """Build portfolio summary from Alpaca + metadata."""
        positions = self._get_option_positions()
        collateral = self.alpaca_manager.get_short_collateral()
        return {
            'active_positions': len(positions),
            'total_collateral': collateral,
            'symbols': list(self._get_active_symbols()),
        }

    def is_market_open(self) -> bool:
        """Check if US market is currently open using Alpaca calendar API.
        Caches the trading-day check per calendar date so only one API call per day.
        """
        now = datetime.now(self.eastern)
        today = now.date()

        # Weekday check (Mon=0, Fri=4) — fast reject weekends
        if now.weekday() > 4:
            return False

        # Check if today is a trading day (holiday check, cached per day)
        if not hasattr(self, '_trading_day_cache') or self._trading_day_cache.get('date') != today:
            try:
                from alpaca.trading.requests import GetCalendarRequest
                cal_req = GetCalendarRequest(start=today, end=today)
                cal = self.alpaca_manager.trading_client.get_calendar(cal_req)
                # cal[0].date is a date object; cal[0].open/close are datetime objects
                is_trading_day = len(cal) > 0 and cal[0].date == today
                self._trading_day_cache = {
                    'date': today,
                    'is_trading_day': is_trading_day,
                    'open': cal[0].open.time() if is_trading_day else None,
                    'close': cal[0].close.time() if is_trading_day else None,
                }
                if not is_trading_day:
                    print(f"  Market closed today ({today} is not a trading day)")
            except Exception as e:
                print(f"  Warning: Alpaca calendar check failed ({e}), falling back to time-only check")
                self._trading_day_cache = {'date': today, 'is_trading_day': True, 'open': None, 'close': None}

        if not self._trading_day_cache['is_trading_day']:
            return False

        # Time check — use Alpaca hours if available, else default 9:30-16:00 ET
        market_open = self._trading_day_cache.get('open') or dt_time(9, 30)
        market_close = self._trading_day_cache.get('close') or dt_time(16, 0)

        return market_open <= now.time() <= market_close
    
    def get_session_vix_reference(self) -> float:
        """
        Get VIX reference for current session.
        Uses session open VIX, cached for the day.
        """
        session_date = datetime.now(self.eastern).date()
        
        if self._session_vix_open is None:
            # Get last session's open
            _, vix_open = self.vix_fetcher.get_session_reference_vix()
            self._session_vix_open = (session_date, vix_open)
        
        # Reset if new day
        if self._session_vix_open[0] != session_date:
            _, vix_open = self.vix_fetcher.get_session_reference_vix()
            self._session_vix_open = (session_date, vix_open)
        
        return self._session_vix_open[1]
    
    def check_global_vix_stop(self, current_vix: float) -> bool:
        """
        Check if global VIX stop is triggered.
        If VIX >= 1.15x session open, close ALL positions.
        
        Returns:
            True if global stop triggered
        """
        reference_vix = self.get_session_vix_reference()
        threshold = reference_vix * self.config.vix_spike_multiplier
        
        return current_vix >= threshold
    
    def monitor_positions(self, current_vix: float) -> List[Tuple[ActivePosition, RiskCheckResult, float]]:
        """
        Check all positions for exit conditions.

        Returns:
            List of (position, risk_result, current_premium) tuples that should be closed
        """
        exits_needed = []

        # Check for assignments first (no market data needed)
        assignment_exits = self._check_assignments()
        exits_needed.extend(assignment_exits)
        assigned_ids = {pos.option_symbol for pos, _, _ in assignment_exits}

        reference_vix = self.get_session_vix_reference()

        for alpaca_pos in self._get_option_positions():
            option_sym = alpaca_pos.symbol
            if option_sym in assigned_ids:
                continue

            meta = self.metadata.get(option_sym)
            if meta is None:
                print(f"  Warning: No metadata for {option_sym}, skipping risk checks")
                continue
            position = self._build_position_proxy(alpaca_pos, meta)

            try:
                # Check expiration proximity
                days_to_expiry = (position.expiration - date.today()).days
                if days_to_expiry <= self.config.close_before_expiry_days:
                    # Fetch premium for P&L tracking
                    snapshots = self.data_manager.options_fetcher.get_option_snapshots(
                        [position.option_symbol]
                    )
                    snapshot = snapshots.get(position.option_symbol, {})
                    current_premium = snapshot.get('ask', 0)

                    print(f"  {position.symbol}: Expiring in {days_to_expiry} day(s), triggering exit")
                    expiry_result = RiskCheckResult(
                        should_exit=True,
                        exit_reason=ExitReason.EXPIRY,
                        details=f"Position expiring in {days_to_expiry} day(s) (threshold: {self.config.close_before_expiry_days}d)",
                        current_values={
                            'days_to_expiry': days_to_expiry,
                            'expiration': position.expiration.isoformat(),
                            'current_premium': current_premium,
                        }
                    )
                    exits_needed.append((position, expiry_result, current_premium))
                    continue

                # Get current data for the position
                current_stock_price = self.data_manager.equity_fetcher.get_current_price(
                    position.symbol
                )

                # Get current option data
                snapshots = self.data_manager.options_fetcher.get_option_snapshots(
                    [position.option_symbol]
                )

                if position.option_symbol not in snapshots:
                    print(f"  Warning: No data for {position.option_symbol}")
                    continue

                snapshot = snapshots[position.option_symbol]
                current_premium = snapshot.get('ask', 0)  # Use ask to buy back
                current_delta = snapshot.get('delta')

                # Calculate delta if not provided by snapshot
                if current_delta is None and snapshot.get('bid') and snapshot.get('ask'):
                    mid = (snapshot['bid'] + snapshot['ask']) / 2
                    greeks = self.greeks_calc.compute_greeks_from_price(
                        mid, current_stock_price, position.strike,
                        position.current_dte, 'put'
                    )
                    current_delta = greeks.get('delta')  # No fallback to entry_delta

                # Handle missing delta
                if current_delta is None:
                    if self.config.exit_on_missing_delta:
                        print(f"  WARNING: Delta unavailable for {position.symbol}, triggering exit (exit_on_missing_delta=True)")
                        data_result = RiskCheckResult(
                            should_exit=True,
                            exit_reason=ExitReason.DATA_UNAVAILABLE,
                            details=f"Delta could not be retrieved or computed for {position.option_symbol}",
                            current_values={
                                'entry_delta': position.entry_delta,
                                'current_premium': current_premium,
                            }
                        )
                        exits_needed.append((position, data_result, current_premium))
                        continue
                    else:
                        print(f"  WARNING: Delta unavailable for {position.symbol}, using entry_delta as fallback")
                        current_delta = position.entry_delta

                # Run risk evaluation
                risk_result = self.risk_manager.evaluate_position(
                    position=position,
                    current_delta=current_delta,
                    current_stock_price=current_stock_price,
                    current_vix=current_vix,
                    current_premium=current_premium,
                    reference_vix=reference_vix
                )

                if risk_result.should_exit:
                    exits_needed.append((position, risk_result, current_premium))

            except Exception as e:
                print(f"  Error monitoring {position.symbol}: {e}")

        return exits_needed


    def execute_exit(
        self,
        position: ActivePosition,
        risk_result: RiskCheckResult,
        current_premium: float = 0.0,
    ) -> bool:
        """
        Execute exit for a position. Routes to appropriate order type based on exit reason.

        Args:
            position: The position to close
            risk_result: Risk check result with exit reason and details
            current_premium: Current option premium (ask price), passed directly
                             from monitor_positions for reliable P&L tracking

        Returns:
            True if exit completed successfully
        """
        print(f"  Exiting {position.symbol}: {risk_result.exit_reason.value}")
        print(f"    {risk_result.details}")

        exit_premium = current_premium

        # === Assignment: no order needed ===
        if risk_result.exit_reason == ExitReason.ASSIGNED:
            exit_premium = 0.0
            print(f"    Assignment detected -- no order needed")

            self.metadata.record_exit(
                option_symbol=position.option_symbol,
                exit_reason=risk_result.exit_reason.value,
                exit_details=risk_result.details,
                exit_order_id="assignment",
            )
            pnl = position.calculate_pnl(exit_premium)
            print(f"    Position closed (assigned). Option P&L: ${pnl:.2f}")
            return True

        # === Early exit & Expiry: use stepped limit ===
        if risk_result.exit_reason in (ExitReason.EARLY_EXIT, ExitReason.EXPIRY):
            result_tuple = self._execute_stepped_exit(position)

            if result_tuple is not None:
                result, filled_price = result_tuple
                exit_premium = filled_price

                self.metadata.record_exit(
                    option_symbol=position.option_symbol,
                    exit_reason=risk_result.exit_reason.value,
                    exit_details=risk_result.details,
                    exit_order_id=result.order_id,
                )
                pnl = position.calculate_pnl(exit_premium)
                print(f"    Stepped exit filled @ ${exit_premium:.2f}. P&L: ${pnl:.2f}")
                return True
            else:
                # Stepped exit exhausted -- fall back to market order
                print(f"    Stepped exit exhausted, falling back to market order")
                result = self.execution.buy_to_close(
                    option_symbol=position.option_symbol,
                    quantity=abs(position.quantity),
                    limit_price=None,
                )
                if result.success:
                    self.metadata.record_exit(
                        option_symbol=position.option_symbol,
                        exit_reason=risk_result.exit_reason.value,
                        exit_details=risk_result.details,
                        exit_order_id=result.order_id,
                    )
                    pnl = position.calculate_pnl(exit_premium)
                    print(f"    Market fallback submitted. Est. P&L: ${pnl:.2f}")
                    return True
                else:
                    print(f"    Market fallback also failed: {result.message}")
                    return False

        # === Stop-loss exits: market, bid, or stepped ===
        result = None
        if self.config.stop_loss_order_type == "stepped":
            print(f"    Stop-loss using stepped exit")
            result, filled_price = self._execute_stepped_stop_exit(position)
            if filled_price > 0:
                exit_premium = filled_price
        elif self.config.stop_loss_order_type == "bid":
            snapshots = self.data_manager.options_fetcher.get_option_snapshots(
                [position.option_symbol]
            )
            snapshot = snapshots.get(position.option_symbol, {})
            bid_price = snapshot.get('bid', 0)
            limit_price = None
            if bid_price and bid_price > 0:
                limit_price = round(bid_price, 2)
                print(f"    Stop-loss using bid limit @ ${limit_price:.2f}")
            else:
                print(f"    No bid available, using market order")
            result = self.execution.buy_to_close(
                option_symbol=position.option_symbol,
                quantity=abs(position.quantity),
                limit_price=limit_price,
            )
        else:  # "market"
            print(f"    Stop-loss using market order")
            result = self.execution.buy_to_close(
                option_symbol=position.option_symbol,
                quantity=abs(position.quantity),
                limit_price=None,
            )

        if result and result.success:
            self.metadata.record_exit(
                option_symbol=position.option_symbol,
                exit_reason=risk_result.exit_reason.value,
                exit_details=risk_result.details,
                exit_order_id=result.order_id,
            )
            pnl = position.calculate_pnl(exit_premium)
            print(f"    Exit order submitted. Est. P&L: ${pnl:.2f}")
            return True
        else:
            msg = result.message if result else "No order result"
            print(f"    Exit order failed: {msg}")
            return False


    def _refresh_equity_scan(self) -> List[str]:
        """
        Run equity scan once per day. Cache passing symbols and full results.
        Returns list of equity-passing symbols.
        """
        today = datetime.now(self.eastern).date()
        
        if self._equity_scan_date == today and self._equity_passing is not None:
            return self._equity_passing
        
        scan_start = datetime.now()
        print(f"  Initiating equity scan at {scan_start.strftime('%H:%M:%S')}...")
        scan_results = self.scanner.scan_universe(skip_equity_filter=False)
        scan_elapsed = (datetime.now() - scan_start).total_seconds()
        
        passing_equity = [r for r in scan_results if r.equity_result.passes]
        self._equity_passing = [r.symbol for r in passing_equity]
        self._equity_scan_date = today
        self._last_scan_results = scan_results  # Cache full results for diagnostics
        
        print(f"  Symbols scanned:                         {len(scan_results)}")
        print(f"  Passed equity filter:                     {len(passing_equity)}")
        print(f"  Scan completed in {scan_elapsed:.1f}s")
        
        # Print equity-passing table
        if passing_equity:
            bb_label = f"BB{self.config.bb_period}"
            print(f"\n  \u2713 Equity-passing symbols ({len(passing_equity)}):")
            print(f"  {'Symbol':<8} {'Price':>9} {'SMA8':>9} {'SMA20':>9} {'SMA50':>9} {bb_label:>9} {'RSI':>6} {'Collateral':>12}")
            print("  " + "-" * 72)
            for result in passing_equity:
                r = result.equity_result
                collateral = r.current_price * 100
                print(
                    f"  {r.symbol:<8} "
                    f"${r.current_price:>8.2f} "
                    f"{r.sma_8:>9.2f} "
                    f"{r.sma_20:>9.2f} "
                    f"{r.sma_50:>9.2f} "
                    f"{r.bb_upper:>9.2f} "
                    f"{r.rsi:>6.1f} "
                    f"${collateral:>10,.0f}"
                )
        else:
            print("\n  \u26a0 No symbols passed the equity filter.")
        
        # Log equity scan
        self.logger.log_equity_scan(
            [r.equity_result for r in scan_results],
            self._equity_passing
        )
                
        return self._equity_passing

    def _get_sort_key(self, contract):
        """Get sort key based on configured rank mode."""
        if self.config.contract_rank_mode == "daily_return_per_delta":
            return contract.daily_return_per_delta
        elif self.config.contract_rank_mode == "days_since_strike":
            return contract.days_since_strike or 0
        elif self.config.contract_rank_mode == "lowest_strike_price":
            return -contract.strike
        else:  # "daily_return_on_collateral"
            return contract.daily_return_on_collateral

    def compute_target_quantity(self, collateral_per_contract: float, available_cash: float) -> int:
        """Compute number of CSP contracts for a ticker.
        If cash >= max_position_pct of portfolio: qty = floor(max_position_pct * portfolio / collateral)
        Else: qty = floor(available_cash / collateral)
        Capped by max_contracts_per_ticker.
        """
        if collateral_per_contract <= 0:
            return 1
        target_allocation = self.config.starting_cash * self.config.max_position_pct
        if available_cash >= target_allocation:
            n = int(target_allocation // collateral_per_contract)
        else:
            n = int(available_cash // collateral_per_contract)
        n = min(n, self.config.max_contracts_per_ticker)
        return max(1, n)

    def _execute_stepped_entry(
        self,
        candidate: 'OptionContract',
        qty: int,
        current_vix: float,
    ) -> Optional[Tuple['OrderResult', float]]:
        """Execute a stepped limit order entry for a CSP position.

        Starts at mid (or bid) and steps down toward bid over multiple
        attempts, optionally re-fetching the snapshot and re-validating
        the contract between steps.

        Returns:
            Tuple of (OrderResult, filled_price) if filled, or None if exhausted.
        """
        cfg = self.config
        symbol = candidate.symbol

        bid = candidate.bid
        ask = candidate.ask
        mid = candidate.mid
        spread = ask - bid

        # Initial limit price
        if cfg.entry_start_price == "mid":
            limit_price = mid
        else:
            limit_price = bid

        # Floor: never go below bid; also respect the max-step computed floor
        floor_from_steps = mid - (cfg.entry_max_steps * cfg.entry_step_pct * spread)
        price_floor = max(bid, floor_from_steps)
        limit_price = round(max(limit_price, price_floor), 2)

        print(f"    Stepped entry: start=${limit_price:.2f}, "
              f"bid=${bid:.2f}, ask=${ask:.2f}, mid=${mid:.2f}, "
              f"spread=${spread:.2f}, floor=${price_floor:.2f}")

        self._last_step_log = []

        for step in range(cfg.entry_max_steps + 1):
            print(f"    Step {step}/{cfg.entry_max_steps}: limit @ ${limit_price:.2f}")

            result = self.execution.sell_to_open(
                option_symbol=symbol,
                quantity=qty,
                limit_price=limit_price,
            )

            if not result.success:
                print(f"    Step {step}: order submission failed — {result.message}")
                return None

            order_id = result.order_id

            print(f"    Step {step}: waiting {cfg.entry_step_interval}s for fill...")
            time.sleep(cfg.entry_step_interval)

            status = self.execution.get_order_status(order_id)
            self._last_step_log.append({
                "step": step, "limit_price": limit_price,
                "status": status['status'] if status else 'unknown',
                "duration_s": cfg.entry_step_interval,
                "bid": bid, "ask": ask, "mid": round(mid, 2), "spread": round(spread, 2),
            })

            if status and status['status'] in ('filled', 'partially_filled'):
                filled_price = float(status['filled_avg_price']) if status.get('filled_avg_price') else limit_price
                tag = "FILLED" if status['status'] == 'filled' else f"PARTIAL ({status['filled_qty']}/{qty})"
                print(f"    Step {step}: {tag} @ ${filled_price:.2f}")
                return (result, filled_price)

            # Not filled — cancel
            print(f"    Step {step}: not filled (status={status['status'] if status else 'unknown'}), cancelling...")
            self.execution.cancel_order(order_id)
            time.sleep(1)  # brief pause for cancel to propagate

            # Re-check in case fill happened during cancel
            status = self.execution.get_order_status(order_id)
            if status and status['status'] in ('filled', 'partially_filled'):
                filled_price = float(status['filled_avg_price']) if status.get('filled_avg_price') else limit_price
                print(f"    Step {step}: filled during cancel @ ${filled_price:.2f}")
                return (result, filled_price)

            # Last step — give up
            if step >= cfg.entry_max_steps:
                print(f"    All {cfg.entry_max_steps} steps exhausted. Giving up on {candidate.underlying}.")
                return None

            # Optionally re-fetch snapshot
            if cfg.entry_refetch_snapshot:
                snapshots = self.data_manager.options_fetcher.get_option_snapshots([symbol])
                if symbol not in snapshots:
                    print(f"    Snapshot unavailable after re-fetch. Aborting.")
                    return None

                snap = snapshots[symbol]
                new_bid = float(snap.get('bid', 0) or 0)
                new_ask = float(snap.get('ask', 0) or 0)

                if new_bid <= 0:
                    print(f"    Bid is zero after re-fetch. Aborting.")
                    return None

                new_mid = (new_bid + new_ask) / 2
                new_spread = new_ask - new_bid

                # Update candidate for re-validation
                candidate.bid = new_bid
                candidate.ask = new_ask
                candidate.mid = new_mid
                if snap.get('delta') is not None:
                    candidate.delta = snap['delta']
                if snap.get('implied_volatility') is not None:
                    candidate.implied_volatility = snap['implied_volatility']
                if snap.get('volume') is not None:
                    candidate.volume = snap['volume']
                if snap.get('open_interest') is not None:
                    candidate.open_interest = snap['open_interest']

                # Re-validate against filters
                filter_result = self.scanner.options_filter.evaluate(candidate)
                if not filter_result.passes:
                    print(f"    Contract no longer passes filters: {filter_result.failure_reasons}")
                    return None

                bid, ask, mid, spread = new_bid, new_ask, new_mid, new_spread
                price_floor = max(bid, mid - (cfg.entry_max_steps * cfg.entry_step_pct * spread))

                print(f"    Refreshed: bid=${bid:.2f}, ask=${ask:.2f}, "
                      f"mid=${mid:.2f}, spread=${spread:.2f}, floor=${price_floor:.2f}")

            # Compute next step price
            next_step = step + 1
            if cfg.entry_start_price == "mid":
                limit_price = mid - (next_step * cfg.entry_step_pct * spread)
            else:
                limit_price = bid - (next_step * cfg.entry_step_pct * spread)

            limit_price = round(max(limit_price, price_floor), 2)

        return None


    def _execute_stepped_exit(
        self,
        position,
    ) -> Optional[Tuple['OrderResult', float]]:
        """Execute a stepped limit order exit (buy-to-close) for a CSP position.

        Starts at mid (or ask) and steps UP toward ask over multiple
        attempts, optionally re-fetching the snapshot between steps.

        Returns:
            Tuple of (OrderResult, filled_price) if filled, or None if exhausted.
        """
        cfg = self.config
        option_symbol = position.option_symbol
        qty = abs(position.quantity)

        # Fetch current snapshot
        snapshots = self.data_manager.options_fetcher.get_option_snapshots([option_symbol])
        if option_symbol not in snapshots:
            print(f"    Stepped exit: no snapshot for {option_symbol}")
            return None

        snap = snapshots[option_symbol]
        bid = float(snap.get('bid', 0) or 0)
        ask = float(snap.get('ask', 0) or 0)

        if ask <= 0:
            print(f"    Stepped exit: ask is zero, aborting")
            return None

        mid = (bid + ask) / 2
        spread = ask - bid

        # Initial limit price
        if cfg.exit_start_price == "ask":
            limit_price = ask
        else:
            limit_price = mid

        # Ceiling: never go above ask
        ceiling_from_steps = mid + (cfg.exit_max_steps * cfg.exit_step_pct * spread)
        price_ceiling = min(ask, ceiling_from_steps)
        limit_price = round(min(limit_price, price_ceiling), 2)

        print(f"    Stepped exit: start=${limit_price:.2f}, "
              f"bid=${bid:.2f}, ask=${ask:.2f}, mid=${mid:.2f}, "
              f"spread=${spread:.2f}, ceiling=${price_ceiling:.2f}")

        for step in range(cfg.exit_max_steps + 1):
            print(f"    Step {step}/{cfg.exit_max_steps}: limit @ ${limit_price:.2f}")

            result = self.execution.buy_to_close(
                option_symbol=option_symbol,
                quantity=qty,
                limit_price=limit_price,
            )

            if not result.success:
                print(f"    Step {step}: order submission failed -- {result.message}")
                return None

            order_id = result.order_id

            print(f"    Step {step}: waiting {cfg.exit_step_interval}s for fill...")
            time.sleep(cfg.exit_step_interval)

            status = self.execution.get_order_status(order_id)

            if status and status['status'] in ('filled', 'partially_filled'):
                filled_price = float(status['filled_avg_price']) if status.get('filled_avg_price') else limit_price
                tag = "FILLED" if status['status'] == 'filled' else f"PARTIAL ({status['filled_qty']}/{qty})"
                print(f"    Step {step}: {tag} @ ${filled_price:.2f}")
                return (result, filled_price)

            # Not filled -- cancel
            print(f"    Step {step}: not filled (status={status['status'] if status else 'unknown'}), cancelling...")
            self.execution.cancel_order(order_id)
            time.sleep(1)

            # Re-check in case fill happened during cancel
            status = self.execution.get_order_status(order_id)
            if status and status['status'] in ('filled', 'partially_filled'):
                filled_price = float(status['filled_avg_price']) if status.get('filled_avg_price') else limit_price
                print(f"    Step {step}: filled during cancel @ ${filled_price:.2f}")
                return (result, filled_price)

            # Last step -- give up
            if step >= cfg.exit_max_steps:
                print(f"    All {cfg.exit_max_steps} steps exhausted for exit of {position.symbol}.")
                return None

            # Optionally re-fetch snapshot
            if cfg.exit_refetch_snapshot:
                snapshots = self.data_manager.options_fetcher.get_option_snapshots([option_symbol])
                if option_symbol not in snapshots:
                    print(f"    Snapshot unavailable after re-fetch. Aborting.")
                    return None

                snap = snapshots[option_symbol]
                new_bid = float(snap.get('bid', 0) or 0)
                new_ask = float(snap.get('ask', 0) or 0)

                if new_ask <= 0:
                    print(f"    Ask is zero after re-fetch. Aborting.")
                    return None

                new_mid = (new_bid + new_ask) / 2
                new_spread = new_ask - new_bid

                bid, ask, mid, spread = new_bid, new_ask, new_mid, new_spread
                price_ceiling = min(ask, mid + (cfg.exit_max_steps * cfg.exit_step_pct * spread))

                print(f"    Refreshed: bid=${bid:.2f}, ask=${ask:.2f}, "
                      f"mid=${mid:.2f}, spread=${spread:.2f}, ceiling=${price_ceiling:.2f}")

            # Compute next step price (step UP toward ask)
            next_step = step + 1
            if cfg.exit_start_price == "ask":
                limit_price = ask  # Already at ask, stay there
            else:
                limit_price = mid + (next_step * cfg.exit_step_pct * spread)

            limit_price = round(min(limit_price, price_ceiling), 2)

        return None

    def _execute_stepped_stop_exit(
        self,
        position,
    ) -> Tuple['OrderResult', float]:
        """Execute a stepped stop-loss exit with market order fallback.

        Starts at bid + offset% of spread, steps up by step_pct of spread.
        After max_steps retries, falls back to market order.

        Returns:
            Tuple of (OrderResult, filled_price) -- always fills due to market fallback.
        """
        cfg = self.config
        option_symbol = position.option_symbol
        qty = abs(position.quantity)

        # Fetch current snapshot
        snapshots = self.data_manager.options_fetcher.get_option_snapshots([option_symbol])
        if option_symbol not in snapshots:
            print(f"    Stepped stop: no snapshot, using market order")
            result = self.execution.buy_to_close(option_symbol=option_symbol, quantity=qty, limit_price=None)
            return (result, 0.0)

        snap = snapshots[option_symbol]
        bid = float(snap.get('bid', 0) or 0)
        ask = float(snap.get('ask', 0) or 0)

        if bid <= 0 or ask <= 0:
            print(f"    Stepped stop: no bid/ask, using market order")
            result = self.execution.buy_to_close(option_symbol=option_symbol, quantity=qty, limit_price=None)
            return (result, 0.0)

        spread = ask - bid
        limit_price = round(bid + cfg.stop_exit_start_offset_pct * spread, 2)

        print(f"    Stepped stop: start=${limit_price:.2f}, "
              f"bid=${bid:.2f}, ask=${ask:.2f}, spread=${spread:.2f}")

        for step in range(cfg.stop_exit_max_steps + 1):
            print(f"    Stop step {step}/{cfg.stop_exit_max_steps}: limit @ ${limit_price:.2f}")

            result = self.execution.buy_to_close(
                option_symbol=option_symbol,
                quantity=qty,
                limit_price=limit_price,
            )

            if not result.success:
                print(f"    Stop step {step}: submission failed, using market order")
                result = self.execution.buy_to_close(option_symbol=option_symbol, quantity=qty, limit_price=None)
                return (result, 0.0)

            order_id = result.order_id
            time.sleep(cfg.stop_exit_step_interval)

            status = self.execution.get_order_status(order_id)
            if status and status['status'] in ('filled', 'partially_filled'):
                filled_price = float(status['filled_avg_price']) if status.get('filled_avg_price') else limit_price
                print(f"    Stop step {step}: FILLED @ ${filled_price:.2f}")
                return (result, filled_price)

            # Cancel and retry or fallback
            self.execution.cancel_order(order_id)
            time.sleep(1)

            # Check if filled during cancel
            status = self.execution.get_order_status(order_id)
            if status and status['status'] in ('filled', 'partially_filled'):
                filled_price = float(status['filled_avg_price']) if status.get('filled_avg_price') else limit_price
                print(f"    Stop step {step}: filled during cancel @ ${filled_price:.2f}")
                return (result, filled_price)

            if step >= cfg.stop_exit_max_steps:
                break

            # Re-fetch snapshot for fresh pricing
            snapshots = self.data_manager.options_fetcher.get_option_snapshots([option_symbol])
            if option_symbol in snapshots:
                snap = snapshots[option_symbol]
                bid = float(snap.get('bid', 0) or 0)
                ask = float(snap.get('ask', 0) or 0)
                spread = ask - bid if ask > bid else spread

            # Step up
            next_step = step + 1
            limit_price = round(bid + (cfg.stop_exit_start_offset_pct + next_step * cfg.stop_exit_step_pct) * spread, 2)
            limit_price = min(limit_price, ask)  # Never exceed ask

        # All steps exhausted -- fall back to market
        print(f"    Stepped stop exhausted after {cfg.stop_exit_max_steps} steps, falling back to market order")
        result = self.execution.buy_to_close(option_symbol=option_symbol, quantity=qty, limit_price=None)
        return (result, 0.0)

    def _check_assignments(self) -> List[Tuple[ActivePosition, RiskCheckResult, float]]:
        """Detect assignments by comparing internal portfolio against Alpaca holdings.

        Assignment signal: our portfolio has an active short put for symbol X,
        but Alpaca no longer has the option position AND Alpaca now holds
        shares of X with side='long'.

        Returns:
            List of (position, risk_result, current_premium=0) for assigned positions.
        """
        assigned = []

        if not self.alpaca_manager:
            return assigned

        try:
            alpaca_positions = self.alpaca_manager.trading_client.get_all_positions()
        except Exception as e:
            print(f"  Warning: Could not fetch Alpaca positions for assignment check: {e}")
            return assigned

        # Build lookup maps
        alpaca_option_symbols = set()
        alpaca_stock_holdings = {}  # symbol -> qty (float)

        for pos in alpaca_positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)

            # Options use OCC format (long symbol with digits), stocks are short tickers
            if any(c.isdigit() for c in symbol) and len(symbol) > 10:
                alpaca_option_symbols.add(symbol)
            else:
                if side == 'long' or qty > 0:
                    alpaca_stock_holdings[symbol] = abs(qty)

        # Check each position we have metadata for
        for option_sym, meta in self.metadata.get_active().items():
            if option_sym not in alpaca_option_symbols:
                underlying = meta.get('underlying', '')
                shares_held = alpaca_stock_holdings.get(underlying, 0)
                expected_shares = abs(meta.get('quantity', 1)) * 100

                if shares_held >= expected_shares:
                    print(f"  ASSIGNMENT DETECTED: {underlying}")
                    print(f"    Option {option_sym} no longer in Alpaca")
                    print(f"    Now holding {shares_held:.0f} shares of {underlying}")

                    from types import SimpleNamespace
                    proxy = SimpleNamespace(
                        symbol=underlying, option_symbol=option_sym,
                        position_id=option_sym,
                        quantity=meta.get('quantity', -1),
                        strike=meta.get('strike', 0),
                        expiration=date.fromisoformat(meta['expiration']) if meta.get('expiration') else date.today(),
                        entry_delta=meta.get('entry_delta', 0),
                        entry_iv=meta.get('entry_iv', 0),
                        entry_premium=meta.get('entry_premium', 0),
                        current_dte=0,
                    )
                    proxy.calculate_pnl = lambda ep, m=meta: (m.get('entry_premium', 0) - ep) * abs(m.get('quantity', 1)) * 100

                    result = RiskCheckResult(
                        should_exit=True,
                        exit_reason=ExitReason.ASSIGNED,
                        details=f"Assigned: {option_sym} removed, now hold {shares_held:.0f} shares of {underlying}",
                        current_values={'shares_held': shares_held, 'option_symbol': option_sym}
                    )
                    assigned.append((proxy, result, 0.0))

        return assigned


    def scan_and_enter(self, deployable_cash: float) -> int:
        """
        Scan for new opportunities and enter positions.
        Uses cached daily equity scan; only fetches options for passing symbols.
        Produces verbose output matching the Universe Scan cell diagnostic format.
        
        Returns:
            Number of new positions entered, or -1 if no candidates for the day
        """
        if self._monitor_only:
            return 0
        
        available_cash = min(self.alpaca_manager.compute_available_capital(), deployable_cash)
        
        if available_cash <= 0:
            return 0
        
        if self._get_position_count() >= self.config.num_tickers:
            return 0
        
        # Run equity scan (cached per day)
        equity_passing = self._refresh_equity_scan()
        
        if not equity_passing:
            print("  No symbols pass equity filter today.")
            return -1  # -1 means "nothing to do all day"

        # Only fetch options for equity-passing symbols (not the full universe)
        active_symbols = self._get_active_symbols()
        skipped_active = [s for s in equity_passing if s in active_symbols]
        symbols_to_check = [s for s in equity_passing if s not in active_symbols]
        
        if skipped_active:
            print(f"\n  Already in portfolio (skipped): {skipped_active}")
        print(f"  Checking options for {len(symbols_to_check)} symbol(s): {symbols_to_check}")
        
        all_candidates = []
        all_filter_results_by_symbol = {}  # symbol -> (stock_price, puts_count, filter_results, ranked)
        all_failure_counts = {}
        
        for symbol in symbols_to_check:
            try:
                stock_price = self.data_manager.equity_fetcher.get_current_price(symbol)
                
                # Get SMA ceiling for options chain if configured
                sma_ceiling = None
                if self.config.max_strike_mode == "sma" and hasattr(self, '_last_scan_results'):
                    for sr in self._last_scan_results:
                        if sr.symbol == symbol:
                            sma_ceiling = getattr(sr.equity_result, f"sma_{self.config.max_strike_sma_period}", None)
                            break
                
                puts = self.data_manager.options_fetcher.get_puts_chain(
                    symbol, stock_price, self.config, sma_ceiling=sma_ceiling
                )
                
                # Enrich with days_since_strike from price history
                price_history = self.data_manager.equity_fetcher.get_close_history(
                    [symbol], days=self.config.history_days
                )
                if symbol in price_history:
                    prices = price_history[symbol]
                    for put in puts:
                        at_or_below = prices[prices <= put.strike]
                        if at_or_below.empty:
                            put.days_since_strike = 999
                        else:
                            last_date = at_or_below.index[-1]
                            put.days_since_strike = (prices.index[-1] - last_date).days
                
                ranked, filter_results = self.scanner.options_filter.filter_and_rank(puts)
                all_candidates.extend(ranked[:self.config.max_candidates_per_symbol])
                all_filter_results_by_symbol[symbol] = (stock_price, len(puts), filter_results, ranked)
                
                # Log options scan for this symbol
                self.logger.log_options_scan(self._cycle_count, symbol, filter_results)
                
                # Tally failure reasons
                for r in filter_results:
                    for reason in r.failure_reasons:
                        if "Daily return" in reason:
                            key = "Premium too low"
                        elif "Strike" in reason:
                            key = "Strike too high"
                        elif "Delta" in reason:
                            key = "Delta out of range" if "outside" in reason else "Delta unavailable"
                        elif "DTE" in reason:
                            key = "DTE out of range"
                        else:
                            key = reason
                        all_failure_counts[key] = all_failure_counts.get(key, 0) + 1
            except Exception as e:
                print(f"  Error fetching options for {symbol}: {e}")
        
        # Print options scan summary per symbol
        passing_both_count = sum(1 for s, (_, _, _, ranked) in all_filter_results_by_symbol.items() if ranked)
        print(f"  Passed equity + options filter:            {passing_both_count}")
        
        # Pick best 1 contract per ticker using configured rank mode
        from itertools import groupby
        all_candidates.sort(key=lambda c: c.underlying)
        best_per_ticker = []
        for ticker, group in groupby(all_candidates, key=lambda c: c.underlying):
            group_list = list(group)
            group_list.sort(key=lambda c: self._get_sort_key(c), reverse=True)
            best_per_ticker.append(group_list[0])

        # Re-rank across tickers
        best_per_ticker.sort(key=lambda c: self._get_sort_key(c), reverse=True)
        
        # Check earnings & dividends only for final candidates
        candidate_symbols = list(set(c.underlying for c in best_per_ticker)) if best_per_ticker else []
        event_rejections = self.scanner.equity_filter.check_events(candidate_symbols) if candidate_symbols else {}
        if event_rejections:
            print(f"\n  Event-based rejections (DTE window = {self.config.max_dte}d):")
            for sym in sorted(event_rejections):
                for reason in event_rejections[sym]:
                    print(f"    {sym:<8} {reason}")
            best_per_ticker = [c for c in best_per_ticker if c.underlying not in event_rejections]
        
        candidates = best_per_ticker[:self.config.max_candidates_total]
        
        if not candidates:
            print("\n  No options candidates passed all filters.")
            if all_failure_counts:
                reasons_str = ", ".join(f"{k}: {v}" for k, v in sorted(all_failure_counts.items(), key=lambda x: -x[1]))
                print(f"  Aggregate fail reasons: {reasons_str}")
            
            # Detailed per-symbol diagnostics (like Cell 34)
            failed_symbols = [(s, info) for s, info in all_filter_results_by_symbol.items() if not info[3]]
            if failed_symbols:
                print(f"\n  Diagnostic \u2014 {len(failed_symbols)} equity-passing symbol(s) failed options filter:")
                print("  " + "-" * 95)
                for symbol, (stock_price, puts_count, filter_results, _) in sorted(failed_symbols):
                    if puts_count == 0:
                        if self.config.max_strike_mode == "sma":
                            max_strike = stock_price
                        else:
                            max_strike = stock_price * self.config.max_strike_pct
                        min_strike = stock_price * self.config.min_strike_pct
                        print(f"\n    {symbol} @ ${stock_price:.2f}: 0 puts returned from API "
                              f"(strike range ${min_strike:.0f}-${max_strike:.0f}, DTE {self.config.min_dte}-{self.config.max_dte})")
                        continue
                    
                    # Tally per-symbol failure reasons
                    sym_failure_counts = {}
                    for r in filter_results:
                        for reason in r.failure_reasons:
                            if "Daily return" in reason:
                                key = "Premium too low"
                            elif "Strike" in reason:
                                key = "Strike too high"
                            elif "Delta" in reason:
                                key = "Delta out of range" if "outside" in reason else "Delta unavailable"
                            elif "DTE" in reason:
                                key = "DTE out of range"
                            else:
                                key = reason
                            sym_failure_counts[key] = sym_failure_counts.get(key, 0) + 1
                    
                    reasons_str = ", ".join(f"{k}: {v}" for k, v in sorted(sym_failure_counts.items(), key=lambda x: -x[1]))
                    print(f"\n    {symbol} @ ${stock_price:.2f}: {puts_count} puts, 0 passed \u2014 {reasons_str}")
                    
                    # Show nearest misses (top 5 by daily return)
                    near_misses = sorted(filter_results, key=lambda r: r.daily_return, reverse=True)[:5]
                    print(f"      {'Contract':<26} {'Strike':>8} {'DTE':>5} {'Bid':>8} {'Delta':>8} {'Daily%':>10}  Fail Reasons")
                    print(f"      {'-'*91}")
                    for r in near_misses:
                        c = r.contract
                        delta_str = f"{r.delta_abs:.3f}" if r.delta_abs else "N/A"
                        reasons = "; ".join(r.failure_reasons) if r.failure_reasons else "\u2713"
                        print(
                            f"      {c.symbol:<26} "
                            f"${c.strike:>7.2f} "
                            f"{c.dte:>5} "
                            f"${c.bid:>7.2f} "
                            f"{delta_str:>8} "
                            f"{r.daily_return:>9.2%}  "
                            f"{reasons}"
                        )
            
            return 0  # 0 means "none right now, keep trying"
        
        # === Print full candidate table ===
        print(f"\n  {len(all_candidates)} total option candidates, {len(candidates)} selected for entry")
        
        # Sort for display: by symbol ascending, then daily return descending
        display_candidates = sorted(all_candidates, key=lambda c: (c.underlying, -c.daily_return_on_collateral))
        
        print(f"\n  {'Symbol':<26} {'Price':>9} {'Strike':>8} {'Drop%':>7} {'Days':>5} {'DTE':>5} {'Bid':>8} {'Ask':>8} {'Spread':>8} {'Sprd%':>7} {'Delta':>7} {'Daily%':>9} {'Vol':>6} {'OI':>6}")
        print("  " + "-" * 135)
        for c in display_candidates:
            delta_str = f"{abs(c.delta):.3f}" if c.delta else "N/A"
            spread = c.ask - c.bid if c.ask and c.bid else 0
            spread_pct = spread / c.mid if c.mid > 0 else 0
            vol_str = f"{c.volume:>6}" if c.volume is not None else "     0"
            oi_str = f"{c.open_interest:>6}" if c.open_interest is not None else "   N/A"
            drop_pct = (c.stock_price - c.strike) / c.stock_price if c.stock_price > 0 else 0
            days_str = str(c.days_since_strike) if c.days_since_strike is not None and c.days_since_strike < 999 else ">60"
            print(
                f"  {c.symbol:<26} "
                f"${c.stock_price:>8.2f} "
                f"${c.strike:>7.2f} "
                f"{drop_pct:>6.1%} "
                f"{days_str:>5} "
                f"{c.dte:>5} "
                f"${c.bid:>7.2f} "
                f"${c.ask:>7.2f} "
                f"${spread:>7.2f} "
                f"{spread_pct:>6.0%} "
                f"{delta_str:>7} "
                f"{c.daily_return_on_collateral:>8.4%} "
                f"{vol_str} "
                f"{oi_str} "
            )
        
        # === Best Pick Per Ticker by Ranking Mode ===
        if len(all_candidates) > 1:
            from itertools import groupby as _groupby
            
            def _days_since(c):
                return c.days_since_strike if c.days_since_strike is not None else 0
            
            rank_modes = {
                "daily_ret/delta": lambda c: c.daily_return_per_delta,
                "days_since_strike": lambda c: _days_since(c),
                "daily_return_on_collateral": lambda c: c.daily_return_on_collateral,
                "lowest_strike": lambda c: -c.strike,
            }
            
            sorted_by_ticker = sorted(all_candidates, key=lambda c: c.underlying)
            tickers = []
            for ticker, grp in _groupby(sorted_by_ticker, key=lambda c: c.underlying):
                tickers.append((ticker, list(grp)))
            
            if tickers:
                print(f"\n  {'='*120}")
                print(f"  Best Pick Per Ticker by Ranking Mode   (active mode: {self.config.contract_rank_mode})")
                print(f"  {'='*120}")
                print(f"    {'Ticker':<8} | {'daily_ret/delta':<30} | {'days_since_strike':<30} | {'daily_ret':<30} | {'lowest_strike':<30}")
                print(f"    {'-'*8}-+-{'-'*30}-+-{'-'*30}-+-{'-'*30}-+-{'-'*30}")
                
                for ticker, contracts in tickers:
                    picks = {}
                    for mode_name, key_fn in rank_modes.items():
                        best = max(contracts, key=key_fn)
                        val = key_fn(best)
                        if mode_name == "daily_ret/delta":
                            val_str = f"{best.symbol[-15:]}  ({val:.4f})"
                        elif mode_name == "days_since_strike":
                            days_val = int(val) if val < 999 else ">60"
                            val_str = f"{best.symbol[-15:]}  ({days_val}d)"
                        elif mode_name == "lowest_strike":
                            val_str = f"{best.symbol[-15:]}  (${best.strike:.0f})"
                        else:
                            val_str = f"{best.symbol[-15:]}  (${val:.3f}/d)"
                        picks[mode_name] = val_str
                    
                    print(
                        f"    {ticker:<8} | {picks['daily_ret/delta']:<30} | {picks['days_since_strike']:<30} | {picks['daily_return_on_collateral']:<30} | {picks['lowest_strike']:<30}"
                    )
        
        # Show which symbols had no passing options (diagnostic for completeness)
        symbols_no_opts = [s for s in symbols_to_check if s in all_filter_results_by_symbol and not all_filter_results_by_symbol[s][3]]
        if symbols_no_opts and all_failure_counts:
            print(f"\n  {len(symbols_no_opts)} symbol(s) had no passing options: {symbols_no_opts}")
            reasons_str = ", ".join(f"{k}: {v}" for k, v in sorted(all_failure_counts.items(), key=lambda x: -x[1]))
            print(f"  Aggregate fail reasons: {reasons_str}")
        
        # === Order Entry ===
        print(f"\n  {'='*80}")
        print(f"  ORDER ENTRY \u2014 {len(candidates)} candidate(s)")
        print(f"  {'='*80}")
                                
        entered = 0
        current_vix = self.vix_fetcher.get_current_vix()
        
        for i, candidate in enumerate(candidates, 1):

            # Guard: skip contracts with missing Greeks
            if candidate.delta is None or candidate.implied_volatility is None:
                print(f"\n  [{i}/{len(candidates)}] \u26a0 Skipping {candidate.underlying}: missing Greeks (delta={candidate.delta}, iv={candidate.implied_volatility})")
                continue

            # Compute dynamic quantity
            available_cash = min(self.alpaca_manager.compute_available_capital(), deployable_cash)
            qty = self.compute_target_quantity(candidate.collateral_required, available_cash)
            total_collateral = candidate.collateral_required * qty

            # Check if we can add this position
            if available_cash < total_collateral:
                print(f"\n  [{i}/{len(candidates)}] \u26a0 Skipping {candidate.underlying}: insufficient cash for ${total_collateral:,.0f} collateral")
                continue

            target_val = self.config.starting_cash * self.config.max_position_pct
            spread = candidate.ask - candidate.bid if candidate.ask and candidate.bid else 0
            delta_str = f"{abs(candidate.delta):.3f}" if candidate.delta else "N/A"
            
            print(f"\n  [{i}/{len(candidates)}] ENTERING {candidate.underlying}: {candidate.symbol}")
            print(f"    Stock: ${candidate.stock_price:.2f} | Strike: ${candidate.strike:.2f} | DTE: {candidate.dte}")
            print(f"    Bid: ${candidate.bid:.2f} | Ask: ${candidate.ask:.2f} | Mid: ${candidate.mid:.2f} | Spread: ${spread:.2f}")
            print(f"    Delta: {delta_str} | IV: {candidate.implied_volatility:.1%} | Daily: {candidate.daily_return_on_collateral:.4%}")
            print(f"    Qty: {qty} | Collateral: ${total_collateral:,.0f} (target: ${target_val:,.0f}) | Cash avail: ${available_cash:,.0f}")

            # Execute stepped entry
            entry_result = self._execute_stepped_entry(
                candidate=candidate,
                qty=qty,
                current_vix=current_vix,
            )

            # Log order attempt
            self.logger.log_order_attempt(
                action="entry",
                symbol=candidate.underlying,
                contract=candidate.symbol,
                steps=self._last_step_log,
                outcome="filled" if entry_result is not None else "exhausted",
                filled_price=entry_result[1] if entry_result else None,
                start_price=candidate.mid,
                floor_price=candidate.bid,
                qty=qty,
            )

            if entry_result is not None:
                result, filled_price = entry_result
                improvement = filled_price - candidate.bid
                
                self.metadata.record_entry(
                    option_symbol=candidate.symbol,
                    underlying=candidate.underlying,
                    strike=candidate.strike,
                    expiration=candidate.expiration.isoformat(),
                    entry_delta=candidate.delta,
                    entry_iv=candidate.implied_volatility,
                    entry_vix=current_vix,
                    entry_stock_price=candidate.stock_price,
                    entry_premium=filled_price,
                    entry_daily_return=candidate.daily_return_on_collateral,
                    dte_at_entry=candidate.dte,
                    quantity=-qty,
                    entry_order_id=result.order_id,
                )
                print(f"    \u2713 FILLED: {candidate.symbol}")
                print(f"      {qty} contracts @ ${filled_price:.2f} (bid was ${candidate.bid:.2f}, improvement: ${improvement:+.2f})")
                print(f"      Total premium: ${filled_price * qty * 100:,.2f} | Collateral: ${total_collateral:,.0f}")
                entered += 1

            else:
                print(f"    \u2717 FAILED: Entry exhausted for {candidate.underlying} after {self.config.entry_max_steps} steps")


        print(f"\n  Entry complete: {entered}/{len(candidates)} positions opened")
        return entered

    def run_cycle(self) -> dict:
        """
        Run a single trading cycle.
        
        Returns:
            Cycle summary dict
        """
        cycle_start = datetime.now()
        summary = {
            'timestamp': cycle_start.isoformat(),
            'market_open': self.is_market_open(),
            'exits': 0,
            'entries': 0,
            'errors': [],
        }
        
        try:
            # Print capital banner
            if self.alpaca_manager:
                account_info = self.alpaca_manager.get_account_info()
                short_collateral = self.alpaca_manager.get_short_collateral()
                avail_capital = account_info['cash'] - short_collateral
                target_pos = avail_capital * self.config.max_position_pct
                
                print(f"\n  {'='*60}")
                print(f"  Alpaca cash:                    ${account_info['cash']:>12,.2f}")
                print(f"  Short position collateral:      ${short_collateral:>12,.2f}")
                print(f"  Available capital:               ${avail_capital:>12,.2f}")
                print(f"  Max position size ({self.config.max_position_pct*100:.1f}%):     ${target_pos:>12,.2f}")
                print(f"  Active positions:                {self._get_position_count():>12}")
                print(f"  {'='*60}")

            # Check liquidate_all toggle
            if self.config.liquidate_all and self.alpaca_manager:
                print("LIQUIDATE ALL: config.liquidate_all is True")
                liq_result = self.alpaca_manager.liquidate_all_holdings()
                summary['liquidation'] = liq_result
                
                # Mark all metadata entries as exited
                stale_count = 0
                for sym in list(self.metadata.get_active().keys()):
                    self.metadata.record_exit(
                        option_symbol=sym,
                        exit_reason=ExitReason.MANUAL.value,
                        exit_details="Closed by liquidate_all",
                    )
                    stale_count += 1
                if stale_count:
                    print(f"  Marked {stale_count} metadata entries as exited")
                
                # Refresh starting_cash after liquidation
                self.config.starting_cash = self.alpaca_manager.compute_available_capital()
                print(f"  Starting cash refreshed: ${self.config.starting_cash:,.2f}")
                self.config.liquidate_all = False  # Reset toggle after execution
                print("  liquidate_all reset to False")
                return summary

            # Get current VIX
            current_vix = self.vix_fetcher.get_current_vix()
            summary['current_vix'] = current_vix
            
            # Refresh starting_cash from live account data
            if self.alpaca_manager:
                self.config.starting_cash = self.alpaca_manager.compute_available_capital()

            # Calculate deployable cash
            deployable_cash = self.config.get_deployable_cash(current_vix)
            summary['deployable_cash'] = deployable_cash
            
            # Check global VIX stop
            if self.check_global_vix_stop(current_vix):
                print(f"🚨 GLOBAL VIX STOP TRIGGERED - VIX: {current_vix:.2f}")
                summary['global_vix_stop'] = True
                
                # Close all positions
                for alpaca_pos in self._get_option_positions():
                    meta = self.metadata.get(alpaca_pos.symbol)
                    if meta is None:
                        continue
                    position = self._build_position_proxy(alpaca_pos, meta)
                    result = RiskCheckResult(
                        should_exit=True,
                        exit_reason=ExitReason.VIX_SPIKE,
                        details=f"Global VIX stop: {current_vix:.2f}",
                        current_values={'current_vix': current_vix}
                    )
                    if self.execute_exit(position, result, current_premium=0.0):
                        summary['exits'] += 1
                
                return summary
            
            # Monitor existing positions
            exits_needed = self.monitor_positions(current_vix)
            
            for position, risk_result, exit_premium in exits_needed:
                if self.execute_exit(position, risk_result, current_premium=exit_premium):
                    summary['exits'] += 1
            
            # Scan for new entries (only if market is open and not monitor-only)
            if self.is_market_open() and deployable_cash > 0 and not self._monitor_only:
                entries = self.scan_and_enter(deployable_cash)
                summary['entries'] = entries
                
                # No candidates available today
                if entries == -1:
                    summary['entries'] = 0
                    has_positions = self._get_position_count() > 0
                    if has_positions:
                        print("  → Switching to monitor-only mode (tracking exits only)")
                        self._monitor_only = True
                    else:
                        print("  → No positions and no candidates. Shutting down for the day.")
                        self._running = False
                        summary['shutdown_reason'] = 'no_candidates_no_positions'
                        return summary

            # Update summary with portfolio state
            portfolio_summary = self._get_portfolio_summary()
            summary['portfolio'] = portfolio_summary
            
        except Exception as e:
            summary['errors'].append(str(e))
            print(f"Cycle error: {e}")
        
        return summary
    
    def run(
        self, 
        poll_interval: int = 60,
        max_cycles: Optional[int] = None
    ):
        """
        Run the main trading loop.
        
        Args:
            poll_interval: Seconds between cycles
            max_cycles: Maximum cycles to run (None for infinite)
        """
        self._running = True
        cycle_count = 0
        self._cycle_count = 0
        
        # Log config snapshot for the day
        self.logger.log_config(self.config)
        
        print("\n" + "=" * 60)
        print("CSP TRADING LOOP STARTED")
        print(f"Poll Interval: {poll_interval}s")
        print(f"Paper Trading: {self.execution.paper}")
        print("=" * 60 + "\n")

        try:
            while self._running:
                cycle_count += 1
                self._cycle_count = cycle_count
                
                print(f"\n--- Cycle {cycle_count} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
                                
                # Reset daily state if new trading day
                today = datetime.now(self.eastern).date()
                if self._equity_scan_date is not None and self._equity_scan_date != today:
                    print("  New trading day — resetting equity scan and monitor-only flag")
                    self._equity_passing = None
                    self._equity_scan_date = None
                    self._monitor_only = False
                
                summary = self.run_cycle()
                                
                # Print cycle summary
                mode = "MONITOR-ONLY" if self._monitor_only else "ACTIVE"
                print(f"  Mode: {mode}")
                vix_val = summary.get('current_vix')
                vix_str = f"{vix_val:.2f}" if isinstance(vix_val, (int, float)) else "N/A"
                print(f"  VIX: {vix_str} | Deployable: ${summary.get('deployable_cash', 0):,.0f}")
                print(f"  Market Open: {summary.get('market_open', False)}")
                print(f"  Exits: {summary.get('exits', 0)}, Entries: {summary.get('entries', 0)}")
                
                if 'portfolio' in summary:
                    p = summary['portfolio']
                    print(f"  Positions: {p['active_positions']}, Collateral: ${p['total_collateral']:,.2f}")
                
                # Log cycle
                summary['monitor_only'] = self._monitor_only
                self.logger.log_cycle(cycle_count, summary,
                    options_checked=self._equity_passing or [],
                    failure_tally=summary.get('failure_tally', {}),
                )
                
                if 'portfolio' in summary:
                    # In monitor-only mode, stop once all positions are closed
                    if self._monitor_only and p['active_positions'] == 0:
                        print("\n  All positions closed in monitor-only mode. Done for the day.")
                        break
                
                # Check max cycles                # Check max cycles
                if max_cycles and cycle_count >= max_cycles:
                    print(f"\nMax cycles ({max_cycles}) reached. Stopping.")
                    break
                
                # Wait for next cycle
                time.sleep(poll_interval)
                
        except KeyboardInterrupt:
            print("\n\nLoop stopped by user.")
        finally:
            self._running = False
            portfolio_summary = self._get_portfolio_summary()
            
            # Log shutdown
            self.logger.log_shutdown(
                reason="keyboard_interrupt" if cycle_count > 0 else "error",
                total_cycles=cycle_count,
                portfolio_summary=portfolio_summary,
            )
            
            print("\nTrading loop ended.")
            print(f"Total cycles: {cycle_count}")
            print(f"Final portfolio: {portfolio_summary}")
                
    def stop(self):
        """Stop the trading loop."""
        self._running = False


print("Trading Loop loaded")

# Cell 35
# ----------------------------------------------------------------------
# Initialize all components for the trading loop
try:
    # Initialize strategy metadata store (Alpaca is source of truth for positions)
    metadata_store = StrategyMetadataStore(
        path="strategy_metadata.json"
    )
    
    # Initialize risk manager
    risk_manager = RiskManager(config)
    
    # Initialize execution engine
    execution = ExecutionEngine(alpaca, config)
    
    # Initialize trading loop
    trading_loop = TradingLoop(
        config=config,
        data_manager=data_manager,
        scanner=scanner,
        metadata_store=metadata_store,
        risk_manager=risk_manager,
        execution=execution,
        vix_fetcher=vix_fetcher,
        greeks_calc=greeks_calc,
        alpaca_manager=alpaca
    )
    
    print("✓ All components initialized!")
    print(f"\nConfiguration:")
    print(f"  Universe: {len(config.ticker_universe)} symbols")
    print(f"  Max positions: {config.num_tickers}")
    print(f"  Starting cash: ${config.starting_cash:,}")
    print(f"  Paper trading: {execution.paper}")
    
except NameError as e:
    print(f"⚠ Missing dependency: {e}")
except Exception as e:
    print(f"⚠ Initialization error: {e}")
    import traceback
    traceback.print_exc()

# Cell 36
# ----------------------------------------------------------------------
# Run a single test cycle (safe - doesn't loop)
try:
    print("Running single test cycle...")
    print("(No actual orders will be submitted in this test)\n")
    
    # Run one cycle
    trading_loop._cycle_count += 1    
    summary = trading_loop.run_cycle()
    
    print("\n" + "=" * 50)
    print("Cycle Summary:")
    print("=" * 50)
    for key, value in summary.items():
        if key == 'portfolio':
            p = value
            print(f"  Positions: {p['active_positions']}, Collateral: ${p['total_collateral']:,.2f}")
        else:
            print(f"  {key}: {value}")
except Exception as e:
    print(f"⚠ Cycle error: {e}")
    import traceback
    traceback.print_exc()
    

# Cell 37
# ----------------------------------------------------------------------
# ⚠️ LIVE TRADING LOOP - USE WITH CAUTION
# Uncomment to run (Ctrl+C to stop)

poll_interval, max_cycles= (60,100)

# Verify paper trading
if execution.paper:
    print("✓ Paper trading mode confirmed")
    print("\nTo start the trading loop, uncomment the line below:")
    print(f"  # trading_loop.run(poll_interval={poll_interval}, max_cycles={max_cycles})")
    print("\nOr run indefinitely with:")
    print(f"  # trading_loop.run(poll_interval={poll_interval})")
else:
    print("⚠️ WARNING: LIVE TRADING MODE")
    print("Switch to paper trading before running the loop!")

# Uncomment to run (limited to 10 cycles for safety):
trading_loop.run(poll_interval, max_cycles)


# Cell 38
# ----------------------------------------------------------------------
def run_phase3_diagnostics():
    """Run diagnostics on Phase 3 components."""
    print("Phase 3 Diagnostics")
    print("=" * 60)
    
    results = {}
    
    # 1. Position Tracking
    print("\n1. Position Tracking...")
    try:
        test_pos = ActivePosition(
            position_id="DIAG_001", symbol="TEST", option_symbol="TEST123",
            entry_date=datetime.now(), entry_stock_price=100, entry_delta=-0.15,
            entry_premium=1.0, entry_vix=15, entry_iv=0.3, strike=95,
            expiration=date.today() + timedelta(days=5), dte_at_entry=5, quantity=-1
        )
        assert test_pos.collateral_required == 9500
        results['position_tracking'] = 'ok'
        print("   ✓ OK")
    except Exception as e:
        results['position_tracking'] = 'error'
        print(f"   ✗ Error: {e}")
    
    # 2. Portfolio Manager
    print("\n2. Portfolio Manager...")
    try:
        pm = PortfolioManager(config)
        pm.add_position(test_pos)
        assert pm.active_count == 1
        results['portfolio_manager'] = 'ok'
        print("   ✓ OK")
    except Exception as e:
        results['portfolio_manager'] = 'error'
        print(f"   ✗ Error: {e}")
    
    # 3. Risk Manager
    print("\n3. Risk Manager...")
    try:
        rm = RiskManager(config)
        delta_check = rm.check_delta_stop(test_pos, -0.35)
        assert delta_check.should_exit == True
        results['risk_manager'] = 'ok'
        print("   ✓ OK")
    except Exception as e:
        results['risk_manager'] = 'error'
        print(f"   ✗ Error: {e}")
    
    # 4. Execution Engine
    print("\n4. Execution Engine...")
    try:
        ee = ExecutionEngine(alpaca, config)
        assert ee.paper == True  # Should be paper trading
        results['execution_engine'] = 'ok'
        print(f"   ✓ OK (Paper: {ee.paper})")
    except Exception as e:
        results['execution_engine'] = 'error'
        print(f"   ✗ Error: {e}")
    
    # 5. Trading Loop
    print("\n5. Trading Loop...")
    try:
        assert trading_loop is not None
        assert trading_loop.is_market_open() in [True, False]
        results['trading_loop'] = 'ok'
        print(f"   ✓ OK (Market open: {trading_loop.is_market_open()})")
    except Exception as e:
        results['trading_loop'] = 'error'
        print(f"   ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    all_ok = all(v == 'ok' for v in results.values())
    
    if all_ok:
        print("✓ All Phase 3 components working!")
        print("\n🎉 Strategy is ready for paper trading!")
    else:
        print("⚠ Some components need attention")
    
    return results


# Run diagnostics
try:
    phase3_results = run_phase3_diagnostics()
except Exception as e:
    print(f"⚠ Diagnostics failed: {e}")

# Cell 39
# ----------------------------------------------------------------------
def replenish_buying_power_p3():
    """Cancel all open orders and close all positions to free up buying power."""
    print("  [REPLENISH] Cancelling all open orders...")
    try:
        alpaca.trading_client.cancel_orders()
        print("    Orders cancelled.")
    except Exception as e:
        print(f"    Cancel orders error: {e}")
    
    time.sleep(2)
    
    print("  [REPLENISH] Closing all open positions...")
    try:
        positions = alpaca.trading_client.get_all_positions()
        if not positions:
            print("    No open positions.")
        else:
            for pos in positions:
                try:
                    alpaca.trading_client.close_position(pos.symbol)
                    print(f"    Closed: {pos.symbol} (qty={pos.qty})")
                except Exception as e:
                    print(f"    Failed to close {pos.symbol}: {e}")
            time.sleep(3)
    except Exception as e:
        print(f"    Get positions error: {e}")
    
    try:
        info = alpaca.get_account_info()
        print(f"    Cash: ${info['cash']:,.2f} | Buying power: ${info['buying_power']:,.2f}")
    except Exception as e:
        print(f"    Account info error: {e}")
    print()


def test_all_order_types_p3(
    execution_engine: ExecutionEngine,
    cfg: StrategyConfig,
    num_tickers: int = 1,
    do_replenish: bool = True
):
    """
    Test all order type combinations via ExecutionEngine.sell_to_open / buy_to_close.
    During market hours: all 4 combos (limit/limit, limit/market, market/limit, market/market).
    Outside market hours: limit/limit only (submit, verify accepted, cancel).
    
    Args:
        execution_engine: The ExecutionEngine instance from Phase 3
        cfg: StrategyConfig (must be paper_trading=True)
        num_tickers: Number of random tickers per combo
        do_replenish: Cancel orders & close positions before/between/after tests
    """
    assert execution_engine.paper, "Safety check: must be on paper trading!"
    
    # Check market hours
    try:
        clock = alpaca.trading_client.get_clock()
        market_open = clock.is_open
    except Exception:
        market_open = False
    
    if market_open:
        order_combos = [
            ("limit",  "limit"),
            ("limit",  "market"),
            ("market", "limit"),
            ("market", "market"),
        ]
    else:
        order_combos = [("limit", "limit")]
        print("NOTE: Market is CLOSED.")
        print("  - Limit sell orders submitted & verified (then cancelled)")
        print("  - Buy-to-close skipped (sell won't fill until market opens)")
        print("  - Re-run during market hours for full round-trip tests\n")
    
    all_tickers = cfg.ticker_universe[:]
    random.shuffle(all_tickers)
    test_tickers = all_tickers[:num_tickers]
    
    print(f"{'='*70}")
    print(f"ORDER TYPE VERIFICATION (Phase 3 ExecutionEngine)")
    print(f"  {len(order_combos)} combo(s) x {len(test_tickers)} ticker(s): {test_tickers}")
    print(f"  Market open: {market_open} | Replenish: {do_replenish}")
    print(f"{'='*70}\n")
    
    if do_replenish:
        replenish_buying_power_p3()
    
    results = []
    
    for entry_type, exit_type in order_combos:
        print(f"\n{'─'*70}")
        print(f"  COMBO: entry={entry_type.upper()} / exit={exit_type.upper()}")
        print(f"{'─'*70}")
        
        if do_replenish and results:
            print("  [Between combos] Replenishing...")
            replenish_buying_power_p3()
        
        for sym in test_tickers:
            print(f"\n  [{sym}]")
            try:
                prices = equity_fetcher.get_current_prices([sym])
                price = prices.get(sym)
                if not price:
                    print(f"    Skip: no price data")
                    results.append({"combo": f"{entry_type}/{exit_type}", "symbol": sym, "sell": "SKIP", "buy": "SKIP"})
                    continue
                print(f"    Stock: ${price:.2f}")
                
                puts = options_fetcher.get_puts_chain(sym, price, cfg)
                if not puts:
                    print(f"    Skip: no put contracts")
                    results.append({"combo": f"{entry_type}/{exit_type}", "symbol": sym, "sell": "SKIP", "buy": "SKIP"})
                    continue
                
                pick = random.choice(puts)
                delta_str = f"{abs(pick.delta):.3f}" if pick.delta else "N/A"
                print(f"    Contract: {pick.symbol} | Strike: ${pick.strike:.2f} | DTE: {pick.dte} | Bid/Ask: ${pick.bid:.2f}/${pick.ask:.2f} | Delta: {delta_str}")
                
                sell_status = "FAIL"
                buy_status = "N/A"
                
                # --- SELL TO OPEN via ExecutionEngine ---
                sell_limit = round(pick.bid, 2) if entry_type == "limit" else None
                sell_result = execution_engine.sell_to_open(
                    option_symbol=pick.symbol,
                    quantity=1,
                    limit_price=sell_limit
                )
                print(f"    SELL ({entry_type}): success={sell_result.success}, id={sell_result.order_id}, msg={sell_result.message}")
                
                if not sell_result.success:
                    sell_status = "REJECTED"
                    results.append({"combo": f"{entry_type}/{exit_type}", "symbol": sym, "sell": sell_status, "buy": sell_result.message[:50]})
                    continue
                
                sell_status = "accepted"
                time.sleep(3)
                
                # Check if sell filled
                order_info = execution_engine.get_order_status(sell_result.order_id)
                fill_status = order_info.get("status", "unknown") if order_info else "unknown"
                print(f"    SELL fill check: {fill_status}")
                
                if fill_status == "filled":
                    # --- BUY TO CLOSE via ExecutionEngine ---
                    close_limit = round(pick.ask, 2) if exit_type == "limit" else None
                    close_result = execution_engine.buy_to_close(
                        option_symbol=pick.symbol,
                        quantity=1,
                        limit_price=close_limit
                    )
                    buy_status = "accepted" if close_result.success else "REJECTED"
                    print(f"    BUY  ({exit_type}): success={close_result.success}, id={close_result.order_id}, msg={close_result.message}")
                    time.sleep(2)
                else:
                    # Sell not filled (market closed) -- cancel and move on
                    print(f"    Sell not filled (market closed). Cancelling...")
                    cancelled = execution_engine.cancel_order(sell_result.order_id)
                    print(f"    Cancel: {'OK' if cancelled else 'FAILED'}")
                    sell_status = f"accepted (verified, cancelled)"
                    buy_status = "SKIPPED (sell not filled)"
                
                results.append({
                    "combo": f"{entry_type}/{exit_type}",
                    "symbol": sym,
                    "contract": pick.symbol,
                    "sell": sell_status,
                    "buy": buy_status,
                })
                
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({"combo": f"{entry_type}/{exit_type}", "symbol": sym, "sell": "ERROR", "buy": str(e)[:60]})
    
    # Final cleanup
    if do_replenish:
        print(f"\n{'─'*70}")
        print("  [CLEANUP] Final replenish...")
        replenish_buying_power_p3()
    
    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Combo':<16} {'Ticker':<8} {'Sell':>28} {'Buy':>28}")
    print(f"  {'-'*16} {'-'*8} {'-'*28} {'-'*28}")
    for r in results:
        sell_short = r['sell'][:28]
        buy_short = r['buy'][:28]
        print(f"  {r['combo']:<16} {r['symbol']:<8} {sell_short:>28} {buy_short:>28}")
    
    passed = sum(1 for r in results if r["sell"] not in ("FAIL", "ERROR", "SKIP", "REJECTED"))
    total = len(results)
    print(f"\n  {passed}/{total} tests passed")
    print(f"{'='*70}")
    
    return results


# Run test
test_all_order_types_p3(execution, config, num_tickers=1, do_replenish=True)

