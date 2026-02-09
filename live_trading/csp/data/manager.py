"""Unified data manager combining all data sources."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from csp.data.equity import EquityDataFetcher
from csp.data.models import MarketSnapshot, OptionContract
from csp.data.options import OptionsDataFetcher
from csp.data.vix import VixDataFetcher

if TYPE_CHECKING:
    from csp.clients import AlpacaClientManager
    from csp.config import StrategyConfig


class DataManager:
    """
    Unified data manager that combines all data sources.
    Provides a clean interface for strategy logic.
    """

    def __init__(
        self,
        alpaca_manager: Optional[AlpacaClientManager],
        config: StrategyConfig,
    ) -> None:
        self.config = config
        self.vix_fetcher = VixDataFetcher()

        if alpaca_manager:
            self.equity_fetcher = EquityDataFetcher(alpaca_manager)
            self.options_fetcher = OptionsDataFetcher(alpaca_manager)
        else:
            self.equity_fetcher = None
            self.options_fetcher = None

    def get_market_snapshot(self) -> MarketSnapshot:
        """Get complete market state for strategy decision-making."""
        vix_current = self.vix_fetcher.get_current_vix()
        _, vix_open = self.vix_fetcher.get_session_reference_vix()
        deployable_cash = self.config.get_deployable_cash(vix_current)

        if self.equity_fetcher:
            equity_history = self.equity_fetcher.get_close_history(
                self.config.ticker_universe,
                days=self.config.history_days,
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
            equity_history=equity_history,
        )

    def get_puts_for_symbol(
        self,
        symbol: str,
        stock_price: float,
    ) -> List[OptionContract]:
        """Get filtered put options for a symbol."""
        if not self.options_fetcher:
            raise RuntimeError("Options fetcher not configured")

        return self.options_fetcher.get_puts_chain(
            symbol,
            stock_price,
            self.config,
        )

    def refresh_option_data(self, option_symbol: str) -> Optional[dict]:
        """
        Refresh data for a single option (for position monitoring).
        Returns raw snapshot dict; parsing OCC format for full OptionContract is not implemented.
        """
        if not self.options_fetcher:
            return None

        snapshots = self.options_fetcher.get_option_snapshots([option_symbol])
        return snapshots.get(option_symbol)
