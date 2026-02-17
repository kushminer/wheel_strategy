"""Unified data manager composing all data sources."""

from datetime import datetime
from typing import List, Optional

from csp.config import StrategyConfig
from csp.data.equity import EquityDataFetcher
from csp.data.models import MarketSnapshot
from csp.data.options import OptionContract
from csp.data.vix import VixDataFetcher


class DataManager:
    """Unified data manager that combines all data sources."""

    def __init__(self, alpaca_manager, config: StrategyConfig):
        self.config = config
        self.vix_fetcher = VixDataFetcher()

        if alpaca_manager:
            self.equity_fetcher = EquityDataFetcher(alpaca_manager)
            # Delay import to avoid circular deps
            from csp.data.options import OptionsDataFetcher
            self.options_fetcher = OptionsDataFetcher(alpaca_manager)
        else:
            self.equity_fetcher = None
            self.options_fetcher = None

    def get_market_snapshot(self) -> MarketSnapshot:
        vix_current = self.vix_fetcher.get_current_vix()
        _, vix_open = self.vix_fetcher.get_session_reference_vix()
        deployable_cash = self.config.get_deployable_cash(vix_current)

        if self.equity_fetcher:
            equity_history = self.equity_fetcher.get_close_history(
                self.config.ticker_universe, days=self.config.history_days
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
        if not self.options_fetcher:
            raise RuntimeError("Options fetcher not configured")
        return self.options_fetcher.get_puts_chain(symbol, stock_price, self.config)

    def refresh_option_data(self, option_symbol: str) -> Optional[OptionContract]:
        if not self.options_fetcher:
            return None
        snapshots = self.options_fetcher.get_option_snapshots([option_symbol])
        if option_symbol not in snapshots:
            return None
        return snapshots[option_symbol]
