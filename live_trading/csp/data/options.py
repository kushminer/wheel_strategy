"""Option contract dataclass and options data fetching."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pytz

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest, OptionBarsRequest, OptionLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import ContractType, AssetStatus
from alpaca.trading.requests import GetOptionContractsRequest


@dataclass
class OptionContract:
    """Represents a single option contract with relevant data."""
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
    entry_time: Optional[datetime] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    days_since_strike: Optional[int] = None

    @property
    def premium(self) -> float:
        """Premium received when selling (use bid price)."""
        return self.bid

    @property
    def effective_dte(self) -> float:
        """Pro-rata DTE: fractional day remaining today + whole DTE days."""
        TRADING_MINUTES_PER_DAY = 390
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
        """Cost basis = stock price * 100."""
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
    """Fetches options chain data from Alpaca.

    Handles contract discovery, quote retrieval, and snapshot fetching.
    """

    def __init__(self, alpaca_manager):
        self.trading_client = alpaca_manager.trading_client
        self.data_client = OptionHistoricalDataClient(
            api_key=alpaca_manager.api_key,
            secret_key=alpaca_manager.secret_key,
        )

    def get_option_contracts(
        self,
        underlying: str,
        contract_type: str = "put",
        min_dte: int = 7,
        max_dte: int = 45,
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None,
    ) -> List[dict]:
        """Get available option contracts for an underlying."""
        today = date.today()
        min_expiry = today + timedelta(days=min_dte)
        max_expiry = today + timedelta(days=max_dte)

        request_params = {
            "underlying_symbols": [underlying],
            "status": AssetStatus.ACTIVE,
            "type": ContractType.PUT if contract_type == "put" else ContractType.CALL,
            "expiration_date_gte": min_expiry,
            "expiration_date_lte": max_expiry,
        }

        if min_strike is not None:
            request_params["strike_price_gte"] = str(min_strike)
        if max_strike is not None:
            request_params["strike_price_lte"] = str(max_strike)

        request = GetOptionContractsRequest(**request_params)
        response = self.trading_client.get_option_contracts(request)

        contracts = []
        if response.option_contracts:
            for contract in response.option_contracts:
                contracts.append({
                    "symbol": contract.symbol,
                    "underlying": contract.underlying_symbol,
                    "strike": float(contract.strike_price),
                    "expiration": contract.expiration_date,
                    "contract_type": contract_type,
                    "open_interest": int(contract.open_interest) if getattr(contract, "open_interest", None) else None,
                })

        return contracts

    def get_option_quotes(self, option_symbols: List[str]) -> Dict[str, dict]:
        """Get current quotes for option contracts."""
        if not option_symbols:
            return {}

        chunk_size = 100
        all_quotes = {}

        for i in range(0, len(option_symbols), chunk_size):
            chunk = option_symbols[i : i + chunk_size]
            try:
                request = OptionLatestQuoteRequest(symbol_or_symbols=chunk, feed="indicative")
                quotes = self.data_client.get_option_latest_quote(request)
                for symbol, quote in quotes.items():
                    all_quotes[symbol] = {
                        "bid": float(quote.bid_price) if quote.bid_price else 0.0,
                        "ask": float(quote.ask_price) if quote.ask_price else 0.0,
                        "bid_size": quote.bid_size,
                        "ask_size": quote.ask_size,
                    }
            except Exception as e:
                print(f"  Warning: Quote fetch error for chunk: {e}")

        return all_quotes

    def get_option_snapshots(self, option_symbols: List[str]) -> Dict[str, dict]:
        """Get snapshots including Greeks for option contracts."""
        if not option_symbols:
            return {}

        chunk_size = 100
        all_snapshots = {}

        for i in range(0, len(option_symbols), chunk_size):
            chunk = option_symbols[i : i + chunk_size]
            try:
                request = OptionSnapshotRequest(symbol_or_symbols=chunk, feed="indicative")
                snapshots = self.data_client.get_option_snapshot(request)

                for symbol, snapshot in snapshots.items():
                    greeks = snapshot.greeks if snapshot.greeks else None
                    quote = snapshot.latest_quote if snapshot.latest_quote else None

                    all_snapshots[symbol] = {
                        "bid": float(quote.bid_price) if quote and quote.bid_price else 0.0,
                        "ask": float(quote.ask_price) if quote and quote.ask_price else 0.0,
                        "delta": float(greeks.delta) if greeks and greeks.delta else None,
                        "gamma": float(greeks.gamma) if greeks and greeks.gamma else None,
                        "theta": float(greeks.theta) if greeks and greeks.theta else None,
                        "vega": float(greeks.vega) if greeks and greeks.vega else None,
                        "implied_volatility": float(snapshot.implied_volatility) if snapshot.implied_volatility else None,
                    }

                # Fetch daily bars for volume
                try:
                    bar_request = OptionBarsRequest(
                        symbol_or_symbols=chunk,
                        timeframe=TimeFrame.Day,
                        feed="indicative",
                    )
                    bars = self.data_client.get_option_bars(bar_request)
                    for symbol in chunk:
                        if symbol in all_snapshots:
                            try:
                                bar_list = bars[symbol]
                                if bar_list:
                                    all_snapshots[symbol]["volume"] = int(bar_list[-1].volume)
                                    all_snapshots[symbol]["open_interest"] = (
                                        int(bar_list[-1].trade_count)
                                        if hasattr(bar_list[-1], "trade_count")
                                        else None
                                    )
                            except (KeyError, IndexError):
                                pass
                except Exception as e:
                    print(f"  Warning: Option bars fetch error: {e}")
            except Exception as e:
                print(f"  Warning: Snapshot fetch error for chunk: {e}")

        return all_snapshots

    def get_puts_chain(
        self,
        underlying: str,
        stock_price: float,
        config,
        sma_ceiling: float = None,
    ) -> List[OptionContract]:
        """Get filtered put options chain with full data."""
        if config.max_strike_mode == "sma" and sma_ceiling is not None:
            max_strike = sma_ceiling
        else:
            max_strike = stock_price * config.max_strike_pct
        min_strike = stock_price * config.min_strike_pct

        contracts = self.get_option_contracts(
            underlying=underlying,
            contract_type="put",
            min_dte=config.min_dte,
            max_dte=config.max_dte,
            min_strike=min_strike,
            max_strike=max_strike,
        )

        if not contracts:
            return []

        symbols = [c["symbol"] for c in contracts]
        snapshots = self.get_option_snapshots(symbols)

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
                    entry_time=datetime.now(pytz.timezone("US/Eastern")),
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
                print(f"  Warning: Options fetch error for {contract.get('symbol')}: {e}")
                continue

        return result
