"""Options chain fetcher and Greeks calculator."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest
from py_vollib.black_scholes.greeks.analytical import delta as bs_delta
from py_vollib.black_scholes.greeks.analytical import gamma as bs_gamma
from py_vollib.black_scholes.greeks.analytical import theta as bs_theta
from py_vollib.black_scholes.greeks.analytical import vega as bs_vega
from py_vollib.black_scholes.implied_volatility import implied_volatility

from csp.data.models import OptionContract

if TYPE_CHECKING:
    from csp.clients import AlpacaClientManager
    from csp.config import StrategyConfig

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.04


class GreeksCalculator:
    """
    Calculates IV and Greeks using Black-Scholes via py_vollib.
    Falls back gracefully when calculation fails.
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE) -> None:
        self.r = risk_free_rate

    def compute_iv(
        self,
        option_price: float,
        stock_price: float,
        strike: float,
        dte: int,
        option_type: str = "put",
    ) -> Optional[float]:
        """Compute implied volatility from option price."""
        t = dte / 365.0
        flag = "p" if option_type == "put" else "c"

        if not all(
            [
                np.isfinite(option_price),
                np.isfinite(stock_price),
                np.isfinite(strike),
                t > 0,
                option_price > 0,
                stock_price > 0,
                strike > 0,
            ]
        ):
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
        iv: Optional[float],
        option_type: str = "put",
    ) -> Optional[float]:
        """Compute delta from IV."""
        if iv is None or not np.isfinite(iv) or iv <= 0:
            return None

        t = dte / 365.0
        flag = "p" if option_type == "put" else "c"
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
        option_type: str = "put",
    ) -> dict:
        """Compute both IV and delta in one call."""
        iv = self.compute_iv(option_price, stock_price, strike, dte, option_type)
        delta = self.compute_delta(stock_price, strike, dte, iv, option_type) if iv else None

        return {
            "iv": iv,
            "delta": delta,
            "delta_abs": abs(delta) if delta else None,
        }

    def compute_all_greeks(
        self,
        stock_price: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: str = "put",
    ) -> Dict[str, Optional[float]]:
        """Compute all Greeks from IV."""
        result: Dict[str, Optional[float]] = {
            "delta": None,
            "gamma": None,
            "theta": None,
            "vega": None,
        }

        if iv is None or not np.isfinite(iv) or iv <= 0:
            return result

        t = dte / 365.0
        flag = "p" if option_type == "put" else "c"
        if t <= 0:
            return result

        try:
            result["delta"] = bs_delta(flag, stock_price, strike, t, self.r, iv)
            result["gamma"] = bs_gamma(flag, stock_price, strike, t, self.r, iv)
            result["theta"] = bs_theta(flag, stock_price, strike, t, self.r, iv)
            result["vega"] = bs_vega(flag, stock_price, strike, t, self.r, iv)
        except Exception:
            pass

        return result

    def compute_greeks_from_price(
        self,
        option_price: float,
        stock_price: float,
        strike: float,
        dte: int,
        option_type: str = "put",
    ) -> Dict[str, Optional[float]]:
        """Compute IV and all Greeks from option price in one call."""
        iv = self.compute_iv(option_price, stock_price, strike, dte, option_type)

        result: Dict[str, Optional[float]] = {
            "iv": iv,
            "delta": None,
            "gamma": None,
            "theta": None,
            "vega": None,
        }

        if iv:
            greeks = self.compute_all_greeks(stock_price, strike, dte, iv, option_type)
            result.update(greeks)

        return result


class OptionsDataFetcher:
    """
    Fetches options chain data from Alpaca.
    Handles contract discovery and quote retrieval.
    """

    def __init__(self, alpaca_manager: AlpacaClientManager) -> None:
        self.trading_client = alpaca_manager.trading_client
        self.data_client = OptionHistoricalDataClient(
            api_key=alpaca_manager.api_key,
            secret_key=alpaca_manager.secret_key,
        )

    def get_option_contracts(
        self,
        underlying: str,
        contract_type: str = "put",
        min_dte: int = 1,
        max_dte: int = 10,
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
                contracts.append(
                    {
                        "symbol": contract.symbol,
                        "underlying": contract.underlying_symbol,
                        "strike": float(contract.strike_price),
                        "expiration": contract.expiration_date,
                        "contract_type": contract_type,
                    }
                )

        return contracts

    def get_option_snapshots(self, option_symbols: List[str]) -> Dict[str, dict]:
        """Get snapshots including Greeks for option contracts."""
        if not option_symbols:
            return {}

        chunk_size = 100
        all_snapshots: Dict[str, dict] = {}

        for i in range(0, len(option_symbols), chunk_size):
            chunk = option_symbols[i : i + chunk_size]
            try:
                request = OptionSnapshotRequest(symbol_or_symbols=chunk)
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
                        "implied_volatility": (
                            float(snapshot.implied_volatility) if snapshot.implied_volatility else None
                        ),
                    }
            except Exception as e:
                logger.warning("Snapshot fetch error for chunk: %s", e)

        return all_snapshots

    def get_puts_chain(
        self,
        underlying: str,
        stock_price: float,
        config: StrategyConfig,
    ) -> List[OptionContract]:
        """Get filtered put options chain with full data."""
        max_strike = stock_price * config.max_strike_pct
        min_strike = stock_price * 0.70

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
        result: List[OptionContract] = []

        for contract in contracts:
            symbol = contract["symbol"]
            snapshot = snapshots.get(symbol, {})

            bid = snapshot.get("bid", 0.0)
            ask = snapshot.get("ask", 0.0)

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
                delta=snapshot.get("delta"),
                gamma=snapshot.get("gamma"),
                theta=snapshot.get("theta"),
                vega=snapshot.get("vega"),
                implied_volatility=snapshot.get("implied_volatility"),
            )

            result.append(option)

        return result
