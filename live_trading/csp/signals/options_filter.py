"""Options filter for the CSP strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from csp.data.models import OptionContract
from csp.data.options import GreeksCalculator

if TYPE_CHECKING:
    from csp.config import StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class OptionsFilterResult:
    """
    Result of options filter for a single contract.
    """

    contract: OptionContract
    passes: bool
    daily_return: float
    delta_abs: Optional[float]
    failure_reasons: List[str]

    def __str__(self) -> str:
        status = "PASS" if self.passes else "FAIL"
        delta_str = f"{self.delta_abs:.3f}" if self.delta_abs is not None else "N/A"
        return f"{status} {self.contract.symbol} | delta={delta_str} | ret={self.daily_return:.4%}"


class OptionsFilter:
    """
    Filters and ranks options based on strategy criteria.

    Filter Rules:
    1. Daily return on cost basis >= min_daily_return (config)
    2. Strike <= max_strike_pct of stock price
    3. |Delta| between delta_min and delta_max
    4. DTE between min_dte and max_dte

    Ranking: By premium per day (descending)
    """

    def __init__(self, config: "StrategyConfig", greeks_calculator: GreeksCalculator) -> None:
        self.config = config
        self.greeks_calc = greeks_calculator

    def _ensure_greeks(self, contract: OptionContract) -> OptionContract:
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
            option_type=contract.contract_type,
        )

        # Update contract with calculated Greeks (only if missing)
        if contract.implied_volatility is None and greeks.get("iv") is not None:
            contract.implied_volatility = greeks["iv"]
        if contract.delta is None and greeks.get("delta") is not None:
            contract.delta = greeks["delta"]
        if contract.gamma is None and greeks.get("gamma") is not None:
            contract.gamma = greeks["gamma"]
        if contract.theta is None and greeks.get("theta") is not None:
            contract.theta = greeks["theta"]
        if contract.vega is None and greeks.get("vega") is not None:
            contract.vega = greeks["vega"]

        return contract

    def evaluate(self, contract: OptionContract) -> OptionsFilterResult:
        """
        Evaluate a single option contract against filter criteria.

        Args:
            contract: OptionContract to evaluate

        Returns:
            OptionsFilterResult with pass/fail and details
        """
        contract = self._ensure_greeks(contract)

        failure_reasons: List[str] = []

        daily_return = contract.daily_return_on_cost_basis
        delta_abs = abs(contract.delta) if contract.delta is not None else None
        strike_pct = contract.strike / contract.stock_price

        # 1. Premium filter: daily return >= min_daily_return (config stores as decimal, e.g. 0.15 = 15%)
        if daily_return < self.config.min_daily_return:
            failure_reasons.append(
                f"Daily return {daily_return:.4%} < {self.config.min_daily_return:.1%}"
            )

        # 2. Strike filter: strike <= max_strike_pct of stock price
        if strike_pct > self.config.max_strike_pct:
            failure_reasons.append(
                f"Strike {strike_pct:.1%} > {self.config.max_strike_pct:.1%} of stock"
            )

        # 3. Delta filter: |delta| between delta_min and delta_max
        if delta_abs is None:
            failure_reasons.append("Delta unavailable")
        elif not (self.config.delta_min <= delta_abs <= self.config.delta_max):
            failure_reasons.append(
                f"Delta {delta_abs:.3f} outside [{self.config.delta_min}, {self.config.delta_max}]"
            )

        # 4. DTE filter (should already be filtered, but double-check)
        if not (self.config.min_dte <= contract.dte <= self.config.max_dte):
            failure_reasons.append(
                f"DTE {contract.dte} outside [{self.config.min_dte}, {self.config.max_dte}]"
            )

        passes = len(failure_reasons) == 0

        return OptionsFilterResult(
            contract=contract,
            passes=passes,
            daily_return=daily_return,
            delta_abs=delta_abs,
            failure_reasons=failure_reasons,
        )

    def filter_and_rank(
        self,
        contracts: List[OptionContract],
    ) -> Tuple[List[OptionContract], List[OptionsFilterResult]]:
        """
        Filter contracts and rank passing ones by premium per day.

        Args:
            contracts: List of OptionContracts to evaluate

        Returns:
            Tuple of (ranked_passing_contracts, all_results)
        """
        results: List[OptionsFilterResult] = []
        passing: List[OptionContract] = []

        for contract in contracts:
            result = self.evaluate(contract)
            results.append(result)
            if result.passes:
                passing.append(result.contract)

        # Rank by premium_per_day descending
        passing.sort(key=lambda c: c.premium_per_day, reverse=True)

        return passing, results

    def get_best_candidates(
        self,
        contracts: List[OptionContract],
        max_candidates: int = 10,
    ) -> List[OptionContract]:
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
