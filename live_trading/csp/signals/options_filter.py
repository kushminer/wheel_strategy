"""Options contract filter and ranking."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from csp.config import StrategyConfig
from csp.data.greeks import GreeksCalculator
from csp.data.options import OptionContract


@dataclass
class OptionsFilterResult:
    """Result of options filter for a single contract."""
    contract: OptionContract
    passes: bool
    daily_return: float
    delta_abs: Optional[float]
    failure_reasons: List[str]


class OptionsFilter:
    """Filters and ranks options based on strategy criteria."""

    def __init__(self, config: StrategyConfig, greeks_calculator: GreeksCalculator):
        self.config = config
        self.greeks_calc = greeks_calculator

    def _ensure_greeks(self, contract: OptionContract) -> OptionContract:
        """Ensure contract has Greeks, calculating if missing."""
        if contract.delta is not None and contract.implied_volatility is not None:
            return contract

        greeks = self.greeks_calc.compute_greeks_from_price(
            option_price=contract.mid,
            stock_price=contract.stock_price,
            strike=contract.strike,
            dte=contract.dte,
            option_type=contract.contract_type,
        )

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

    def evaluate(self, contract: OptionContract) -> OptionsFilterResult:
        """Evaluate a single option contract against filter criteria."""
        contract = self._ensure_greeks(contract)

        failure_reasons = []
        daily_return = contract.daily_return_on_collateral
        delta_abs = abs(contract.delta) if contract.delta else None
        strike_pct = contract.strike / contract.stock_price

        if self.config.enable_premium_check:
            if daily_return < self.config.min_daily_return:
                failure_reasons.append(
                    f"Daily return {daily_return:.4%} < {self.config.min_daily_return:.4%}"
                )

        if strike_pct > self.config.max_strike_pct:
            failure_reasons.append(
                f"Strike {strike_pct:.1%} > {self.config.max_strike_pct:.1%} of stock"
            )
        if strike_pct < self.config.min_strike_pct:
            failure_reasons.append(
                f"Strike {strike_pct:.1%} < {self.config.min_strike_pct:.1%} of stock"
            )

        if self.config.enable_delta_check:
            if delta_abs is None:
                failure_reasons.append("Delta unavailable")
            elif not (self.config.delta_min <= delta_abs <= self.config.delta_max):
                failure_reasons.append(
                    f"Delta {delta_abs:.3f} outside [{self.config.delta_min}, {self.config.delta_max}]"
                )

        if self.config.enable_dte_check:
            if not (self.config.min_dte <= contract.dte <= self.config.max_dte):
                failure_reasons.append(
                    f"DTE {contract.dte} outside [{self.config.min_dte}, {self.config.max_dte}]"
                )

        if self.config.enable_volume_check and self.config.min_volume > 0:
            vol = contract.volume or 0
            if vol < self.config.min_volume:
                failure_reasons.append(f"Volume {vol} < {self.config.min_volume}")

        if self.config.enable_open_interest_check and self.config.min_open_interest > 0:
            oi = contract.open_interest or 0
            if oi < self.config.min_open_interest:
                failure_reasons.append(f"OI {oi} < {self.config.min_open_interest}")

        if self.config.enable_spread_check and self.config.max_spread_pct < 1.0:
            if contract.mid > 0:
                spread_pct = (contract.ask - contract.bid) / contract.mid
                if spread_pct > self.config.max_spread_pct:
                    failure_reasons.append(f"Spread {spread_pct:.1%} > {self.config.max_spread_pct:.1%}")

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
        """Filter contracts and rank passing ones."""
        results = []
        passing = []

        for contract in contracts:
            result = self.evaluate(contract)
            results.append(result)
            if result.passes:
                passing.append(result.contract)

        def _sort_key(c):
            if self.config.contract_rank_mode == "daily_return_per_delta":
                return c.daily_return_per_delta
            elif self.config.contract_rank_mode == "days_since_strike":
                return c.days_since_strike or 0
            elif self.config.contract_rank_mode == "lowest_strike_price":
                return -c.strike
            else:
                return c.daily_return_on_collateral

        passing.sort(key=_sort_key, reverse=True)
        return passing, results

    def get_best_candidates(
        self,
        contracts: List[OptionContract],
        max_candidates: int,
    ) -> List[OptionContract]:
        """Get top N candidates after filtering and ranking."""
        passing, _ = self.filter_and_rank(contracts)
        return passing[:max_candidates]
