"""Risk manager - stop-loss and early exit logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from csp.trading.models import ActivePosition, ExitReason, RiskCheckResult

if TYPE_CHECKING:
    from csp.config import StrategyConfig

logger = logging.getLogger(__name__)


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

    def __init__(self, config: "StrategyConfig") -> None:
        self.config = config

    def check_delta_stop(
        self,
        position: ActivePosition,
        current_delta: float,
    ) -> RiskCheckResult:
        """
        Check if delta has doubled from entry.
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
                "entry_delta": entry_delta_abs,
                "current_delta": current_delta_abs,
                "threshold": threshold,
            },
        )

    def check_stock_drop_stop(
        self,
        position: ActivePosition,
        current_stock_price: float,
    ) -> RiskCheckResult:
        """
        Check if stock has dropped 5% from entry.
        """
        threshold = position.entry_stock_price * (
            1 - self.config.stock_drop_stop_pct
        )
        drop_pct = (
            position.entry_stock_price - current_stock_price
        ) / position.entry_stock_price

        triggered = current_stock_price <= threshold

        return RiskCheckResult(
            should_exit=triggered,
            exit_reason=ExitReason.STOCK_DROP if triggered else None,
            details=f"Stock ${current_stock_price:.2f} {'≤' if triggered else '>'} ${threshold:.2f} ({drop_pct:.1%} drop)",
            current_values={
                "entry_stock_price": position.entry_stock_price,
                "current_stock_price": current_stock_price,
                "threshold": threshold,
                "drop_pct": drop_pct,
            },
        )

    def check_vix_spike_stop(
        self,
        position: ActivePosition,
        current_vix: float,
        reference_vix: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        Check if VIX has spiked 15% from reference.
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
                "reference_vix": ref_vix,
                "current_vix": current_vix,
                "threshold": threshold,
                "spike_pct": spike_pct,
            },
        )

    def check_early_exit(
        self,
        position: ActivePosition,
        current_premium: float,
    ) -> RiskCheckResult:
        """
        Check if premium has decayed enough for early exit.

        Early exit if: capture_pct >= expected_capture + buffer
        Where expected_capture = days_held / dte_at_entry
        """
        days_held = position.days_held
        if days_held <= 0:
            return RiskCheckResult(
                should_exit=False,
                exit_reason=None,
                details="Position just opened, no early exit check",
                current_values={},
            )

        premium_captured = position.entry_premium - current_premium
        capture_pct = (
            premium_captured / position.entry_premium
            if position.entry_premium > 0
            else 0
        )

        expected_capture = days_held / position.dte_at_entry
        target_capture = expected_capture + self.config.early_exit_buffer

        triggered = capture_pct >= target_capture

        return RiskCheckResult(
            should_exit=triggered,
            exit_reason=ExitReason.EARLY_EXIT if triggered else None,
            details=f"Captured {capture_pct:.1%} {'≥' if triggered else '<'} target {target_capture:.1%} (expected {expected_capture:.1%} + {self.config.early_exit_buffer:.0%} buffer)",
            current_values={
                "entry_premium": position.entry_premium,
                "current_premium": current_premium,
                "premium_captured": premium_captured,
                "capture_pct": capture_pct,
                "expected_capture": expected_capture,
                "target_capture": target_capture,
                "days_held": days_held,
            },
        )

    def check_all_stops(
        self,
        position: ActivePosition,
        current_delta: float,
        current_stock_price: float,
        current_vix: float,
        reference_vix: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        Check all stop-loss conditions.
        Returns first triggered condition, or no-exit result.
        """
        delta_check = self.check_delta_stop(position, current_delta)
        if delta_check.should_exit:
            return delta_check

        stock_check = self.check_stock_drop_stop(position, current_stock_price)
        if stock_check.should_exit:
            return stock_check

        vix_check = self.check_vix_spike_stop(
            position, current_vix, reference_vix
        )
        if vix_check.should_exit:
            return vix_check

        return RiskCheckResult(
            should_exit=False,
            exit_reason=None,
            details="All stop-loss checks passed",
            current_values={
                "delta": delta_check.current_values,
                "stock": stock_check.current_values,
                "vix": vix_check.current_values,
            },
        )

    def evaluate_position(
        self,
        position: ActivePosition,
        current_delta: float,
        current_stock_price: float,
        current_vix: float,
        current_premium: float,
        reference_vix: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        Full risk evaluation: check stops first, then early exit.
        """
        stop_result = self.check_all_stops(
            position,
            current_delta,
            current_stock_price,
            current_vix,
            reference_vix,
        )
        if stop_result.should_exit:
            return stop_result

        early_result = self.check_early_exit(position, current_premium)
        if early_result.should_exit:
            return early_result

        return RiskCheckResult(
            should_exit=False,
            exit_reason=None,
            details="Position healthy, continue holding",
            current_values={
                "stops": stop_result.current_values,
                "early_exit": early_result.current_values,
            },
        )
