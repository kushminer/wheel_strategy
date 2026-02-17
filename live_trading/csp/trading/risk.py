"""Risk management: stop-loss and early exit logic."""

from typing import Dict, Optional

from csp.config import StrategyConfig
from csp.trading.models import ExitReason, RiskCheckResult


class RiskManager:
    """
    Manages risk checks for positions.
    Implements stop-loss and early exit logic.

    Stop-Loss Conditions (ANY triggers exit):
    1. Current delta >= 2x entry delta
    2. Stock price <= 95% of entry stock price
    3. Current VIX >= 1.15x entry VIX (or session open VIX)

    Early Exit Condition:
    - Premium captured >= expected decay + buffer
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def check_delta_stop(
        self,
        position,
        current_delta: float,
    ) -> RiskCheckResult:
        """Check if delta has doubled from entry."""
        entry_delta_abs = abs(position.entry_delta)
        current_delta_abs = abs(current_delta)
        threshold = entry_delta_abs * self.config.delta_stop_multiplier

        triggered = current_delta_abs >= threshold

        return RiskCheckResult(
            should_exit=triggered,
            exit_reason=ExitReason.DELTA_STOP if triggered else None,
            details=f"Delta {current_delta_abs:.3f} {'>=' if triggered else '<'} {threshold:.3f} (2x entry {entry_delta_abs:.3f})",
            current_values={
                'entry_delta': entry_delta_abs,
                'current_delta': current_delta_abs,
                'threshold': threshold,
            },
        )

    def check_delta_absolute_stop(
        self,
        position,
        current_delta: float,
    ) -> RiskCheckResult:
        """Check if delta has reached the absolute ceiling (e.g. 0.40)."""
        current_delta_abs = abs(current_delta)
        threshold = self.config.delta_absolute_stop
        triggered = current_delta_abs >= threshold

        return RiskCheckResult(
            should_exit=triggered,
            exit_reason=ExitReason.DELTA_ABSOLUTE if triggered else None,
            details=f"Delta {current_delta_abs:.3f} {'>=' if triggered else '<'} {threshold:.3f} (absolute cap)",
            current_values={
                'current_delta': current_delta_abs,
                'threshold': threshold,
            },
        )

    def check_stock_drop_stop(
        self,
        position,
        current_stock_price: float,
    ) -> RiskCheckResult:
        """Check if stock has dropped 5% from entry."""
        threshold = position.entry_stock_price * (1 - self.config.stock_drop_stop_pct)
        drop_pct = (position.entry_stock_price - current_stock_price) / position.entry_stock_price

        triggered = current_stock_price <= threshold

        return RiskCheckResult(
            should_exit=triggered,
            exit_reason=ExitReason.STOCK_DROP if triggered else None,
            details=f"Stock ${current_stock_price:.2f} {'<=' if triggered else '>'} ${threshold:.2f} ({drop_pct:.1%} drop)",
            current_values={
                'entry_stock_price': position.entry_stock_price,
                'current_stock_price': current_stock_price,
                'threshold': threshold,
                'drop_pct': drop_pct,
            },
        )

    def check_vix_spike_stop(
        self,
        position,
        current_vix: float,
        reference_vix: Optional[float] = None,
    ) -> RiskCheckResult:
        """Check if VIX has spiked 15% from reference."""
        ref_vix = reference_vix or position.entry_vix
        threshold = ref_vix * self.config.vix_spike_multiplier
        spike_pct = (current_vix - ref_vix) / ref_vix

        triggered = current_vix >= threshold

        return RiskCheckResult(
            should_exit=triggered,
            exit_reason=ExitReason.VIX_SPIKE if triggered else None,
            details=f"VIX {current_vix:.2f} {'>=' if triggered else '<'} {threshold:.2f} ({spike_pct:+.1%} from ref {ref_vix:.2f})",
            current_values={
                'reference_vix': ref_vix,
                'current_vix': current_vix,
                'threshold': threshold,
                'spike_pct': spike_pct,
            },
        )

    def check_early_exit(
        self,
        position,
        current_premium: float,
    ) -> RiskCheckResult:
        """
        Check if premium has decayed enough for early exit.

        Formula:
            daily_return = position.entry_daily_return OR config.min_daily_return
            expected_capture = days_held * daily_return * strike  ($ per share)
            buffer = expected_capture * early_exit_buffer_pct
            Exit if: premium_captured >= expected_capture + buffer
        """
        days_held = position.days_held
        if days_held <= 0:
            return RiskCheckResult(
                should_exit=False,
                exit_reason=None,
                details="Position just opened, no early exit check",
                current_values={},
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
                f"Captured ${premium_captured:.4f}/sh {'>=' if triggered else '<'} "
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
            },
        )

    def check_all_stops(
        self,
        position,
        current_delta: float,
        current_stock_price: float,
        current_vix: float,
        reference_vix: Optional[float] = None,
    ) -> RiskCheckResult:
        """Check all stop-loss conditions. Returns first triggered or no-exit."""
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
            },
        )

    def evaluate_position(
        self,
        position,
        current_delta: float,
        current_stock_price: float,
        current_vix: float,
        current_premium: float,
        reference_vix: Optional[float] = None,
    ) -> RiskCheckResult:
        """Full risk evaluation: check stops first, then early exit."""
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
            },
        )
