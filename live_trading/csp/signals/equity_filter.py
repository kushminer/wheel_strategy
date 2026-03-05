"""Equity technical filter and event calendars.

This module re-exports from equity_screener for backward compatibility.
Import directly from equity_screener.filter and equity_screener.calendars instead.
"""

from equity_screener.filter import EquityFilter, EquityFilterResult
from equity_screener.calendars import (
    EarningsCalendar,
    DividendCalendar,
    FomcCalendar,
)

__all__ = [
    "EquityFilter",
    "EquityFilterResult",
    "EarningsCalendar",
    "DividendCalendar",
    "FomcCalendar",
]
