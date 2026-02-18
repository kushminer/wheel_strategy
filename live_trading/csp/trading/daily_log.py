"""Daily JSON log for strategy activity."""

import json
import time
from datetime import date, datetime
from typing import Dict, List

from csp.storage import LocalStorage, StorageBackend


class DailyLog:
    """Single-file daily JSON log.

    Accumulates log entries in memory and only writes to the storage
    backend when flush() is called. This avoids GCS rate-limit errors
    (1 write/sec/object) that occur when logging many events rapidly.
    """

    def __init__(self, log_dir: str = "logs", backend: StorageBackend = None):
        self.log_dir = log_dir
        self._backend = backend or LocalStorage()
        self._backend.mkdir(self.log_dir)
        self._data = None
        self._current_date = None
        self._dirty = False

    def _get_path(self) -> str:
        return f"{self.log_dir}/{date.today().isoformat()}.json"

    def _ensure_loaded(self):
        """Load or initialize today's log."""
        today = date.today()
        if self._current_date == today and self._data is not None:
            return

        path = self._get_path()
        if self._backend.exists(path):
            self._data = json.loads(self._backend.read(path))
        else:
            self._data = {
                "date": today.isoformat(),
                "config_snapshot": {},
                "equity_scan": {},
                "options_scans": [],
                "cycles": [],
                "order_attempts": [],
                "shutdown": {},
            }
        self._current_date = today

    def _save(self):
        """Write current data to storage with retry on rate-limit errors."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._backend.write(self._get_path(), json.dumps(self._data, indent=2))
                self._dirty = False
                return
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def flush(self):
        """Flush accumulated log data to storage. Call at cycle boundaries."""
        if self._dirty and self._data is not None:
            self._save()

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
            "exit_start_price": config.exit_start_price,
            "exit_step_interval": config.exit_step_interval,
            "exit_step_pct": config.exit_step_pct,
            "exit_max_steps": config.exit_max_steps,
            "exit_refetch_snapshot": config.exit_refetch_snapshot,
            "close_before_expiry_days": config.close_before_expiry_days,
            "exit_on_missing_delta": config.exit_on_missing_delta,
            "early_exit_return_source": config.early_exit_return_source,
            "early_exit_buffer_pct": config.early_exit_buffer_pct,
            "delta_stop_multiplier": config.delta_stop_multiplier,
            "delta_absolute_stop": config.delta_absolute_stop,
            "stock_drop_stop_pct": config.stock_drop_stop_pct,
            "vix_spike_multiplier": config.vix_spike_multiplier,
            "contract_rank_mode": config.contract_rank_mode,
            "universe_rank_mode": config.universe_rank_mode,
        }
        self._dirty = True

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
        self._dirty = True

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
        self._dirty = True

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
        self._dirty = True

    def log_order_attempt(
        self,
        action: str,
        symbol: str,
        contract: str,
        steps: List[dict],
        outcome: str,
        **kwargs,
    ):
        """Log a full stepped order attempt (entry or exit)."""
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
        self._dirty = True

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
        self._dirty = True
        self.flush()  # Always persist shutdown

    @property
    def today_path(self) -> str:
        return str(self._get_path())
