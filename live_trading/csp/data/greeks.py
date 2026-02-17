"""Greeks calculator using Black-Scholes via py_vollib."""

from typing import Dict, Optional

import numpy as np
from py_vollib.black_scholes.greeks.analytical import (
    delta as bs_delta,
    gamma as bs_gamma,
    theta as bs_theta,
    vega as bs_vega,
)
from py_vollib.black_scholes.implied_volatility import implied_volatility


class GreeksCalculator:
    """
    Calculates IV and Greeks using Black-Scholes via py_vollib.
    Used to fill in missing Greeks from Alpaca data.
    """

    def __init__(self, risk_free_rate: float = 0.04):
        self.r = risk_free_rate

    def compute_iv(
        self,
        option_price: float,
        stock_price: float,
        strike: float,
        dte: int,
        option_type: str = 'put',
    ) -> Optional[float]:
        """Compute implied volatility from option price. Returns None on failure."""
        t = dte / 365.0
        flag = 'p' if option_type == 'put' else 'c'

        if not all([
            np.isfinite(option_price),
            np.isfinite(stock_price),
            np.isfinite(strike),
            t > 0,
            option_price > 0,
            stock_price > 0,
            strike > 0,
        ]):
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
        iv: float,
        option_type: str = 'put',
    ) -> Optional[float]:
        """Compute delta from IV. Returns None on failure."""
        if iv is None or not np.isfinite(iv) or iv <= 0:
            return None

        t = dte / 365.0
        flag = 'p' if option_type == 'put' else 'c'

        if t <= 0:
            return None

        try:
            d = bs_delta(flag, stock_price, strike, t, self.r, iv)
            return d if np.isfinite(d) else None
        except Exception:
            return None

    def compute_all_greeks(
        self,
        stock_price: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: str = 'put',
    ) -> Dict[str, Optional[float]]:
        """Compute all Greeks from IV."""
        result = {'delta': None, 'gamma': None, 'theta': None, 'vega': None}

        if iv is None or not np.isfinite(iv) or iv <= 0:
            return result

        t = dte / 365.0
        flag = 'p' if option_type == 'put' else 'c'

        if t <= 0:
            return result

        try:
            result['delta'] = bs_delta(flag, stock_price, strike, t, self.r, iv)
            result['gamma'] = bs_gamma(flag, stock_price, strike, t, self.r, iv)
            result['theta'] = bs_theta(flag, stock_price, strike, t, self.r, iv)
            result['vega'] = bs_vega(flag, stock_price, strike, t, self.r, iv)
        except Exception:
            pass

        return result

    def compute_greeks_from_price(
        self,
        option_price: float,
        stock_price: float,
        strike: float,
        dte: int,
        option_type: str = 'put',
    ) -> Dict[str, Optional[float]]:
        """Compute IV and all Greeks from option price in one call."""
        iv = self.compute_iv(option_price, stock_price, strike, dte, option_type)

        result = {'iv': iv, 'delta': None, 'gamma': None, 'theta': None, 'vega': None}

        if iv:
            greeks = self.compute_all_greeks(stock_price, strike, dte, iv, option_type)
            result.update(greeks)

        return result
