"""
var_cvar.py
-----------
VaR and CVaR calculator supporting both historical simulation
and Monte Carlo methods.

Definitions
-----------
VaR (Value at Risk) at confidence level alpha:
    The loss not exceeded with probability alpha.
    VaR_alpha = -quantile(returns, 1 - alpha)

CVaR (Conditional Value at Risk / Expected Shortfall) at alpha:
    The expected loss given that the loss exceeds VaR.
    CVaR_alpha = -E[returns | returns <= -VaR_alpha]

CVaR is a more informative risk measure than VaR because it captures
the shape of the tail rather than just a single threshold. Two portfolios
can have the same VaR but very different CVaR if their tail distributions
differ — which matters for understanding worst-case exposure.

Methods
-------
- Historical simulation: uses realized return distribution directly,
  making no distributional assumptions. Robust but limited by
  the length and representativeness of the history window.

- Monte Carlo: uses simulated return paths from the GBM engine,
  allowing scenario extension beyond the historical window and
  explicit correlation modeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .simulation import SimulationResult


@dataclass
class VaRResult:
    """
    VaR and CVaR estimates for a single instrument or portfolio.

    Attributes
    ----------
    name : str
    confidence_level : float
    var_pct : float
        VaR as a percentage of current value (positive = loss).
    cvar_pct : float
        CVaR as a percentage of current value (positive = loss).
    var_abs : float
        VaR in absolute currency units.
    cvar_abs : float
        CVaR in absolute currency units.
    method : str
        "historical" or "monte_carlo".
    n_observations : int
        Number of observations / simulations used.
    tail_returns : np.ndarray
        Returns in the tail (beyond VaR threshold).
    """
    name: str
    confidence_level: float
    var_pct: float
    cvar_pct: float
    var_abs: float
    cvar_abs: float
    method: str
    n_observations: int
    tail_returns: np.ndarray

    def summary(self) -> dict:
        return {
            "name": self.name,
            "method": self.method,
            "confidence_level": self.confidence_level,
            "var_pct": round(self.var_pct * 100, 3),
            "cvar_pct": round(self.cvar_pct * 100, 3),
            "var_abs": round(self.var_abs, 2),
            "cvar_abs": round(self.cvar_abs, 2),
            "n_observations": self.n_observations,
        }


class VaRCalculator:
    """
    Computes VaR and CVaR using historical simulation or Monte Carlo paths.

    Parameters
    ----------
    confidence_levels : list[float]
        Confidence levels to compute. Default [0.95, 0.99].
    """

    def __init__(self, confidence_levels: Optional[list[float]] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99]

    def _compute(
        self,
        returns: np.ndarray,
        name: str,
        position_value: float,
        confidence_level: float,
        method: str,
    ) -> VaRResult:
        """Core VaR/CVaR computation from a return distribution."""
        alpha = 1 - confidence_level
        var_pct = -np.quantile(returns, alpha)
        tail_mask = returns <= -var_pct
        tail_returns = returns[tail_mask]
        cvar_pct = -tail_returns.mean() if len(tail_returns) > 0 else var_pct

        return VaRResult(
            name=name,
            confidence_level=confidence_level,
            var_pct=var_pct,
            cvar_pct=cvar_pct,
            var_abs=var_pct * position_value,
            cvar_abs=cvar_pct * position_value,
            method=method,
            n_observations=len(returns),
            tail_returns=tail_returns,
        )

    def historical(
        self,
        returns_history: np.ndarray,
        position_value: float,
        name: str = "position",
        confidence_level: Optional[float] = None,
    ) -> list[VaRResult]:
        """
        Compute VaR/CVaR via historical simulation.

        Parameters
        ----------
        returns_history : np.ndarray
            Array of historical daily log returns.
        position_value : float
            Current position value in currency units.
        name : str
        confidence_level : float, optional
            Single confidence level. Computes all levels if None.
        """
        levels = [confidence_level] if confidence_level else self.confidence_levels
        return [
            self._compute(returns_history, name, position_value, cl, "historical")
            for cl in levels
        ]

    def monte_carlo(
        self,
        sim_result: SimulationResult,
        position_values: dict[str, float],
        confidence_level: Optional[float] = None,
    ) -> list[VaRResult]:
        """
        Compute VaR/CVaR from Monte Carlo simulation paths.

        Parameters
        ----------
        sim_result : SimulationResult
            Output of MonteCarloEngine.simulate().
        position_values : dict[str, float]
            {instrument_name: position_value_in_currency}
        confidence_level : float, optional
        """
        levels = [confidence_level] if confidence_level else self.confidence_levels
        results = []

        for i, name in enumerate(sim_result.instruments):
            if name not in position_values:
                continue
            terminal_rets = sim_result.terminal_returns[i]
            pos_value = position_values[name]
            for cl in levels:
                results.append(
                    self._compute(terminal_rets, name, pos_value, cl, "monte_carlo")
                )

        return results

    def portfolio_monte_carlo(
        self,
        sim_result: SimulationResult,
        position_values: dict[str, float],
        confidence_level: Optional[float] = None,
    ) -> list[VaRResult]:
        """
        Compute portfolio-level VaR/CVaR from Monte Carlo paths.

        Portfolio P&L is computed as the sum of P&L across all positions
        in each simulation, preserving cross-instrument correlations.

        Parameters
        ----------
        sim_result : SimulationResult
        position_values : dict[str, float]
        confidence_level : float, optional
        """
        levels = [confidence_level] if confidence_level else self.confidence_levels

        # Portfolio P&L per simulation
        portfolio_pnl = np.zeros(sim_result.n_simulations
                                  if hasattr(sim_result, 'n_simulations')
                                  else sim_result.terminal_prices.shape[1])

        total_value = 0.0
        for i, name in enumerate(sim_result.instruments):
            if name not in position_values:
                continue
            pos_value = position_values[name]
            terminal_rets = sim_result.terminal_returns[i]
            portfolio_pnl += terminal_rets * pos_value
            total_value += pos_value

        portfolio_returns = portfolio_pnl / total_value if total_value > 0 else portfolio_pnl

        results = []
        for cl in levels:
            results.append(
                self._compute(portfolio_returns, "portfolio", total_value, cl, "monte_carlo")
            )
        return results

    def to_dataframe(self, results: list[VaRResult]) -> pd.DataFrame:
        return pd.DataFrame([r.summary() for r in results])
