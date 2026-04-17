"""
mc_risk
-------
Monte Carlo risk engine: correlated GBM simulation, VaR/CVaR,
stress testing, and portfolio risk aggregation.

Modules
-------
simulation      : Correlated GBM Monte Carlo engine (Cholesky decomposition)
var_cvar        : VaR and CVaR via historical simulation and Monte Carlo
stress_test     : Discrete scenario shocks layered on simulation paths
portfolio_risk  : Portfolio aggregator and risk report
"""

from .simulation import MonteCarloEngine, InstrumentSpec, SimulationResult
from .var_cvar import VaRCalculator, VaRResult
from .stress_test import StressTester, StressScenario, StressResult
from .portfolio_risk import PortfolioRiskAggregator, PortfolioRiskReport

__all__ = [
    "MonteCarloEngine", "InstrumentSpec", "SimulationResult",
    "VaRCalculator", "VaRResult",
    "StressTester", "StressScenario", "StressResult",
    "PortfolioRiskAggregator", "PortfolioRiskReport",
]
