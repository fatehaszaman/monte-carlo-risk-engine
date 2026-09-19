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

from .portfolio_risk import PortfolioRiskAggregator, PortfolioRiskReport
from .simulation import InstrumentSpec, MonteCarloEngine, SimulationResult
from .stress_test import StressResult, StressScenario, StressTester
from .var_cvar import VaRCalculator, VaRResult

__all__ = [
    "InstrumentSpec",
    "MonteCarloEngine",
    "PortfolioRiskAggregator",
    "PortfolioRiskReport",
    "SimulationResult",
    "StressResult",
    "StressScenario",
    "StressTester",
    "VaRCalculator",
    "VaRResult",
]
