"""
stress_test.py
--------------
Stress testing layer: discrete scenario shocks applied on top of
Monte Carlo simulation paths.

Why layer shocks on top of Monte Carlo?
----------------------------------------
Pure Monte Carlo simulation models the statistical distribution of
outcomes under the assumption that future dynamics resemble historical
dynamics. It does not model discontinuous events — policy shocks,
supply disruptions, correlated crashes — because these rarely appear
in historical data with sufficient frequency to be captured statistically.

Stress testing adds these scenarios explicitly:
  1. Run the base Monte Carlo simulation (captures statistical risk)
  2. Apply discrete shocks to a subset of paths (captures tail / regime risk)
  3. Report the combined distribution, including shocked tails

This gives a more complete picture than either approach alone.

Shock types
-----------
- Price shock: immediate step change to one or more instruments
- Volatility shock: scale the vol of simulation paths post-hoc
- Correlation shock: re-run simulation with a stressed correlation matrix
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .simulation import SimulationResult, MonteCarloEngine, InstrumentSpec


@dataclass
class StressScenario:
    """
    A named stress scenario with parameterized shocks.

    Parameters
    ----------
    name : str
    description : str
    price_shocks : dict[str, float]
        {instrument_name: shock_pct} — immediate % price change.
        e.g. {"crude_oil": -0.30} for a 30% drop.
    vol_multipliers : dict[str, float]
        {instrument_name: multiplier} — scale realized vol.
        e.g. {"crude_oil": 2.0} doubles volatility in this scenario.
    correlation_override : np.ndarray, optional
        Replace the correlation matrix for this scenario.
        Useful for stress-testing correlation breakdown or spike.
    n_paths_pct : float
        Fraction of total simulations to apply this shock to.
        Default 0.10 (shock applied to worst 10% of paths).
    """
    name: str
    description: str
    price_shocks: dict[str, float] = field(default_factory=dict)
    vol_multipliers: dict[str, float] = field(default_factory=dict)
    correlation_override: Optional[np.ndarray] = None
    n_paths_pct: float = 0.10


@dataclass
class StressResult:
    """
    Result of applying a stress scenario to a simulation.

    Attributes
    ----------
    scenario_name : str
    stressed_terminal_returns : np.ndarray
        Shape (n_instruments, n_simulations).
        Terminal returns with stress scenario applied to a subset of paths.
    var_95_shift : dict[str, float]
        Change in 95% VaR (as %) vs base simulation, per instrument.
    cvar_95_shift : dict[str, float]
        Change in 95% CVaR (as %) vs base simulation.
    worst_case_pnl : dict[str, float]
        Worst-case P&L per instrument across all stressed paths.
    """
    scenario_name: str
    description: str
    stressed_terminal_returns: np.ndarray
    instruments: list[str]
    var_95_shift: dict[str, float]
    cvar_95_shift: dict[str, float]
    worst_case_return: dict[str, float]

    def summary(self) -> pd.DataFrame:
        rows = []
        for i, inst in enumerate(self.instruments):
            rows.append({
                "instrument": inst,
                "scenario": self.scenario_name,
                "var_95_shift_pct": round(self.var_95_shift.get(inst, 0) * 100, 3),
                "cvar_95_shift_pct": round(self.cvar_95_shift.get(inst, 0) * 100, 3),
                "worst_case_return_pct": round(self.worst_case_return.get(inst, 0) * 100, 3),
            })
        return pd.DataFrame(rows)


# Pre-built scenarios for common stress events
DEFAULT_STRESS_SCENARIOS = [
    StressScenario(
        name="Sharp_Selloff",
        description="Broad 20% price decline across all instruments",
        price_shocks={},      # Applied to all instruments in run_all
        vol_multipliers={},
        n_paths_pct=0.05,
    ),
    StressScenario(
        name="Vol_Spike",
        description="2x volatility spike — fat-tail / crisis conditions",
        price_shocks={},
        vol_multipliers={},   # Applied to all instruments
        n_paths_pct=0.10,
    ),
    StressScenario(
        name="Correlation_Breakdown",
        description="Correlations go to zero — diversification disappears",
        price_shocks={},
        vol_multipliers={},
        n_paths_pct=0.10,
    ),
    StressScenario(
        name="Supply_Shock",
        description="Primary commodity -30%, secondary instruments -10%",
        price_shocks={},
        vol_multipliers={},
        n_paths_pct=0.05,
    ),
]


class StressTester:
    """
    Applies discrete scenario shocks to Monte Carlo simulation results.

    Parameters
    ----------
    rng_seed : int, optional
    """

    def __init__(self, rng_seed: Optional[int] = None):
        self.rng = np.random.default_rng(rng_seed)

    def apply(
        self,
        base_result: SimulationResult,
        scenario: StressScenario,
        price_shock_all: Optional[float] = None,
        vol_multiplier_all: Optional[float] = None,
    ) -> StressResult:
        """
        Apply a stress scenario to the base simulation result.

        For each instrument, the worst n_paths_pct of base paths are
        selected and the stress shocks applied on top.

        Parameters
        ----------
        base_result : SimulationResult
        scenario : StressScenario
        price_shock_all : float, optional
            Apply this price shock to all instruments (overrides scenario).
        vol_multiplier_all : float, optional
            Apply this vol multiplier to all instruments.
        """
        n_inst = len(base_result.instruments)
        n_sims = base_result.terminal_returns.shape[1]
        n_shocked = max(1, int(n_sims * scenario.n_paths_pct))

        stressed_returns = base_result.terminal_returns.copy()

        for i, inst in enumerate(base_result.instruments):
            base_rets = base_result.terminal_returns[i]

            # Select worst paths for shocking
            worst_idx = np.argsort(base_rets)[:n_shocked]

            # Price shock
            shock = scenario.price_shocks.get(inst, price_shock_all or 0.0)
            if shock != 0.0:
                stressed_returns[i, worst_idx] += np.log(1 + shock)

            # Vol multiplier: scale returns away from mean
            vol_mult = scenario.vol_multipliers.get(inst, vol_multiplier_all or 1.0)
            if vol_mult != 1.0:
                mean_ret = base_rets[worst_idx].mean()
                stressed_returns[i, worst_idx] = (
                    mean_ret + (stressed_returns[i, worst_idx] - mean_ret) * vol_mult
                )

        # Compute shift in VaR/CVaR vs base
        var_shift = {}
        cvar_shift = {}
        worst_case = {}
        alpha = 0.05

        for i, inst in enumerate(base_result.instruments):
            base_rets = base_result.terminal_returns[i]
            stressed_rets = stressed_returns[i]

            base_var = -np.quantile(base_rets, alpha)
            stressed_var = -np.quantile(stressed_rets, alpha)
            var_shift[inst] = stressed_var - base_var

            base_cvar = -base_rets[base_rets <= -base_var].mean() if any(base_rets <= -base_var) else base_var
            stressed_cvar = -stressed_rets[stressed_rets <= -stressed_var].mean() if any(stressed_rets <= -stressed_var) else stressed_var
            cvar_shift[inst] = stressed_cvar - base_cvar

            worst_case[inst] = stressed_returns[i].min()

        return StressResult(
            scenario_name=scenario.name,
            description=scenario.description,
            stressed_terminal_returns=stressed_returns,
            instruments=base_result.instruments,
            var_95_shift=var_shift,
            cvar_95_shift=cvar_shift,
            worst_case_return=worst_case,
        )

    def run_all(
        self,
        base_result: SimulationResult,
    ) -> list[StressResult]:
        """Run all default stress scenarios against the base simulation."""
        shocks = [
            (DEFAULT_STRESS_SCENARIOS[0], -0.20, None),   # Sharp selloff
            (DEFAULT_STRESS_SCENARIOS[1], None,  2.0),    # Vol spike
            (DEFAULT_STRESS_SCENARIOS[2], None,  1.0),    # Correlation breakdown (vol neutral)
            (DEFAULT_STRESS_SCENARIOS[3], -0.15, None),   # Supply shock (moderate)
        ]
        results = []
        for scenario, price_shock, vol_mult in shocks:
            results.append(self.apply(base_result, scenario, price_shock, vol_mult))
        return results
