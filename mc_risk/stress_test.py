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
- Correlation shocks are unsupported here and raise NotImplementedError.
  They require a separately parameterized simulation, not a terminal-return edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .simulation import SimulationResult


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
        Reserved for a future re-simulation API. Any non-None value raises
        NotImplementedError; it is never silently ignored.
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
        Change in the 95% loss quantile of log returns, per instrument.
    cvar_95_shift : dict[str, float]
        Change in log-return tail loss vs base simulation.
    worst_case_return : dict[str, float]
        Minimum log return, not dollar P&L or arithmetic return.
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
        description="20% price decline on each instrument's worst 5% of paths",
        price_shocks={},      # Applied to all instruments in run_all
        vol_multipliers={},
        n_paths_pct=0.05,
    ),
    StressScenario(
        name="Vol_Spike",
        description="2x dispersion about the mean within each instrument's worst 10% of log-return paths",
        price_shocks={},
        vol_multipliers={},   # Applied to all instruments
        n_paths_pct=0.10,
    ),
    StressScenario(
        name="Supply_Shock",
        description="15% price decline on each instrument's worst 5% of paths",
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
            Default shock for instruments without a scenario-specific shock.
        vol_multiplier_all : float, optional
            Default multiplier for instruments without a scenario-specific value.
        """
        if scenario.correlation_override is not None:
            raise NotImplementedError("Correlation stress requires re-simulation; terminal-return overrides are unsupported")
        if not np.isfinite(scenario.n_paths_pct) or not 0 <= scenario.n_paths_pct <= 1:
            raise ValueError("n_paths_pct must be between 0 and 1")
        shocks = list(scenario.price_shocks.values()) + ([] if price_shock_all is None else [price_shock_all])
        multipliers = list(scenario.vol_multipliers.values()) + ([] if vol_multiplier_all is None else [vol_multiplier_all])
        if any(not np.isfinite(x) or x <= -1 for x in shocks):
            raise ValueError("Price shocks must be finite and greater than -1")
        if any(not np.isfinite(x) or x < 0 for x in multipliers):
            raise ValueError("Volatility multipliers must be finite and non-negative")
        n_sims = base_result.terminal_returns.shape[1]
        n_shocked = 0 if scenario.n_paths_pct == 0 else max(1, int(n_sims * scenario.n_paths_pct))

        stressed_returns = base_result.terminal_returns.copy()

        for i, inst in enumerate(base_result.instruments):
            base_rets = base_result.terminal_returns[i]

            # Select worst paths for shocking
            worst_idx = np.argsort(base_rets)[:n_shocked]
            if n_shocked == 0:
                continue

            # Price shock
            shock = scenario.price_shocks.get(inst, price_shock_all or 0.0)
            if shock != 0.0:
                stressed_returns[i, worst_idx] += np.log(1 + shock)

            # Vol multiplier: scale returns away from mean
            vol_mult = scenario.vol_multipliers.get(inst, 1.0 if vol_multiplier_all is None else vol_multiplier_all)
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
            (DEFAULT_STRESS_SCENARIOS[2], -0.15, None),   # Supply shock (moderate)
        ]
        results = []
        for scenario, price_shock, vol_mult in shocks:
            results.append(self.apply(base_result, scenario, price_shock, vol_mult))
        return results
