"""
validate.py
-----------
Lightweight validation script for the Monte Carlo risk engine.

Runs a small set of property checks that any correctly implemented
GBM + VaR/CVaR pipeline should pass, and prints a wall-clock benchmark
for the default simulation size. Designed to be cheap (< a few seconds
on a laptop) and dependency-free beyond what is already in
requirements.txt.

Checks performed
----------------
1. Reproducibility: two runs with the same seed produce identical paths.
2. Cholesky / correlation: empirical correlation of the simulated log
   returns is close to the input correlation matrix.
3. GBM mean: empirical mean of terminal log returns is close to the
   theoretical value (mu - 0.5 * sigma**2) * T for large n_simulations.
4. CVaR >= VaR for every result returned by the VaR calculator.
5. Portfolio VaR <= sum of individual VaRs when correlations are not
   uniformly +1 (diversification benefit is non-negative).

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from mc_risk import (
    MonteCarloEngine,
    InstrumentSpec,
    VaRCalculator,
    PortfolioRiskAggregator,
)


def _check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


def main() -> int:
    rng = np.random.default_rng(0)
    n_hist = 500
    cov = np.array([
        [0.02**2,            0.5 * 0.02 * 0.025,  0.0],
        [0.5 * 0.02 * 0.025, 0.025**2,           -0.3 * 0.025 * 0.015],
        [0.0,               -0.3 * 0.025 * 0.015, 0.015**2],
    ])
    means = np.array([0.0002, 0.0001, 0.0003])
    hist = rng.multivariate_normal(means, cov, size=n_hist)

    specs = [
        InstrumentSpec("A", 100.0, returns_history=hist[:, 0]),
        InstrumentSpec("B", 200.0, returns_history=hist[:, 1]),
        InstrumentSpec("C", 50.0,  returns_history=hist[:, 2]),
    ]

    engine = MonteCarloEngine(n_simulations=20_000, n_steps=252, seed=123)

    print("Running validation simulation...")
    t0 = time.perf_counter()
    sim_a = engine.simulate(specs)
    elapsed = time.perf_counter() - t0
    print(f"  20,000 paths x 252 steps x 3 instruments: {elapsed:.2f}s\n")

    print("Property checks:")
    all_ok = True

    # 1. Reproducibility
    engine_b = MonteCarloEngine(n_simulations=20_000, n_steps=252, seed=123)
    sim_b = engine_b.simulate(specs)
    repro_ok = np.array_equal(sim_a.paths, sim_b.paths)
    all_ok &= _check("reproducibility (same seed -> identical paths)", repro_ok)

    # 2. Correlation recovery
    step_log_returns = np.diff(np.log(sim_a.paths), axis=2)
    flat = step_log_returns.reshape(len(specs), -1)
    empirical_corr = np.corrcoef(flat)
    corr_err = np.abs(empirical_corr - sim_a.correlation_matrix).max()
    all_ok &= _check(
        "correlation recovery (|empirical - input| <= 0.02)",
        corr_err <= 0.02,
        f"max abs error = {corr_err:.4f}",
    )

    # 3. GBM terminal mean
    mus = np.array(sim_a.params["mus"])
    sigmas = np.array(sim_a.params["sigmas"])
    T = sim_a.params["n_steps"] * sim_a.params["dt"]
    theoretical = (mus - 0.5 * sigmas**2) * T
    empirical = sim_a.terminal_returns.mean(axis=1)
    mean_err = np.abs(empirical - theoretical).max()
    all_ok &= _check(
        "GBM terminal mean within Monte Carlo tolerance",
        mean_err < 0.02,
        f"max abs error = {mean_err:.4f}",
    )

    # 4. CVaR >= VaR
    positions = {"A": 1_000_000, "B": 500_000, "C": 300_000}
    var_calc = VaRCalculator(confidence_levels=[0.95, 0.99])
    results = var_calc.monte_carlo(sim_a, positions)
    cvar_ok = all(r.cvar_pct >= r.var_pct - 1e-12 for r in results)
    all_ok &= _check("CVaR >= VaR for all per-instrument results", cvar_ok)

    # 5. Diversification benefit non-negative
    aggregator = PortfolioRiskAggregator(confidence_level=0.95)
    report = aggregator.build_report(sim_a, positions)
    div_ok = report.diversification_benefit_var >= -1e-6
    all_ok &= _check(
        "diversification benefit >= 0 (sum indiv VaR >= portfolio VaR)",
        div_ok,
        f"benefit = ${report.diversification_benefit_var:,.0f}",
    )

    print()
    if all_ok:
        print("All checks passed.")
        return 0
    print("One or more checks FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
