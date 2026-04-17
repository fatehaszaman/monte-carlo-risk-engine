"""
demo.py
-------
End-to-end demonstration of the Monte Carlo risk engine.

Simulates a portfolio of four instruments with realistic correlation
structure, computes VaR/CVaR at 95% and 99%, runs stress scenarios,
and prints a full portfolio risk report.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from mc_risk import (
    MonteCarloEngine, InstrumentSpec,
    VaRCalculator,
    StressTester,
    PortfolioRiskAggregator,
)

# ── 1. Define portfolio ───────────────────────────────────────────────────────

rng = np.random.default_rng(42)
n_history = 504  # ~2 years of daily returns

# Simulate correlated historical returns (would be real data in production)
# Instrument A and B are positively correlated (0.65)
# Instrument C is weakly correlated with A/B (0.20)
# Instrument D is negatively correlated with A (FX hedge, -0.30)

cov_matrix = np.array([
    [0.018**2,  0.65*0.018*0.022,  0.20*0.018*0.015, -0.30*0.018*0.012],
    [0.65*0.018*0.022, 0.022**2,   0.25*0.022*0.015, -0.20*0.022*0.012],
    [0.20*0.018*0.015, 0.25*0.022*0.015, 0.015**2,    0.10*0.015*0.012],
    [-0.30*0.018*0.012, -0.20*0.022*0.012, 0.10*0.015*0.012, 0.012**2],
])
means = np.array([0.0003, 0.0002, 0.0004, -0.0001])
hist_returns = rng.multivariate_normal(means, cov_matrix, size=n_history)

instruments = [
    InstrumentSpec("INST_A", current_price=250.0, returns_history=hist_returns[:, 0]),
    InstrumentSpec("INST_B", current_price=180.0, returns_history=hist_returns[:, 1]),
    InstrumentSpec("INST_C", current_price=95.0,  returns_history=hist_returns[:, 2]),
    InstrumentSpec("INST_D", current_price=120.0, returns_history=hist_returns[:, 3]),
]

position_values = {
    "INST_A": 1_200_000,
    "INST_B":   800_000,
    "INST_C":   500_000,
    "INST_D":   400_000,
}

print("=" * 60)
print("  PORTFOLIO SETUP")
print("=" * 60)
for name, val in position_values.items():
    print(f"  {name:<10} ${val:>12,.0f}")
print(f"  {'TOTAL':<10} ${sum(position_values.values()):>12,.0f}")

# ── 2. Run Monte Carlo simulation ─────────────────────────────────────────────

print("\nRunning 10,000-path Monte Carlo simulation...")
engine = MonteCarloEngine(n_simulations=10_000, n_steps=252, seed=42)
sim_result = engine.simulate(instruments)

print(f"Simulated {sim_result.params['n_simulations']:,} paths × "
      f"{sim_result.params['n_steps']} steps")
print(f"\nEstimated annual vols:")
for i, inst in enumerate(sim_result.instruments):
    print(f"  {inst}: {sim_result.params['sigmas'][i]*100:.1f}%")

print(f"\nCorrelation matrix:")
corr_df_str = "\n".join(
    f"  {sim_result.instruments[i]}: " +
    "  ".join(f"{sim_result.correlation_matrix[i,j]:+.2f}"
              for j in range(len(sim_result.instruments)))
    for i in range(len(sim_result.instruments))
)
print(corr_df_str)

# ── 3. VaR / CVaR ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  VaR / CVaR (Monte Carlo, per instrument)")
print("=" * 60)

var_calc = VaRCalculator(confidence_levels=[0.95, 0.99])
mc_results = var_calc.monte_carlo(sim_result, position_values)
var_df = var_calc.to_dataframe(mc_results)
print(var_df[["name", "confidence_level", "var_pct", "cvar_pct",
              "var_abs", "cvar_abs"]].to_string(index=False))

# Historical VaR on first instrument for comparison
print(f"\n  Historical VaR/CVaR (INST_A, for comparison):")
hist_results = var_calc.historical(
    hist_returns[:, 0], position_values["INST_A"], "INST_A (hist)"
)
hist_df = var_calc.to_dataframe(hist_results)
print(hist_df[["name", "confidence_level", "var_pct", "cvar_pct"]].to_string(index=False))

# ── 4. Stress testing ─────────────────────────────────────────────────────────

print("\nRunning stress scenarios...")
stress_tester = StressTester(rng_seed=42)
stress_results = stress_tester.run_all(sim_result)

print(f"\n  Stress scenario CVaR shifts (vs base, 95%):")
for sr in stress_results:
    avg_shift = sum(sr.cvar_95_shift.values()) / len(sr.cvar_95_shift)
    worst = min(sr.worst_case_return.values())
    print(f"  {sr.scenario_name:<30} avg CVaR shift: {avg_shift*100:>+6.2f}%  "
          f"worst case return: {worst*100:>+6.2f}%")

# ── 5. Portfolio risk report ──────────────────────────────────────────────────

aggregator = PortfolioRiskAggregator(confidence_level=0.95)
report = aggregator.build_report(sim_result, position_values, stress_results)
report.print_report()
