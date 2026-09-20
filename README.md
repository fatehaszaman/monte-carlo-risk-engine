# Monte Carlo Risk Engine

[Algorithm guide: pseudocode, time complexity, and memory](docs/ALGORITHM_GUIDE.md).

[![CI](https://github.com/fatehaszaman/monte-carlo-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/fatehaszaman/monte-carlo-risk-engine/actions/workflows/ci.yml)

A Python library for quantitative portfolio risk analysis: correlated Monte Carlo simulation, VaR / CVaR, stress testing, and portfolio aggregation. The engine is designed to be small, transparent, and reproducible, with an emphasis on the property checks and engineering discipline that systematic-research and risk-infrastructure work depend on.

## Why Monte Carlo for risk

Historical simulation is bounded by what actually happened inside your data window. A two-year history will not contain a 30% overnight shock or a correlation breakdown, but those are exactly the scenarios a risk system has to size for.

This engine combines two complementary approaches:

- **Statistical risk** -- a large set of correlated Geometric Brownian Motion (GBM) paths generated via Cholesky decomposition of the input correlation matrix. This captures the joint distribution implied by the calibration.
- **Scenario risk** -- discrete price and volatility shocks layered onto the worst-tail subset of paths. (Correlation-regime shocks are described in the code but not implemented; a correlation change cannot be applied to terminal returns after the fact, it requires re-simulating.) This makes tail events explicit rather than relying on them to appear by chance in the statistical sample.

Both are reported alongside a historical-simulation baseline so the dependence on modelling assumptions is visible rather than hidden.

## Features

- Correlated multi-instrument GBM simulation with Cholesky-based correlation handling, and a positive-semidefinite repair step for numerically noisy correlation matrices *estimated from history*. A matrix passed in directly by the caller is not repaired and must already be positive definite -- see [KNOWN_ISSUES.md](KNOWN_ISSUES.md) #4.
- Per-instrument and portfolio-level VaR and CVaR at configurable confidence levels, via either historical simulation or Monte Carlo paths.
- Pluggable stress-test layer for price and volatility shocks, with a `StressScenario` dataclass for defining custom ones. Three of the four built-in scenarios do something; `Correlation_Breakdown` is currently inert, and correlation shocks are not implemented at all -- see [KNOWN_ISSUES.md](KNOWN_ISSUES.md) #6 and #7.
- Portfolio aggregator that computes diversification benefit and per-instrument VaR concentration.
- Deterministic seeding throughout (`numpy.random.default_rng`) for reproducible runs.
- A property-based validation script that checks correlation recovery, GBM terminal moments, CVaR-vs-VaR ordering, and diversification non-negativity.

## Architecture

```
mc_risk/
├── simulation.py       # Correlated GBM engine: builds correlation, Cholesky, paths
├── var_cvar.py         # Historical and Monte Carlo VaR / CVaR (per-instrument and portfolio)
├── stress_test.py      # Scenario shocks applied to worst-tail subset of paths
├── portfolio_risk.py   # Aggregation, diversification benefit, concentration, report
└── __init__.py
examples/
├── demo.py             # End-to-end run on a 4-instrument synthetic portfolio
└── validate.py         # Property checks + a small benchmark
```

The data flow is one-directional: `simulation` -> `var_cvar` -> `portfolio_risk`, with `stress_test` reading a `SimulationResult` and producing a `StressResult` that the aggregator consumes. Each module is independently usable and has no hidden state beyond an explicitly seeded RNG.

## How to run

Requires Python 3.10+. Install runtime dependencies (numpy, pandas) and run the demo or the validation script:

```bash
pip install -r requirements.txt

# End-to-end demo: 10,000 paths x 252 steps, 4 instruments, full report
PYTHONPATH=. python examples/demo.py

# Property checks + small benchmark
PYTHONPATH=. python examples/validate.py
```

The demo prints the simulation parameters, the recovered correlation matrix, per-instrument VaR/CVaR at 95% and 99%, a historical comparison, the four stress scenarios, and a consolidated portfolio risk report.

`examples/validate.py` runs a 20,000-path simulation and asserts five properties: identical paths under a fixed seed, correlation recovery within 0.02 of the input matrix, terminal-return mean within Monte Carlo tolerance of the GBM theoretical value, `CVaR >= VaR` everywhere, and non-negative diversification benefit.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

70 tests across four files. They are written against the analytic properties of
the process the engine claims to simulate rather than against recorded output,
so they stay meaningful through a refactor:

- GBM moments checked in closed form: `E[S_T] = S_0 e^{mu T}` and
  `Var[log(S_T/S_0)] = sigma^2 T`.
- VaR and CVaR for a normal sample checked against `1.645 sigma` and the
  analytic expected shortfall `sigma phi(z_alpha)/alpha`.
- Coherence properties that must hold for any correct implementation: CVaR is
  never below VaR, VaR is monotone in the confidence level, heavier tails widen
  the CVaR-to-VaR ratio, and diversification cannot increase risk.
- Realised correlation of simulated returns is checked directly against the
  requested correlation, not only indirectly through VaR.

Writing them surfaced eight defects. The two highest-severity portfolio VaR
issues are fixed: terminal log returns are converted to simple returns before
currency P&L aggregation, and long/short books are normalised by gross exposure.
The six unresolved items are recorded in [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
with severity and a proposed fix.

Tests that assert a defect say so in the docstring and cite the issue number.
They pass by pinning the current behaviour, so each fix replaces the defect test
with a regression test for the corrected result.

## Reproducibility

- Every entry point that uses randomness takes an explicit `seed` and constructs a `numpy.random.default_rng(seed)`. The same seed produces bit-identical paths (asserted by `examples/validate.py`).
- All simulation parameters used in a run are echoed back inside `SimulationResult.params` (`n_simulations`, `n_steps`, `dt`, drift and volatility per instrument), so a given result is self-describing.
- No network calls, no implicit data downloads, no hidden global state. The demo generates its own synthetic history with a fixed seed.
- `requirements.txt` pins lower bounds on numpy and pandas only -- no other runtime dependencies, no optional accelerators that would change numerical output.

## Performance

Wall-clock times for the bundled simulation engine, re-measured on a small cloud VM (x86_64, 2 vCPU, numpy 2.5.3). These are illustrative; absolute numbers will vary by hardware.

| Configuration | Wall clock |
|---|---|
| 4 instruments x 10,000 paths x 252 steps | 0.29 s |
| 4 instruments x 50,000 paths x 252 steps | 1.63 s |
| 10 instruments x 10,000 paths x 252 steps | 0.69 s |

The engine is single-threaded NumPy with a Python-level loop over time steps; it has not been tuned for production-scale workloads. Vectorising the time loop or moving to a compiled backend would be the obvious next step if throughput mattered.

## Validation properties

`examples/validate.py` is the executable specification of what the engine claims to do. The current checks are:

1. **Reproducibility** -- two runs with the same seed produce identical path arrays.
2. **Correlation recovery** -- empirical correlation of simulated log returns is within 0.02 of the input correlation matrix for 20,000 paths.
3. **GBM terminal mean** -- empirical mean of terminal log returns is within Monte Carlo tolerance of `(mu - 0.5 * sigma^2) * T`.
4. **Tail ordering** -- `CVaR >= VaR` for every per-instrument result.
5. **Diversification non-negativity** -- sum of individual VaRs is at least portfolio VaR for the test portfolio.

These are properties, not regression snapshots, so they remain meaningful if the implementation is refactored.

## Design notes

**Cholesky decomposition for correlation.** Independent standard normals `Z` are transformed by `L`, the lower-triangular Cholesky factor of the correlation matrix `Sigma = L L^T`. The transformed draws `L Z` have the target correlation by construction. If a matrix *estimated from history* is not positive definite (a common artefact of short or misaligned histories), it is shifted by a small multiple of the identity and renormalised before factorisation. This repair is not applied to a matrix supplied directly by the caller, so passing perfect correlation raises `LinAlgError`.

**Layering stress on simulation.** A pure Monte Carlo system understates tail risk for rare, high-impact events because those events do not appear in calibrated distributions with the frequency they appear in the world. Stress scenarios make these inputs explicit. The engine applies them to the worst `n_paths_pct` of paths so the resulting distribution is interpretable: the base distribution unchanged in the body, augmented in the tail by the scenario you defined.

The cost of that choice is worth stating plainly. Because only the tail is shocked, `var_95_shift` moves very little for a scenario confined to the worst 5% of paths, while `cvar_95_shift` moves a lot. Read the CVaR shift, not the VaR shift, as the measure of scenario impact.

**Historical vs Monte Carlo VaR.** Historical VaR is shown alongside Monte Carlo so that the dependence on model choice is visible. Historical VaR is bounded by the history window and cannot describe events outside it; Monte Carlo VaR can, but only inside whatever model you calibrated. Reporting both and letting the reader compare is the honest framing.

## Scope and limitations

This is a clean, transparent implementation of a textbook risk pipeline. It is suitable for research, teaching, and as a baseline to build against. It is not a production trading-firm risk system -- it has no instrument-specific pricing models, no jump-diffusion or stochastic-vol dynamics, no incremental / marginal VaR, and no integration with a position or P&L source of truth. Anyone using it for real risk decisions should treat the GBM and constant-correlation assumptions as the load-bearing simplifications they are.

## Requirements

- Python 3.10 or newer
- numpy >= 1.24
- pandas >= 2.0

## License

MIT. See [LICENSE](LICENSE).
