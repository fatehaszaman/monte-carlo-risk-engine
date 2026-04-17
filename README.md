# Monte Carlo Risk Engine

A Python risk engine for computing VaR, CVaR, and stress-test exposure across a correlated portfolio of instruments.

## Why Monte Carlo for risk?

Historical simulation is limited by what actually happened in your data window. A two-year history won't contain a 30% overnight shock or a correlation breakdown -- but those are exactly the scenarios you need to size for.

This engine uses Monte Carlo simulation to generate a large synthetic distribution of outcomes (10,000+ paths), preserving the correlation structure between instruments via Cholesky decomposition. Discrete stress scenarios are layered on top to model tail events that don't appear in the statistical distribution.

## Components

### `mc_risk/simulation.py` -- Correlated GBM Engine

Simulates joint price paths for N instruments using Geometric Brownian Motion with a shared correlation structure:

1. Estimate drift and volatility from historical returns (or accept user-supplied parameters)
2. Build the correlation matrix from historical return data
3. Cholesky-decompose the correlation matrix to produce correlated normal draws
4. Simulate paths: `S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)`

Handles numerical stability (positive semidefinite correction) and supports arbitrary instrument counts.

### `mc_risk/var_cvar.py` -- VaR / CVaR Calculator

Computes VaR and CVaR at configurable confidence levels via two methods:

| Method | Description |
|---|---|
| Historical simulation | Uses realized return distribution directly -- no distributional assumptions |
| Monte Carlo | Uses simulated paths from the GBM engine |

Both per-instrument and portfolio-level (correlation-preserving) VaR/CVaR are supported.

CVaR over VaR: two portfolios can have identical VaR but very different tail shapes. CVaR captures the expected loss beyond the VaR threshold, which gives a more complete picture of tail risk.

### `mc_risk/stress_test.py` -- Stress Testing Layer

Applies discrete scenario shocks to a subset of Monte Carlo paths:

| Scenario | Description |
|---|---|
| `Sharp_Selloff` | Broad price decline across all instruments |
| `Vol_Spike` | 2x volatility -- fat-tail / crisis conditions |
| `Correlation_Breakdown` | Correlations go to zero -- diversification disappears |
| `Supply_Shock` | Primary commodity shock with secondary spillover |

Custom scenarios are defined via `StressScenario` with per-instrument price shocks, vol multipliers, and optional correlation overrides.

### `mc_risk/portfolio_risk.py` -- Portfolio Aggregator

Aggregates results into a portfolio risk report:
- Portfolio-level VaR/CVaR (preserving correlations)
- Diversification benefit: sum of individual VaRs minus portfolio VaR
- Concentration table: % contribution to total portfolio VaR per instrument
- Stress scenario summary: CVaR shift and worst-case return per scenario

## Quickstart

```bash
pip install -r requirements.txt
PYTHONPATH=. python examples/demo.py
```

The demo runs a 10,000-path simulation across a four-instrument portfolio, computes VaR/CVaR at 95% and 99%, runs four stress scenarios, and prints a full risk report.

## Project Structure

```
mc_risk_engine/
├── mc_risk/
│   ├── simulation.py       # Correlated GBM Monte Carlo engine
│   ├── var_cvar.py         # VaR and CVaR calculator
│   ├── stress_test.py      # Stress scenario engine
│   ├── portfolio_risk.py   # Portfolio aggregator and risk report
│   └── __init__.py
├── examples/
│   └── demo.py             # End-to-end demo
├── requirements.txt
└── README.md
```

## Requirements

Python 3.10+, pandas, numpy

## Design Notes

**Cholesky decomposition for correlation**
To generate correlated random variables, the correlation matrix is Cholesky-decomposed into a lower triangular matrix L such that L * L^T = sigma. Independent standard normals are then transformed by L to produce correlated draws. Numerically stable and exact for positive definite matrices.

**Layering stress on simulation**
Pure Monte Carlo underestimates tail risk for rare, high-impact events because they appear infrequently in calibrated distributions. Stress scenarios model these explicitly -- not as statistical outcomes but as deliberate "what if" inputs -- giving a combined view of statistical and scenario-based risk.

**Historical vs Monte Carlo VaR**
Historical simulation is shown alongside Monte Carlo for comparison. Historical VaR is constrained by the length and representativeness of the data window; Monte Carlo extends the distribution but depends on the validity of the GBM model. Neither is sufficient alone.
