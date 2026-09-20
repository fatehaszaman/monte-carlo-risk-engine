# Algorithm guide

These cards describe selected implemented algorithms, not a separate executable
version. Costs assume fixed-width numerical arithmetic; library work, retained
outputs, and preprocessing are separated where they matter.

## Correlated price paths

Implementation: [`MonteCarloEngine.simulate`](../mc_risk/simulation.py).

```text
# Monte Carlo / Dense Correlated GBM
# Goal: generate correlated paths and terminal log returns.
# Input: A instruments, P paths, T time steps, parameters, correlation matrix
# Output: paths[A, P, T+1], terminal prices and log returns
# Time: O(A^3 + T*A^2*P), given resolved parameters and correlation
# Memory: O(A*P*T + A^2 + A*P), including retained paths and draw buffers
# Auxiliary beyond returned arrays: O(A^2 + A*P)

RESOLVE drift and volatility for each instrument
BUILD or accept the correlation matrix
L = CHOLESKY(correlation)
ALLOCATE all paths and initialize their starting prices
FOR each time step:
    Z = independent standard-normal draws[A, P]
    correlated_Z = L @ Z
    FOR each instrument:
        paths[next] = paths[current] * EXP(drift + scaled correlated_Z)
RETURN paths, final prices, LOG(final / initial), parameters
```

Why: each dense matrix multiplication costs O(A²P), and the implementation
retains every time step rather than just terminal prices. NumPy vectorization
does not remove those arithmetic or storage costs.

Preprocessing: scanning supplied histories costs O(H_total). If all histories
are used to estimate correlation, an A-by-H aligned matrix needs O(AH) memory,
correlation costs O(A²H), and the eigenvalue check costs O(A³). H is the shortest
history length; the history alignment here is positional, not date-aware.

Watch: Cholesky can fail for an invalid or singular matrix. Positive prices,
sensible step/path counts, valid volatility inputs, and correlation conditioning
need separate validation. Simulation output is not proof of model calibration.

## Tail-risk summary

Implementation: [`VaRCalculator._compute`](../mc_risk/var_cvar.py).

```text
# Empirical Quantile / Tail Mean
# Input: N returns, confidence alpha, position value
# Output: VaR, CVaR, and the retained tail returns
# Time: O(N) scans + Q(N), where Q is the library quantile cost
# Memory: O(N), including quantile workspace, mask, and returned tail copy

threshold = QUANTILE(returns, 1 - alpha)
tail = returns WHERE return <= threshold
VaR = -threshold
CVaR = -MEAN(tail) if tail is nonempty, otherwise VaR
RETURN percentages, currency-scaled values, and tail
```

A sorting-based quantile gives a conservative O(N log N) implementation model;
selection-based routines can do better. Do not treat that model as a measured
NumPy runtime guarantee. Repeating this for L confidence levels repeats the work
and may retain O(LN) tail values.

Ties at the threshold all enter the tail, so its size need not be exactly
(1-alpha)N. Empty arrays and NaNs require caller care. The simulation path
supplies log returns; the VaR calculator converts them to simple returns before
computing signed position P&L and normalises portfolio results by gross exposure.
