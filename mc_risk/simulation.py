"""
simulation.py
-------------
Correlated Monte Carlo simulation engine using Geometric Brownian Motion.

Generates joint price paths for a portfolio of instruments, preserving
the correlation structure between them. This matters because naive
independent simulations underestimate portfolio risk when instruments
are correlated — e.g. a commodity and its input costs moving together.

Simulation approach
-------------------
1. Estimate drift (mu) and volatility (sigma) from historical return data,
   or accept user-supplied parameters.
2. Compute the historical correlation matrix from return data.
3. Apply Cholesky decomposition to the correlation matrix to produce
   correlated standard normal draws.
4. Simulate N paths of T steps using the GBM discretization:
     S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
   where Z is a correlated standard normal vector.

The output is a SimulationResult containing the full path array and
derived statistics used by the VaR/CVaR calculator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class InstrumentSpec:
    """
    Specification for a single simulated instrument.

    Parameters
    ----------
    name : str
    current_price : float
        Starting price (S0).
    mu : float, optional
        Annual drift. Estimated from history if None.
    sigma : float, optional
        Annual volatility. Estimated from history if None.
    returns_history : np.ndarray, optional
        Historical simple daily returns, (P_t / P_{t-1}) - 1, used to estimate
        mu/sigma and correlation. Passing log returns understates drift by
        sigma^2 / 2. See KNOWN_ISSUES.md #9.
    """
    name: str
    current_price: float
    mu: Optional[float] = None
    sigma: Optional[float] = None
    returns_history: Optional[np.ndarray] = None


@dataclass
class SimulationResult:
    """
    Output of a Monte Carlo simulation run.

    Attributes
    ----------
    instruments : list[str]
        Names of simulated instruments.
    paths : np.ndarray
        Shape (n_instruments, n_simulations, n_steps + 1).
        paths[i, j, t] = price of instrument i in simulation j at step t.
    terminal_prices : np.ndarray
        Shape (n_instruments, n_simulations).
        Final price of each instrument in each simulation.
    terminal_returns : np.ndarray
        Shape (n_instruments, n_simulations).
        Log return from S0 to terminal price.
    correlation_matrix : np.ndarray
        Estimated or supplied correlation matrix used in simulation.
    params : dict
        Simulation parameters (n_sims, n_steps, dt, mus, sigmas).
    """
    instruments: list[str]
    paths: np.ndarray
    terminal_prices: np.ndarray
    terminal_returns: np.ndarray
    correlation_matrix: np.ndarray
    params: dict


def _correlation_factor(corr: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Return a factor F with F @ F.T == corr for a valid correlation matrix.

    Validates shape, symmetry, unit diagonal and entry bounds. Uses Cholesky
    when the matrix is positive definite (O(n^3), cheapest and exact). When it
    is only positive semidefinite (for example perfect correlation, a valid
    stress assumption), falls back to an eigendecomposition factor
    V * sqrt(max(lambda, 0)), which reproduces the input exactly. Matrices with
    eigenvalues below -tol are not correlation matrices and raise ValueError
    instead of being silently altered.
    """
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"correlation matrix must be square, got {corr.shape}")
    if not np.allclose(corr, corr.T, atol=tol):
        raise ValueError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=tol):
        raise ValueError("correlation matrix must have a unit diagonal")
    if np.any(np.abs(corr) > 1.0 + tol):
        raise ValueError("correlation entries must lie in [-1, 1]")
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr)
        if eigvals.min() < -tol:
            raise ValueError(
                f"correlation matrix is not positive semidefinite "
                f"(min eigenvalue {eigvals.min():.3g})"
            )
        return eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))


class MonteCarloEngine:
    """
    Runs correlated GBM Monte Carlo simulations for a portfolio of instruments.

    Parameters
    ----------
    n_simulations : int
        Number of simulation paths. Default 10,000.
    n_steps : int
        Number of time steps per path. Default 252 (one trading year).
    dt : float
        Time step size in years. Default 1/252 (one trading day).
    seed : int, optional
        Random seed for reproducibility.
    annualization_factor : int
        Periods per year in historical data. Default 252 (daily).
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        n_steps: int = 252,
        dt: float = 1 / 252,
        seed: Optional[int] = None,
        annualization_factor: int = 252,
    ):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.dt = dt
        self.annualization_factor = annualization_factor
        self.rng = np.random.default_rng(seed)

    def _estimate_params(self, returns: np.ndarray) -> tuple[float, float]:
        """Estimate annualized drift and volatility from daily returns."""
        mu_daily = returns.mean()
        sigma_daily = returns.std()
        mu_annual = mu_daily * self.annualization_factor
        sigma_annual = sigma_daily * np.sqrt(self.annualization_factor)
        return mu_annual, sigma_annual

    def _build_correlation_matrix(
        self,
        specs: list[InstrumentSpec],
    ) -> np.ndarray:
        """Build correlation matrix from historical returns."""
        n = len(specs)
        histories = []
        for spec in specs:
            if spec.returns_history is not None:
                histories.append(spec.returns_history)

        if len(histories) < n:
            # Not all instruments have history — use identity
            return np.eye(n)

        min_len = min(len(h) for h in histories)
        returns_matrix = np.array([h[-min_len:] for h in histories])

        corr = np.corrcoef(returns_matrix)
        # Ensure positive semidefinite (numerical stability)
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < 0:
            corr += (-eigvals.min() + 1e-8) * np.eye(n)
            # Re-normalize to correlation matrix
            d = np.sqrt(np.diag(corr))
            corr = corr / np.outer(d, d)

        return corr

    def simulate(
        self,
        specs: list[InstrumentSpec],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> SimulationResult:
        """
        Run correlated GBM simulation for all instruments.

        Parameters
        ----------
        specs : list[InstrumentSpec]
        correlation_matrix : np.ndarray, optional
            Override correlation matrix. Estimated from history if None.

        Returns
        -------
        SimulationResult
        """
        n_inst = len(specs)

        # Resolve parameters
        mus = []
        sigmas = []
        for spec in specs:
            if spec.mu is not None and spec.sigma is not None:
                mus.append(spec.mu)
                sigmas.append(spec.sigma)
            elif spec.returns_history is not None:
                mu, sigma = self._estimate_params(spec.returns_history)
                mus.append(mu)
                sigmas.append(sigma)
            else:
                # Default: zero drift, 20% vol
                mus.append(0.0)
                sigmas.append(0.20)

        mus = np.array(mus)
        sigmas = np.array(sigmas)

        # Correlation matrix
        if correlation_matrix is None:
            correlation_matrix = self._build_correlation_matrix(specs)

        # Cholesky decomposition for correlated draws
        L = _correlation_factor(correlation_matrix)

        # Simulate paths: shape (n_inst, n_simulations, n_steps + 1)
        paths = np.zeros((n_inst, self.n_simulations, self.n_steps + 1))
        for i, spec in enumerate(specs):
            paths[i, :, 0] = spec.current_price

        sqrt_dt = np.sqrt(self.dt)

        for t in range(1, self.n_steps + 1):
            # Draw independent standard normals: (n_inst, n_simulations)
            Z_indep = self.rng.standard_normal((n_inst, self.n_simulations))
            # Apply Cholesky to correlate: L @ Z_indep
            Z_corr = L @ Z_indep  # (n_inst, n_simulations)

            for i in range(n_inst):
                drift = (mus[i] - 0.5 * sigmas[i] ** 2) * self.dt
                diffusion = sigmas[i] * sqrt_dt * Z_corr[i]
                paths[i, :, t] = paths[i, :, t - 1] * np.exp(drift + diffusion)

        terminal_prices = paths[:, :, -1]
        s0 = np.array([spec.current_price for spec in specs])
        terminal_returns = np.log(terminal_prices / s0[:, np.newaxis])

        return SimulationResult(
            instruments=[spec.name for spec in specs],
            paths=paths,
            terminal_prices=terminal_prices,
            terminal_returns=terminal_returns,
            correlation_matrix=correlation_matrix,
            params={
                "n_simulations": self.n_simulations,
                "n_steps": self.n_steps,
                "dt": self.dt,
                "mus": mus.tolist(),
                "sigmas": sigmas.tolist(),
            },
        )
