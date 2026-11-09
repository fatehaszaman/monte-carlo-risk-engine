"""Reproduce the wall-clock figures in the README's Performance section.

Run from the repository root:

    python scripts/benchmark.py          # 4 instruments x 10,000 paths x 252 steps
    python scripts/benchmark.py --all    # every configuration in the README table

Each configuration builds instruments with fixed drift and volatility, uses a
fixed correlation matrix and seed 42, and times MonteCarloEngine.simulate()
with time.perf_counter() over several runs. The median is reported because it
is less sensitive to a single slow run than the mean. Results vary by hardware.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_risk.simulation import InstrumentSpec, MonteCarloEngine  # noqa: E402

SEED = 42
N_STEPS = 252
CONFIGS = [(4, 10_000), (4, 50_000), (10, 10_000)]


def correlation_matrix(n: int, rho: float = 0.3) -> np.ndarray:
    """Equicorrelation matrix: 1 on the diagonal, rho elsewhere. Positive definite for 0 <= rho < 1."""
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return corr


def instruments(n: int) -> list[InstrumentSpec]:
    return [
        InstrumentSpec(name=f"ASSET_{i}", current_price=100.0, mu=0.05, sigma=0.20 + 0.02 * i)
        for i in range(n)
    ]


def time_config(n_instruments: int, n_paths: int, runs: int) -> list[float]:
    specs = instruments(n_instruments)
    corr = correlation_matrix(n_instruments)
    times = []
    for _ in range(runs):
        engine = MonteCarloEngine(n_simulations=n_paths, n_steps=N_STEPS, seed=SEED)
        start = time.perf_counter()
        engine.simulate(specs, correlation_matrix=corr)
        times.append(time.perf_counter() - start)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="run every README configuration")
    parser.add_argument("--runs", type=int, default=5, help="timed runs per configuration (default 5)")
    args = parser.parse_args()

    print(f"Python    {platform.python_version()} ({platform.python_implementation()})")
    print(f"Platform  {platform.platform()}")
    print(f"Processor {platform.processor() or platform.machine()}")
    print(f"NumPy     {np.__version__}")
    print(f"Seed {SEED}, {N_STEPS} steps, median of {args.runs} runs\n")

    configs = CONFIGS if args.all else CONFIGS[:1]
    for n_instruments, n_paths in configs:
        times = time_config(n_instruments, n_paths, args.runs)
        print(
            f"{n_instruments:>2} instruments x {n_paths:>6,} paths x {N_STEPS} steps: "
            f"median {statistics.median(times):.3f} s "
            f"(min {min(times):.3f}, max {max(times):.3f})"
        )


if __name__ == "__main__":
    main()
