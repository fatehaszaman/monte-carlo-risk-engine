"""Fixtures with analytically known properties.

Where possible the expected answer comes from theory (GBM moments, normal
quantiles) rather than from a recorded output of this code.
"""
import numpy as np
import pytest

from mc_risk import InstrumentSpec, MonteCarloEngine

SEED = 20260918


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def normal_returns(rng):
    """10k draws from N(0, 1%). Normal quantiles give the expected VaR."""
    return rng.normal(0.0, 0.01, 10_000)


@pytest.fixture
def engine():
    return MonteCarloEngine(n_simulations=4000, n_steps=50, seed=SEED)


@pytest.fixture
def two_specs():
    return [
        InstrumentSpec("A", 100.0, mu=0.0, sigma=0.20),
        InstrumentSpec("B", 100.0, mu=0.0, sigma=0.20),
    ]


@pytest.fixture
def sim(engine, two_specs):
    return engine.simulate(two_specs, correlation_matrix=np.eye(2))
