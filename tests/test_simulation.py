"""GBM engine. Tested against the analytic moments of the process it claims to
simulate, not against recorded output.
"""
import numpy as np
import pytest

from mc_risk import InstrumentSpec, MonteCarloEngine


def test_paths_have_the_documented_shape(sim):
    assert sim.paths.shape == (2, 4000, 51)
    assert sim.terminal_prices.shape == (2, 4000)


def test_all_paths_start_at_the_current_price(sim):
    assert np.all(sim.paths[:, :, 0] == 100.0)


def test_gbm_prices_stay_strictly_positive(sim):
    assert np.all(sim.paths > 0), "GBM cannot reach zero; a negative price is a bug"


def test_expected_terminal_price_matches_gbm_theory():
    """E[S_T] = S_0 * exp(mu * T). With mu=0 the expectation is S_0."""
    specs = [InstrumentSpec("A", 100.0, mu=0.0, sigma=0.20)]
    sim = MonteCarloEngine(40_000, 252, seed=11).simulate(specs, np.eye(1))
    assert sim.terminal_prices[0].mean() == pytest.approx(100.0, rel=0.02)


def test_positive_drift_raises_the_expected_terminal_price():
    specs = [InstrumentSpec("A", 100.0, mu=0.10, sigma=0.20)]
    sim = MonteCarloEngine(40_000, 252, seed=12).simulate(specs, np.eye(1))
    assert sim.terminal_prices[0].mean() == pytest.approx(100 * np.exp(0.10), rel=0.03)


def test_terminal_log_return_variance_matches_sigma_squared_t():
    """Var[log(S_T/S_0)] = sigma^2 * T."""
    specs = [InstrumentSpec("A", 100.0, mu=0.0, sigma=0.20)]
    sim = MonteCarloEngine(40_000, 252, seed=13).simulate(specs, np.eye(1))
    assert sim.terminal_returns[0].var() == pytest.approx(0.20**2, rel=0.05)


def test_zero_volatility_produces_a_deterministic_path():
    specs = [InstrumentSpec("A", 100.0, mu=0.0, sigma=0.0)]
    sim = MonteCarloEngine(50, 10, seed=14).simulate(specs, np.eye(1))
    assert np.allclose(sim.terminal_prices[0], 100.0)


def test_realised_correlation_matches_the_requested_correlation():
    """The Cholesky step is the whole point of the engine, so it gets a direct
    check rather than an indirect one through VaR."""
    specs = [
        InstrumentSpec("A", 100.0, mu=0.0, sigma=0.20),
        InstrumentSpec("B", 100.0, mu=0.0, sigma=0.20),
    ]
    target = np.array([[1.0, 0.7], [0.7, 1.0]])
    sim = MonteCarloEngine(20_000, 100, seed=15).simulate(specs, target)
    realised = np.corrcoef(sim.terminal_returns)[0, 1]
    assert realised == pytest.approx(0.7, abs=0.03)


def test_independent_instruments_show_near_zero_correlation():
    specs = [
        InstrumentSpec("A", 100.0, mu=0.0, sigma=0.20),
        InstrumentSpec("B", 100.0, mu=0.0, sigma=0.20),
    ]
    sim = MonteCarloEngine(20_000, 100, seed=16).simulate(specs, np.eye(2))
    assert abs(np.corrcoef(sim.terminal_returns)[0, 1]) < 0.03


def test_same_seed_reproduces_identical_paths():
    specs = [InstrumentSpec("A", 100.0, mu=0.05, sigma=0.2)]
    a = MonteCarloEngine(500, 20, seed=99).simulate(specs, np.eye(1))
    b = MonteCarloEngine(500, 20, seed=99).simulate(specs, np.eye(1))
    assert np.array_equal(a.terminal_prices, b.terminal_prices)


def test_different_seeds_give_different_paths():
    specs = [InstrumentSpec("A", 100.0, mu=0.05, sigma=0.2)]
    a = MonteCarloEngine(500, 20, seed=1).simulate(specs, np.eye(1))
    b = MonteCarloEngine(500, 20, seed=2).simulate(specs, np.eye(1))
    assert not np.array_equal(a.terminal_prices, b.terminal_prices)


def test_parameters_are_estimated_and_annualised_from_history():
    rng = np.random.default_rng(3)
    daily = rng.normal(0.0004, 0.0126, 5000)   # ~10% drift, ~20% vol annualised
    sim = MonteCarloEngine(100, 10, seed=1).simulate(
        [InstrumentSpec("A", 100.0, returns_history=daily)], np.eye(1)
    )
    assert sim.params["sigmas"][0] == pytest.approx(0.20, rel=0.06)
    assert sim.params["mus"][0] == pytest.approx(0.10, rel=0.20)


def test_missing_parameters_fall_back_to_documented_defaults():
    sim = MonteCarloEngine(50, 5, seed=1).simulate(
        [InstrumentSpec("A", 100.0)], np.eye(1)
    )
    assert sim.params["mus"][0] == 0.0
    assert sim.params["sigmas"][0] == 0.20


def test_estimated_correlation_recovers_a_known_relationship():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 0.01, 4000)
    b = 0.8 * a + np.sqrt(1 - 0.8**2) * rng.normal(0, 0.01, 4000)
    sim = MonteCarloEngine(100, 5, seed=1).simulate([
        InstrumentSpec("A", 100.0, returns_history=a),
        InstrumentSpec("B", 100.0, returns_history=b),
    ])
    assert sim.correlation_matrix[0, 1] == pytest.approx(0.8, abs=0.04)


def test_partial_history_silently_discards_known_correlation():
    """DEFECT: if any instrument lacks returns_history, _build_correlation_matrix
    returns the identity and throws away the correlation it could have measured
    for the others.

    Assuming independence understates portfolio risk, which is the dangerous
    direction for a risk engine, and nothing in the output records that the
    fallback was taken. See KNOWN_ISSUES.md #5.
    """
    rng = np.random.default_rng(5)
    a = rng.normal(0, 0.01, 1000)
    b = 0.9 * a + 0.1 * rng.normal(0, 0.01, 1000)
    sim = MonteCarloEngine(100, 5, seed=1).simulate([
        InstrumentSpec("A", 100.0, returns_history=a),
        InstrumentSpec("B", 100.0, returns_history=b),
        InstrumentSpec("C", 100.0, mu=0.0, sigma=0.2),   # no history
    ])
    assert np.array_equal(sim.correlation_matrix, np.eye(3)), (
        "A and B are 0.9 correlated but the whole matrix collapsed to identity"
    )


def test_volatility_estimate_uses_population_standard_deviation():
    """numpy defaults to ddof=0 while pandas defaults to ddof=1. Which one is in
    use slightly changes every downstream VaR, so it is pinned deliberately."""
    daily = np.array([0.01, -0.02, 0.015, 0.0, -0.005])
    eng = MonteCarloEngine(10, 2, seed=1)
    _mu, sigma = eng._estimate_params(daily)
    assert sigma == pytest.approx(daily.std(ddof=0) * np.sqrt(252))


def test_simulation_result_lacks_the_n_simulations_attribute_var_cvar_probes_for(sim):
    """var_cvar.portfolio_monte_carlo tests `hasattr(sim_result,'n_simulations')`,
    but SimulationResult keeps that value inside `params`. The attribute branch
    is therefore dead code and the shape fallback always runs.
    Harmless, but it is dead code that reads as a safeguard."""
    assert not hasattr(sim, "n_simulations")
    assert sim.params["n_simulations"] == 4000
