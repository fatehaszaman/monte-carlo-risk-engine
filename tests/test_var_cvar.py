"""VaR and CVaR.

These are the numbers the repository exists to produce, so they are tested
against closed-form normal quantiles and against the coherence properties any
correct implementation must satisfy.
"""
import numpy as np
import pytest
from scipy import stats

from mc_risk import VaRCalculator

# --------------------------------------------------------- analytic benchmarks

def test_historical_var_matches_the_normal_quantile(normal_returns):
    """For N(0, 1%), 95% VaR is 1.645 * sigma."""
    res = VaRCalculator().historical(normal_returns, 1_000_000, confidence_level=0.95)[0]
    expected = -stats.norm.ppf(0.05) * 0.01
    assert res.var_pct == pytest.approx(expected, rel=0.05)


def test_historical_cvar_matches_the_normal_expected_shortfall(normal_returns):
    """Closed form: ES = sigma * phi(z_alpha) / alpha."""
    res = VaRCalculator().historical(normal_returns, 1_000_000, confidence_level=0.95)[0]
    alpha = 0.05
    expected = 0.01 * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    assert res.cvar_pct == pytest.approx(expected, rel=0.05)


def test_var_scales_linearly_with_position_size(normal_returns):
    calc = VaRCalculator()
    a = calc.historical(normal_returns, 1_000_000, confidence_level=0.95)[0]
    b = calc.historical(normal_returns, 2_000_000, confidence_level=0.95)[0]
    assert b.var_abs == pytest.approx(2 * a.var_abs)
    assert b.var_pct == pytest.approx(a.var_pct), "percentage VaR is size-invariant"


# ------------------------------------------------------- coherence properties

def test_cvar_is_never_below_var(normal_returns):
    """THE core invariant. Expected shortfall is an average over the tail beyond
    VaR, so it cannot be smaller. If this fails, the tail selection is wrong."""
    for res in VaRCalculator([0.90, 0.95, 0.99]).historical(normal_returns, 1e6):
        assert res.cvar_pct >= res.var_pct - 1e-12, (
            f"CVaR {res.cvar_pct:.6f} < VaR {res.var_pct:.6f} at "
            f"{res.confidence_level}"
        )


def test_var_increases_with_confidence_level(normal_returns):
    """Monotonicity in alpha: a 99% loss threshold exceeds a 95% one."""
    r = VaRCalculator([0.90, 0.95, 0.99]).historical(normal_returns, 1e6)
    vars_ = [x.var_pct for x in r]
    assert vars_ == sorted(vars_), f"VaR not monotone in confidence: {vars_}"


def test_fatter_tails_raise_cvar_more_than_var():
    """A t-distribution and a normal can be scaled to similar VaR while
    differing in CVaR. This is the reason CVaR is reported at all, so it is
    worth asserting the engine actually distinguishes them."""
    rng = np.random.default_rng(1)
    normal = rng.normal(0, 0.01, 200_000)
    t = rng.standard_t(3, 200_000) * 0.01 / np.sqrt(3.0)
    calc = VaRCalculator()
    n = calc.historical(normal, 1e6, confidence_level=0.99)[0]
    h = calc.historical(t, 1e6, confidence_level=0.99)[0]
    assert (h.cvar_pct / h.var_pct) > (n.cvar_pct / n.var_pct), (
        "heavy tails must widen the CVaR-to-VaR ratio"
    )


def test_riskless_position_has_zero_var():
    res = VaRCalculator().historical(np.zeros(1000), 1e6, confidence_level=0.99)[0]
    assert res.var_pct == pytest.approx(0.0)
    assert res.cvar_pct == pytest.approx(0.0)


def test_tail_observation_count_is_about_alpha(normal_returns):
    res = VaRCalculator().historical(normal_returns, 1e6, confidence_level=0.95)[0]
    assert len(res.tail_returns) == pytest.approx(500, rel=0.2)


def test_n_observations_is_recorded(normal_returns):
    res = VaRCalculator().historical(normal_returns, 1e6, confidence_level=0.95)[0]
    assert res.n_observations == 10_000


def test_default_confidence_levels_produce_two_results(normal_returns):
    assert len(VaRCalculator().historical(normal_returns, 1e6)) == 2


def test_summary_is_serialisable(normal_returns):
    s = VaRCalculator().historical(normal_returns, 1e6, confidence_level=0.95)[0].summary()
    assert set(s) >= {"name", "method", "var_pct", "cvar_pct", "n_observations"}


def test_to_dataframe_has_one_row_per_result(normal_returns):
    calc = VaRCalculator([0.95, 0.99])
    df = calc.to_dataframe(calc.historical(normal_returns, 1e6))
    assert len(df) == 2


# -------------------------------------------------------------- monte carlo

def test_monte_carlo_var_is_positive_and_finite(sim):
    for r in VaRCalculator([0.95]).monte_carlo(sim, {"A": 1e6, "B": 1e6}):
        assert 0 < r.var_pct < 10 and np.isfinite(r.cvar_pct)


def test_monte_carlo_skips_instruments_without_positions(sim):
    out = VaRCalculator([0.95]).monte_carlo(sim, {"A": 1e6})
    assert [r.name for r in out] == ["A"]


def test_monte_carlo_cvar_never_below_var(sim):
    for r in VaRCalculator([0.95, 0.99]).monte_carlo(sim, {"A": 1e6, "B": 1e6}):
        assert r.cvar_pct >= r.var_pct - 1e-12


def test_longer_horizon_increases_var(two_specs):
    """GBM diffusion grows with sqrt(t), so a 100-day VaR exceeds a 10-day one."""
    from mc_risk import MonteCarloEngine
    calc = VaRCalculator([0.95])
    short = calc.monte_carlo(
        MonteCarloEngine(3000, 10, seed=5).simulate(two_specs, np.eye(2)), {"A": 1e6}
    )[0]
    long = calc.monte_carlo(
        MonteCarloEngine(3000, 100, seed=5).simulate(two_specs, np.eye(2)), {"A": 1e6}
    )[0]
    assert long.var_pct > short.var_pct


# --------------------------------------------------------- portfolio

def test_diversification_reduces_portfolio_var(two_specs):
    """Two uncorrelated positions must be less risky than two identical ones.
    This is the property the Cholesky machinery exists to preserve."""
    from mc_risk import MonteCarloEngine
    calc = VaRCalculator([0.95])
    pos = {"A": 1e6, "B": 1e6}

    indep = calc.portfolio_monte_carlo(
        MonteCarloEngine(6000, 50, seed=9).simulate(two_specs, np.eye(2)), pos
    )[0]
    near_perfect = np.array([[1.0, 0.99], [0.99, 1.0]])
    perfect = calc.portfolio_monte_carlo(
        MonteCarloEngine(6000, 50, seed=9).simulate(two_specs, near_perfect), pos
    )[0]
    assert indep.var_pct < perfect.var_pct, (
        "uncorrelated positions should diversify; correlation is not reaching VaR"
    )


def test_portfolio_var_converts_log_returns_to_simple_pnl(two_specs):
    """Portfolio currency P&L must use exact simple returns, not log returns."""
    from mc_risk import MonteCarloEngine
    sim = MonteCarloEngine(4000, 50, seed=3).simulate(two_specs, np.eye(2))
    res = VaRCalculator([0.99]).portfolio_monte_carlo(sim, {"A": 1e6, "B": 1e6})[0]

    log_rets = sim.terminal_returns
    simple_rets = np.exp(log_rets) - 1.0
    pnl_simple = (simple_rets * 1e6).sum(axis=0) / 2e6
    var_simple = -np.quantile(pnl_simple, 0.01)

    assert res.var_pct == pytest.approx(var_simple, rel=1e-9)
    assert res.var_abs == pytest.approx(var_simple * 2e6, rel=1e-9)


def test_portfolio_normalises_long_short_pnl_by_gross_exposure(two_specs):
    """A market-neutral book still has risk and a well-defined percentage basis."""
    from mc_risk import MonteCarloEngine
    sim = MonteCarloEngine(2000, 20, seed=4).simulate(two_specs, np.eye(2))
    res = VaRCalculator([0.95]).portfolio_monte_carlo(sim, {"A": 1e6, "B": -1e6})[0]
    simple_rets = np.expm1(sim.terminal_returns)
    expected_returns = (simple_rets[0] * 1e6 - simple_rets[1] * 1e6) / 2e6
    expected_var = -np.quantile(expected_returns, 0.05)

    assert res.var_pct == pytest.approx(expected_var, rel=1e-9)
    assert res.var_abs == pytest.approx(expected_var * 2e6, rel=1e-9)


def test_portfolio_rejects_zero_gross_exposure(sim):
    with pytest.raises(ValueError, match="gross exposure"):
        VaRCalculator([0.95]).portfolio_monte_carlo(sim, {"A": 0.0, "B": 0.0})


def test_cvar_silently_equals_var_when_the_tail_is_empty():
    """DEFECT: `cvar_pct = var_pct if len(tail)==0`.

    With a degenerate distribution no observation falls strictly beyond the
    threshold, and CVaR is reported as equal to VaR with no indication that it
    was never computed. A caller cannot distinguish this from a genuine result.
    See KNOWN_ISSUES.md #3.
    """
    res = VaRCalculator().historical(
        np.array([0.0] * 100), 1e6, confidence_level=0.99
    )[0]
    assert res.cvar_pct == res.var_pct
    assert len(res.tail_returns) > 0 or res.cvar_pct == res.var_pct


def test_user_supplied_correlation_bypasses_the_psd_repair(two_specs):
    """DEFECT: _build_correlation_matrix repairs a non-positive-definite
    estimated matrix by eigenvalue shifting, but simulate() applies Cholesky
    directly to a caller-supplied matrix with no such repair.

    Perfect correlation is a legitimate stress assumption — "assume everything
    moves together" — and it raises LinAlgError instead of simulating.
    See KNOWN_ISSUES.md #4.
    """
    from mc_risk import MonteCarloEngine
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        MonteCarloEngine(100, 5, seed=1).simulate(two_specs, np.ones((2, 2)))
