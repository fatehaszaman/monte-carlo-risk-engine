"""Stress scenarios.

Regression tests cover actual tail transformations and unsupported operations.
"""
import numpy as np
import pytest

from mc_risk import StressScenario, StressTester
from mc_risk.stress_test import DEFAULT_STRESS_SCENARIOS


def test_price_shock_worsens_the_tail(sim):
    st = StressTester(rng_seed=1)
    res = st.apply(sim, StressScenario("S", "d", n_paths_pct=0.10), price_shock_all=-0.30)
    assert res.worst_case_return["A"] < sim.terminal_returns[0].min()


def test_price_shock_is_applied_on_a_log_basis(sim):
    """A -30% shock adds log(0.7) to the affected paths, keeping the return
    space consistent with terminal_returns."""
    st = StressTester(rng_seed=1)
    res = st.apply(sim, StressScenario("S", "d", n_paths_pct=0.05), price_shock_all=-0.30)
    expected = sim.terminal_returns[0].min() + np.log(0.7)
    assert res.worst_case_return["A"] == pytest.approx(expected)


def test_zero_shock_leaves_returns_untouched(sim):
    st = StressTester(rng_seed=1)
    res = st.apply(sim, StressScenario("S", "d"), price_shock_all=0.0)
    assert np.array_equal(res.stressed_terminal_returns, sim.terminal_returns)


def test_vol_multiplier_widens_the_shocked_paths(sim):
    st = StressTester(rng_seed=1)
    res = st.apply(sim, StressScenario("S", "d", n_paths_pct=0.10), vol_multiplier_all=3.0)
    assert res.worst_case_return["A"] < sim.terminal_returns[0].min()


def test_cvar_shift_is_never_negative_for_a_loss_shock(sim):
    """A downward shock cannot reduce expected shortfall."""
    st = StressTester(rng_seed=1)
    res = st.apply(sim, StressScenario("S", "d", n_paths_pct=0.10), price_shock_all=-0.25)
    for inst in sim.instruments:
        assert res.cvar_95_shift[inst] >= -1e-9


def test_shock_only_touches_the_requested_fraction_of_paths(sim):
    st = StressTester(rng_seed=1)
    res = st.apply(sim, StressScenario("S", "d", n_paths_pct=0.10), price_shock_all=-0.30)
    changed = (res.stressed_terminal_returns[0] != sim.terminal_returns[0]).sum()
    assert changed == pytest.approx(400, rel=0.02)


def test_per_instrument_shocks_are_targeted(sim):
    st = StressTester(rng_seed=1)
    res = st.apply(sim, StressScenario("S", "d", price_shocks={"A": -0.4}, n_paths_pct=0.1))
    assert not np.array_equal(res.stressed_terminal_returns[0], sim.terminal_returns[0])
    assert np.array_equal(res.stressed_terminal_returns[1], sim.terminal_returns[1])


def test_summary_dataframe_has_a_row_per_instrument(sim):
    st = StressTester(rng_seed=1)
    df = st.apply(sim, StressScenario("S", "d"), price_shock_all=-0.2).summary()
    assert len(df) == 2 and "var_95_shift_pct" in df.columns


def test_run_all_returns_every_default_scenario(sim):
    assert len(StressTester(rng_seed=1).run_all(sim)) == 3


def test_stress_is_deterministic(sim):
    a = StressTester(rng_seed=7).apply(sim, StressScenario("S", "d"), price_shock_all=-0.2)
    b = StressTester(rng_seed=7).apply(sim, StressScenario("S", "d"), price_shock_all=-0.2)
    assert np.array_equal(a.stressed_terminal_returns, b.stressed_terminal_returns)


# ------------------------------------------------------------------- defects

def test_correlation_override_fails_explicitly_without_mutating_base(sim):
    """Regression #6: unsupported stress must not look like a zero impact."""
    st = StressTester(rng_seed=1)
    original = sim.terminal_returns.copy()
    with pytest.raises(NotImplementedError, match="re-simulation"):
        st.apply(sim, StressScenario("S", "d", correlation_override=np.eye(2)),
                 price_shock_all=-0.1)
    assert np.array_equal(original, sim.terminal_returns)


def test_defaults_do_not_include_an_inert_correlation_scenario(sim):
    """Regression #7: every shipped scenario transforms the test paths."""
    results = {r.scenario_name: r for r in StressTester(rng_seed=1).run_all(sim)}
    assert "Correlation_Breakdown" not in results
    for r in results.values():
        assert not np.array_equal(r.stressed_terminal_returns, sim.terminal_returns)


def test_default_descriptions_match_applied_price_shocks(sim):
    """Regression #8: the named percentage and path fraction are accurate."""
    selloff = DEFAULT_STRESS_SCENARIOS[0]
    supply = DEFAULT_STRESS_SCENARIOS[2]
    assert "20%" in selloff.description and "worst 5%" in selloff.description
    assert "15%" in supply.description and "worst 5%" in supply.description
    results = StressTester().run_all(sim)
    for result, shock in [(results[0], -0.2), (results[2], -0.15)]:
        changed = result.stressed_terminal_returns - sim.terminal_returns
        for row in changed:
            assert np.count_nonzero(row) == int(row.size * 0.05)
            assert np.allclose(row[row != 0], np.log1p(shock))


def test_zero_path_fraction_changes_nothing(sim):
    r = StressTester().apply(sim, StressScenario("zero", "", n_paths_pct=0),
                             price_shock_all=-0.2, vol_multiplier_all=2)
    assert np.array_equal(r.stressed_terminal_returns, sim.terminal_returns)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
def test_invalid_path_fraction_is_rejected(sim, fraction):
    with pytest.raises(ValueError):
        StressTester().apply(sim, StressScenario("bad", "", n_paths_pct=fraction))


@pytest.mark.parametrize("shock", [-1, -2, float("inf")])
def test_invalid_log_price_shock_is_rejected(sim, shock):
    with pytest.raises(ValueError):
        StressTester().apply(sim, StressScenario("bad", ""), price_shock_all=shock)


def test_shocking_only_the_worst_paths_barely_moves_var(sim):
    """Consequence worth knowing, arguably by design: because only the worst
    5% of paths are shocked, the 95% VaR quantile hardly moves while CVaR moves
    a lot. Anyone reading var_95_shift as "the scenario impact" will understate
    it. Documented rather than called a bug."""
    st = StressTester(rng_seed=1)
    res = st.apply(sim, StressScenario("S", "d", n_paths_pct=0.05), price_shock_all=-0.30)
    assert res.cvar_95_shift["A"] > res.var_95_shift["A"]
