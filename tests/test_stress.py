"""Stress scenarios.

Several scenario descriptions do not match what the code does. Those gaps are
asserted here so the documentation and the implementation cannot drift further
apart silently.
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
    assert len(StressTester(rng_seed=1).run_all(sim)) == 4


def test_stress_is_deterministic(sim):
    a = StressTester(rng_seed=7).apply(sim, StressScenario("S", "d"), price_shock_all=-0.2)
    b = StressTester(rng_seed=7).apply(sim, StressScenario("S", "d"), price_shock_all=-0.2)
    assert np.array_equal(a.stressed_terminal_returns, b.stressed_terminal_returns)


# ------------------------------------------------------------------- defects

def test_correlation_override_is_declared_but_never_read(sim):
    """DEFECT: StressScenario.correlation_override is documented as replacing
    the correlation matrix for a scenario, and StressTester.apply never reads
    it.

    Passing it has no effect whatsoever. See KNOWN_ISSUES.md #6.
    """
    import inspect

    src = inspect.getsource(StressTester.apply)
    assert "correlation_override" not in src

    st = StressTester(rng_seed=1)
    with_override = st.apply(
        sim,
        StressScenario("S", "d", correlation_override=np.eye(2), n_paths_pct=0.1),
        price_shock_all=-0.1,
    )
    without = st.apply(
        sim, StressScenario("S", "d", n_paths_pct=0.1), price_shock_all=-0.1
    )
    assert np.array_equal(
        with_override.stressed_terminal_returns, without.stressed_terminal_returns
    )


def test_correlation_breakdown_scenario_is_a_no_op(sim):
    """DEFECT, consequence of #6: the shipped Correlation_Breakdown scenario
    changes nothing at all.

    run_all passes it price_shock=None and vol_mult=1.0, and it carries no
    price_shocks or vol_multipliers of its own, so every reported shift is
    exactly zero. One of four advertised scenarios is inert.
    See KNOWN_ISSUES.md #7.
    """
    results = {r.scenario_name: r for r in StressTester(rng_seed=1).run_all(sim)}
    cb = results["Correlation_Breakdown"]
    assert np.array_equal(cb.stressed_terminal_returns, sim.terminal_returns)
    assert all(v == pytest.approx(0.0) for v in cb.var_95_shift.values())
    assert all(v == pytest.approx(0.0) for v in cb.cvar_95_shift.values())


def test_scenario_descriptions_do_not_match_what_is_applied(sim):
    """DEFECT: descriptions overstate the scenarios.

    - Sharp_Selloff says "Broad 20% price decline across all instruments" but
      only the worst 5% of paths are shocked, so the central distribution is
      untouched.
    - Supply_Shock says "Primary commodity -30%, secondary instruments -10%"
      but run_all applies a flat -15% to everything, with no notion of primary
      versus secondary.

    See KNOWN_ISSUES.md #8.
    """
    selloff = DEFAULT_STRESS_SCENARIOS[0]
    assert "across all instruments" in selloff.description
    assert selloff.n_paths_pct == 0.05, "only 5% of paths, not all of them"

    supply = DEFAULT_STRESS_SCENARIOS[3]
    assert "-30%" in supply.description and "-10%" in supply.description
    assert supply.price_shocks == {}, "no per-instrument shocks are actually defined"


def test_shocking_only_the_worst_paths_barely_moves_var(sim):
    """Consequence worth knowing, arguably by design: because only the worst
    5% of paths are shocked, the 95% VaR quantile hardly moves while CVaR moves
    a lot. Anyone reading var_95_shift as "the scenario impact" will understate
    it. Documented rather than called a bug."""
    st = StressTester(rng_seed=1)
    res = st.apply(sim, StressScenario("S", "d", n_paths_pct=0.05), price_shock_all=-0.30)
    assert res.cvar_95_shift["A"] > res.var_95_shift["A"]
