"""Portfolio aggregation and the diversification benefit figure."""
import numpy as np
import pytest

from mc_risk import (
    InstrumentSpec,
    MonteCarloEngine,
    PortfolioRiskAggregator,
    StressScenario,
    StressTester,
)

POS = {"A": 1_000_000.0, "B": 1_000_000.0}


def _sim(corr, seed=21, n=6000):
    specs = [
        InstrumentSpec("A", 100.0, mu=0.0, sigma=0.20),
        InstrumentSpec("B", 100.0, mu=0.0, sigma=0.20),
    ]
    return MonteCarloEngine(n, 50, seed=seed).simulate(specs, corr)


def test_total_portfolio_value_sums_the_positions(sim):
    rep = PortfolioRiskAggregator().build_report(sim, POS)
    assert rep.total_portfolio_value == 2_000_000.0


def test_long_short_report_uses_gross_exposure(sim):
    rep = PortfolioRiskAggregator().build_report(sim, {"A": 1e6, "B": -1e6})
    assert rep.total_portfolio_value == 2_000_000.0
    assert rep.portfolio_var > 0


def test_every_instrument_gets_a_var_and_cvar(sim):
    rep = PortfolioRiskAggregator().build_report(sim, POS)
    assert set(rep.individual_var) == {"A", "B"}
    assert set(rep.individual_cvar) == {"A", "B"}


def test_portfolio_cvar_is_at_least_portfolio_var(sim):
    rep = PortfolioRiskAggregator().build_report(sim, POS)
    assert rep.portfolio_cvar >= rep.portfolio_var - 1e-9


def test_individual_cvar_is_at_least_individual_var(sim):
    rep = PortfolioRiskAggregator().build_report(sim, POS)
    for inst in ("A", "B"):
        assert rep.individual_cvar[inst] >= rep.individual_var[inst] - 1e-9


def test_uncorrelated_positions_show_a_positive_diversification_benefit():
    rep = PortfolioRiskAggregator().build_report(_sim(np.eye(2)), POS)
    assert rep.diversification_benefit_var > 0


def test_diversification_benefit_shrinks_as_correlation_rises():
    indep = PortfolioRiskAggregator().build_report(_sim(np.eye(2)), POS)
    tight = PortfolioRiskAggregator().build_report(
        _sim(np.array([[1.0, 0.98], [0.98, 1.0]])), POS
    )
    assert tight.diversification_benefit_var < indep.diversification_benefit_var, (
        "near-perfectly correlated positions offer almost no diversification"
    )


def test_concentration_table_contributions_sum_to_one_hundred(sim):
    rep = PortfolioRiskAggregator().build_report(sim, POS)
    assert rep.concentration["var_contribution_pct"].sum() == pytest.approx(100.0)


def test_concentration_table_is_sorted_by_contribution():
    rep = PortfolioRiskAggregator().build_report(_sim(np.eye(2)), {"A": 5e6, "B": 1e5})
    col = rep.concentration["var_contribution_pct"].tolist()
    assert col == sorted(col, reverse=True)
    assert rep.concentration.iloc[0]["instrument"] == "A"


def test_larger_position_carries_more_var():
    rep = PortfolioRiskAggregator().build_report(_sim(np.eye(2)), {"A": 5e6, "B": 1e5})
    assert rep.individual_var["A"] > rep.individual_var["B"]


def test_higher_confidence_level_raises_portfolio_var(sim):
    lo = PortfolioRiskAggregator(0.95).build_report(sim, POS)
    hi = PortfolioRiskAggregator(0.99).build_report(sim, POS)
    assert hi.portfolio_var > lo.portfolio_var


def test_stress_results_are_folded_into_the_report(sim):
    stress = StressTester(rng_seed=1).apply(
        sim, StressScenario("S", "d", n_paths_pct=0.1), price_shock_all=-0.2
    )
    rep = PortfolioRiskAggregator().build_report(sim, POS, stress_results=[stress])
    assert rep.stress_summary is not None and len(rep.stress_summary) == 2


def test_stress_summary_is_none_when_no_stress_is_supplied(sim):
    assert PortfolioRiskAggregator().build_report(sim, POS).stress_summary is None


def test_print_report_runs_without_error(sim, capsys):
    PortfolioRiskAggregator().build_report(sim, POS).print_report()
    assert "VaR" in capsys.readouterr().out


def test_report_is_reproducible_for_a_fixed_seed():
    a = PortfolioRiskAggregator().build_report(_sim(np.eye(2), seed=42), POS)
    b = PortfolioRiskAggregator().build_report(_sim(np.eye(2), seed=42), POS)
    assert a.portfolio_var == b.portfolio_var


def test_an_instrument_with_no_position_drops_out_of_individual_var(sim):
    rep = PortfolioRiskAggregator().build_report(sim, {"A": 1e6})
    assert "B" not in rep.individual_var
    assert rep.concentration["var_abs"].iloc[-1] == 0.0


def test_diversification_benefit_uses_consistent_simple_return_risk(sim):
    rep = PortfolioRiskAggregator().build_report(sim, POS)
    simple_rets = np.expm1(sim.terminal_returns)
    expected_individual_var = {
        name: -np.quantile(simple_rets[i], 0.05) * POS[name]
        for i, name in enumerate(sim.instruments)
    }
    assert rep.individual_var == pytest.approx(expected_individual_var)
    assert rep.diversification_benefit_var == pytest.approx(
        sum(rep.individual_var.values()) - rep.portfolio_var
    )
