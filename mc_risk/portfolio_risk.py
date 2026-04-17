"""
portfolio_risk.py
-----------------
Portfolio aggregator and risk report generator.

Aggregates per-instrument VaR/CVaR results into a portfolio-level
risk report, including diversification benefit, concentration analysis,
and a consolidated summary table.

Diversification benefit
-----------------------
Naive (sum of individual VaRs) vs portfolio VaR.
The difference is the diversification benefit — how much correlation
between instruments reduces total portfolio risk vs holding each
position in isolation.

If diversification benefit is near zero, positions are highly correlated
and provide little hedging value against each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .simulation import SimulationResult
from .var_cvar import VaRResult, VaRCalculator
from .stress_test import StressResult


@dataclass
class PortfolioRiskReport:
    """
    Consolidated risk report for a portfolio.

    Attributes
    ----------
    instruments : list[str]
    position_values : dict[str, float]
    total_portfolio_value : float
    individual_var : dict[str, float]      # 95% VaR per instrument (abs)
    individual_cvar : dict[str, float]     # 95% CVaR per instrument (abs)
    portfolio_var : float                  # Portfolio-level 95% VaR
    portfolio_cvar : float                 # Portfolio-level 95% CVaR
    diversification_benefit_var : float    # Sum(indiv VaR) - portfolio VaR
    concentration : pd.DataFrame           # % contribution to portfolio VaR
    stress_summary : pd.DataFrame          # Stress scenario impacts
    """
    instruments: list[str]
    position_values: dict[str, float]
    total_portfolio_value: float
    individual_var: dict[str, float]
    individual_cvar: dict[str, float]
    portfolio_var: float
    portfolio_cvar: float
    diversification_benefit_var: float
    concentration: pd.DataFrame
    stress_summary: Optional[pd.DataFrame] = None

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print("  PORTFOLIO RISK REPORT")
        print("=" * 60)

        print(f"\n  Portfolio Value       : ${self.total_portfolio_value:>15,.2f}")
        print(f"  Portfolio VaR (95%)   : ${self.portfolio_var:>15,.2f}  "
              f"({self.portfolio_var / self.total_portfolio_value * 100:.2f}%)")
        print(f"  Portfolio CVaR (95%)  : ${self.portfolio_cvar:>15,.2f}  "
              f"({self.portfolio_cvar / self.total_portfolio_value * 100:.2f}%)")
        print(f"  Diversification Benefit: ${self.diversification_benefit_var:>14,.2f}")

        print(f"\n  {'Instrument':<20} {'Position':>12} {'VaR 95%':>12} {'CVaR 95%':>12} {'Conc.':>8}")
        print("  " + "-" * 68)
        for inst in self.instruments:
            pos = self.position_values.get(inst, 0)
            var = self.individual_var.get(inst, 0)
            cvar = self.individual_cvar.get(inst, 0)
            conc = var / sum(self.individual_var.values()) * 100 if sum(self.individual_var.values()) > 0 else 0
            print(f"  {inst:<20} ${pos:>11,.0f} ${var:>11,.0f} ${cvar:>11,.0f} {conc:>7.1f}%")

        if self.stress_summary is not None and not self.stress_summary.empty:
            print(f"\n  Stress Scenarios (CVaR 95% shift vs base):")
            print("  " + "-" * 68)
            for _, row in self.stress_summary.drop_duplicates("scenario").iterrows():
                # Average shift across instruments
                scenario_rows = self.stress_summary[
                    self.stress_summary["scenario"] == row["scenario"]
                ]
                avg_cvar_shift = scenario_rows["cvar_95_shift_pct"].mean()
                worst = scenario_rows["worst_case_return_pct"].min()
                print(f"  {row['scenario']:<30} avg CVaR shift: {avg_cvar_shift:>+6.2f}%  "
                      f"worst case: {worst:>+6.2f}%")

        print("=" * 60 + "\n")


class PortfolioRiskAggregator:
    """
    Aggregates simulation results into a portfolio risk report.

    Parameters
    ----------
    confidence_level : float
        Primary confidence level for VaR/CVaR. Default 0.95.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.var_calc = VaRCalculator(confidence_levels=[confidence_level])

    def build_report(
        self,
        sim_result: SimulationResult,
        position_values: dict[str, float],
        stress_results: Optional[list[StressResult]] = None,
    ) -> PortfolioRiskReport:
        """
        Build a portfolio risk report from simulation and stress results.

        Parameters
        ----------
        sim_result : SimulationResult
        position_values : dict[str, float]
            {instrument_name: position_value_in_currency}
        stress_results : list[StressResult], optional
        """
        # Individual instrument VaR/CVaR
        indiv_var_results = self.var_calc.monte_carlo(
            sim_result, position_values, self.confidence_level
        )
        individual_var = {r.name: r.var_abs for r in indiv_var_results}
        individual_cvar = {r.name: r.cvar_abs for r in indiv_var_results}

        # Portfolio-level VaR/CVaR (preserves correlations)
        port_results = self.var_calc.portfolio_monte_carlo(
            sim_result, position_values, self.confidence_level
        )
        portfolio_var = port_results[0].var_abs if port_results else 0.0
        portfolio_cvar = port_results[0].cvar_abs if port_results else 0.0

        # Diversification benefit
        sum_indiv_var = sum(individual_var.values())
        diversification_benefit = sum_indiv_var - portfolio_var

        # Concentration table
        total_var = sum_indiv_var if sum_indiv_var > 0 else 1.0
        concentration = pd.DataFrame([{
            "instrument": inst,
            "position_value": position_values.get(inst, 0),
            "var_abs": individual_var.get(inst, 0),
            "cvar_abs": individual_cvar.get(inst, 0),
            "var_contribution_pct": individual_var.get(inst, 0) / total_var * 100,
        } for inst in sim_result.instruments]).sort_values(
            "var_contribution_pct", ascending=False
        )

        # Stress summary
        stress_summary = None
        if stress_results:
            frames = [r.summary() for r in stress_results]
            stress_summary = pd.concat(frames, ignore_index=True)

        total_value = sum(position_values.values())

        return PortfolioRiskReport(
            instruments=sim_result.instruments,
            position_values=position_values,
            total_portfolio_value=total_value,
            individual_var=individual_var,
            individual_cvar=individual_cvar,
            portfolio_var=portfolio_var,
            portfolio_cvar=portfolio_cvar,
            diversification_benefit_var=diversification_benefit,
            concentration=concentration,
            stress_summary=stress_summary,
        )
