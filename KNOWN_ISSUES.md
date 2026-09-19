# Known issues

Found by writing the test suite. Each item below has a test that pins the
current behaviour and cites this file by number.

Convention: a test that asserts a defect says so in its docstring. It passes
today. Fixing the underlying bug will break that test, which is the signal to
delete the test and the entry together.

Nothing here is fixed yet. They are recorded rather than patched so the
behaviour of the published version is documented, and so the fix and the
description of the fix land in the same commit.

| # | Severity | Area | Issue |
|---|----------|------|-------|
| 1 | High | `var_cvar.portfolio_monte_carlo` | Log returns used as arithmetic P&L weights |
| 2 | High | `var_cvar.portfolio_monte_carlo` | Short positions break the normalisation denominator |
| 3 | Medium | `var_cvar` | CVaR silently reported as equal to VaR when the tail is empty |
| 4 | Medium | `simulation.simulate` | Caller-supplied correlation matrices skip the PSD repair |
| 5 | Medium | `simulation._build_correlation_matrix` | Partial history collapses the whole matrix to identity |
| 6 | Medium | `stress_test` | `correlation_override` is documented but never read |
| 7 | Medium | `stress_test` | The shipped `Correlation_Breakdown` scenario is a no-op |
| 8 | Low | `stress_test` | Scenario descriptions overstate what is applied |

---

### 1. Log returns used as arithmetic P&L weights (High)

`SimulationResult.terminal_returns` is `log(S_T / S_0)`, stated in the
docstring. `portfolio_monte_carlo` then forms portfolio P&L as:

```python
portfolio_pnl += terminal_rets * pos_value
```

That is the formula for simple returns. For a position to lose `x` percent of
its value the simple return must be `-x`; the log return is `log(1-x)`, which is
more negative. A 50% price fall is booked as a 69.3% loss.

The error grows with the size of the move, so it is largest exactly in the deep
tail where VaR and CVaR are measured. The direction is conservative — reported
risk is too high, not too low — but the output is not the quantity it is
labelled as, and it is inconsistent with the per-instrument path in the same
module, which reports a log-return quantile.

This also contaminates `PortfolioRiskAggregator.diversification_benefit_var`,
which subtracts a portfolio number computed one way from a sum of individual
numbers computed the other way.

Fix: convert once at the boundary, `simple = np.exp(terminal_returns) - 1`, and
weight with that. Alternatively aggregate on terminal prices rather than
returns, which avoids the conversion entirely.

Tests: `test_portfolio_var_treats_log_returns_as_arithmetic_pnl`,
`test_diversification_benefit_mixes_two_different_return_conventions`.

### 2. Short positions break the normalisation denominator (High)

`total_value = sum(position_values.values())` and the percentage conversion is
guarded by `if total_value > 0`. A short booked as a negative position value
shrinks the denominator; a hedged book sums to zero and takes the else branch,
which returns `var_pct` as a raw currency amount in a field named as a
percentage. There is no error and no warning.

A long/short book is the normal case for the kind of portfolio this library
describes, so this is not an edge case.

Fix: normalise by gross exposure, `sum(abs(v) for v in position_values.values())`,
and raise rather than silently changing units when it is zero.

Test: `test_portfolio_normalisation_breaks_with_short_positions`.

### 3. CVaR silently reported as equal to VaR when the tail is empty (Medium)

The tail is selected with a strict comparison against the VaR threshold, and
when nothing falls beyond it the code assigns `cvar_pct = var_pct`. A caller
cannot distinguish "expected shortfall equals VaR" from "expected shortfall was
never computed". This fires on degenerate or heavily discretised distributions,
and on very small samples at high confidence.

Fix: return `None`, or include the threshold observation in the tail and record
`n_tail` on the result so the caller can see the sample size behind the number.

Test: `test_cvar_silently_equals_var_when_the_tail_is_empty`.

### 4. Caller-supplied correlation matrices skip the PSD repair (Medium)

`_build_correlation_matrix` repairs a non-positive-definite estimated matrix by
shifting eigenvalues and renormalising. `simulate()` applies
`np.linalg.cholesky` directly to a matrix passed in by the caller, with no such
repair, so it raises `LinAlgError`.

Perfect correlation is a legitimate and common stress assumption — "assume
everything moves together" — and it is exactly the input that fails. The README
advertises the repair step without noting it only applies to estimated
matrices.

Fix: extract the repair into a function and route both paths through it.

Test: `test_user_supplied_correlation_bypasses_the_psd_repair`.

### 5. Partial history collapses the whole matrix to identity (Medium)

If any instrument lacks `returns_history`, `_build_correlation_matrix` returns
`np.eye(n)` and discards the correlation it could have measured for the
instruments that do have history.

Assuming independence understates portfolio risk, which is the dangerous
direction for a risk engine, and nothing in `SimulationResult` records that the
fallback was taken.

Fix: estimate the block that has history, use a caller-supplied default for the
missing entries, and record on the result which pairs were assumed rather than
measured.

Test: `test_partial_history_silently_discards_known_correlation`.

### 6. `correlation_override` is documented but never read (Medium)

`StressScenario.correlation_override` is declared and documented as replacing
the correlation matrix for a scenario. `StressTester.apply` never references it.
Passing it has no effect.

Correlation regime change cannot in fact be applied post hoc to
`terminal_returns` the way price and vol shocks are; it requires re-simulating
with a different Cholesky factor. So this is not a one-line fix — it needs
`apply` to hold a reference to the engine and specs, or a separate entry point
that re-runs the simulation under an overridden matrix.

Fix, minimum viable: raise `NotImplementedError` when the field is set, so it
cannot be passed silently. Properly: a `stress_correlation` method that
re-simulates.

Test: `test_correlation_override_is_declared_but_never_read`.

### 7. The shipped `Correlation_Breakdown` scenario is a no-op (Medium)

Consequence of #6. The scenario carries no `price_shocks` and no
`vol_multipliers`, and `run_all` passes it `price_shock=None, vol_mult=1.0`.
Both shock branches are skipped, the returns are copied unchanged, and every
reported shift is exactly zero.

One of the four advertised built-in scenarios does nothing, and its zero row in
the stress summary reads as "this scenario had no impact" rather than "this
scenario did not run".

Fix: with #6 addressed, implement it as a re-simulation at identity
correlation. Until then, remove it from `DEFAULT_STRESS_SCENARIOS` rather than
shipping an inert row.

Test: `test_correlation_breakdown_scenario_is_a_no_op`.

### 8. Scenario descriptions overstate what is applied (Low)

- `Sharp_Selloff` is described as "Broad 20% price decline across all
  instruments". It is applied to the worst 5% of paths only; the body of the
  distribution is untouched.
- `Supply_Shock` is described as "Primary commodity -30%, secondary instruments
  -10%". `run_all` applies a flat -15% to every instrument. There is no notion
  of primary versus secondary anywhere in the code.

Restricting shocks to the tail is a deliberate and defensible design choice,
documented in the README. The problem is only that the scenario descriptions do
not say so, and a reader comparing the description to `var_95_shift` will
conclude the shock had little effect when in fact it was never applied to the
quantile being reported.

Fix: reword the descriptions to state the path fraction, and either implement
the primary/secondary distinction or describe the flat shock.

Tests: `test_scenario_descriptions_do_not_match_what_is_applied`,
`test_shocking_only_the_worst_paths_barely_moves_var`.
