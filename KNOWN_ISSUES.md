# Known issues

Tests distinguish corrected behavior from unresolved modeling limitations.
Original issue numbers are retained so earlier review references remain useful.

## Unresolved issues

### #3: Empty-tail CVaR fallback

The instrument-level calculator uses a strict tail comparison and reports VaR
as CVaR when that tail is empty. Degenerate, discrete, or small samples can
therefore hide the lack of tail observations. Consider an inclusive threshold
and explicit tail count rather than silently substituting VaR.

Test: `test_cvar_silently_equals_var_when_the_tail_is_empty`.

### #4: Caller-supplied correlation conditioning

Estimated correlation matrices have a repair path, but caller-supplied
matrices go directly to Cholesky. They must be positive definite; a singular
perfect-correlation matrix raises `LinAlgError`. Validation and a documented
positive-semidefinite factorization policy remain unimplemented.

Test: `test_user_supplied_correlation_bypasses_the_psd_repair`.

### #5: Partial return history

If any instrument lacks history, correlation estimation falls back to an
identity matrix for the entire book, discarding observed relationships.
This can misstate portfolio risk; the direction depends on positions and
correlations. A future implementation should preserve the observed block and
report which pairs use explicit assumptions.

Test: `test_partial_history_silently_discards_known_correlation`.

## Corrected defects and unsupported functionality

- **#1 and #2:** Earlier fixes convert log returns to arithmetic returns for
  currency P&L and use gross exposure to normalize long/short portfolios.
- **#6:** A non-None `correlation_override` now raises `NotImplementedError`
  before touching the base result. Correlation stress is still unsupported,
  not implemented by this change. It needs a separate simulation with the
  stressed dependence assumptions.
- **#7:** The inert `Correlation_Breakdown` default has been removed. Default
  stress runs now contain three scenarios rather than a misleading zero-impact
  fourth row.
- **#8:** Descriptions match the actual transforms: 20% and 15% price drops
  on each instrument's worst 5% of paths, and doubled dispersion around the
  mean of its worst 10% of log-return paths.

The regressions in `tests/test_stress.py` check explicit failure, preservation
of base data, actual changed-path counts and magnitudes, and description units.

## Interpretation of stress output

Price shocks are added in log-return space using `log(1 + shock)`.
Reported stress quantiles, tail averages and minima are log-return statistics,
not dollar P&L or arithmetic-return percentages. Legacy summary columns ending
in `_pct` multiply these log-return statistics by 100.

Each instrument selects its own worst paths. This is a marginal tail
sensitivity, not a coherent joint-event or correlation re-simulation.
Shocking only the worst 5% can leave the 95% quantile nearly unchanged while
substantially worsening expected shortfall. Passing these tests does not
validate crisis probabilities or empirical stress calibration.
