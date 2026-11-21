# AutoGluon challenger

User authorized bringing AutoGluon into the project. Senior Data Scientist skill
consulted: preserve the reference, isolate dependencies, use temporal evaluation,
and verify scoring parity before trusting automated selection.

## Scope

- Optional `autogluon.timeseries==1.6.3` environment; leave `.venv` unchanged.
- Own-shop, units-only forecasts with a common runner for A-D and target.
- Default CPU shortlist: Naive, AutoETS (nonseasonal), DirectTabular (LightGBM).
  DeepAR is opt-in and bounded to 5 epochs/20 batches per epoch. No automatic
  foundation-weight downloads or broad model presets.
- DirectTabular has no calendar features and no implicit annual differencing.
  This is an integration baseline, not a claim that those features never help.
- Outer cuts use the existing `validate.split` convention: four nonoverlapping
  13-week periods where feasible. Internal selection uses up to three prior
  13-week windows, never the outer holdout; short histories reduce windows.
- The adapter uses adjacent-week history scaling, excludes constant/empty scales,
  clips negative forecasts, and optimizes a median point estimate. Saved outputs
  are independently scored by `src/validate.py`, including exclusion/coverage counts.
- Export every successfully trained model as well as the internally selected one;
  outer per-model scores are diagnostics, not untouched evidence after selection.
- No pooled learning, model promotion, full-history submission fit, or upload in
  this integration. The reported -4.8 foundation ensemble is not present locally;
  a matched comparison requires its historical predictions/configuration.

## Commands

```bash
bash env/setup_autogluon.sh
.venv-autogluon/bin/python tests/test_autogluon.py

# Small plumbing check; not a representative accuracy claim
.venv-autogluon/bin/python src/run_autogluon.py --shops test --cuts 0 --limit-series 32 --time-limit 60

# Target-shop historical comparison, 300 seconds soft fit budget per cut
.venv-autogluon/bin/python src/run_autogluon.py --shops test --cuts 0 1 2 3

# Same recipe on A-D; expensive, especially D; review resource scope first
.venv-autogluon/bin/python src/run_autogluon.py --shops A B C D --cuts 0

# Explicit neural challenger; does not run by default
.venv-autogluon/bin/python src/run_autogluon.py --shops test --cuts 0 --models Naive DeepAR --time-limit 300
```

Runs get fresh directories under `runs/` (ignored by Git); `--output PATH` must be
new. They contain model artifacts, forecasts, internal rankings, project metric
scores, per-series errors, statuses, input/source hashes, configuration and full
environment freeze. Time limits are soft fit budgets; imports, validation and
prediction add overhead. Failures are retained, not silently zero-filled.

AutoGluon may skip requested models within its budget. Inspect its logs and the
recorded trained-model list; a successful fallback baseline is not proof that
every requested challenger trained. B/D's second outer cut cannot fit AG's
standard internal-validation workflow and is explicitly skipped.

## Success criteria

1. Isolated environment installs and dependency consistency check passes.
2. Adapter agrees with hand-computed and saved-reference MASE, preserves distinct
   series IDs, and rejects incomplete/nonfinite forecasts.
3. A real fit/predict/score run succeeds and saves artifacts; report scope honestly.
4. Existing training/preparation/submission paths remain untouched.

References: [installation](https://auto.gluon.ai/stable/install.html),
[custom metric](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html),
[fit/validation](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.fit.html).
