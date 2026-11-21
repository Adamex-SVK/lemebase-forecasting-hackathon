# AutoGluon integration — 2026-11-21

## Outcome

AutoGluon 1.6.3 is installed in `.venv-autogluon` and integrated as an optional
challenger. The first bounded CPU shortlist **does not improve the existing
LightGBM reference**. No model was promoted and no submission was generated.

Consulted the Senior Data Scientist skill for temporal validation, leakage and
evaluation discipline. Kept existing preprocessing and D26 evaluator unchanged.

## Implementation

- `env/requirements_autogluon.txt` and `env/setup_autogluon.sh`: isolated setup.
  Installed TimeSeries rather than the full multimodal framework; its dependencies
  include neural libraries. The reference `.venv` was not modified.
- `src/autogluon_adapter.py`: collision-resistant shop/store/article IDs, units-only
  conversion, strict forecast-key checking, clipping, and custom selection metric.
  The metric uses adjacent-week scale and excludes zero/undefined histories;
  AutoGluon's negative MASE ranking is distinct from the displayed project score.
- `src/run_autogluon.py`: fixed model shortlist, seed and resource budgets;
  internal prior windows; outer holdouts; median and optional saved LightGBM
  comparisons; forecasts, errors, fingerprints, environment and failure records.
- `tests/test_autogluon.py`: 5 checks covering metric parity, constant histories,
  clipping, alignment, ID uniqueness, short histories, and saved-reference parity.
- Model binaries and large run artifacts stay in ignored `runs/`. Small numerical
  evidence and configurations are retained under `evidence/`.

## Verification

1. `python3 -m venv .venv-autogluon` followed by
   `.venv-autogluon/bin/python -m pip install 'autogluon.timeseries==1.6.3'`.
   Dependency resolution installed LightGBM 4.7.0 and pyarrow 24.0.0; the setup
   requirements pin them explicitly. `pip check`: no broken requirements.
2. `.venv-autogluon/bin/python tests/test_autogluon.py`: all 5 tests passed.
   Saved full target-shop cut-0 forecasts match the project MASE to 10 decimals.
3. `.venv/bin/python tests/test_validate.py`: all existing metric checks passed.
4. `bash -n env/setup_autogluon.sh` and `git diff --check`: passed.
5. All 21 raw/prepared input hashes still match the data-evaluation manifest.

The installed environment is recorded in [environment.txt](evidence/environment.txt).
macOS sandbox CPU-discovery, cache, and existing timedelta warnings appeared but
did not prevent completed fits/tests. No certificate verification was disabled.

## Smoke runs (plumbing evidence only)

```bash
.venv-autogluon/bin/python src/run_autogluon.py --shops test --cuts 0 --limit-series 32 --time-limit 60 --cpus 1 --output iterations/06-autogluon/runs/smoke32
.venv-autogluon/bin/python src/run_autogluon.py --shops test --cuts 0 --models Naive DeepAR --limit-series 8 --val-windows 1 --time-limit 40 --cpus 1 --output iterations/06-autogluon/runs/neural_smoke8
.venv-autogluon/bin/python src/run_autogluon.py --shops A B C D --cuts 0 --models Naive --limit-series 16 --time-limit 20 --cpus 1 --output iterations/06-autogluon/runs/shop_smoke16
```

The first fitted Naive, DirectTabular, AutoETS and WeightedEnsemble. The neural
check fitted Naive, DeepAR and WeightedEnsemble. The final check completed all
four shop paths, reducing internal windows to one for B/D. Subsets were seeded
and explicitly flagged; their scores are not representative accuracy estimates.
The first smoke preceded adding `n_jobs` to Naive; its exact configuration and
source hashes are retained. No unexpected skipped/failed models were observed.

## Full target-shop four-cut comparison

```bash
.venv-autogluon/bin/python src/run_autogluon.py --shops test --cuts 0 1 2 3 --time-limit 60 --cpus 2 --output iterations/06-autogluon/runs/target_four_cuts
```

All 2,904 products were forecast for all 13 weeks at each outer cut. Internal
selection used three preceding 13-week windows. Each run trained all three
requested models plus an ensemble, within its 60-second soft fit budget. Outer
scores use the unchanged project evaluator; exclusions and coverage are present
in every result row. This is a limited units-only configuration, not a broad HPO
or pretrained-model evaluation. Refit-full was off, as recorded in config.

| Model | Cut 0 | Cut 1 | Cut 2 | Cut 3 | Mean score |
|---|---:|---:|---:|---:|---:|
| Saved reference LightGBM | -4.83 | -28.45 | -13.35 | -15.44 | **-15.52** |
| Median13 | -6.68 | -30.53 | -12.59 | -16.34 | -16.54 |
| AutoGluon DirectTabular | -35.85 | -55.00 | -37.04 | -37.78 | -41.42 |
| AutoGluon WeightedEnsemble | -36.19 | -55.00 | -37.04 | -37.78 | -41.50 |
| AutoETS | -56.90 | -92.57 | -60.06 | -59.41 | -67.24 |
| Naive | -58.26 | -97.58 | -70.44 | -46.64 | -68.23 |

Higher scores are better. Internal selection chose WeightedEnsemble on cut 0
and DirectTabular on cuts 1–3, yielding a mean outer score of **-41.50**.
This is worse than the reference on every cut. No outer-cut winner was substituted
for the internally selected model after observing results.

Authoritative numerical files: [results.csv](evidence/results.csv),
[comparison.csv](evidence/comparison.csv), [status.json](evidence/status.json),
[config.json](evidence/config.json). Config includes code/input/reference hashes.
Saved historical LightGBM predictions were scored, not retrained here.

## Interpretation and next checkpoint

The tool integration works; this initial shortlist is not competitive. Do not
replace the reference or add these forecasts to the current ensemble on the
strength of AutoGluon's internal rankings. Metric alignment does not automatically
make each underlying model's training objective/normalization identical to the
reference, and our initial configuration also omits stock and uses a smaller
feature set. These are possible explanations, not established causes.

DeepAR is smoke-tested but not full-benchmarked. Pretrained challengers, stock
covariates, richer feature configurations, and pooled learning remain explicit
future experiments. Full A-D accuracy/transfer was not evaluated by the subset
plumbing tests, so no A-D average or transfer claim is made.

The user's reported Moirai/TiRex/Chronos plus LightGBM ensemble (-4.8) is not in
the local artifacts and its score window is unconfirmed. It was not compared
numerically or overwritten. Obtain its historical component predictions before
making claims about improvement over that ensemble.

Integration and verification are complete. Review these results before selecting
the next experiment. Changes remain uncommitted; nothing pushed or submitted.
