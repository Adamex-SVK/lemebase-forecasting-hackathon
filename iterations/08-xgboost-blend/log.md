# Fixed XGBoost blend test — 2026-11-21

## Outcome

The requested **60% foundation average + 30% LightGBM + 10% XGBoost** blend
did not improve the current 70/30 ensemble. It lost on all four historical
13-week periods, by **0.1841 points on average**. Keep the existing submission.

This was one fixed XGBoost configuration, not evidence that every XGBoost model
or blend must fail. Per-cut paired intervals include zero; the small declines
are not proof of a statistically established population effect. Nevertheless,
there is no positive evidence here to promote this configuration.

## Execution

User requested this local test. Senior Data Scientist skill consulted; weights
were announced before execution and held fixed. Weight interpretation reserves
30% for LightGBM and 10% for XGBoost, leaving 60% for the three equally weighted
foundation models (20% each).

```bash
.venv-autogluon/bin/python iterations/08-xgboost-blend/run.py
```

XGBoost 3.3.0 was already installed. It was trained on this laptop, using the
existing features and origin-specific MASE-normalized targets with L1 loss.
CPU histogram training, seed 0, 2 threads; exact settings in
[config.json](evidence/config.json). Temporal early stopping used earlier
windows with disjoint target weeks, then a refit on all available pre-cut data.
Best inner rounds: 332, 389, 57, 67; final refit rounds: 366, 428, 63, 74.
Total cut runtimes were approximately 186, 189, 31, and 28 seconds.

Foundation inference was not rerun. Immutable saved forecasts were loaded from
commit `71b82f33f0fe5647a13d3786cd17d11b4c4e5493`. LightGBM row order and values
were checked against keyed saved predictions. The foundation average receives
the same 8-week zero rule before blending; final forecasts are rounded once and
clipped at zero. The reference recipe matches iteration 07's scores.

## Results

Higher scores are better. All 2,904 products receive 13 predictions per cut;
constant histories are excluded and counted by the unchanged D26 evaluator.

| Cut | Current 70/30 | Candidate 60/30/10 | Candidate change | XGBoost alone, rounded |
|---|---:|---:|---:|---:|
| 0 | -3.7249 | -3.8278 | -0.1028 | -3.4585 |
| 1 | -25.9074 | -26.1190 | -0.2116 | -29.0765 |
| 2 | -11.0106 | -11.1904 | -0.1798 | -12.7581 |
| 3 | -14.0208 | -14.2630 | -0.2422 | -15.4044 |
| Mean | **-13.6659** | **-13.8501** | **-0.1841** | **-15.1744** |

Mean score over historical weeks 1–7: current -12.2933, candidate -12.5419.
Mean over weeks 8–13: current -15.2673, candidate -15.3763. Neither segment
improves. These are local practice periods, not actual leaderboard/private scores.

Exact scores, exclusion/coverage counts and per-cut paired bootstrap intervals
are in [results.csv](evidence/results.csv). Summary:
[summary.csv](evidence/summary.csv). Training counts/purges:
[status.json](evidence/status.json). Full environment:
[environment.txt](../06-autogluon/evidence/environment.txt) (identical shared
environment; the duplicate snapshot was removed during pre-commit cleanup).

## Verification and preservation

- Existing feature/leakage checks and metric tests passed; script compiles and
  `git diff --check` passes. Existing NumPy deprecation and sandbox CPU-cache
  warnings did not prevent completion.
- All 12 saved forecast tables have exactly 37,752 unique keys, finite
  nonnegative values, and complete product/horizon coverage.
- All four completed fits satisfy the 13-week purge assertion. Code and target
  input hashes match the recorded configuration after execution.
- The original winning submission's SHA256 is unchanged:
  `af2d49198248e54f4fe5b175484f019fb9c4b22ec1f0b6f916d344f01bfb5639`.
- Model binaries, XGBoost arrays and keyed forecasts are retained locally in
  ignored `runs/fixed_60_30_10/`. Small results/configuration evidence is tracked
  separately. A rerun needs a fresh `--output` directory.

These target-shop cuts were used in previous development, so this is exploratory
comparison, not fresh confirmation or unseen-shop validation. No new final
submission was generated; nothing uploaded, committed, merged or pushed.

## User-requested final CSV — 2026-11-21

After reviewing the historical results, the user explicitly requested a CSV
including XGBoost. Generated a separate experimental file; this is not a promotion
over the incumbent. Reproduction:

```bash
.venv-autogluon/bin/python iterations/08-xgboost-blend/make_final.py
```

XGBoost was refitted on all 143 observed weeks through 2023-04-17 (1,877,824
training rows), with unchanged parameters and 366 trees fixed from the newest
historical validation cut. No hidden outcomes were used. Combined raw predictions
as 60% foundation average, 30% original LightGBM, 10% XGBoost, with the same zero
rule, rounding, and clipping as the local test.

Output: `submissions/ensemble_top3_lgbm30_xgb10_v1.csv`, 2,904 products, w1–w13
covering 2023-04-24 through 2023-07-17. It passes the existing submission validator,
and CSV round-trip values exactly match the intended mixture. Relative to the
incumbent, 636 products change; mean absolute change is 0.1017 units per product
week. Incumbent file checksum is unchanged. Full fitting model and arrays remain
in ignored `runs/final_60_30_10/`; fingerprints and provenance are preserved in
[final_metadata.json](evidence/final_metadata.json).

The builder refuses to overwrite an existing CSV or artifact directory. Choose
fresh `--output` and `--artifacts` paths for another run. No upload was performed.

## Pre-commit cleanup — 2026-11-21

Removed the duplicate `evidence/environment.txt` after verifying byte-for-byte
equality with iteration 06's committed snapshot; use the link above. Kept the
plan, scripts, compact evidence, and dependency pin for reproducibility. Models
and bulky generated forecasts remain local under ignored `runs/`.

The user renamed the final CSV to [mayas1.csv](../../submissions/mayas1.csv).
Its SHA256 still matches `output_sha256` in `evidence/final_metadata.json`; the
original generated filename in that metadata records its creation provenance.
The renamed file is preserved and remains eligible for committing.
