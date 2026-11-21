# The four clean checks before an upload (2026-11-21, `kilo` session)

All numbers are the D26 metric on the standard 4 rolling test-shop cuts (`src/meta/clean_checks.py`),
higher is better. Baseline = the winning D40 recipe: mean(moirai-2.0-small, tirex, chronos-t5-small)
raw forecasts, 8-week zero rule, 0.3 x own LightGBM + 0.7 x ensemble, rounded. Raw table:
`results_kilo_clean_checks.csv`.

## Baseline check

D40 mean over the 4 cuts **-13.67** (cuts -3.72 / -25.91 / -11.01 / -14.02). This is the exact reference
for every test below. It is the recipe whose upload scored **live -4.8**; local mean and live weeks 1-7
are not on the same scale, so "-4.1" cannot be confirmed from local numbers alone.

## Test A — no rounding + 16-week zero rule

| Variant | cut0 | cut1 | cut2 | cut3 | mean | vs baseline |
|---|---|---|---|---|---|---|
| baseline: rounded, 8-week zero | -3.72 | -25.91 | -11.01 | -14.02 | **-13.67** | 0.00 |
| A1: rounded, 16-week zero | -3.72 | -25.92 | -11.02 | -14.04 | -13.68 | -0.01 |
| A2: no rounding, 8-week zero | -5.56 | -28.32 | -14.10 | -14.92 | -15.72 | -2.05 |
| A3: no rounding, 16-week zero | -5.58 | -28.36 | -14.16 | -14.99 | -15.77 | -2.10 |

Relaxing the cutoff to 16 weeks changes nothing (the extra dead series were already forecast ~0).
Removing whole-unit rounding costs **2.05 points**: the metric rewards integer medians on count data.
Fractions are accepted by the submission validator, but they are clearly worse. Reject A.

## Test B — Chronos-2 instead of Chronos-T5-small, weights from local strength

Solo local scores (rounded + 8-week zero): moirai-2.0-small -14.72, tirex -15.97, chronos-2 -22.96.
Weighting by inverse-MASE gives 0.342 / 0.339 / 0.319; by inverse-|score| 0.390 / 0.360 / 0.250.

| Variant (0.3 own LightGBM kept) | cut0 | cut1 | cut2 | cut3 | mean | vs baseline |
|---|---|---|---|---|---|---|
| baseline (T5-small, equal weights) | -3.72 | -25.91 | -11.01 | -14.02 | **-13.67** | 0.00 |
| B1 Chronos-2, equal weights | -4.06 | -26.17 | -11.57 | -14.17 | -13.99 | -0.33 |
| B2 Chronos-2, inverse-MASE weights | -4.08 | -26.18 | -11.59 | -13.90 | -13.94 | -0.27 |
| B3 Chronos-2, inverse-|score| weights | -3.96 | -26.10 | -11.49 | -14.08 | -13.91 | -0.24 |

Chronos-2 is the larger/newer member of the Chronos family but a weaker forecaster here, and weighting it
down does not recover the loss. The equal-weight T5-small blend stays best. Reject B. The file built
earlier from this idea (`submissions/ensemble_top_series_v1.csv`, B1) is **not** an improvement.

## Test C — add the cross-shop LightGBM

The cross-shop arm is the pooled model of iteration 04 (`pooled_norich`): test-shop windows before the
cut plus **all weeks of shops A-D**, equal shop weights, 300 trees; generated for all 4 cuts by
`src/meta/cross_shop_forecasts.py` (files in `forecasts_lgbm_cross/`). Scores below are rounded.

| Model / blend | cut0 | cut1 | cut2 | cut3 | mean | vs baseline |
|---|---|---|---|---|---|---|
| own LightGBM alone | -4.03 | -28.06 | -13.08 | -15.31 | -15.12 | — |
| cross-shop LightGBM alone | -4.90 | -29.71 | -12.98 | -15.76 | -15.84 | — |
| C1 70% pretrained + 30% own | -3.72 | -25.91 | -11.01 | -14.02 | **-13.67** | 0.00 |
| C2 55% pretrained + 20% own + 25% cross-shop | -3.47 | -26.41 | -11.65 | -14.27 | -13.95 | -0.28 |
| C3 as C2 with the 8-week zero rule also on cross-shop | -3.47 | -26.41 | -11.65 | -14.26 | -13.95 | -0.28 |

Borrowing the other shops hurts the LightGBM on its own (own -15.12 vs cross-shop -15.84 mean; the same
own -4.83 / pooled -5.84 / others-only -23.07 pattern seen in step 1.3 at cut 0) and adding 25% of it to
the blend is **-0.28** on the mean, better on only cut 0 (1/4). It does not clear the adoption bar.
The own-only vs cross-shop comparison is still useful for the extra-credit write-up: pooling adds data
but also shop-specific scale/noise differences, and the test shop has its own long history, so the
borrowed signal is not worth its transfer cost. Reject C.

## Upload candidates built on the real weeks (2026-11-21, after the checks)

Requested because the leaderboard values the live weeks differently. Recipe as above but on the actual
13 weeks 2023-04-24 .. 2023-07-17, rounded and clipped; `src/meta/build_upload_candidates.py`.

| File | Recipe | Local 4-cut mean | Diff vs D40 file |
|---|---|---|---|
| `submissions/ensemble_zero16_v1.csv` | A: 16-week zero rule, 30% own LGBM | -13.68 | 0.002 units/article-week, 0.1% weeks |
| `submissions/ensemble_c70_own30_v1.csv` | C: 70% pretrained + 30% own LGBM | -13.67 | **byte-identical** to `ensemble_top3_blend_v1.csv` (live -4.8) |
| `submissions/ensemble_c55_own20_cross25_v1.csv` | C: 55% pretrained + 20% own + 25% cross-shop | -13.95 | 0.209 units/article-week, 11.0% weeks |

Notes:
- The 16-week rule has almost no effect on the file: the extra 213 series (860 dead under 16 weeks vs
  1,073 under 8) are forecast 0 by the foundation models anyway, so only 0.1% of weeks move.
- C55 uses the cross-shop (pooled) LightGBM trained on all test-shop history + all weeks of shops A-D,
  equal shop weights, 300 trees (`cross_shop_forecasts.py --final`).
- C70 reproduces the uploaded -4.8 file exactly, so it is a control, not a new candidate.
- The live board scores weeks 1-7 only; the final score uses weeks 8-13. Uploads and the scores are for a
  person to log in `submissions/UPLOADS.md`.

## Live outcome of the upload candidates (reported by the user, 2026-11-21)

The user uploaded the candidate(s) and reported **no improvement: the best live score stays -4.8**. This
is what the local evidence predicted: C70 is byte-identical to the -4.8 file, A (16-week zero rule)
differs in only 0.1% of cells, and C55 was 0.28 points worse on the 4-cut mean. The board keeps each
team's best score, so a worse upload still displays -4.8. No candidate is adopted; D40 remains the
submission.

Takeaway: the local ordering transferred (again). Gains below about 0.3 on the 4-cut mean have not
survived a live check in any attempt (blend probe +0.05, gate +0.06, NHITS +0.04, FFORMA +0.23, three
uploads now), so further recombinations of the same forecasts should not be uploaded.

## Verdict

**None of A, B or C improves the winning recipe; keep D40 (rounded, 8-week zero rule, 0.3 own LightGBM).**
The baseline beats the previous uploaded file's live -4.8 only if the live window cooperates; locally it
is the best of every variant tested here. Do not upload `ensemble_top_series_v1.csv` hoping for a better
score; it is -0.33 on the same 4 cuts.
