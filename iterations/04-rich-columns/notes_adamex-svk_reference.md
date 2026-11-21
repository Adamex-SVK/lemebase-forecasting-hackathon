# Reference runs: own vs pooled without rich columns (iteration 04, step 4.2)

Date: 2026-11-21. Branch: `it03/adamex-svk/foundation-models` (uncommitted). Handle: `adamex-svk`.

## What was run
- Command: `.venv/bin/python src/run_rich.py own,pooled_norich --seeds 0,1 --topic reference`
- Results: `iterations/04-rich-columns/results_adamex-svk_reference.csv`; per-article MASE in `detail/<arm>_cut<k>_seed<s>.parquet`.
- Yardstick: the 4 rolling cuts of the test shop (cut `k` drops the last `13*k` weeks), D26 score `(1 - MASE) * 100`.
- `own`: test-shop training windows only, early stopping with the last 7 origins as validation, final fit = `int(best_iter*1.1)+1` trees (seed 0: 276/224/47/83; seed 1: 178/140/60/81 for cuts 0-3).
- `pooled_norich`: test-shop cut-truncated windows (origin step 2) + donor shops A, B, C, D with **all** weeks (D27), origin step 4 (D31), `sample_inactive` for D, shop weights `equal` (D30), fixed 300 trees.
- Donor windows (seed 1): A 257,517; B 487,500; C 259,662; D 2,255,162 rows.
- Seed handling: the seed sets the LightGBM seed **and** the window rng (D inactive-series sampling; rich-dropout masks in the dropout arms).
- Run note: the first background job (PID 89350) did not save per-article detail and was stopped; the run was restarted with detail output. That restart was killed after `pooled_norich` seed 1 cut 1 (likely transient memory pressure on the shared laptop); the two missing cuts were resumed with `pooled_norich --seeds 1 --cuts 2,3`. Scores match the earlier partial run where they overlap (e.g. cut 0 seed 0 `-5.69`).

## Scores by cut and seed (D26, higher is better)

| arm | cut | seed 0 | seed 1 | mean over seeds |
|---|---|---|---|---|
| own | 0 | -4.83 | -4.31 | -4.57 |
| own | 1 | -28.45 | -29.15 | -28.80 |
| own | 2 | -13.35 | -12.82 | -13.09 |
| own | 3 | -15.44 | -15.55 | -15.50 |
| pooled_norich | 0 | -5.69 | -5.77 | -5.73 |
| pooled_norich | 1 | -30.48 | -30.96 | -30.72 |
| pooled_norich | 2 | -13.92 | -13.38 | -13.65 |
| pooled_norich | 3 | -16.14 | -16.17 | -16.16 |

Mean over the 4 cuts, per seed: own -15.52 (seed 0) / -15.46 (seed 1); pooled_norich -16.56 / -16.57.
**Mean over the 4 cuts and both seeds: own -15.49, pooled_norich -16.56.**
Seed spread (max minus min over seeds) per cut: own 0.11-0.69, pooled_norich 0.03-0.54.

## Comparison `pooled_norich - own`
Paired per-article 95% bootstrap interval (1,000 resamples; seeds averaged per article) of the score difference:

| cut | pooled_norich | own | diff | 95% CI | excludes 0 |
|---|---|---|---|---|---|
| 0 | -5.73 | -4.57 | -1.16 | [-2.86, +0.29] | no |
| 1 | -30.72 | -28.80 | -1.92 | [-2.83, -1.23] | yes (negative) |
| 2 | -13.65 | -13.09 | -0.56 | [-0.88, -0.24] | yes (negative) |
| 3 | -16.16 | -15.50 | -0.66 | [-0.95, -0.34] | yes (negative) |

Cuts won by `pooled_norich` over `own`: **0 of 4**.

## Conclusion
Pooling the donor shops (without rich columns) does **not** help the test shop: `pooled_norich` is worse than `own` on all four cuts, by 1.08 points on the mean, and the paired interval excludes 0 (negative) on cuts 1-3. This matches the honest risk stated in the design (the pooled arm had not beaten own on cut 0 before).

Consequence for feature dropout: the rich arm must first recover this ~1.1-point pooling deficit and then clear `own` by the adoption bar (+0.5 mean, at least 3 of 4 cuts). That is a demanding target; a null result for dropout would be a legitimate outcome.
