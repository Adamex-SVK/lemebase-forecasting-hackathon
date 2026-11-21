# Rich columns + feature dropout on the test shop (iteration 04, step 4.3) — results and decision gate

Date: 2026-11-21. Branch: `it04/adamex-svk/rich-columns`. Handle: `adamex-svk`.

## Question
Does a pooled LightGBM (test shop + donor shops A-D) that also sees the donors' rich columns
(price, discount share, promotion flag, markdown level), with feature dropout, forecast the test shop
better than our current best? The test shop has no rich columns, so at forecast time they are NaN;
feature dropout masks them on a random share of donor windows so the trees learn to cope.

## What was run
- Arms (all `--topic dropout`, seeds 0 and 1, cuts 0-3, 300 trees, D26 score):
  - `pooled_rich_nodrop` — rich features, no dropout.
  - `pooled_rich_drop50` — rich features, 50% of donor windows masked.
  - `pooled_rich_drop80` — rich features, 80% of donor windows masked.
- Command pattern (run in per-arm/seed chunks so a memory/slowdown failure is isolated):
  `.venv/bin/python src/run_rich.py <arm> --seeds <s> --cuts 0,1,2,3 --topic dropout`
- Results: `iterations/04-rich-columns/results_adamex-svk_dropout.csv`; per-article MASE in `detail/`.
- Pooling: donor shops A-D all weeks (D27), origin step 4 (D31), `sample_inactive` for D, weights `equal`
  (D30); test-shop windows origin step 2, cut-truncated. Seed sets the LightGBM seed and the window rng.
- Run note: one chunk (`drop80` seed 1, cut 3) hit the 20-minute shell timeout; the runner saves after
  every cut, so that single cut was resumed (`--cuts 3`). All 24 rows are present.

## Scores (D26, higher is better)
### By arm x cut x seed
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
| pooled_rich_nodrop | 0 | -5.60 | -5.15 | -5.37 |
| pooled_rich_nodrop | 1 | -30.01 | -30.05 | -30.03 |
| pooled_rich_nodrop | 2 | -13.35 | -13.02 | -13.18 |
| pooled_rich_nodrop | 3 | -16.47 | -16.64 | -16.56 |
| pooled_rich_drop50 | 0 | -5.61 | -5.12 | -5.37 |
| pooled_rich_drop50 | 1 | -30.78 | -30.94 | -30.86 |
| pooled_rich_drop50 | 2 | -13.22 | -13.15 | -13.18 |
| pooled_rich_drop50 | 3 | -16.43 | -16.16 | -16.30 |
| pooled_rich_drop80 | 0 | -5.81 | -5.44 | -5.62 |
| pooled_rich_drop80 | 1 | -30.74 | -30.27 | -30.50 |
| pooled_rich_drop80 | 2 | -13.14 | -13.24 | -13.19 |
| pooled_rich_drop80 | 3 | -15.83 | -16.27 | -16.05 |

### Mean over the 4 cuts (per seed) and overall
| arm | seed 0 | seed 1 | mean over cuts and seeds |
|---|---|---|---|
| own | -15.52 | -15.46 | **-15.49** |
| pooled_norich | -16.56 | -16.57 | -16.56 |
| pooled_rich_nodrop | -16.36 | -16.22 | **-16.29** |
| pooled_rich_drop50 | -16.51 | -16.34 | -16.43 |
| pooled_rich_drop80 | -16.38 | -16.31 | -16.34 |

## Comparison (a): each rich arm vs `pooled_norich` (does the rich information help?)
Paired per-article 95% bootstrap interval (1,000 resamples, seeds averaged per article), diff = arm - reference.

| arm | cut 0 | cut 1 | cut 2 | cut 3 | mean diff | cuts won |
|---|---|---|---|---|---|---|
| nodrop | +0.36 [0.13, 0.62] | +0.69 [0.34, 1.10] | +0.47 [0.24, 0.70] | -0.40 [-0.64, -0.14] | **+0.28** | 3/4 |
| drop50 | +0.36 [0.05, 0.73] | -0.14 [-0.32, 0.05] | +0.46 [0.32, 0.60] | -0.14 [-0.44, 0.15] | +0.14 | 2/4 |
| drop80 | +0.11 [-0.11, 0.34] | +0.22 [0.01, 0.44] | +0.46 [0.25, 0.64] | +0.11 [-0.06, 0.30] | +0.22 | 4/4 |

All three rich arms are better than `pooled_norich` on the mean; the gain is statistically positive on
cuts 0-2 for the no-dropout arm, and `drop80` wins all four cuts (small but consistent). So the rich
columns carry real (if modest) signal beyond parity.

## Comparison (b): each rich arm vs `own` (does it beat our current best?)
| arm | cut 0 | cut 1 | cut 2 | cut 3 | mean diff | cuts won |
|---|---|---|---|---|---|---|
| nodrop | -0.80 [-2.51, 0.70] | -1.23 [-1.79, -0.78] | -0.10 [-0.36, 0.17] | -1.06 [-1.33, -0.78] | **-0.80** | 0/4 |
| drop50 | -0.80 [-2.22, 0.56] | -2.06 [-2.94, -1.38] | -0.10 [-0.38, 0.20] | -0.80 [-1.19, -0.38] | -0.94 | 0/4 |
| drop80 | -1.05 [-2.73, 0.42] | -1.71 [-2.56, -1.08] | -0.10 [-0.43, 0.22] | -0.55 [-0.80, -0.26] | -0.85 | 0/4 |

**No rich arm beats `own`.** Every arm loses on all four cuts by 0.80-0.94 points on the mean, and the
loss is statistically significant (interval excludes 0, negative) on cuts 1 and 3 for all arms.

## Feature dropout vs no dropout
Dropout does **not** help. The no-dropout rich arm has the best mean (-16.29); masking 50% or 80% of donor
windows is slightly worse (-16.43 / -16.34). `drop80` is more consistent per cut (4/4 vs `pooled_norich`)
but its mean is below `pooled_rich_nodrop`. The hypothesis that dropout is needed to survive the test
shop's all-NaN rich columns is not supported: the no-dropout rich model is already the best rich arm.

## Feature importance (gain), best rich arm `pooled_rich_nodrop`, cut 0, seed 0
Rich features together = **13.0% of total gain**. Top rich features: `markdown_last` 3.05%,
`price_ratio4` 2.47%, `markdown_max4` 1.99%, `disc13` 1.50%, `weeks_since_promo` 1.39%, `disc4` 1.11%,
`promo13` 0.93%, `promo4` 0.58%, `rich_missing` 0.00%. The strongest features overall remain the test-shop
base features (`zshare26`, `oh_last_log`, `series_age`, ...). Rich columns are a real but secondary signal.
(`rich_missing` gain 0 is expected: it is constant NaN-indicator for the test shop and A.)

## Seed noise
Seed spread (max - min over seeds), per arm x cut: own 0.11-0.69; pooled_norich 0.03-0.54;
nodrop 0.04-0.45; drop50 0.08-0.49; drop80 0.09-0.47. The mean differences above (0.28-0.94) are of the
same order as single-cut seed noise, which is why the mean over cuts and the paired intervals matter.

## Verdict
- **No arm clears the adoption bar against `own`** (needs +0.5 mean AND >=3/4 cuts; all are negative and
  0/4). Feature dropout, as implemented here, does not improve the test-shop forecast over our current best.
- The rich columns do help *within the pooled family* (nodrop +0.28 mean, 3/4 cuts vs `pooled_norich`), but
  this only partially offsets the ~1.0-point penalty that pooling the donor shops already carries
  (see `notes_adamex-svk_reference.md`: `pooled_norich` -16.56 vs `own` -15.49).
- Null result, reported plainly. Nothing was adopted; the owner recorded the closure as **D43** (rich-column path not adopted).

## Decision (recorded as D43: not adopted, path closed)
The owner decided (D43). Evidence for the record: all rich/dropout arms trail `own` by 0.80-0.94 points on
the mean and lose all 4 cuts; rich features add ~0.28 over pooled-without-rich; dropout adds nothing over
no-dropout.

Options:
1. **Close the rich-columns / feature-dropout line (recommended).** Record the null result; keep `own` as
   the reference. Do not adopt. This matches the D19 sequence: technique 1 is tested and does not clear the
   bar, so do not build techniques 2-4 on an unproven premise without a new reason.
2. **Continue D19 techniques 2-4 as pooled arms** (cleaning, auxiliary tasks, teacher-student). Expect the
   same ~0.8-1.0-point pooling headwind that the rich columns only partly close; low probability of
   clearing the bar if the main model keeps training on pooled donor data.
3. **Change the structure, not the schema** (more promising if the owner wants to continue): keep the main
   model trained on the test shop (`own`) and use donor shops only to (a) train auxiliary heads that predict
   promo/markdown from the four columns and become features for the test shop, or (b) act as a teacher in a
   teacher-student setup. This sidesteps the pooling penalty that dominates the current results. It needs a
   new plan and its own decision gate.

Recommendation (recorded as D43): option 1 (do not adopt). If the team later wants to pursue D18/D19, prefer
the structure in option 3 over more pooled arms, because the binding constraint is pooling, not the rich
columns.

## Reproduction
```
.venv/bin/python src/run_rich.py pooled_rich_nodrop --seeds 0 --cuts 0,1,2,3 --topic dropout
.venv/bin/python src/run_rich.py pooled_rich_nodrop --seeds 1 --cuts 0,1,2,3 --topic dropout
.venv/bin/python src/run_rich.py pooled_rich_drop50 --seeds 0 --cuts 0,1,2,3 --topic dropout
.venv/bin/python src/run_rich.py pooled_rich_drop50 --seeds 1 --cuts 0,1,2,3 --topic dropout
.venv/bin/python src/run_rich.py pooled_rich_drop80 --seeds 0 --cuts 0,1,2,3 --topic dropout
.venv/bin/python src/run_rich.py pooled_rich_drop80 --seeds 1 --cuts 0,1,2,3 --topic dropout
.venv/bin/python iterations/04-rich-columns/analyse_rich.py \
    --results iterations/04-rich-columns/results_adamex-svk_dropout.csv \
    --also iterations/04-rich-columns/results_adamex-svk_reference.csv \
    --arms pooled_rich_nodrop,pooled_rich_drop50,pooled_rich_drop80 --refs pooled_norich,own --seeds 0,1
.venv/bin/python iterations/04-rich-columns/feature_importance_rich.py --seed 0 --cut 0
```
