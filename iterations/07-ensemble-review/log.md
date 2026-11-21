# Ensemble review — 2026-11-21

## Result

Keep the current recipe: the tested removals and alternative LightGBM weights
do not improve its four-cut average. This review independently reproduces the
saved evidence; it is not a new untouched validation experiment.

The user confirmed **-4.8 is a public leaderboard score**. The friend's branch
records the same score in `submissions/UPLOADS.md`. It improves on the reported
LightGBM -7.6 by 2.8 points on the public period; this is not a measured private
weeks 8–13 gain.

## Evidence recovery and preservation

`git fetch origin` revealed `origin/it03/adamex-svk/foundation-models` at
`71b82f33f0fe5647a13d3786cd17d11b4c4e5493`. No branch was checked out or merged.
Its historical/final component Parquets and LightGBM arrays were read directly
from Git objects. No donor code was executed. The donor's `prepare.py` and
`validate.py` have no differences from the versions on this branch.

`evaluate.py` verifies complete, unique and finite nonnegative keyed forecasts,
exact forecast dates, NPY row order against keyed local LightGBM forecasts, and
agreement with the existing project scorer. It reconstructs the final forecast
and compares it with the committed submission: **maximum difference 0 units**.

The original `submissions/ensemble_top3_blend_v1.csv` was copied byte-for-byte
from the immutable commit into this branch. It was not rebuilt with new weights
or uploaded. All source-blob fingerprints are in [metadata.json](results/metadata.json).

## Exact incumbent recipe

1. Average raw Moirai 2.0-small, TiRex, and Chronos-T5-small forecasts equally.
2. Set that foundation average to zero for products with no sales in the last
   eight observed weeks.
3. Blend 70% foundation average with 30% original LightGBM.
4. Apply NumPy rounding to whole units, then clip at zero.

The zero rule affects the foundation average before blending, not a separate
forced zeroing of all final forecasts. Rounding individual components first
would be a different recipe. Effective unrounded weights are 23.33% per
foundation model plus 30% LightGBM.

The previously poor local `logs/chronos_four` run is **Chronos-Bolt-small**, not
the **Chronos-T5-small** used here. Do not generalize that result across the family.

## Reproduced current scores

| Historical cut | Full 13 weeks | Weeks 1–7 | Weeks 8–13 |
|---|---:|---:|---:|
| 0 | -3.72 | -3.02 | -4.54 |
| 1 | -25.91 | -31.14 | -19.80 |
| 2 | -11.01 | -5.29 | -17.69 |
| 3 | -14.02 | -9.72 | -19.04 |

Exact values are in [ablation_by_cut.csv](results/ablation_by_cut.csv). The
mean score is **-13.6659**, versus raw LightGBM **-15.5204** and rounded LightGBM
**-15.1192**. Current ensemble beats both references on all four cuts. Private-like
historical weeks 8–13 also improve in their four-cut mean, but are not the actual
hidden competition weeks. Scored counts/coverage follow the unchanged D26 metric.

## Component and processing ablations

Leave-one-out variants keep 30% LightGBM and redistribute the 70% foundation
share equally over the remaining models; they are not separately weight-tuned.

| Variant | Four-cut mean | Change versus incumbent |
|---|---:|---:|
| Current ensemble | **-13.6659** | 0 |
| Remove zero-sales rule | -13.6823 | -0.0164 |
| Remove Chronos-T5-small | -13.7941 | -0.1282 |
| Remove TiRex | -14.0573 | -0.3914 |
| Remove Moirai 2.0-small | -14.0883 | -0.4224 |
| Remove LightGBM | -14.0483 | -0.3824 |
| Moirai 2.0-small alone, processed | -14.7162 | -1.0503 |
| TiRex alone, processed | -15.9667 | -2.3008 |
| Chronos-T5-small alone, processed | -17.9397 | -4.2738 |
| Current ensemble without rounding | -15.7211 | -2.0551 |

Sources: [ablation_summary.csv](results/ablation_summary.csv), per-cut paired
intervals in `ablation_by_cut.csv`, activity breakdown in
[activity_groups.csv](results/activity_groups.csv), and signed scaled-error
correlations in [error_correlations.csv](results/error_correlations.csv).
Intervals resample products within a cut (500 replicates, seed 0); they do not
capture temporal uncertainty, repeated-search bias, or multiplicity.

Interpretation: component diversity contributes, but Chronos's marginal benefit
is small in this test. These removals establish the tested blends' behavior,
not an intrinsic causal importance or an optimal subset under all possible weights.
Rounding is the largest processing effect measured here; it is an empirical
finding on these count-like series, not a general guarantee for MASE problems.

## LightGBM weight check

The fixed grid 0.0, 0.1, ..., 1.0 selects **0.3** using earlier cuts 1–3 alone.
It is also best by the four-cut mean. Selected comparison points:

| LightGBM share | Four-cut mean |
|---|---:|
| 0% | -14.0483 |
| 20% | -13.7936 |
| **30%** | **-13.6659** |
| 40% | -13.8785 |
| 50% | -13.9385 |
| 100% | -15.1192 |

Full grid: [weights_summary.csv](results/weights_summary.csv) and
[weights_by_cut.csv](results/weights_by_cut.csv). Although cut 0 alone favors a
different weight, choosing it after viewing its score would be further tuning.
Do not replace the current weights based on that one period.

## Next action and limits

Keep AutoGluon optional and the current ensemble unchanged. Do not spend the next
round on a finer weight grid: the measured opportunity is small and all four cuts
have already influenced development. Stronger evidence would come from additional
historical component forecasts with selection on prior nonoverlapping targets,
followed by reserved confirmation; overlapping windows require purging and must
not be counted as independent samples. The friend's branch discusses additional
GPU jobs, but this review did not verify their live status or retrieve remote
machine outputs. No jobs were started or stopped.

The target-shop review cannot establish performance across A-D or unseen shops.
No new model or weight configuration was promoted. No commit, push, or submission
was performed by this review.
