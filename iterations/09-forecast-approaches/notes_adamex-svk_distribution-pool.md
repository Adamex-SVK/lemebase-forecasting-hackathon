# Iteration 09 / task 3 — distribution pooling (adamex-svk)

**Status: RUN on GPU, REJECTED (below the +0.25 kill gate).**

## What was done

Quantile forecasts (levels 0.1..0.9) were exported on the GPU machine for the three D40 members
(`moirai-2.0-small`, `tirex`, `chronos-t5-small`) at cuts 0-3 and the final origin, with

```bash
python src/foundation/run_model.py MODEL test [--cut-index K | --final] --emit-quantiles
```

writing `iterations/09-forecast-approaches/quantiles/{model}__test__{cutK|final}__padded.npz`
(`quantiles [2904, 13, 9]`, `levels`, string `store_id`/`article_id`, `week`; each file < 1 MB).
`src/improve/i09_distribution_pool.py` then builds the equal-weight linear pool of the three CDFs,
reconstructs the mixture median by monotone inverse-CDF interpolation, and scores it.

The additive code path changes no default: `src/foundation/models.py` gains `QUANTILE_LEVELS` and
`forecast_quantiles(...)` (per-family quantile paths); `src/foundation/run_model.py` gains opt-in
`--emit-quantiles`; `forecast()` and every `SPECS` default are unchanged.

## Results (4 standard cuts, round + clip; D40 = -13.67)

| variant | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 | cuts won |
|---|---|---|---|---|---|---|---|
| D40 | -3.72 | -25.91 | -11.01 | -14.02 | **-13.67** | — | — |
| pooled median alone | -3.74 | -25.85 | -12.52 | -14.48 | -14.15 | -0.48 | 1/4 |
| pooled median in D40 recipe | -3.28 | -25.49 | -12.25 | -14.34 | **-13.84** | **-0.18** | 2/4 |

`pooled_in_D40` = zero rule -> `0.3*LightGBM + 0.7*pooled median` -> round -> clip.
Paired per-article bootstrap (1000 resamples, seed 0), `pooled_in_D40` vs D40: **[-0.81, +0.53]** (covers 0).

## Calibration checks

- Quantile monotonicity: **0 crossings** (all models, all cuts).
- 80% interval coverage (target 0.80): moirai 0.77-0.83, tirex 0.77-0.82, **chronos-t5-small 0.26-0.33**
  (severely under-covered; 80% width ~11 vs ~15-18 for the others), pooled 0.75-0.81.
- Tail dependence (median crossing needing an unobserved <0.1 or >0.9 tail): 0.02 (cut0) .. 0.11 (cut3);
  ~0 at the final origin.

## Verdict

**REJECT.** The mixture median is not better than D40's mean-of-medians: the best variant gains -0.18,
below the +0.25 kill gate and far below the +0.5 / 3-of-4 adoption bar, with the interval covering zero.
The three members' medians are already close, and chronos-t5-small's miscalibrated quantiles drag the
equal-weight pool down. The pooled point forecasts were saved as pseudo-model `mixture-median-top3`
(cuts + final) for completeness, and a below-bar demonstration submission
`submissions/ensemble_pooled_median_v1.csv` (pooled median in the D40 recipe) was built at the person's
request for a live sanity check only. No recipe change.

## Fixes made during the run

- `run_model.py --emit-quantiles` was fixed to save quantiles aligned to all 2,904 series (inactive
  all-zero-history series keep zero quantiles) and string ids; otherwise the consumer mis-aligned and
  would fail on cuts where some series have no history.
- `i09_distribution_pool.py` had a reporting bug (`KeyError: 'D40'`); fixed by also recording the D40
  score row so the summary table has its reference row.

## Reproduction

```bash
# on the GPU machine (each model in its own venv), then copy the npz files back:
python src/foundation/run_model.py moirai-2.0-small test --cut-index K --emit-quantiles   # K = 0..3
python src/foundation/run_model.py moirai-2.0-small test --final      --emit-quantiles
# ... same for tirex and chronos-t5-small ...
.venv/bin/python src/improve/i09_distribution_pool.py
```
