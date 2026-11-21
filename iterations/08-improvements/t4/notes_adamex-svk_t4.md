# T4 - flatten the forecast over the 13-week horizon (iteration 08)

Verdict: **none of the three flat variants passes the adoption bar; keep D40.**
All three lose on all four cuts. The horizon-growth premise is only weakly true and the
week-to-week wiggle in the D40 blend is too small to be worth removing (it mostly disappears
at the mandatory rounding step anyway).

## Method

Per standard cut k (D26 metric, cuts 0-3), the un-rounded D40 blend is rebuilt exactly as
`src/meta/ensemble_recipe.py evaluate` does it (same reference reproduced: -3.72 / -25.91 /
-11.01 / -14.02, mean -13.67):

    Fl   = LightGBM (iterations/02-robust-eval/forecasts/base_cut{k}_seed0.npy)
    mod  = mean(moirai-2.0-small, tirex, chronos-t5-small) raw forecasts, 0 where no sale in last 8 weeks
    D40  = 0.3 * Fl + 0.7 * mod

Three parameter-free replacements of each article's 13 values, then the usual `round`:

| variant | definition |
|---|---|
| `flat_median` | repeat median(D40 over the 13 weeks) |
| `flat_mean` | repeat mean(D40 over the 13 weeks) |
| `flat_median_fm` | 0.3 * Fl (week-by-week) + 0.7 * repeat median(mod) |

No tuning, no fitted parameters. Code: `src/improve/t4_flatten.py`.

## Results (score, higher is better)

| variant | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 | cuts won | verdict |
|---|---|---|---|---|---|---|---|---|
| D40 (reference) | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 | - | - |
| flat_median | -3.87 | -26.42 | -11.33 | -14.24 | -13.96 | -0.30 | 0/4 | keep D40 |
| flat_mean | -4.01 | -26.00 | -11.98 | -14.25 | -14.06 | -0.39 | 0/4 | keep D40 |
| flat_median_fm | -3.82 | -26.29 | -11.31 | -14.18 | -13.90 | -0.24 | 0/4 | keep D40 |

Adoption bar: mean gain >= +0.5 and better on >= 3 of 4 cuts. Closest variant
(`flat_median_fm`) is -0.24 with 0/4.

## Why it fails

1. **The error does grow with the horizon, but only mildly and flattening does not remove it.**
   Mean absolute error by horizon week (average over the 4 cuts) for D40:
   4.58, 4.80, 4.68, 4.19, 4.56, 4.41, 4.60, 4.57, 4.61, 4.81, 4.62, 5.01, 5.37.
   Late/early (weeks 10-13 / weeks 1-4) ratio is 1.086 for D40 and 1.076-1.078 for the flat
   variants - a ~1% reduction, i.e. nothing. Weeks 12-13 are high for every variant, so the
   growth is a calendar/level effect the flat level cannot fix.
2. **The wiggle is below the rounding resolution.** After `round`, 71-73% of articles have the
   identical 13-week forecast under D40 and `flat_median`; flattening only ever touches the
   remaining ~27%.
3. **Among the articles that change, wins and losses roughly cancel in count but not in MASE.**
   `flat_median` is better for 13.8-15.6% of articles and worse for 12.5-13.7%. MASE divides
   each article's error by its own week-to-week scale, so the losses concentrate in low-scale
   articles and are amplified: mean score change by scale quartile (small -> large, negative =
   flattening hurt) is -0.35 / +0.22 / -0.65 / +0.20 (cut 0), -0.74 / -0.71 / -0.71 / +0.13
   (cut 1), -1.55 / -0.07 / +0.23 / +0.13 (cut 2), -1.05 / -0.22 / +0.21 / +0.21 (cut 3).
4. **Keeping the LightGBM part week-by-week is the least bad variant** (-0.24), consistent with
   the foundation part carrying more of the removable noise than the GBM part. Still negative.

## Recommendation

Keep D40. If flattening is revisited, it would need to be conditional (e.g. only for stable
high-volume articles, or before the 0.3/0.7 blend with a tuned weight), which is outside this
task's parameter-free scope and is not supported by the per-scale breakdown above.

## Files, runtime, failures

- Added: `src/improve/t4_flatten.py`, `iterations/08-improvements/t4/results_adamex-svk_t4.csv`,
  `iterations/08-improvements/t4/notes_adamex-svk_t4.md`.
- Runtime: ~5 s for the 4 cuts (plus ~1 s import); evaluation is pure numpy over cached parquet forecasts.
- Failures: none; `bash scripts/check.sh` passes. No leakage: only the cut's own history and its
  pre-existing forecast files are read. Nothing was built or uploaded (no variant passed).
