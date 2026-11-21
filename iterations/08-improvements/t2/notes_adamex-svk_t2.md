# T2 - post-processing tuning (iteration 08)

Owner: adamex-svk (worker T2), branch `it08/adamex-svk/t2`, worktree `/Users/adomi/Documents/hack-t2`.
Code + results commit: `83ab4a6`.

## What was run

```bash
.venv/bin/python src/improve/t2_postproc.py
```

`src/improve/t2_postproc.py` rebuilds, for each of the 4 standard cuts, the **un-rounded D40 blend**
exactly as `src/meta/ensemble_recipe.py evaluate` does:

```
mean(raw forecasts of moirai-2.0-small, tirex, chronos-t5-small)
-> 0 for articles with no sale in the last 8 training weeks (zero rule)
-> 0.3 * LightGBM + 0.7 * ensemble
```

and then scores one post-processing variant at a time on the D26 metric (`validate.score` via
`experiments_02.score_F`). Reproduced D40 on the same cuts:
**-3.72 / -25.91 / -11.01 / -14.02, mean -13.67** (matches the shared reference exactly).

* Parameters are chosen **leave-one-cut-out**: for cut K the parameter is selected on the other three
  cuts and then scored on cut K; the honest table is the mean of those four honest scores.
* The **in-sample optimum** (parameter chosen on all four cuts) is listed separately and is *not*
  evidence for adoption.
* Zero-rule windows use only training weeks before the cut's first holdout week (no leakage).
* The per-segment offset exploits that the metric is a mean of per-article MASE: the three segments'
  contributions are separable, so each segment's offset is chosen on its own mean MASE over the
  other cuts.
* Grids: offset c = 0.30..0.70 step 0.05; shrink a = 0.80..1.05 step 0.05; zero window
  W in {4,6,8,10,13,26}; small-value cut t in {0.5,0.75,1.0}; per-segment offset c for
  mean-of-last-13-weeks segments <1, 1-5, >5.

Runtime: ~48 s (one Python process, local CPU).

## Honest table (leave-one-cut-out), vs D40

| variant (honest) | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 | cuts won | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D40 reference | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 | - | - |
| rounding offset `floor(x+c)` | -4.02 | -25.91 | -11.01 | -14.02 | -13.74 | -0.07 | 0/4 | reject |
| shrink `round(a*x)` | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 | 0/4 | reject |
| zero-rule window W | -3.70 | -25.86 | -11.00 | -13.97 | -13.63 | **+0.03** | **4/4** | reject (below +0.5) |
| small-value cut t | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 | 0/4 | reject |
| per-segment offset (c per segment) | -4.13 | -26.47 | -12.11 | -13.83 | -14.14 | -0.47 | 1/4 | reject |

Selected parameters per cut (honest): offset `0.60/0.50/0.50/0.50`; shrink `1.00/1.00/1.00/1.00`;
zero window `4/4/4/4`; small cut `0.5/0.5/0.5/0.5`; per-segment
`0.70/0.30/0.35 | 0.70/0.30/0.70 | 0.70/0.30/0.60 | 0.70/0.70/0.60`.

## In-sample optima (NOT honest; upper bound only)

| variant | best parameter | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 |
|---|---|---:|---:|---:|---:|---:|---:|
| rounding offset | c = 0.50 | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 |
| shrink | a = 1.00 | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 |
| zero-rule window | W = 4 | -3.70 | -25.86 | -11.00 | -13.97 | -13.63 | +0.03 |
| small-value cut | t = 0.5 | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 |
| per-segment offset | c = 0.70/0.30/0.70 | -4.24 | -26.47 | -12.12 | -14.14 | -14.24 | -0.58 |

The whole grid, including every individual parameter value, is in
`results_adamex-svk_t2.csv` (`kind = in_sample_grid`).

## Conclusion

**No variant passes the adoption bar** (mean gain >= +0.5 and better on >= 3 of 4 cuts).

* D40's rounding (`c = 0.5`) is already the in-sample and leave-one-cut-out optimum; every other
  offset trades cut3 for cut0 and loses on the mean.
* Shrinking before rounding never helps; `a = 1.0` is optimal.
* The zero rule is very flat: `W = 4` wins all four cuts by a tiny +0.03, which is well inside
  seed/sampling noise. Not worth a recipe change.
* Zeroing small positive forecasts (`t >= 0.75`) helps cut0 but hurts the other cuts strongly.
* Per-segment offsets are unstable leave-one-cut-out (cut0 wants small offsets, cut3 wants large
  ones), so they lose overall.

D40 is left unchanged.

## Files

* `src/improve/t2_postproc.py` - experiment code.
* `iterations/08-improvements/t2/results_adamex-svk_t2.csv` - full wide results table
  (D40 reference, every in-sample grid point, honest LOOCV rows, in-sample optima).
* `iterations/08-improvements/t2/notes_adamex-svk_t2.md` - this note.

No leakage: for every cut only weeks before its first holdout week are used (the zero-rule window is
computed from `ShopMatrix.U`, which is built from `df[df["week"] < cut]`).

## Failures / suspicious

None. `bash scripts/check.sh` passed before starting.
