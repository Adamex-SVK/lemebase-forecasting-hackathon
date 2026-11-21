# Iteration 09, task 1 — error and headroom atlas for D40

**Commit:** `dfc50f1`
**Script:** `src/improve/i09_error_atlas.py`
**Results:** `iterations/09-forecast-approaches/results_adamex-svk_error-atlas.csv`

## Command

```bash
cd /Users/adomi/Documents/hack-t11
.venv/bin/python src/improve/i09_error_atlas.py
```

This document only writes up the output of that run; nothing was re-run for these notes.

## What was run

The script rebuilds the unrounded D40 matrices exactly as `src/meta/ensemble_recipe.py` does
(Moirai 2.0-small + TiRex + Chronos-T5-small, zero rule, `0.3 * LightGBM + 0.7 * ensemble`), then:

* scores D40 and its members, `median13`, zero, NHITS, Sundial, the q40 variants and the `last52`
  variants on the 4 standard cuts and on the 13 overlapping cuts (offsets 0, 4, ..., 48);
* decomposes D40's scaled error (MASE contribution, not raw MAE) by scale quartile, zero share,
  weeks since sale, forecast level, horizon week and calendar week;
* measures the concentration of excess scaled error in the worst 1%, 5% and 10% of articles;
* computes a **non-adoptable** per-article oracle (choose the best model for each article using the
  realised holdout) and a **purged pre-origin** LightGBM multiclass selector that tries to predict
  that choice from pre-origin features only (donors are restricted to origins whose whole 13-week
  holdout ends before the target origin, i.e. `W >= target + 13`);
* checks whether the per-article error ranking persists across adjacent overlapping cuts;
* compares the padded and `last52` D40 forecasts for complementarity.

## D40 reproduction (standard cuts)

| cut | first holdout week | D40 score |
|---|---|---:|
| 0 | 2023-01-23 | **-3.72** |
| 1 | 2022-10-24 | **-25.91** |
| 2 | 2022-07-25 | **-11.01** |
| 3 | 2022-04-25 | **-14.02** |
| mean | — | **-13.67** |

D40 reproduces exactly on all four standard cuts. `ensemble_recipe.py evaluate` prints the same
by-cut scores, and `ensemble_recipe.py build` reproduces `submissions/ensemble_top3_blend_v1.csv`
byte-identically; both were verified separately.

## Where the error is concentrated

Share of total D40 scaled error carried by the worst articles (standard cuts):

| cut | top 1% (n) | top 5% (n) | top 10% (n) |
|---|---:|---:|---:|
| 0 | **26.3%** (28) | **47.3%** (142) | **59.1%** (284) |
| 1 | 31.0% (28) | 53.0% (138) | 63.6% (276) |
| 2 | 25.6% (27) | 49.1% (134) | 60.1% (268) |
| 3 | 33.2% (26) | 53.2% (130) | 63.2% (260) |

On cut 0, 1% of articles carry 26.3% of the scaled error, 5% carry 47.3% and 10% carry 59.1%.
The decomposition also shows that the smallest scale quartile (Q1) carries 37.0% / 33.6% / 41.9% /
45.3% of the total error on cuts 0-3, so most of the loss sits in a small, low-scale,
intermittent tail rather than being spread uniformly across the catalogue.

## Oracle headroom (non-adoptable diagnostic)

The per-article oracle chooses the best candidate model for each article using the realised holdout.
It is **not a forecast and not adoptable**; it only measures how much error a perfect per-article
model choice would remove.

| cut kind | per-cut oracle gain vs D40 | mean |
|---|---|---:|
| standard | +12.44 / +11.70 / +10.49 / +10.78 | **+11.35** |
| overlapping (13 cuts) | +6.20 to +9.85 | **+8.34** |

So there is a large theoretical headroom (about +8 to +11 points) if the best model could be picked
per article.

## Purged pre-origin model selector (non-adoptable diagnostic)

A LightGBM multiclass model over the overlapping cuts was trained to pick the best model per article
from pre-origin features only, using purged donors (`W >= target + 13`). It is a diagnostic of whether
the oracle headroom is predictable, not an adoptable recipe.

| target | donor offsets | selector score | D40 score | gain | accuracy / majority |
|---|---:|---:|---:|---:|---:|
| 0 | 16..48 | -5.12 | -4.17 | -0.95 | 0.700 / 0.673 |
| 4 | 20..48 | -2.00 | -2.17 | **+0.18** | 0.707 / 0.676 |
| 8 | 24..48 | -15.98 | -11.98 | -4.00 | 0.697 / 0.669 |
| 12 | 28..48 | -21.82 | -20.88 | -0.94 | 0.691 / 0.669 |
| 16 | 32..48 | -39.81 | -36.26 | -3.55 | 0.688 / 0.668 |
| 20 | 36..48 | -27.38 | -26.90 | -0.48 | 0.694 / 0.675 |
| 24 | 40..48 | -2.84 | -1.01 | -1.84 | 0.686 / 0.669 |
| 28 | 44..48 | -24.20 | -23.05 | -1.15 | 0.690 / 0.676 |
| mean | — | **-17.40** | **-15.81** | **-1.59** | **0.694 / 0.672** |

The selector wins **1 of 8** purged targets (target 4, +0.18) and loses **1.59 points** on average.
Its accuracy (0.694) is barely above the majority-class rate (0.672), so it is not learning a usable
model-choice signal. The large oracle headroom is therefore **not predictable from pre-origin
features**.

## Gate for dynamic blending (plan task 6)

The task-6 gate required oracle overlap headroom >= +1.0 **and** selector wins >= 3 of the purged
targets. Oracle overlap is +8.34 (passes the first condition), but the selector wins 1/8 (fails the
second).

**Gate: CLOSED.** Dynamic / learned per-article blending is not worth pursuing and task 6 is gated off.

## Persistence of per-article error

Adjacent overlapping cuts (w4 vs w0, ..., w48 vs w44):

* mean Spearman correlation of per-article D40 error: **0.876** (range 0.862-0.896);
* mean top-decile overlap: **0.795** (range 0.723-0.813).

Hard articles are consistently hard across cuts. That persistence is real, but the selector result
above shows it does not translate into a working per-article model selector: knowing that an article
is hard does not tell us which model will be better for it before the origin.

## Padded vs last52 D40 complementarity

| cut | Spearman rho | fraction last52 better | gain_last52 |
|---|---:|---:|---:|
| 0 | 0.992 | 0.148 | -0.47 |
| 1 | 0.994 | 0.153 | +0.33 |
| 2 | 0.993 | 0.150 | -2.01 |
| 3 | 0.993 | 0.157 | -0.05 |

The two D40 variants rank articles almost identically (rho 0.992-0.994) and `last52` is better for only
about 15% of articles. Apart from cut 2 they are essentially the same forecast, so they are **not
complementary** and there is no useful blend to build from the pair.

## Verdict

**The error atlas is the useful output of this task; there is no recipe change.**

* D40 is reproduced exactly (-13.67 on the 4 standard cuts), so the atlas is built on the current
  reference forecast.
* Error is highly concentrated: a small fraction of articles (top 1% / 5% / 10% carry about
  26% / 47% / 59% on cut 0) accounts for most of the scaled loss, mostly in low-scale, intermittent
  series.
* Error is persistent across cuts (rho 0.876, top-decile overlap 0.795), but the per-article best model
  is not predictable pre-origin: the purged selector gains -1.59 and wins 1/8, despite an oracle
  headroom of +8.34 (overlap) / +11.35 (standard).
* Therefore dynamic or learned per-article blending is **not worth pursuing**; the plan's task 6 is
  **gated off** (gate closed).
* The oracle and selector numbers are **non-adoptable diagnostics**, not candidate recipes.

## Leakage and limits

* The oracle uses the realised holdout by construction and is labelled non-adoptable everywhere.
* The selector uses only pre-origin features and purged donors (`W >= target + 13`), so every donor
  future ends before the target origin; there is no holdout leakage into its features.
* The padded-vs-last52 comparison is descriptive only.
* All scores are the D26 scaled metric, round + clip, as in D40.

## Verification

D40 reproduction was verified separately with `ensemble_recipe.py evaluate` (by-cut -3.72 / -25.91 /
-11.01 / -14.02, mean -13.67) and `ensemble_recipe.py build` (byte-identical
`submissions/ensemble_top3_blend_v1.csv`). The atlas CSV is the output of commit `dfc50f1`; it was not
re-run for this write-up.
