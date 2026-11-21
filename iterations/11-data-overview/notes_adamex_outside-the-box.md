# Outside the box: where does the best recipe (D40) actually lose? (2026-11-21, adamex)

Question from the user: the leaders "must have found something we are missing". Instead of another model variant,
take D40's local forecasts apart and test the cheap hypotheses directly. D40 cut forecasts are rebuilt with
`src/probes/d40_cuts_dump.py` (reproduces -3.72 / -25.91 / -11.01 / -14.02, mean -13.67 exactly).
Analysis snippets were run inline; numbers below are from those runs.

## 1. Is the test shop a copy of a sibling set? No.
No article ids overlap. Sales fingerprints (6-week runs with >= 5 selling weeks and >= 15 units) do coincide
(A 1,012, B 6,430, C 87, D 18 of 68,581 test patterns), but no (test series, sibling series, week shift) pair
lines up on more than 4-5 overlapping windows, which is chance for small integers; a copy would match over 100+ weeks.

## 2. The loss is concentrated in a few near-dead articles
- The top 1% of articles (about 27) carry 25-33% of the total MASE on every cut; the top 10% carry 59-64%.
- The lowest-scale fifth (scale about 0.05: one or two sales in the whole history) carries 25-41% of the loss.
  One such article that wakes up (e.g. scale 0.02, 1.7 units/week in the holdout, forecast 0) costs about 4 score points alone.
- **These wake-ups are not predictable from stock.** Among articles with no sale in the last 8 weeks, those with stock on
  hand sell in only 8-11% of holdout weeks (0.6-0.7% without stock). Forecasting 1 instead of 0 for them would cost
  -130 to -370 points; even the sharpest signal (stock arriving last week, 1-6 articles) loses. Forecasting 0 is right.
- An oracle that knows the best flat level for every dead article gains only +0.74 on the mean. Most of this loss is luck.

## 3. The under-forecast is correct, not a bug
D40's total forecast is 78-89% of actual sales on every cut. But MAE is minimised by the median, which for skewed
counts sits below the mean. Scaling the forecast up (x1.05-1.3) or moving the rounding threshold (0.3-0.6) never
beats D40 on the mean; leave-one-cut-out picks "no change" on 3 of 4 cuts (mean -0.07). Scaling only the
regularly selling articles loses too (-0.04 to -1.13).

## 4. Where the headroom is
| Oracle (uses the holdout; ceiling only) | cut0 | cut1 | cut2 | cut3 | mean |
|---|---|---|---|---|---|
| D40 | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 |
| D40, perfect flat level on dead articles only | -2.45 | -24.61 | -10.93 | -13.72 | -12.93 |
| D40, perfect flat level on non-dead articles only | 8.73 | -8.53 | 2.02 | 0.63 | 0.71 |
| Best flat line per article (median of actuals) | 10.01 | -7.24 | 2.10 | 0.93 | 1.45 |
| Pick the best of our models per article | 5.84 | -17.06 | -4.12 | -5.02 | -5.09 |

The room is in the **level of articles that are still selling** (up to +14 with a perfect flat level), not in
week-to-week shape and not in dead articles. Picking the right existing model per article is worth up to +8.6, but
FFORMA (D42) could not predict the pick. Anything new should therefore target "what will this active article's median
weekly sales be over the next 13 weeks", for example with signals our models do not see: price/promotion, assortment
changes, sibling articles. The test shop has only units and on_hand, so the remaining levers are few.

## Ruled out today (do not repeat)
Copy/leak from sibling sets; stock-based wake-up override of the zero rule; global or active-only bias multiplier;
rounding threshold; hard trim of early training weeks (D46). Soft trim and 3-seed trim: see `notes_adamex_trim.md`.

## 5. Follow-up: a model trained directly on the level of active articles (`src/experiments_level.py`)
LightGBM, one row per (active series, origin), default features without `h`, target = median of the next 13 weeks / scale
(the metric's best flat forecast), L1 loss, 3 seeds averaged; active = a sale in the last 8 weeks, dead = 0.
Log `logs/exp_level.txt`, results `results_adamex_level.csv`.

| Variant (rounded) | cut0 | cut1 | cut2 | cut3 | mean | vs D40 |
|---|---|---|---|---|---|---|
| D40 | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | |
| level model alone | -4.17 | -27.65 | -12.06 | -14.89 | -14.69 | -1.02 |
| D40 mixed with level on active articles, weight 0.1-1.0 | | | | | | -0.10 to -1.03 |
| D40 mixed with median13 on active articles, weight 0.1-1.0 | | | | | | -0.09 to -2.87 |
| 0.3 level + 0.7 foundation ensemble (level replaces the LightGBM share) | -3.77 | -25.71 | -11.01 | -14.06 | -13.64 | +0.03, 1/4 |

The level model alone beats our per-week LightGBM (-15.12 rounded) by about 0.4, but it adds nothing to D40: the
foundation ensemble already estimates the level as well as our 4-column features allow. Leave-one-cut-out picks weight 0
on every cut. **Verdict: the level of active articles is where the headroom is, but our current inputs (units + on_hand
of one shop) do not predict it better than D40. New information is needed to move it, not a new model.**
