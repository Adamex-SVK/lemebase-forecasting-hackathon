# Iteration 02 log (robust evaluation, F6, `on_hand`)

Append-only.

## 2026-11-21 night: F6 investigation, the moving yearly peak (user asked to run it now)
- Code: `src/probes/peak_f6.py` (read-only analysis of the weekly tables). Run: `.venv/bin/python src/probes/peak_f6.py`.
- **The top weeks of the test shop are exactly 50 weeks apart:** 2020-11-16 (53,036), 2021-11-01 (37,396), 2022-10-17 (39,333). The autocorrelation of the detrended log weekly total is broad and weak around one year (lag 50: 0.42, lag 51: 0.47, lag 52: 0.43), while Set A has a sharp annual cycle (lag 52: 0.74; its peaks are on fixed dates).
- **Different articles drive each peak:** 816-971 of 2,904 articles rise above their normal level (+75%, +44%, +56% in total); the top-20 articles are 27-31% of the excess; the top-50 articles overlap only 25-30% between years (Jaccard 0.30, 0.25, 0.25). So it is an event-like surge across many changing articles, not product seasonality that repeats.
- **The 13 forecast weeks (2023-04-24 .. 2023-07-17) contain no peak.** The same calendar weeks 52 weeks earlier (2022) total 24-31k per week, 104 weeks earlier (2021) 23-33k, and the last 13 known weeks 24.7-31.2k. The next peak would fall near 2023-10-02 (50-week spacing) or 2023-10-23 (52 weeks), outside the window.
- **Conclusion:** F6 does not matter for this task. No calendar feature is needed; the caution remains for any model that would use week-of-year. Closed in `decisions-to-make.md`.
- **Side finding (D33): the test shop's `on_hand` changes regime.** Total stock about 955k in 2020, 388k in Jan 2021, about 100k from Apr 2021; stock cover (`on_hand` / weekly units) 27 -> 3; missing share 62% -> 2%; median per article-week 14 (2020), 4 (2021), 5 (2022), 3 (2023). Our raw `oh_last_log` feature therefore has a different scale in early windows. Open decision D33.

## 2026-11-21 night: rolling-origin evaluation (D32), started in the background
- Code: `src/rolling.py` (4 cuts on the test shop: last 13, 26, 39, 52 weeks dropped; zero, naive, mean13, median13, seasonal naive, Top-down, AutoARIMA and LightGBM own arm). Run log `logs/rolling_test.txt`, results `iterations/02-robust-eval/results_rolling_test.csv`. Running alongside step 1.3, so both are slower.
- **Results (finished; `results_rolling_test.csv`, run log `logs/rolling_test.txt`).** Test shop, D26 score, cut 0 = the standard local cut, cut 3 = the earliest; holdout starts 2023-01-23, 2022-10-24, 2022-07-25, 2022-04-25:

  | model | cut 0 | cut 1 | cut 2 | cut 3 | mean |
  |---|---|---|---|---|---|
  | lgbm_own | -4.8 | -28.5 | -13.4 | -15.4 | **-15.5** |
  | median13 | -6.7 | -30.5 | -12.6 | -16.3 | -16.5 |
  | zero | -23.4 | -49.3 | -32.3 | -33.6 | -34.7 |
  | autoarima | -28.8 | -51.4 | -30.8 | -31.0 | -35.5 |
  | topdown_mstl | -28.6 | -54.6 | -31.5 | -30.4 | -36.3 |
  | mean13 | -36.0 | -52.7 | -35.0 | -37.8 | -40.4 |
  | seasonal_naive | -40.9 | -71.0 | -46.7 | -49.6 | -52.0 |
  | naive | -58.3 | -97.6 | -70.4 | -46.6 | -68.2 |

- **LightGBM (own) minus median13, paired over articles, 95% bootstrap interval:** cut 0 +1.85 [-0.22, +3.84]; cut 1 +2.08 [-0.28, +4.16]; cut 2 **-0.76** [-2.27, +0.63]; cut 3 +0.90 [+0.04, +1.78]. Average +1.02 points; LightGBM wins 3 of 4 cuts.
- **Reading:** (1) the ranking of the baselines is stable across all four cuts: median13 and the LightGBM are about 20 points ahead of AutoARIMA, Top-down and the zero forecast, which are about 20 points ahead of naive and seasonal naive. (2) The LightGBM's edge over the median is only about 1 point on average, and only one of four intervals excludes zero: the single-cut gain of 1.85 in step 1.2 was optimistic; the honest estimate is about +1. (3) The difficulty of a window varies a lot (cut 1, which covers Oct 2022 to Jan 2023 including the 2022 peak, is about 20 points harder than cut 0), so one cut cannot be compared with another in absolute terms, only models within a cut. (4) Models should therefore be judged by the mean over the 4 cuts and by wins per cut.

## 2026-11-21 night: quick experiments on the 4 cuts (D33, blend with the median, seeds), user said go
- Code: `src/experiments_02.py` (phases 1 and 2), D33 option in `src/features.py` (`FEATURES_D33`, `mask_oh_before`; default unchanged) with extra tests in `tests/test_features.py` (no leakage with D33, regime mask works, default shape unchanged: all pass). Logs `logs/exp02_phase1.txt`, `logs/exp02_phase2.txt`; results `results_experiments.csv`, `results_experiments_seeds.csv`; forecasts `forecasts/*.npy`. About 6 minutes in total.
- **Score by cut (test shop, D26), cuts 0-3 and mean:**

  | variant | 0 | 1 | 2 | 3 | mean | gain over median13 (mean, cuts won) |
  |---|---|---|---|---|---|---|
  | median13 | -6.68 | -30.53 | -12.59 | -16.34 | -16.54 | |
  | base LightGBM (own, seed 0) | -4.83 | -28.45 | -13.35 | -15.44 | -15.52 | +1.02, 3/4 |
  | D33 (cover features + old regime masked) | -4.66 | -29.00 | -12.98 | -15.40 | -15.51 | +1.03, 3/4 |
  | base + median13 blend (w = 0.7 on LightGBM) | -4.96 | -28.79 | -12.53 | -15.46 | -15.44 | +1.10, 4/4 |
  | D33 + median13 blend (w = 0.7) | -4.79 | -29.15 | -12.37 | -15.43 | -15.44 | +1.10, 4/4 |
  | D33, 3-seed average | -4.68 | -28.58 | -13.05 | -15.44 | -15.44 | |
  | D33 3-seed average + median13 (w = 0.7) | -4.85 | -28.87 | -12.43 | -15.47 | **-15.41** | |

- **D33, paired over articles (D33 minus base, 95% interval):** cut 0 +0.18 [-0.41, +0.83]; cut 1 **-0.54** [-1.05, -0.13]; cut 2 **+0.38** [+0.12, +0.65]; cut 3 +0.05 [-0.24, +0.36]. Mean about 0: no consistent effect. Not recommended for adoption (D33 stays optional).
- **Blend weight** chosen on cuts 1-3 was 0.7 for both variants; the score curve over the weight is very flat (cuts 1-3 mean from -19.8 at w = 0 to -18.9 at w = 0.7). The blend makes the model win all 4 cuts against the median, but the gain is only about +0.08 on the mean.
- **Seed noise is as large as the effects:** on cut 0 the three seeds of D33 scored -4.66, -4.39 and -5.15 (spread 0.8 points); the 3-seed average -4.68.
- **Conclusion:** none of these three ideas moves the mean score by more than about 0.1 point (all variants between -15.4 and -15.5, median13 -16.5). They are within noise. Not worth a new submission upload. The gap to the leader (about 2 points on the live board) will not be closed by tuning this model; it needs a different source of information or a different kind of model (candidates: foundation models as blend members, hyperparameter search, combining several robust level estimators, the rich-column techniques of D18).

## 2026-11-21 night: optional upload file v2 (user asked whether a change invisible locally could still move the live score)
- Prepared `submissions/lgbm_d33_3seed_blend_v2.csv`: LightGBM with the D33 features, 3 seeds averaged (223 trees each, 1.1 x the cut-0 best iteration), trained on ALL test-shop weeks up to 2023-04-17, blended 70/30 with the median of the last 13 weeks. Passes `src/make_submission.py` validation (2,904 articles, w1..w13, non-negative).
- Difference from `lgbm_own_v1.csv` (live -7.6): mean absolute difference 0.55 units per article-week; correlation of per-article means 0.997; 14.5% of articles change by more than 0.5 unit in some week; mean forecast 7.48 -> 7.75.
- Expectation: the local 4-cut evidence says the true effect is about +0.1 point, within noise. Any live difference of about 2 points or less between v1 and v2 should be read as luck (seed spread alone was 0.8 points on one cut), and the leaderboard keeps each team's best score, so uploading many variants inflates the live number without improving the final w8-w13 score. Live score of v2: not yet reported.
- **Live score of `lgbm_d33_3seed_blend_v2.csv` (reported by the user): -8.4**, i.e. 0.8 points WORSE than `lgbm_own_v1.csv` (-7.6); median13 was -9.8. The leaderboard keeps the best, so v1 (-7.6) remains our score.
- Reading: the 30% median in v2 pulls it toward the median's score: 0.7 x (-7.6) + 0.3 x (-9.8) = -8.3, observed -8.4, so on the live weeks the blend behaved like a plain average of the two (no diversification gain), and the D33 features and extra seeds changed nothing visible. On local cut 0 the best weight was also about 0.9-1.0 on the LightGBM (-4.83 at w = 1.0 vs -4.96 at w = 0.7); the weight 0.7 chosen on cuts 1-3 was too low for cut-0-like windows. A 0.8-point difference is within the seed noise we measured, so this is not evidence that D33 or the seeds hurt. Decision: keep v1 as the reference model; do not blend with the median; no further tuning of this model tonight.
