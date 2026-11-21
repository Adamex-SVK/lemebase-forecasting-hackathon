# Leaderboard uploads

One row per upload (see `TEAM-RULES.md` section 6). The live score covers weeks 1-7 only; the final score uses weeks 8-13.

| Date | Who | File | Commit | Live score | Note |
|---|---|---|---|---|---|
| 2026-11-21 | owner | `lgbm_own_v1.csv` | 9081b5a (file created before the repo) | **-7.6** | global LightGBM, own arm, trained on all test-shop history |
| 2026-11-21 | owner | `dry_run_median13.csv` | 9081b5a | -9.8 | median of the last 13 weeks |
| 2026-11-21 | owner | `lgbm_d33_3seed_blend_v2.csv` | 9081b5a | -8.4 | D33 features, 3 seeds, 70/30 blend with the median: no better than v1 |
| 2026-11-21 | owner | `ensemble_top3_blend_v1.csv` | 54ad3e7 | **-4.8** | moirai-2.0-small + tirex + chronos-t5-small averaged, 8-week zero rule, 30% LightGBM, rounded. **Best so far, +2.8 over `lgbm_own_v1`.** |

Reference points on the board at the time: AutoARIMA -33.7, Naive -64.5, leader -5.4 (seen before the -4.8 upload; the board may have changed).

`moirai2_small_v1.csv` (Moirai 2.0-small alone) is built but its live score is not reported yet.
