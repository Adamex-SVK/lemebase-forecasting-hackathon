# Leaderboard uploads

One row per upload (see `docs/TEAM-RULES.md` section 6). The live score covers weeks 1-7 only; the final score uses weeks 8-13.

| Date | Who | File | Commit | Live score | Note |
|---|---|---|---|---|---|
| 2026-11-21 | owner | `lgbm_own_v1.csv` | 9081b5a (file created before the repo) | **-7.6** | global LightGBM, own arm, trained on all test-shop history |
| 2026-11-21 | owner | `dry_run_median13.csv` | 9081b5a | -9.8 | median of the last 13 weeks |
| 2026-11-21 | owner | `lgbm_d33_3seed_blend_v2.csv` | 9081b5a | -8.4 | D33 features, 3 seeds, 70/30 blend with the median: no better than v1 |
| 2026-11-21 | owner | `moirai2_small_v1.csv` | 54ad3e7 | -7.3 | Moirai 2.0-small alone, zero rule, rounded: about the same as the LightGBM (-7.6); the blend (-4.8) is much better, so the gain comes from combining models |
| 2026-11-21 | owner | `ensemble_top3_blend_v1.csv` | 54ad3e7 | **-4.8** | moirai-2.0-small + tirex + chronos-t5-small averaged, 8-week zero rule, 30% LightGBM, rounded. **Best so far, +2.8 over `lgbm_own_v1`.** |

Reference points on the board at the time: AutoARIMA -33.7, Naive -64.5, leader -5.4 (seen before the -4.8 upload; the board may have changed).

| 2026-11-21 | owner | `ensemble_d40_plus_chronos2ft_v1.csv` | it10/adamex-svk/x1-oos-mask | **-4.5** | D40 with the sibling-shop fine-tuned Chronos-2 as a 4th member. **Best live so far (+0.3 over D40)**; 13-cut check +0.14 (9/13 cuts) |
| 2026-11-21 | owner | `chronos2_ft_siblings_v1.csv` | it10/adamex-svk/x1-oos-mask | -4.9 | the sibling-shop fine-tuned Chronos-2 alone: about equal to D40 (-4.8) as a single model |
| 2026-11-21 | owner | `ensemble_d40_half_chronos2ft_own_v1.csv` | it10/adamex-svk/x1-oos-mask | -5.2 | D40 with half the foundation part replaced by the 5-seed own-history fine-tune: worse than D40 live, although it was the best variant on 13 overlapping cuts (+0.24); live differences of 0.3-0.7 are inside the noise of 7 weeks |
