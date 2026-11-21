# Iteration 09 / task 4 — temporal reconciliation (adamex-svk)

- commit: `dfc50f1`
- command: `.venv/bin/python src/improve/i09_temporal_reconcile.py`
- runtime: 64s
- Python: 3.13.13

## What was run

For cuts k=0..3 and aggregation g in {2,4}: aggregate the pre-origin weekly units into non-overlapping g-week blocks anchored so the last complete block ends at the origin (`M.U[:, o-g:o]`), train the existing scaled-L1 LightGBM (`train_gbm.PARAMS`, `num_threads=4`) on `block_units / block_scale`, with block origins from `features.make_origins(B, horizon=ceil(13/g), min_hist=13)` (default step=2), last 7 as validation and training origins `<= val_origins[0] - horizon` (purged; validation count is reduced only when that would leave no training origin, which happens for g=4). Early stopping 50 rounds, then refit on all origins with `round(best*1.1)+1` trees. Forecast the next ceil(13/g) blocks and disaggregate either equally or with the last pre-origin within-block week shares (uniform fallback on a zero block). Apply the zero rule (no sale in last 8 weeks) and clip; reconcile with the UNROUNDED D40: `F = (1-lambda)*D40 + lambda*F_agg`; round + clip; score.

Block features: lag_1..lag_13, mean2/4/8/13, median13, max13, zshare13, mean26, zshare26, weeks_since_sale (cap 104), series_age, log_scale, log_mean13, plus h (all scaled by the block scale where applicable).

## Results (wide; scores are (1-MASE)*100, higher is better)

| variant | g | lambda | disaggregation | cut0 | cut1 | cut2 | cut3 | mean | gain_vs_d40 | cuts_won | label | is_d40 | in_sample_optimum | loco_selected_for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g2_equal_lam0.25 | 2 | 0.25 | equal | -4.56 | -26.02 | -11.01 | -14.35 | -13.98 | -0.32 | 1 | search_grid | False | True | 0,1,2,3 |
| g2_equal_lam0.5 | 2 | 0.50 | equal | -5.45 | -26.69 | -11.25 | -14.56 | -14.49 | -0.82 | 0 | fixed_lambda0.5 | False | False |  |
| g2_equal_lam0.75 | 2 | 0.75 | equal | -6.60 | -28.04 | -11.63 | -15.04 | -15.33 | -1.66 | 0 | search_grid | False | False |  |
| g2_equal_lam1.0 | 2 | 1.00 | equal | -8.62 | -29.71 | -12.50 | -15.80 | -16.66 | -2.99 | 0 | search_grid | False | False |  |
| g2_shares_lam0.25 | 2 | 0.25 | shares | -4.68 | -26.22 | -11.40 | -14.71 | -14.25 | -0.58 | 0 | search_grid | False | False |  |
| g2_shares_lam0.5 | 2 | 0.50 | shares | -6.61 | -28.17 | -12.91 | -16.00 | -15.92 | -2.26 | 0 | fixed_lambda0.5 | False | False |  |
| g2_shares_lam0.75 | 2 | 0.75 | shares | -9.78 | -31.26 | -14.87 | -18.06 | -18.50 | -4.83 | 0 | search_grid | False | False |  |
| g2_shares_lam1.0 | 2 | 1.00 | shares | -13.40 | -34.79 | -17.26 | -20.35 | -21.45 | -7.79 | 0 | search_grid | False | False |  |
| g4_equal_lam0.25 | 4 | 0.25 | equal | -5.07 | -25.88 | -11.16 | -14.46 | -14.14 | -0.48 | 1 | search_grid | False | False |  |
| g4_equal_lam0.5 | 4 | 0.50 | equal | -6.75 | -27.01 | -11.86 | -15.27 | -15.23 | -1.56 | 0 | fixed_lambda0.5 | False | False |  |
| g4_equal_lam0.75 | 4 | 0.75 | equal | -7.04 | -28.93 | -12.80 | -15.59 | -16.09 | -2.42 | 0 | search_grid | False | False |  |
| g4_equal_lam1.0 | 4 | 1.00 | equal | -10.08 | -31.04 | -14.53 | -16.84 | -18.12 | -4.46 | 0 | search_grid | False | False |  |
| g4_shares_lam0.25 | 4 | 0.25 | shares | -5.23 | -27.53 | -11.92 | -15.25 | -14.98 | -1.31 | 0 | search_grid | False | False |  |
| g4_shares_lam0.5 | 4 | 0.50 | shares | -10.17 | -31.61 | -15.09 | -18.08 | -18.74 | -5.07 | 0 | fixed_lambda0.5 | False | False |  |
| g4_shares_lam0.75 | 4 | 0.75 | shares | -17.09 | -39.07 | -19.63 | -22.67 | -24.61 | -10.95 | 0 | search_grid | False | False |  |
| g4_shares_lam1.0 | 4 | 1.00 | shares | -22.76 | -45.70 | -24.76 | -27.55 | -30.19 | -16.53 | 0 | search_grid | False | False |  |
| D40 | 0 | 0.00 | none | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 | 0 | reference | True | False |  |
| LOCO_selected | per-cut | per-cut | per-cut | -4.56 | -26.02 | -11.01 | -14.35 | -13.98 | -0.32 | 1 | LOCO_honest_adoptable | False | False | 0,1,2,3 |
| IN_SAMPLE[g2_equal_lam0.25] | 2 | 0.25 | equal | -4.56 | -26.02 | -11.01 | -14.35 | -13.98 | -0.32 | 1 | in_sample_non_adoptable | False | True | 0,1,2,3 |

## Fixed lambda=0.5 (evaluated first)

| variant | g | method | cut0 | cut1 | cut2 | cut3 | mean | gain | won |
|---|---|---|---|---|---|---|---|---|---|
| g2_equal_lam0.5 | 2 | equal | -5.45 | -26.69 | -11.25 | -14.56 | -14.49 | -0.82 | 0 |
| g4_equal_lam0.5 | 4 | equal | -6.75 | -27.01 | -11.86 | -15.27 | -15.23 | -1.56 | 0 |
| g2_shares_lam0.5 | 2 | shares | -6.61 | -28.17 | -12.91 | -16.00 | -15.92 | -2.26 | 0 |
| g4_shares_lam0.5 | 4 | shares | -10.17 | -31.61 | -15.09 | -18.08 | -18.74 | -5.07 | 0 |

## LOCO selection (chosen on the other 3 cuts)

| target_cut | selected | score | d40 | gain |
|---|---|---|---|---|
| 0 | g2_equal_lam0.25 | -4.56 | -3.72 | -0.84 |
| 1 | g2_equal_lam0.25 | -26.02 | -25.91 | -0.11 |
| 2 | g2_equal_lam0.25 | -11.01 | -11.01 | 0.00 |
| 3 | g2_equal_lam0.25 | -14.35 | -14.02 | -0.33 |

## Paired bootstrap vs D40 — honest LOCO (95%, 1000 resamples, seed 0, score points)

| cut | selected | n | gain | lo95 | hi95 |
|---|---|---|---|---|---|
| 0 | g2_equal_lam0.25 | 2839 | -0.84 | -1.56 | -0.17 |
| 1 | g2_equal_lam0.25 | 2762 | -0.11 | -0.43 | 0.26 |
| 2 | g2_equal_lam0.25 | 2680 | 0.00 | -0.55 | 0.57 |
| 3 | g2_equal_lam0.25 | 2599 | -0.33 | -1.51 | 0.96 |
| pooled | per-cut | 10880 | -0.32 | -0.73 | 0.05 |

Fixed best lambda=0.5 variant (`g2_equal_lam0.5`) bootstrap:

| variant | cut | gain | lo95 | hi95 |
|---|---|---|---|---|
| g2_equal_lam0.5 | 0 | -1.73 | -3.12 | -0.45 |
| g2_equal_lam0.5 | 1 | -0.78 | -1.50 | -0.08 |
| g2_equal_lam0.5 | 2 | -0.24 | -1.16 | 0.64 |
| g2_equal_lam0.5 | 3 | -0.53 | -1.87 | 1.09 |

## Per-scale-quartile effect (LOCO, averaged over cuts; gain_score_points = score-point change vs D40, positive = candidate better)

| group | n | d40_mase | cand_mase | gain_score_points |
|---|---|---|---|---|
| Q1 smallest | 680.25 | 1.79 | 1.79 | -0.41 |
| Q2 | 680.00 | 1.06 | 1.06 | 0.17 |
| Q3 | 679.50 | 0.89 | 0.90 | -0.56 |
| Q4 largest | 680.25 | 0.81 | 0.81 | -0.46 |

Per cut:

| cut | group | n | d40_mase | cand_mase | gain_score_points |
|---|---|---|---|---|---|
| 0 | Q1 smallest | 710 | 1.54 | 1.55 | -1.45 |
| 0 | Q2 | 710 | 0.86 | 0.86 | -0.02 |
| 0 | Q3 | 709 | 0.89 | 0.90 | -0.50 |
| 0 | Q4 largest | 710 | 0.86 | 0.88 | -1.36 |
| 1 | Q1 smallest | 691 | 1.69 | 1.69 | 0.24 |
| 1 | Q2 | 690 | 1.42 | 1.42 | 0.36 |
| 1 | Q3 | 690 | 1.06 | 1.07 | -0.65 |
| 1 | Q4 largest | 691 | 0.86 | 0.86 | -0.38 |
| 2 | Q1 smallest | 670 | 1.86 | 1.86 | -0.21 |
| 2 | Q2 | 670 | 0.97 | 0.97 | 0.56 |
| 2 | Q3 | 670 | 0.81 | 0.82 | -0.42 |
| 2 | Q4 largest | 670 | 0.79 | 0.79 | 0.08 |
| 3 | Q1 smallest | 650 | 2.06 | 2.07 | -0.22 |
| 3 | Q2 | 650 | 0.98 | 0.98 | -0.23 |
| 3 | Q3 | 649 | 0.80 | 0.81 | -0.68 |
| 3 | Q4 largest | 650 | 0.72 | 0.72 | -0.19 |

## Per-activity effect (sold in last 13 weeks vs not; LOCO, averaged over cuts; gain_score_points positive = candidate better)

| group | n | d40_mase | cand_mase | gain_score_points |
|---|---|---|---|---|
| active_13w | 1958.25 | 1.43 | 1.44 | -0.44 |
| inactive_13w | 761.75 | 0.37 | 0.37 | 0.00 |

Per cut:

| cut | group | n | d40_mase | cand_mase | gain_score_points |
|---|---|---|---|---|---|
| 0 | active_13w | 2006 | 1.32 | 1.33 | -1.18 |
| 0 | inactive_13w | 833 | 0.35 | 0.35 | -0.00 |
| 1 | active_13w | 1992 | 1.59 | 1.59 | -0.15 |
| 1 | inactive_13w | 770 | 0.40 | 0.40 | -0.00 |
| 2 | active_13w | 1937 | 1.41 | 1.41 | 0.00 |
| 2 | inactive_13w | 743 | 0.33 | 0.33 | -0.00 |
| 3 | active_13w | 1898 | 1.41 | 1.42 | -0.45 |
| 3 | inactive_13w | 701 | 0.40 | 0.40 | -0.00 |

## Verdict

**REJECT** — honest LOCO mean gain -0.32 score points, cuts won 1/4 (bar: mean >= +0.5 and >= 3/4). In-sample optimum `g2_equal_lam0.25` (mean -13.98) is reported for reference only and is non-adoptable.

## Notes, warnings and leakage risks

- The final block forecast is trained only on pre-origin blocks; the holdout is never read, so there is no target leakage. Reconciliation mixes only pre-computed D40 and aggregate forecasts.
- For g=4 the block history is short (`B = 22..32` blocks). With `min_hist=13` and the default step=2 in `make_origins`, the purged split often has very few training origins; the validation count is reduced (to 1) so that at least one purged training origin exists. This makes g=4 weak late in the series and is reported honestly rather than tuned away.
- `mean2` is an explicit block feature (the weekly `features_at` has no mean2); the rest of the block feature set mirrors `features_at`.
- D40 is unchanged: the reference recipe command still prints mean -13.67.
- Block-fit early-stopping iterations: cut0/g2=268, cut0/g4=113, cut1/g2=169, cut1/g4=123, cut2/g2=147, cut2/g4=82, cut3/g2=177, cut3/g4=78
