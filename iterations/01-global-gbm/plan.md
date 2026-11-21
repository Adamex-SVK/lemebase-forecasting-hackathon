# Iteration 01: one global LightGBM, own data vs pooled

**Status (2026-11-21):** decisions D27-D29 are settled. Steps 1.1 (builder) and 1.2 (own arm on the test shop: -4.83 vs median13 -6.7, gain not statistically clear) are done, see `log.md`. Next: step 1.3 (all shops, all three arms), waiting for the user's go and for D30 (shop weighting in the pooled arm).

## Hypothesis
A single global LightGBM, trained with an absolute-error loss on windows from many shops, beats the best simple baseline (the median of the last 13 weeks) and AutoARIMA on the test-shop local cut, and we can show for each shop whether borrowing from the other shops helps.

Note: LightGBM is gradient boosting, not a random forest. Both are tree ensembles, but gradient boosting supports the absolute-error (MAE) loss natively and fast, which D20 asks for. A random forest could be added later as a comparison.

## Reference to beat (local cuts, D26 convention, from iteration 00)
Median of last 13 weeks: test -6.7, A 17.4, B 20.0, C 57.3, D 68.8. AutoARIMA: test -28.8, A 8.7, B 20.2, C 40.2, D 58.6. Zero forecast: C 58.8, D 69.1. Full table in `iterations/00-baselines/log.md` and `results.csv`.

## Design (to be confirmed)
- **Target and loss (D29):** predict `units(t+h) / scale`, where `scale` is the series' MASE scale at the forecast origin (mean absolute week-to-week change of its zero-padded history). With an L1 objective this is exactly the per-series MASE that the leaderboard uses, so training optimises the metric directly (D11, D20). Windows whose history scale is 0 are skipped for training.
- **One model for all 13 horizons:** `h` (1..13) is a feature, so the model has one row per (series, origin, h).
- **Inputs (only things the test shop has, plus `h`):** the last 26 weeks of units (scaled), rolling mean and median over 4, 8, 13, 26 and 52 weeks, share of zero weeks in the last 13 and 26 weeks, weeks since the last sale, series age (weeks since the first sale), last `on_hand`, `on_hand` missing flag, out-of-stock flag, and the shop's recent total trend. No calendar or week-of-year feature (F6).
- **Windows:** origins every 2 weeks, at least 13 weeks of history, all 13 target weeks inside the training period. Set D has 50k series, so windows are sampled (D28).
- **Arms per shop X in {A, B, C, D, test} (D16, D27):**
  1. `own`: trained on X's training weeks only.
  2. `pooled`: X's training weeks + all weeks of the other shops.
  3. `others_only`: trained on the other shops only, run on X (zero-shot transfer).
- **Evaluation:** the local cuts (13 weeks) with `src/validate.py` (D26), all 13 weeks. D reported on its scorable series only.
- **Extra experiment (D20):** the model alone vs the model plus an explicit zero rule (no sale in the last 8 weeks -> 0).
- **Laptop:** LightGBM on CPU is enough; expected tens of minutes per arm after sampling.

## Steps (each ends in a checkpoint)
| # | Step | Needs |
|---|---|---|
| 1.1 | Feature and window builder (`src/features.py`), with a leakage check (no input uses weeks at or after the origin) and unit tests | D27-D29 |
| 1.2 | Train the `own` arm for the test shop first and compare with the baselines | |
| 1.3 | All shops and all three arms; fill `results.csv` | |
| 1.4 | Zero-rule experiment; feature importance; write `changes.md` against iteration 00 | |
| 1.5 | Review with the user: what to change for iteration 02 | |

## Not in this iteration
Calendar features (F6), foundation models (GPU, iteration 03), the rich-column techniques (D18, iteration 04), external data.
