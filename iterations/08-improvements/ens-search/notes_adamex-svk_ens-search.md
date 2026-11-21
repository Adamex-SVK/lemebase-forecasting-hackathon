# Ensemble approaches beyond T1-T8, and the full TabPFN verdict (2026-11-21, 16:30)

Author: main session (`it03/adamex-svk/foundation-models`). Code `src/improve/ens_search.py`, results `results_ens_search.csv`.
Yardstick: the 4 standard cuts, D26, every chosen parameter or weight picked **leave-one-cut-out** (LOCO: choose on three cuts, score the fourth). Reference D40 = -3.72 / -25.91 / -11.01 / -14.02, mean **-13.67** (reproduced; the fast scorer matches the official `score_F` on cut 0).

## Pool (19 members, all 4 cuts available)
14 zero-shot models (zero rule applied to each), LightGBM `base` and `d33`, CatBoost (seed average; other session's forecasts), median of the last 13 weeks, zero. Every blend is rounded and clipped.

## Results (honest LOCO numbers)
| approach | cut 0 | cut 1 | cut 2 | cut 3 | mean | gain vs D40 | cuts won | verdict |
|---|---|---|---|---|---|---|---|---|
| greedy ensemble selection with replacement, from empty (maximises the rounded score) | -3.83 | -26.14 | -12.52 | -14.20 | -14.17 | -0.51 | 0/4 | reject |
| greedy selection warm-started at D40 | -3.89 | -25.73 | -12.27 | -13.98 | -13.97 | -0.30 | 2/4 | reject |
| optimised non-negative global weights (SLSQP, unrounded objective) | -4.58 | -27.10 | -11.82 | -14.47 | -14.49 | -0.83 | 0/4 | reject |
| zero voting: D40, 0 where >= half of the 14 models forecast < 0.5 | -3.25 | -25.85 | -11.08 | -14.28 | -13.61 | +0.05 | 2/4 | reject |
| zero voting with D40's 3 models | -3.26 | -26.02 | -10.98 | -14.23 | -13.62 | +0.05 | 2/4 | reject |
| one LightGBM weight per sales segment (slow / mid / fast) | -3.66 | -25.86 | -12.18 | -14.00 | -13.93 | -0.26 | 3/4 | reject |

Reading: **every approach that learns weights from three cuts does worse on the fourth than the fixed D40 weights** (classic overfitting of the combiner with only four, very different windows). The learned weights jump from fold to fold (e.g. `lgbm-d33` 0 to 0.5), another sign that the data cannot pin them down. Zero voting is neutral.

## TabPFN-TS, all 4 cuts (`ensemble_recipe.py evaluate`)
All five jobs finished (cut 3 at 15:32; about 13,500-14,500 s of forecasting each). Files copied to `iterations/03-foundation-model/forecasts/tabpfn-ts-v2__test__cut{0..3}__padded.parquet`.
| set | cut 0 | cut 1 | cut 2 | cut 3 | mean | gain | won |
|---|---|---|---|---|---|---|---|
| D40 + TabPFN (4 members) | -3.65 | -26.23 | -11.25 | -14.03 | -13.79 | -0.13 | 1/4 |
| TabPFN replacing chronos-t5-small (uploaded as `ensemble_tabpfn_swap_v1`, **live -5.2**) | -3.45 | -26.83 | -11.37 | -14.47 | -14.03 | -0.37 | 1/4 |
| TabPFN replacing tirex | -3.85 | -26.83 | -11.68 | -14.16 | -14.13 | -0.46 | 0/4 |
| TabPFN replacing moirai-2.0-small | -3.82 | -26.14 | -11.10 | -14.18 | -13.81 | -0.14 | 0/4 |
**Verdict: reject TabPFN (D36).** The local yardstick and the live board agree (swap: -0.37 locally, -5.2 vs -4.8 live).

## Conclusion
Together with T1-T8 (all rejected) about fifteen independent ways of changing the blend have now failed the bar. D40 sits at a robust optimum for the information we have; further gains need **new information** (a genuinely different strong member: fine-tuning T9, new model families T10), not new ways of weighting the same forecasts.
