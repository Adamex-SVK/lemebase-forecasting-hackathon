# NHITS baseline (own arm, untuned) - adamex-svk, 2026-11-21

Purpose: the first neural result, needed before any hyperparameter search for DL models makes sense.
Status: run and scored; **not adopted** (no variant clears the bar). D41 is still a proposal; this is evidence for it.

## What was run
- `src/neural/run_nf.py --arm own --cuts 0 1 2 3 --models NHITS --seeds 0 1 2 --accelerator cpu` on the **laptop CPU**
  (no GPU needed): defaults from the plan, input_size 52, max_steps 1000, lr 1e-3, 1024 windows per batch,
  MAE on the D26-scaled target, early-stopping patience 300 on the last 13 weeks before the cut.
- 12 fits, 84-135 s each, about 22 min in all. **Every fit stopped at max_steps = 1000; early stopping
  never triggered**, so the model is probably undertrained (first thing to change in a search).
- Scored with `src/neural/score_nf.py` (team scorer, weights chosen on cuts 1-3) and with the new
  `src/neural/eval_nf_d40.py` (D40 code path from `src/meta/ensemble_recipe.py`, **fixed weights, no tuning**;
  reproduces D40 = -13.67). Results: `results_adamex-svk_nhits-baseline.csv`, `results_kilo_neural.csv`.

## Results (D26, 4 standard cuts, rounded + zero rule)
| Variant | cut0 | cut1 | cut2 | cut3 | mean | vs D40 (cuts won) |
|---|---|---|---|---|---|---|
| D40 (0.3 LightGBM + 0.7 mean of 3 FMs) | -3.72 | -25.91 | -11.01 | -14.02 | **-13.67** | - |
| LightGBM alone | -4.03 | -28.06 | -13.08 | -15.31 | -15.12 | -1.45 (0/4) |
| NHITS alone, 3-seed mean | -3.77 | -29.63 | -12.67 | -15.14 | -15.30 | -1.64 (0/4) |
| NHITS single seeds (0/1/2) | | | | | -15.68 / -15.13 / -15.54 | |
| NHITS as 4th FM member | -3.52 | -26.32 | -11.52 | -14.13 | -13.87 | -0.20 (1/4) |
| NHITS replaces LightGBM in the 0.3 slot | -3.39 | -26.16 | -11.20 | -14.03 | -13.70 | -0.03 (1/4) |
| 0.15 LightGBM + 0.15 NHITS + 0.7 FMs | -3.47 | -25.99 | -11.07 | -13.98 | -13.62 | +0.04 (2/4) |
| equal mean of 5 (LGBM, 3 FMs, NHITS) | -3.44 | -26.08 | -11.26 | -14.19 | -13.74 | -0.08 (1/4) |

Best blend vs D40, paired per-article 95% interval: cut0 +0.26 [+0.02, +0.48], cut1 -0.08, cut2 -0.06, cut3 +0.04
(the last three intervals include 0).

## Reading
- Untuned NHITS is **about as good as the LightGBM** (-15.30 vs -15.12; better on 3 of 4 cuts, loses
  1.6 points on cut 1). Seed spread on the mean is 0.55 points, and the 3-seed average beats 2 of 3 single seeds.
- In the blend it is **interchangeable with the LightGBM, not additive**: every combination stays within
  ±0.2 of D40. It carries roughly the same information as the tree model.
- Implication for a hyperparameter search: the bar is +0.5 over D40, not over the LightGBM. A search only
  pays off if it makes NHITS clearly better than the LightGBM alone (cut 1 is the weak spot), or clearly
  different from it. A modest tuning gain is likely to vanish in the blend.

## If we do a search next (suggested, not decided)
- Tune on the overlapping cuts (`--cut-weeks N`, `forecasts_cuts/`) that do not overlap the 4 scored cuts,
  or leave-one-cut-out; score the chosen config once on the 4 standard cuts.
- Small grid first, each knob judged on the 4-cut mean: max_steps {1000, 3000}, input_size {52, 104},
  lr {1e-3, 3e-4}, then the pooled arm (shops A-D, D27). About 2 min per fit on the laptop CPU, 12 fits per config.
- Alternatives to raw HPO that may matter more: the pooled arm, NBEATS / TFT with stock features (plan 3.2, 3.3).
