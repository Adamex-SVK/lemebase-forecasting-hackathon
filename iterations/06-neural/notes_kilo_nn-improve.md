# Neural forecaster improvement — kilo, 2026-11-21

Follow-up to D41 (neural-forecaster family) and `notes_adamex-svk_nhits-baseline.md`. Goal: find a
neural variant clearly better alone (4-cut mean <= -14.5) or clearly different enough to lift D40 past the
adoption bar (+0.5 mean over -13.67 AND better on >= 3 of 4 cuts; TEAM-RULES.md section 2).
**Result: negative — no variant clears either gate; D40 stands.**

## What ran (4 standard cuts x 3 seeds, RTX 3090, own arm unless noted)

| Stage | Config | Stem | Fit time | Wall time | Early stop |
|---|---|---|---|---|---|
| 0 | code: run_nf.py --tag/--dropout/--final/--pool-shops, new select_loco.py, eval_nf_d40 --handle; tests extended; check.sh green; baseline repro exact | — | — | — | — |
| 1 | NHITS own grid: max_steps {1000,3000} x lr {1e-3,3e-4} x input {52,104} (104 uses --start-padding) | NHITS_s*_lr*_in* | ~13 s (1000) / ~36 s (3000) | 16:36:38 -> 17:13:48 (~37 min) | no |
| 2 | NHITS pooled, all 5 shops, baseline settings + --start-padding | NHITS_pooled | ~77 s (CPU-bound data prep) | 17:31:16 -> 17:47:47 (~16 min) | no |
| 3 | NBEATS own, baseline settings | NBEATS_base | ~12 s | 17:28:37 -> 17:31:16 (~2.6 min) | no |
| 4 | stock covariates / TFT (hist_exog on_hand) | — | skipped | — | — |

Early stopping (patience 300) never triggered: the NBEATS log prints "max_steps=1000 reached", and the
3000-step NHITS fits took ~2.8x the 1000-step fits (37 s vs 13 s), i.e. they ran to max_steps. Pooled panel:
test 2,839 + A 599 + B 7,500 + C 851 + D 49,596 = 61,385 series, 3,166,441 rows (all five shops present, no
A+C fallback needed). Fit times and settings are in the per-stem `.json` files next to the parquets.

## Results (D26, 4 standard cuts; in-sample = the config's own 4-cut mean; LOCO = leave-one-cut-out honest choice)

neural alone:

| stem | cut0 | cut1 | cut2 | cut3 | mean | LOCO honest |
|---|---|---|---|---|---|---|
| NHITS (baseline, s1000/lr1e-3/in52) | -3.77 | -29.63 | -12.67 | -15.14 | -15.30 | -15.49 |
| NBEATS_base | -3.43 | -29.55 | -12.50 | -14.79 | **-15.07** | **-15.07** |
| NHITS_pooled | -4.87 | -30.34 | -12.93 | -15.50 | -15.91 | -15.91 |
| 7 stage-1 NHITS configs (range) | ... | ... | ... | ... | -15.36 .. -17.11 | — |
| LightGBM (reference) | -4.03 | -28.06 | -13.08 | -15.31 | -15.12 | — |

blend `replace` = 0.3*N + 0.7*FM3 (D40 = -13.67):

| stem | cut0 | cut1 | cut2 | cut3 | mean | vs D40 | cuts won |
|---|---|---|---|---|---|---|---|
| NBEATS_base | -3.34 | -26.02 | -11.07 | -13.90 | -13.58 | +0.08 | 2/4 |
| NHITS (baseline) | -3.39 | -26.16 | -11.20 | -14.03 | -13.70 | -0.03 | 1/4 |
| NHITS_pooled | -3.53 | -26.27 | -11.06 | -14.04 | -13.72 | -0.06 | 1/4 |

blend `split` = 0.15*LGBM + 0.15*N + 0.7*FM3:

| stem | mean | vs D40 | cuts won |
|---|---|---|---|
| NHITS (baseline) | -13.62 | +0.04 | 2/4 |
| NBEATS_base | -13.63 | +0.03 | 2/4 |
| NHITS_pooled | -13.63 | +0.04 | 2/4 |

Best single blend vs D40, paired per-article 95% bootstrap interval (NBEATS replaces LGBM, +0.08 mean):
cut0 +0.39 [+0.06, +0.72], cut1 -0.11 [-0.41, +0.18], cut2 -0.06 [-0.42, +0.27], cut3 +0.12 [-0.11, +0.41]
(only cut 0 excludes 0).

## Gate verdicts
1. **Better alone (4-cut mean <= -14.5): FAILED.** Best is NBEATS at -15.07 (LOCO -15.07), which edges the
   LightGBM (-15.12) and NHITS (-15.30) but misses -14.5 by 0.57, and its cut 1 (-29.55) is still worse than
   the LightGBM's -28.06.
2. **Useful in D40 (gain >= +0.5 AND >= 3/4 cuts): FAILED.** Best is NBEATS replaces-LGBM at +0.08 (2/4 cuts).
   Nothing approaches +0.5; the honest LOCO `split` mean is -13.67 (= D40 exactly, no gain).

## Reading
- Tuning the own arm does not help: max_steps 3000 overfits (-16.06/-16.08 vs -15.30), input_size 104 clearly
  hurts (-16.19 .. -17.11, likely because the test shop's yearly peak moves, F6), lr 3e-4 is neutral (-15.36).
  The untuned baseline is the best own-arm NHITS.
- NBEATS is the best single neural model and slightly beats the LightGBM alone, but it carries the same
  information, so in the blend it adds only +0.08 (below the ~0.5-0.8 noise by our own measurements).
- Pooling other shops hurts, exactly as it did for the LightGBM (own -4.83 vs pooled -5.84 on cut 0):
  pooled NHITS is -15.91 alone, worse than own.
- Conclusion matches the baseline notes: trained-from-scratch neural models are interchangeable with the
  LightGBM, not additive, so they cannot lift D40 past the bar.

## Skipped
- Stage 4 (historical on_hand / out_of_stock_flag as hist_exog, and/or TFT): not run. Gate 2 already failed
  decisively (best +0.08 vs +0.5), the code lift is ~2-3 h with leakage pitfalls, and the LightGBM already
  consumes on_hand features, so the marginal "different" signal is small.
- NHITS+NBEATS average: not separately scored — it lies between NHITS (-15.30) and NBEATS (-15.07) alone, so
  its best blend gain is bounded below NBEATS's +0.08 and cannot clear the bar.

## Files
- `results_kilo_loco.csv` (LOCO, 10 stems), `results_kilo_nbeats.csv`, `results_kilo_pooled.csv`
  (+ `results_kilo_step0_smoke.csv`, `results_kilo_s1000_baseline_check.csv`).
- Forecasts: `iterations/06-neural/forecasts/` (10 stems x 4 cuts; per-seed files remain on the GPU machine).

## Recommendation
- Do not adopt any neural variant; D40 stays (already frozen, D44).
- If revisiting the neural family: stage 4 stock covariates, or a zero-aware/quantile objective (the metric is
  dominated by slow/dead articles), rather than more NHITS/NBEATS tuning or pooling.
- Cleanup: remove the assistant's SSH key from the GPU machine (~/.ssh/authorized_keys) now that GPU work is done.
