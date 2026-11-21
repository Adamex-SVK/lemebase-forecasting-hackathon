# Iteration 02: robust evaluation and the next model improvements

**Status (2026-11-21, night):** DRAFT. Started early: the rolling-origin evaluation (D32) and the F6 investigation, see `log.md`. The rest needs the user's go after step 1.3 is analysed.

## Hypotheses
1. A score gain of a few points over the median is only believable if it holds across several cuts (D32).
2. Making `on_hand` comparable across the test shop's regime change (D33) helps the model on the active articles.
3. A blend of the LightGBM with the median of the last 13 weeks beats either alone.

## Steps
| # | Step | Needs |
|---|---|---|
| 2.1 | Rolling-origin evaluation of baselines and the own-arm LightGBM (running) | D32 (rec. yes) |
| 2.2 | F6 (done: no calendar features needed) | |
| 2.3 | Features: `on_hand` relative to recent sales and masking of the old regime; evaluate on the 4 cuts against the unchanged features | D33 |
| 2.4 | Blends and variants: LightGBM + median13 (weights chosen on cuts 1-3, tested on cut 0), seeds, per-horizon variants | |
| 2.5 | Review with the user | |

## Not in this iteration
Foundation models (iteration 03), rich-column techniques (iteration 04).
