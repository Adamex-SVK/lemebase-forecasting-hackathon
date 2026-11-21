# Rich columns and feature dropout: result and closure (iteration 04, steps 4.3 and after)

Date: 2026-11-21. Handle: `adamex-svk`. Branch: `it04/adamex-svk/rich-columns`.

## Verdict
**Not adopted; the rich-column path is closed (D43, decided by the user, 2026-11-21).** No rich arm beats our current best (the own-data LightGBM): the best rich arm is 0.80 points worse on the mean over the 4 cuts and wins 0 of 4 cuts. The rich columns recover a little of the loss caused by pooling the donor shops (+0.14 to +0.28), which is below the 0.5-point bar and about the size of seed noise.

## What was run
- Code: rich weekly tables for Sets B, C, D (`src/rich.py`, definitions in `notes_adamex-svk_rich_schema.md`), rich features and per-window dropout (`src/features.py`, `src/train_gbm.py`), runner `src/run_rich.py`. All four columns are NaN for the test shop at forecast time (`rich_missing = 1`).
- Arms (all scored on the 4 rolling cuts of the test shop, D26 score, 2 seeds each; seed sets both LightGBM and the window rng):
  - `own`: test-shop windows only (current best).
  - `pooled_norich`: test shop + donors A-D, default features (reference for the pooled route).
  - `pooled_rich_nodrop`, `pooled_rich_drop50`, `pooled_rich_drop80`: pooled with the rich features, donor windows masked on 0 / 50 / 80% of windows.
- Pooled settings: donor shops use all their weeks (D27), origin step 4 (D31), weighted sample for D (D28), equal shop weights (D30), fixed 300 trees; test-shop windows origin step 2.
- Results: `results_adamex-svk_reference.csv`, `results_adamex-svk_dropout.csv`; per-article MASE in `detail/`; analysis `analyse_rich.py`, full output `analysis_dropout_output.txt`. Reference notes: `notes_adamex-svk_reference.md`.

## Scores (D26, higher is better; mean over 2 seeds)
| arm | cut 0 | cut 1 | cut 2 | cut 3 | mean |
|---|---|---|---|---|---|
| own (current best) | -4.57 | -28.80 | -13.09 | -15.50 | **-15.49** |
| pooled_norich | -5.73 | -30.72 | -13.65 | -16.16 | -16.56 |
| pooled_rich_nodrop | -5.37 | -30.03 | -13.18 | -16.56 | -16.29 |
| pooled_rich_drop50 | -5.37 | -30.86 | -13.18 | -16.30 | -16.43 |
| pooled_rich_drop80 | -5.62 | -30.50 | -13.19 | -16.05 | -16.34 |

Seed spread (max minus min over the 2 seeds) per arm and cut: 0.0 to 0.5.

## Comparisons (mean over cuts, cuts won, paired per-article 95% bootstrap interval, 1,000 resamples)
| arm | vs `pooled_norich`: mean diff, cuts won | vs `own`: mean diff, cuts won |
|---|---|---|
| nodrop | +0.28, 3/4 (interval excludes 0 on cuts 0-2 positive, cut 3 negative) | -0.80, 0/4 (excludes 0 negative on cuts 1 and 3) |
| drop50 | +0.14, 2/4 | -0.94, 0/4 (excludes 0 negative on cuts 1 and 3) |
| drop80 | +0.22, 4/4 (differences +0.11 to +0.46) | -0.85, 0/4 (excludes 0 negative on cuts 1 and 3) |

Per-cut intervals are in `analysis_dropout_output.txt`. Against `own`, no interval excludes 0 on the positive side anywhere; against `pooled_norich` the gains are small (at most +0.69 on one cut).

## What it tells us
1. **Pooling the donor shops hurts the test shop here.** Without rich features the pooled model is 1.07 points worse than own data and loses all 4 cuts (see the reference notes).
2. **The rich features give back a quarter of that loss at best (+0.14 to +0.28).** That is below the adoption bar (+0.5 over a reference) and within the seed spread. More masking does not help: the three rich arms are within 0.14 of each other.
3. **Adoption bar not met.** To improve the current best, a rich arm had to beat `own` by at least 0.5 on the mean and win at least 3 of 4 cuts; the best rich arm is 0.80 below and wins none.
4. **Why this is plausible:** the test shop never has the rich columns at forecast time, so the model can only use them to learn better dynamics from the donor shops; the donor shops differ from the test shop (history length, zero share, stock regime), and that mismatch costs more than the rich columns repair.

## What was not done
- Feature importance of the rich features was not computed (the verdict does not depend on it).
- Cleaning (promo and markdown weeks), auxiliary tasks and teacher-student were not run: they also work through the pooled donor data, whose pooled baseline is already worse than own data, so the expected ceiling is below the bar. If someone wants to try them anyway, the rich tables and features are ready.
- A blend of a pooled-with-rich model into the own-data model was not tested; blends with diverse members helped elsewhere, but it was not part of the agreed plan.
- Not verified: behaviour with more than 2 seeds or with other pooled settings (tree counts, shop weights).

## Reproduce
```bash
bash scripts/check.sh
.venv/bin/python src/run_rich.py own,pooled_norich --seeds 0,1 --topic reference
.venv/bin/python src/run_rich.py pooled_rich_nodrop,pooled_rich_drop50,pooled_rich_drop80 --seeds 0,1 --topic dropout
.venv/bin/python iterations/04-rich-columns/analyse_rich.py --results <reference + dropout results combined> --arms pooled_rich_nodrop,pooled_rich_drop50,pooled_rich_drop80 --refs pooled_norich,own --seeds 0,1
```
(`analyse_rich.py` reads one results file; concatenate the reference and dropout CSVs first.)
