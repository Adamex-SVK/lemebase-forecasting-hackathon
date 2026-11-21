# Iteration 00 log

Append-only. One entry per step: date, what was done, what we saw (numbers), decisions (ID, options, who chose, why), output files.

## 2026-11-21: planning (before any run)
- Profiled all raw files read-only in the scratchpad (not part of the pipeline): shapes, series counts, zero rates, date ranges, `is_return` meaning in C, Set D cold-start series, test-shop statistics.
- Findings and open questions are in `decisions-to-make.md` sections 2 and 3.
- Plan written in `plan.md`. Nothing has been run for this iteration.

## 2026-11-21: step 0.1 environment check (user approved the plan)
- Python 3.10.20 (pyenv), Apple M1, 16 GB, 8 cores, 21 GB free disk (90% full).
- Installed: pandas 2.3.3, pyarrow 25.0.1, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, matplotlib 3.10.9, torch 2.14.1 (MPS available), transformers 5.20.2, pytest, jupyter.
- Not installed: lightgbm, xgboost, catboost, statsforecast, statsmodels, sktime, darts, neuralforecast, chronos-forecasting, timesfm, polars, duckdb, numba.
- Nothing was installed at this point. Open question for the user: what to install (see checkpoint).

## 2026-11-21: step 0.1b install into a project venv (user approved)
- First try: `.venv` on Python 3.10.20 with scipy 1.15.3 (newest available for 3.10). `import scipy.sparse.linalg` fails with a `dlopen` error on `_spropack` ("zero-fill section ... offset field is not zero"), on macOS 27.0.1. The same error occurs with the global 3.10 scipy, so it is a scipy-wheel / macOS incompatibility, not an install error. That venv was deleted.
- Fix: new `.venv` on Python 3.13.13 (already in pyenv, nothing new installed system-wide). It gets scipy 1.18.1, which loads.
- `.venv` (560 MB): pandas 2.3.3, pyarrow 26.0.1, numpy 2.5.5, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, statsforecast 2.1.1, statsmodels 0.15.0, lightgbm 4.7.0. Pinned in `requirements.txt`.
- Smoke test passed: statsforecast AutoARIMA + Naive forecast on a toy weekly series, and a LightGBM `objective='l1'` fit.
- Side note: the global Python 3.10 cannot import scipy.sparse.linalg on this machine, so use `.venv/bin/python` for all project work.

## 2026-11-21: step 0.2 verify raw facts (read-only)
- Done by Claude directly, not by a separate agent (read-only, small, context already loaded); no brief was written. Scripts are scratch files, not part of the pipeline; raw data untouched.
- **D1 (holdout inside files): yes.** Files end at the brief's holdout end in every set (A 2024-06-03; B holdout ends 2025-10-20, data to 11-01; C and D holdout ends 2026-04-13, data to 04-20 / 04-24). A `stock_snapshot` (2024-04-08) lies inside A's holdout window: do not use as a feature.
- **D2 (partial weeks): found at both ends.** Last: B 6/7 days (10-27), C 1 day (04-20), D 5 days (04-20..24). First: B starts Wed, C and D start Sun (1 day). Complete-week counts: B 42, C 120, D 46 (my earlier 44 / 122 / 48 included partial weeks).
- **D4 (codes): inferred.** C `is_return=-1` = no-sale day row (462,376 rows, all zero units). `stock_status` C = reduced-price status, R = regular; all promo/markdown sales happen under C. Not needed for the core model.
- **D21 (weekly on_hand): test yes, C yes (week-end value), D yes with gaps (71% of sales-weeks), B no (events only REPLENISH / STOCKOUT; replenished - sold = end stock in 0.4% of series), A flag only.**
- Open: decision gate 1 (D1-D7, D21, D24) with the user. Details and recommendations are in `decisions-to-make.md`.

## 2026-11-21: step 0.3 decision gate 1 (cleared by the user)
- Settled as recommended: D1, D2 (drop first and last partial weeks), D3, D4, D5, D6, D7, D21, D24, plus the modelling-side items D8-D17, D19, D20, D22, D25. Choices the user made explicitly: **D8 = option (a)** (report D with and without cold-start series; modelling them is follow-up F2); **D9 = report both** (a) non-seasonal-only and (b) short-window variants; **D19 order** = feature dropout, cleaning, auxiliary tasks, teacher-student; **D15** formula as stated on the page.
- The user asked whether D14 collides with D8. Answer: no (D14 defines the cut, D8 how to report series without history). New consequence logged as F1: series with no history or an all-zero history have no MASE scale, so the convention is found by reproducing the brief's baselines in step 0.5.
- All decisions moved to "Decided" in `decisions-to-make.md`; open follow-ups F1-F5 are listed there. Build spec for step 0.4 written in `plan.md`; not yet run.

## 2026-11-21: step 0.4 build weekly tables (user said go)
- Code: `src/prepare.py` (build, 8 s), `src/check_prepare.py` (checks + plot). Output: `data/interim/weekly_{A,B,C,D,test}.parquet`, `data/interim/prepare_stats.json`, plot `iterations/00-baselines/weekly_totals.png`. Raw files untouched.
- Result (rows / series / weeks): A 98,835 / 599 / 165; B 315,000 / 7,500 / 42; C 98,914 / 851 / 120; D 1,713,980 / 50,443 / 46; test 415,272 / 2,904 / 143. Zero rates: A 43.2%, B 23.5%, C 79.0%, D 93.9%, test 64.7%.
- Integrity checks passed: no duplicate keys, all weeks Mondays, no gaps inside a series, all series end at the shop's last complete week, units >= 0. A and the test shop are identical to the raw files (all rows matched, units and on_hand equal).
- D7 grain check: B lines -> baskets is many-to-one with no unmatched lines; C is one row per store x article x day.
- Units reconciliation: B keeps 96.3% of raw units (dropped the two partial weeks). C 101.0% and D 102.1% exceed 100% because netting returns then clipping weekly totals at 0 raises the total; checked for D: 116,995 raw - 1,070 (dropped partial weeks) + 3,512 (clipped negative weeks) = 119,437 = kept, exact. D has 3,491 clipped series-weeks, C 326.
- D5 details: B series without stock event 0, sale before first event 0 (all B series start at week 1). C: first sale before first row 0. D: 2,588 sales series have no stock record; 814 sell before their first stock record (start = the earlier of the two); 4 series dropped (first sale only in the partial last week); 13,109 stock-only series (no sale at all) excluded as all-zero.
- on_hand: missing rate A 100% and B 100% (by design), C 28.2% (weeks without a daily row, mostly after a series' last row), D 39.7%, test 14.2%. D negatives clipped and flagged: 6,687 rows; test 7 rows.
- D8 numbers after preparation (cut 2026-01-19): 9,613 of 50,443 series have no units before the cut; they are 46.8% of holdout units; 6,579 of them have weeks of zeros in the grid before the cut (a stock record exists).
- New observations (plot and counts): (1) the test shop's yearly peak moves about 2 weeks earlier each year (2020-11-16, 2021-11-01, 2022-10-17) while A's peaks sit on fixed dates: logged as F6. (2) 485 of 851 C series stop having rows before the end, which explains C's falling total: F7. (3) B's weekly total is flat and noisy (18-20.5k), consistent with synthetic data; its last complete week (17.9k) is the lowest but not partial (all 7 days present).
- F5 resolved. Not done in this step (by design): features, cuts, metric, baselines.

## 2026-11-21: documentation request, then step 0.5 metric and cuts (user said go)
- Recorded in `decisions-to-make.md`: a data-profile table with the zero-week % per shop (A 43.2%, B 23.5%, C 79.0%, D 93.9%, test 64.7%; the brief's daily zero rates C 96.0%, D 99.3%), and the test shop's moving yearly peak (key finding 10, F6, plus a calendar caution in `pipeline-architecture.md`).
- Code: `src/validate.py` (cuts, history scale, score, summary), `tests/test_validate.py` (9 hand-computed cases: perfect forecast = 100, MASE 0.5 = 50, MASE 1 = 0, negative scores, equal weight across articles, zero-scale and no-history exclusion, late-starting series, `from_first_sale` scale, article level). All pass. Run: `.venv/bin/python tests/test_validate.py`.
- Score implemented as the page states it (D15): per series, mean absolute error over the 13 holdout weeks / mean absolute week-to-week change of the training history; mean over series; score = (1 - MASE) x 100. Conventions the page leaves open are parameters (`level` series|article, `scale_mode` full|from_first_sale, zero scale excluded and counted).
- Cuts (D14): the last 13 complete weeks of each table: A 2024-03-11, B 2025-07-28, C and D 2026-01-19, test 2023-01-23 (130 training weeks). Holdout has one row per series per week, weeks before a series' start count as 0.
- Findings (`.venv/bin/python src/validate.py`): A, B, C fully scorable. D series level: 39,921 of 50,443 scored = 53.1% of holdout units; D article level: 2,169 of 2,326 = 83.8%. Test cut: 65 zero-scale articles excluded (0 in the real test shop). Logged as F1 and a new F8 (grain of "article").
- Not done: baselines (step 0.6), reproduction of the brief's numbers (step 0.7).

## 2026-11-21: step 0.6 baselines (user said go) and the first look at 0.7
- Question asked first: are the 65 all-zero-history test articles "easy zeros"? No: all 65 sell in the holdout (3,021 units, 0.86% of holdout units). They are excluded only because MASE divides by 0 (details in F1).
- Code: `src/baselines.py` (zero, naive, mean13, median13, seasonal naive for >= 65 training weeks, AutoARIMA with no seasonality and the 8-week zero rule, Top-down MSTL with season 52 if >= 2 years else 4, shares from the last 13 weeks, 8-week zero rule), both forecast levels (series, article), both scale modes. Output `iterations/00-baselines/results.csv` (118 rows), run log `logs/baselines_run.txt`. Total run time about 9 minutes on the laptop (D AutoARIMA series level 260 s).
- Local-cut scores, series level, scale = full history, history zero-filled from each series' start (current prepared tables):

  | model | A | B | C | D | test |
  |---|---|---|---|---|---|
  | zero | -1.2 | -8.4 | 61.9 | 77.6 | -23.4 |
  | naive | 3.3 | -3.3 | 46.1 | 51.5 | -58.3 |
  | mean13 | 4.2 | 20.1 | 40.5 | 44.3 | -36.0 |
  | median13 | 17.4 | 20.0 | 60.4 | 77.3 | -6.7 |
  | seasonal_naive | -12.3 | n/a | 20.5 | n/a | -40.9 |
  | autoarima | 8.7 | 20.2 | 44.1 | 70.5 | -28.8 |
  | topdown_mstl | 10.2 | 20.2 | 42.7 | 63.3 | -28.6 |

- Observations: (1) **A reproduces the brief exactly** (8.7 and 10.2). (2) **median13 is the best simple model on the test-shop local cut (-6.7)**, 22 points above AutoARIMA (-28.8); it is also best on A (17.4). This supports the MAE/median hypothesis (D20). The best live-board team is at -7.1 on the hidden test weeks; the windows differ, so this is only a similarity. (3) **Zero beats fitted models on the sparse shops C and D** (61.9 vs 44.1; 77.6 vs 70.5), but loses on A, B and the test shop. (4) Seasonal naive is poor everywhere it applies. (5) The test-shop local cut is a reasonable proxy for the live board: our naive -58.3 and AutoARIMA -28.8 vs the live -64.5 and -33.7.
- Reproduction probe (`src/probes/probe_leading_zeros.py`, not part of results.csv): zero-filling every series from the shop's first week (leading zeros) moves C and D toward the brief: C Top-down 38.7 (brief 37.7; before 42.7), C AutoARIMA 40.2 (brief 41.8; before 44.1), D Top-down 44.5 (brief 44.9; before 63.3), with series-level scoring. Article level does not match (D 7.8 / 20.8 article-level vs brief 57.2 / 44.9). B differs by 0.2 and A matches exactly. D AutoARIMA was not rerun under the probe.
- Also: scale mode matters a lot on the test shop (AutoARIMA -28.8 with full history vs +29.5 with history from the first sale); the live-board references match the full-history version.
- Proposed convention (D26) and its consequence for D8 are in `decisions-to-make.md`. Decision gate 0.7 is open.
- **Update after the user adopted D26 (2026-11-21):** `split()` now pads series with leading zeros by default (new tests pass); the full baseline rerun finished (`results.csv`, 118 rows, column `convention` = "D26 leading-zero grid"; the old results are kept as `results_superseded_series_start_grid.csv`, the old run log as `logs/baselines_run_superseded_series_start_grid.txt`).
- **Reproduction of the brief's baselines (series level, full-history scale):** A AutoARIMA 8.7 (brief 8.7), A Top-down 10.2 (10.2), B AutoARIMA 20.2 (20.4), B Top-down 20.2 (20.4), C AutoARIMA 40.2 (41.8), C Top-down 38.7 (37.7), D AutoARIMA 58.6 (57.2), D Top-down 44.5 (44.9). All within 1.6 points; D is scored on 39,983 of 50,443 series (53% of holdout units); the other 10,460 have a zero-scale history.
- **Final local-cut baseline scores (D26 convention):**

  | model | A | B | C | D | test |
  |---|---|---|---|---|---|
  | zero | -1.2 | -8.4 | 58.8 | 69.1 | -23.4 |
  | naive | 3.3 | -3.3 | 41.9 | 14.9 | -58.3 |
  | mean13 | 4.2 | 20.1 | 36.5 | 13.4 | -36.0 |
  | median13 | 17.4 | 20.0 | 57.3 | 68.8 | **-6.7** |
  | seasonal_naive | -12.3 | n/a | 14.2 | n/a | -40.9 |
  | autoarima | 8.7 | 20.2 | 40.2 | 58.6 | -28.8 |
  | topdown_mstl | 10.2 | 20.2 | 38.7 | 44.5 | -28.6 |

  A, B and the test shop are unchanged from the first run (no leading zeros there). C and D changed; under the new convention naive and mean13 collapse on D (14.9, 13.4) because a short, mostly-zero history gives a tiny scale. On C and D, zero (58.8, 69.1) is now about equal to median13 (57.3, 68.8) and beats AutoARIMA by a wide margin.
- **Gate 0.7 closed:** convention accepted by the user (D26). Iteration 00 is complete.
- Housekeeping: a first probe run printed garbled output because the script lacked a `__main__` guard and AutoARIMA's worker processes re-ran it; rerun correctly, results above are from the rerun.
