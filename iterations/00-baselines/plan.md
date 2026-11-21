# Iteration 00: baselines and a trustworthy metric

**Status (2026-11-21):** plan approved. Steps 0.1, 0.2 and decision gate 1 (0.3) are done; all of D1-D25 are decided. **COMPLETE.** Steps 0.1-0.7 are done; D26 was adopted and the baselines rerun under it reproduce the brief's published scores within 1.6 points (see `log.md`). Step 0.8 (optional sanity upload of a naive submission) is left to the user.

## Hypothesis
If our local MASE reproduces the brief's published baseline scores on A-D, then every later number is comparable, and we know what any model has to beat. We also learn which data-preparation decisions matter before modelling.

## Done means
1. One weekly table per shop in the common schema (`data/interim/weekly_*.parquet`), with every preparation decision logged.
2. A local score function that reproduces the brief's A-D baselines (AutoARIMA A 8.7, B 20.4, C 41.8, D 57.2; Top-down A 10.2, B 20.4, C 37.7, D 44.9) closely enough that the user accepts it (D15).
3. A baseline table on the local cuts of A-D and the test shop (cut 2023-01-23), with D reported with and without the cold-start series (D8).
4. `results.csv`, `log.md`, `changes.md` filled in.

## Steps (each ends with a checkpoint: Claude stops and waits)
| # | Step | Needs decision | Output |
|---|---|---|---|
| 0.1 | **Environment check.** Python version, which of pandas, pyarrow, numpy, lightgbm, statsforecast, torch (+ MPS) are installed; nothing is installed without asking. | D22 | list of what is available |
| 0.2 | **Verify raw facts** (read-only agent): are the holdout weeks inside A-D (D1)? partial last weeks (D2)? what the C codes look like (D4)? can weekly `on_hand` be built for B, C, D (D21)? | D1, D2, D4, D21 | evidence note |
| 0.3 | **Decision gate 1.** User settles D1-D7, D21, D24 (or accepts the recommendations). | D1-D7, D21, D24 | decisions moved to "Decided" |
| 0.4 | **Build weekly tables** per shop (agent runs the agreed code). | as settled | `data/interim/weekly_{A,B,C,D,test}.parquet` + checks (row counts, zero rates, weekly totals before/after) |
| 0.5 | **Metric and cuts.** Implement MASE/score; define the cuts (last 13 full weeks of A-D, 2023-01-23 for the test shop). | D14, D15 | `src/validate.py` |
| 0.6 | **Baselines.** Naive, seasonal naive (where >= 2 years), zero rule (no sale in last 8 weeks -> 0), AutoARIMA, top-down (MSTL), as described on the brief's Data page. Score on each cut. | D8, D20 | baseline table |
| 0.7 | **Decision gate 2.** Compare our numbers to the brief's. If they differ, find out why before going on. | D15 | metric accepted or fixed |
| 0.8 | **Optional: sanity upload.** The user registers the team and uploads a naive `submission.csv` to see the live score (expected near -64.5) and confirm the format. Outward-facing, so the user does it. | D25 | live score noted |

## Agents (briefs saved in `briefs/` before they run)
- **Profiler** (step 0.2): read-only, reports evidence, decides nothing.
- **Builder** (step 0.4): writes only `data/interim/` and `src/prepare.py`, following settled decisions exactly.
- **Baseline runner** (step 0.6): writes only `src/baselines.py` and `results.csv`.

## Step 0.4 build spec (follows the decided D1-D7, D21, D24; nothing built yet)
Output per shop: `data/interim/weekly_{A,B,C,D,test}.parquet` with columns `shop, store_id, article_id, week (Monday), units, on_hand, on_hand_missing, out_of_stock_flag`. Common rules: weeks Monday-based; drop first and last partial weeks (D2); weekly units = sum of units with returns netted, then clipped at 0 (D3); every series is zero-filled from its start week to the shop's last complete week (D5); raw files are never modified.

| Shop | Units | Series start (D5) | `on_hand` / stock flag (D21) | Complete weeks kept |
|---|---|---|---|---|
| A | `weekly.parquet` as is | first week (grid is complete) | `on_hand` missing; `out_of_stock_flag = not in_stock`; snapshot unused (leakage, D1) | 2021-04-12..2024-06-03 |
| B | `sales_lines` joined to `baskets` on `transaction_id` (many lines : 1 basket, checked: no unmatched lines), summed per store x article x week | week of the series' first `stock_events` entry (assumption F5), else first sale | `on_hand` missing; `out_of_stock_flag` = a STOCKOUT event in that week | 2025-01-06..2025-10-20 |
| C | daily rows summed per week (the -1 rows carry 0 units) | first daily row of the series | week-end value = last daily row in the week; missing if no row that week | 2024-01-01..2026-04-13 |
| D | `sales` lines summed per store x article x week | week of the first `stock.parquet` range, else first sale | value of the range covering the week end; missing if none (about 29% of sales-weeks); negatives clipped to 0 and flagged (assumption F5) | 2025-06-02..2026-04-13 |
| test | `history.parquet` as is | first week (grid is complete) | given; missing stays missing + flag; 7 negatives clipped to 0 (D24) | 2020-07-27..2023-04-17 |

Checks after building (reported at the checkpoint): rows per shop, number of series, zero rate, weekly totals before vs after the cuts, and a plot of weekly totals per shop. Assumptions the user can veto: **F5** (B series start from the first stock event; D's negative `on_hand` handled like the test shop's).

## Risks
- The brief's baseline scores may not be reproducible if the exact MASE scale term differs: this is what step 0.7 is for.
- Statistical baselines (AutoARIMA) on 50k D series can be slow on a laptop: if so, score them on a sample of D and log that.
- On B and D the validation windows are short (44-48 weeks of history).

## Not in this iteration
Features, learned models, foundation models, the rich-column techniques, external data. Those start in iteration 01 and later (see `pipeline-architecture.md`).
