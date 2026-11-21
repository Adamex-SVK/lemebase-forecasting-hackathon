# Pipeline architecture

One pass through these stages is one iteration. Each stage lists its input, output, the decisions it depends on (from `decisions-to-make.md`) and its checkpoint.

```text
raw parquet (read-only)
   -> 1 PREPARE   -> data/interim/weekly_{A,B,C,D,test}.parquet
   -> 2 FEATURES  -> data/processed/features_*.parquet
   -> 3 DATASET   -> windows (history, next 13 weeks) + splits
   -> 4 VALIDATE  -> local MASE, cuts, baselines
   -> 5 MODEL     -> forecasts per shop
   -> 6 EXPERIMENTS (own-only vs pooled; rich-column techniques)
   -> 7 SUBMIT    -> submission.csv (user uploads)
   -> results.csv + log.md + changes.md   (every iteration)
```

## Common schema (the "four columns")
Every shop is reshaped to the test shop's schema, plus identifiers:

| column | meaning |
|---|---|
| `shop` | A / B / C / D / test |
| `series_id` | shop + store + article (test shop: article only) |
| `week` | Monday of the week |
| `units` | units sold that week (returns netted, clipped at 0; D3) |
| `on_hand` | stock at week end, where buildable (D21); missing flag otherwise |
| `in_stock` | optional flag (A has it natively; derived elsewhere) |

Richer columns (price, promotions, hierarchy, supplier, return reason) stay in side tables joined by key and are used for cleaning and for the rich-column techniques (D18), never as inputs of the final model.

## Stages

**1 PREPARE.** Input: raw tables. Output: one weekly table per shop in the common schema. Steps: join receipt lines to baskets (B), aggregate daily/line data to weeks (B, C, D), drop partial last weeks (D2), net returns (D3), zero-fill inside each series' active window (D5), build weekly `on_hand` (D21, D24). Decisions: D1-D7, D21, D24. Checkpoint: row counts, zero rates, weekly totals per shop before vs after, plot of weekly totals.

**2 FEATURES.** Only features computable from the four columns (plus calendar): lags, rolling means, share of zero weeks, weeks since last sale, series age, past `on_hand` and out-of-stock flags, week-of-year / trend (aligned per D10), shop-wide weekly total and its seasonal shape. Richer-column features exist only for the teacher side (D18). **Calendar caution (F6):** the test shop's yearly peak moves about two weeks earlier each year (2020-11-16, 2021-11-01, 2022-10-17) while A's peaks are on fixed dates, so no week-of-year feature is used before this is investigated. Decisions: D10, D11, D12. Checkpoint: feature list with how each is computed for the test shop.

**3 DATASET.** Build training windows (history -> next 13 weeks) for every series at many cut points; short shops (B, D) contribute many windows. Splits: time-based only, never random across time. Decisions: D8, D9, D14. Checkpoint: window counts per shop, leakage check (no feature uses weeks after its cut).

**4 VALIDATE.** Implement MASE and the score; cut the last 13 weeks of A-D and of the test shop (2023-01-23); reproduce the brief's published A-D baseline scores (D15). Baselines: naive, seasonal naive, zero rule, AutoARIMA, top-down. Checkpoint: our baselines vs the brief's numbers.

**5 MODEL.** Candidates (D13): global LightGBM with MAE loss (CPU, tonight), foundation model zero-shot (check feasibility on the M1) and fine-tuned (GPU, tomorrow), hybrids/ensembles. Decisions: D13, D20. Checkpoint: per-shop scores vs baselines.

**6 EXPERIMENTS.** (a) Extra credit: own-only vs pooled for each shop (D16). (b) Rich-column techniques (D18/D19): feature dropout, cleaning, auxiliary tasks, teacher-student. Each compared to the previous best on the same cuts.

**7 SUBMIT.** Write `submission.csv` (`article_id, w1..w13`, all 2,904 articles, non-negative per D17) and validate the format. The user uploads it (D25).

## Experiment matrix (the "training schedule")
Rows are models, columns are training data. Filled in `plan.md` before anything runs; results go to `results.csv`.

| | own only | + all other sets | + long shops only (A, C, test) |
|---|---|---|---|
| baselines | x | n/a | n/a |
| LightGBM | x | x | x |
| foundation model | zero-shot | fine-tuned | fine-tuned |
| + D18 techniques | | x | x |

## Iteration roadmap (draft, to be confirmed at each review)
| # | Hypothesis | Where |
|---|---|---|
| 00 baselines | Local metric reproduces the brief; naive / zero rule / AutoARIMA / top-down baselines on A-D and the test-shop cut | laptop |
| 01 global GBM | A pooled LightGBM with MAE loss beats baselines; own-only vs pooled per shop | laptop |
| 02 features | Calendar alignment, `on_hand` features, shop-total signal and the zero rule improve it | laptop |
| 03 foundation model | A pretrained forecasting model, zero-shot then fine-tuned, adds value | GPU |
| 04 rich-column techniques | Feature dropout, cleaning, auxiliary tasks, teacher-student each add score (D18) | GPU |
| 05 ensemble and final | Combine the best; write the extra-credit explanation | either |

## Directory layout
```text
hackathlon/
  hackathon-retail-sets/        raw, read-only
  data/interim/  data/processed/
  src/                          prepare, features, validate, models, submit
  iterations/NN-name/
  decisions-to-make.md  process-rules.md  pipeline-architecture.md
```
