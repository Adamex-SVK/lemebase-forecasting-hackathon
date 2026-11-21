# Iteration history

The final model is in [`../MODEL.md`](../MODEL.md). This folder is the record of how we got there: one folder per iteration, each with its plan, log, results CSVs and notes. Scores are the local mean over 4 rolling cuts (higher is better) unless marked *live* (leaderboard, weeks 1-7). Every decision is in [`../decisions-to-make.md`](../decisions-to-make.md); every upload in [`../submissions/UPLOADS.md`](../submissions/UPLOADS.md).

| # | Folder | Question | Outcome |
|---|---|---|---|
| 00 | `00-baselines` | Weekly tables, local metric, baselines | Metric reproduces the organisers' published baselines |
| 01 | `01-global-gbm` | Global LightGBM: own vs pooled vs others-only data | Own history wins. Live **-7.6** |
| 02 | `02-robust-eval` | Reliable evaluation | 4 rolling cuts and the D26 metric: the yardstick for everything after |
| 03 | `03-foundation-model` | Zero-shot foundation models (Chronos, TimesFM, Moirai, TiRex, TabPFN) | **Blend of 3 models + 30% LightGBM (D40).** Live **-4.8** |
| 04 | `04-rich-columns` | Use of price/stock/promo columns, feature dropout | Not adopted (D43) |
| 05 | `05-meta-ensemble` | FFORMA-style learned weights | Rejected, live -7.9 / -6.2 |
| 06 | `06-neural` | N-HiTS / N-BEATS | Rejected |
| 07 | `07-ensemble-toolkit` | Tools to search ensembles | Used by 08 |
| 08 | `08-improvements` | Ten improvement tasks T1-T10, ensemble search | None beat D40 |
| 09 | `09-forecast-approaches`, `09-different-angles` | Error atlas, distribution pooling, reconciliation, pooled level model | All rejected |
| 10 | `10-external-info`, `10-inventory` | Stock masking, covariates, fine-tuning on sibling shops, inventory features | **Chronos-2 fine-tuned on sibling shops as a 4th member: FINAL model, live -4.5** |
| 11 | `11-data-overview` | Test-shop stock jump, trimming, level | Not adopted (D46) |

## Reading order
D40 recipe: `03-foundation-model/log.md`. Why nothing else beat it: `08-improvements/README.md`, then `09-forecast-approaches/README.md`. How the final member was added: `10-external-info/x1/notes_adamex-svk_unused-info.md`.

## What is and is not in this folder
- Plans, logs, notes and result CSVs for every iteration are here.
- Forecast files needed to rebuild the final model are committed (`03-foundation-model/forecasts_final/`, `forecasts_oos-ft/`). Other bulk forecasts (some cut-level and neural arrays) live on archive branches only.
- Code for rejected experiments is not all on `main`; the branches `it04/adamex-svk/rich-columns`, `it06/kilo/neural-forecasters`, `it09/kimi/different-angles` and `snapshot/2026-11-21-parallel-sessions` keep it (closed, unmerged pull requests #6, #7, #16, #17).
