# Branch map: what each iteration did

Every iteration lives on its own branch (`itNN/<who>/<topic>`). Scores are the local 4-cut D26 mean (higher is better); the current best recipe is **D40**, local -13.67, live -4.8 (`HANDOVER.md`).

| Branch | What it did | Result |
|---|---|---|
| `it03/adamex-svk/foundation-models` | Zero-shot foundation models (Chronos, TimesFM, Moirai, TiRex, TabPFN), blend with LightGBM, submission builders, GPU notes | **D40 recipe found** (-13.67 local, -4.8 live) |
| `it04/adamex-svk/rich-columns` | Use of the rich columns, meta-ensemble / FFORMA on overlapping cuts (`05-meta-ensemble`) | Not adopted (D36, D37) |
| `it06/kilo/neural-forecasters` | N-HiTS / N-BEATS neural forecasters | Not adopted |
| `it08/adamex-svk/improvements` | Integration of T1-T10: all evaluated against D40 | None beat D40 |
| `it08/adamex-svk/t1` ... `t10` | One task each: T1 intermittent-demand classics, T2 post-processing, T3 robust combination, T4 horizon flattening, T5 TabPFN check, T6 quantile variants, T7 context variants, T8 other tree models, T9 fine-tuning Moirai, T10 new models (Toto, Sundial, Chronos-2-small) | All rejected |
| `it09/adamex-svk/forecast-approaches` | Distribution pooling, GPU quantile exports, N-HiTS baseline | Pooling rejected |
| `it09/kimi/different-angles` | Audit and pooled-level model | A/B rejected |
| `snapshot/2026-11-21-parallel-sessions` | Uncommitted work from parallel sessions in the shared checkout (kilo clean checks, FFORMA hedge, CatBoost, neural, it09 notes); not reviewed history | Archive |
| `docs/branch-map` | This file | n/a |

Ensemble search (greedy, SLSQP, segment weights, zero voting, performance-based weights) is documented in `iterations/08-improvements/ens-search/`.

`main` on GitHub is behind these branches; merge by decision (`decisions-to-make.md`), not automatically.
