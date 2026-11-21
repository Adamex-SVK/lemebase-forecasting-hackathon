# LemeBase forecasting hackathon: final model

Forecast weekly units sold per article, 13 weeks ahead, for a test shop (2,904 articles), using four training shops (Sets A-D) and the test shop's own history. Task and data: https://lemebase.com/hackathon. Score = (1 - MASE) x 100, higher is better.

## Final model (`final-model-v1`)
**D40 ensemble + a Chronos-2 fine-tuned on sibling shops. Live score -4.5** (weeks 1-7; next-best known entry -5.4 when last seen).

| Model | Live score |
|---|---|
| **Final: D40 + fine-tuned Chronos-2** | **-4.5** |
| D40: 3 zero-shot foundation models + 30% LightGBM | -4.8 |
| LightGBM alone | -7.6 |
| Median of last 13 weeks | -9.8 |
| AutoARIMA / Naive (organisers) | -33.7 / -64.5 |

- **What it is, how to rebuild it, where it is weak:** [`MODEL.md`](MODEL.md).
- **Submission file:** `submissions/ensemble_d40_plus_chronos2ft_v1.csv`.
- **Rebuild, no GPU needed (byte-identical, sha256 checked):**
```bash
.venv/bin/python src/build_best_model.py --skip-lgbm
```

## How we got here
Eleven iterations, every decision logged, every candidate judged on the same 4 rolling cuts. Combining different models is what worked; about thirty other ideas (neural nets, learned weights, rich columns, gating, tuning) did not clear the adoption bar.
- [`iterations/README.md`](iterations/README.md): one line per iteration and where to read more
- [`decisions-to-make.md`](decisions-to-make.md): every decision with evidence (the final model is D44)
- [`submissions/UPLOADS.md`](submissions/UPLOADS.md): every leaderboard upload and its live score

## Repository map
| Path | Contents |
|---|---|
| `MODEL.md` | The final model card |
| `src/` | Pipeline: `prepare.py`, `validate.py` (metric, cuts), `features.py`, `train_gbm.py`, `build_best_model.py`, `foundation/` (model adapters), `meta/`, `improve/` |
| `tests/`, `scripts/check.sh` | Metric and feature tests (incl. a leakage test); pre-PR check |
| `iterations/` | Plans, logs, results and notes of each iteration |
| `submissions/` | Submission CSVs (`article_id, w1..w13`) |
| `docs/` | Working documents from the hackathon (team rules, process, handover, extra-credit draft); archived |
| `CLAUDE.md` | Rules for AI assistants working in this repo |

## Reproduce from scratch
```bash
huggingface-cli download LemeBase/hackathon-retail-sets --repo-type dataset --local-dir hackathon-retail-sets
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/prepare.py                       # weekly tables -> data/interim
.venv/bin/python tests/test_validate.py && .venv/bin/python tests/test_features.py
.venv/bin/python src/build_best_model.py              # LightGBM + ensemble, validates, checks the hash
```
Regenerating the foundation-model forecasts themselves needs a GPU; the committed ones are enough to rebuild.

## Data licence
The datasets are CC BY 4.0 from several sources (see `ATTRIBUTION.md` in the dataset). They are not included here.
