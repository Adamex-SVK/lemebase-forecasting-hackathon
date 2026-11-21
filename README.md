# LemeBase forecasting hackathon

Forecast weekly units sold per article, 13 weeks ahead, for a test shop (2,904 articles), using
four training shops (Sets A-D) and the test shop's own history. Task, data and scoring are described at
https://lemebase.com/hackathon (score = (1 - MASE) x 100, higher is better).

Work is organised in iterations with every decision tracked.

## Where to look
| File | What it is |
|---|---|
| `decisions-to-make.md` | Every decision: open, tested, decided, with evidence |
| `AGENTS.md` | Workflow rules and advisor-skill guidance |
| `pipeline-architecture.md` | Pipeline stages and historical implementation roadmap |
| `iterations/00-baselines` | Weekly tables, local metric (reproduces the organizers' published baselines), baselines |
| `iterations/01-global-gbm` | Global LightGBM (L1 loss on units / MASE scale): own vs pooled vs others-only arms |
| `iterations/02-robust-eval` | Rolling-origin evaluation (4 cuts), F6 (moving peak) investigation, quick experiments |
| `iterations/03-foundation-model` | Plan and model notes for Chronos, TimesFM, Moirai, TiRex (prepared, not yet run) |
| `iterations/05-data-evaluation` | Reproducible raw/prepared data audit and findings |
| `iterations/06-autogluon` | Optional AutoGluon challenger setup, evaluation commands, and integration evidence |
| `iterations/07-ensemble-review` | Reproduced winning ensemble, component removals, and blend-weight checks |
| `iterations/08-xgboost-blend` | Fixed 60/30/10 foundation/LightGBM/XGBoost test; no improvement over incumbent |
| `iterations/10-benchmark-diagnosis` | Phase-1 ensemble error diagnosis, additional forecast inventory, and next hypothesis |
| `iterations/11-activation-features` | Phase-2 controlled activation-feature ablation, paired retraining and historical checks |
| `iterations/13-recency-weighting` | Isolated fixed 104-week training-decay test; improvement too small/inconsistent to adopt |
| `VALIDATION.md` | Frozen benchmark and temporal validation contract for subsequent experiments |
| `src/` | `prepare.py`, `validate.py` (metric and cuts), `baselines.py`, `features.py`, `train_gbm.py`, `run_arms.py`, `rolling.py`, `make_submission.py`, `foundation/` |
| `tests/` | Hand-computed checks of the metric and the features (including a leakage test) |
| `submissions/` | Submission CSVs (`article_id, w1..w13`) |

## Maya's branch

`Mayas-forecasting-workflow` contains the data evaluation and an optional
AutoGluon challenger (see `iterations/06-autogluon/plan.md`). Preserve the existing
LightGBM reference while comparing other model families and cross-shop training.
Historical iteration plans are context, not instructions to restart unfinished
runs automatically.

## Status (night of 2026-11-21)
Best reported live score: **-4.8**, from `submissions/ensemble_top3_blend_v1.csv`
(Moirai 2.0-small, TiRex, Chronos-T5-small, plus 30% LightGBM; zero rule and rounding).
Confirmed by the user and the friend's upload record; original CSV preserved
from commit `71b82f3`. Historical reconstruction and comparisons are in
`iterations/07-ensemble-review/log.md`. Earlier LightGBM score: -7.6; median13:
-9.8; live references: AutoARIMA -33.7, Naive -64.5. Public and historical
scores cover different periods and must not be compared directly.

Phase 2 activation-feature ablation completed: historical mean -13.6184 versus
incumbent -13.6659, only 2/4 periods improved and both additional periods regressed.
It fails the promotion threshold; the champion remains unchanged. Reproduction,
paired controls and evidence: `iterations/11-activation-features/plan.md` and `log.md`.

## Reproduce
```bash
# 1. data: download the dataset into ./hackathon-retail-sets (set-a ... set-d, test-shop)
huggingface-cli download LemeBase/hackathon-retail-sets --repo-type dataset --local-dir hackathon-retail-sets
# 2. environment (Python 3.13; scipy on Python 3.10 fails to load on macOS 27)
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
# 3. pipeline
.venv/bin/python src/prepare.py            # weekly tables -> data/interim
.venv/bin/python tests/test_validate.py && .venv/bin/python tests/test_features.py
.venv/bin/python src/baselines.py          # baselines on the local cuts
.venv/bin/python src/train_gbm.py test     # LightGBM, own arm, test shop
.venv/bin/python src/make_submission.py    # writes + validates a submission CSV
```

## Data licence
The datasets are CC BY 4.0 from several sources (see `ATTRIBUTION.md` in the dataset). They are not included here.
