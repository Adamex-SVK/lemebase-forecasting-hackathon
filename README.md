# LemeBase forecasting hackathon

Forecast weekly units sold per article, 13 weeks ahead, for a test shop (2,904 articles), using
four training shops (Sets A-D) and the test shop's own history. Task, data and scoring are described at
https://lemebase.com/hackathon (score = (1 - MASE) x 100, higher is better).

Work is organised in iterations with every decision tracked.

## Where to look
| File | What it is |
|---|---|
| `decisions-to-make.md` | Every decision: open, tested, decided, with evidence |
| `next-steps.md` | Current state, what to resume, plan for the next iterations |
| `process-rules.md`, `pipeline-architecture.md` | How we work, the pipeline stages |
| `iterations/00-baselines` | Weekly tables, local metric (reproduces the organizers' published baselines), baselines |
| `iterations/01-global-gbm` | Global LightGBM (L1 loss on units / MASE scale): own vs pooled vs others-only arms |
| `iterations/02-robust-eval` | Rolling-origin evaluation (4 cuts), F6 (moving peak) investigation, quick experiments |
| `iterations/03-foundation-model` | Plan and model notes for Chronos, TimesFM, Moirai, TiRex (prepared, not yet run) |
| `src/` | `prepare.py`, `validate.py` (metric and cuts), `baselines.py`, `features.py`, `train_gbm.py`, `run_arms.py`, `rolling.py`, `make_submission.py`, `foundation/` |
| `tests/` | Hand-computed checks of the metric and the features (including a leakage test) |
| `submissions/` | Submission CSVs (`article_id, w1..w13`) |
| `TEAM-RULES.md`, `WORK-BOARD.md`, `CLAUDE.md`, `scripts/check.sh` | How we cooperate: branches, one yardstick, claiming tasks, uploads, pre-PR checks, rules for AI assistants |
| `submissions/UPLOADS.md` | Log of every leaderboard upload |

## Status (night of 2026-11-21)
Best live score so far: -7.6 (global LightGBM trained on all test-shop history); median of the last 13 weeks: -9.8;
live references: AutoARIMA -33.7, Naive -64.5. Local cuts track the live board well, see `iterations/01-global-gbm/log.md`.

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
