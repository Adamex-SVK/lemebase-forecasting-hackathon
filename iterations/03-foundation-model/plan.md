# Iteration 03: foundation models (GPU)

**Status (2026-11-21):** GPU access received (RTX 3090, JupyterLab); work is on branch `it03/adamex-svk/foundation-models`. Preparation done, nothing run on the GPU yet. Model list decided (D35). Next: step 3.0, follow `gpu-runbook.md` (a person logs in, uploads the bundle, runs the smoke tests and pastes the output). See `log.md`.

## Hypothesis
Pretrained time-series models, used zero-shot and then fine-tuned on our pooled windows, add accuracy on their own or as a blend member with the LightGBM. Our data are mostly zeros, which these models were not designed for, so the expectation is modest.

## What is prepared (tested only with dummy models, no foundation-model package installed yet)
- `src/foundation/models.py`: one adapter per model family with a common interface (see its header; unverified API calls are marked `VERIFY`).
- `src/foundation/run_model.py MODEL SHOP [--cut-index K] [--context padded|first_sale]`: builds the zero-padded history before a cut (D26), sends non-empty series to the model, saves forecasts.
- `src/foundation/score_all.py`: scores all saved forecasts with the D26 metric (main environment) next to the median-of-13-weeks reference.
- `src/foundation/smoke.py`: runs each adapter on 3 toy series, so the first thing on the GPU machine shows which adapter needs a fix.
- `env/setup_gpu.sh` and `env/requirements_*.txt`: one virtual environment per model family (the Moirai pins conflict with modern stacks).
- Plumbing check: `dummy-median13` and `dummy-naive` through `run_model.py` and `score_all.py` reproduce the baselines exactly (-6.68 and -58.26 on the test-shop cut).
- Notes on every model: `model-notes.md`.

## Steps
| # | Step | Needs |
|---|---|---|
| 3.0 | GPU machine: copy the project, `bash env/setup_gpu.sh`, run `smoke.py` per family, fix adapters | access to the GPU, internet for weights |
| 3.1 | Zero-shot on the test shop cut 0 (`--limit 200` first), then all five shops and the 4 rolling cuts for the best few models | D34 (licences) |
| 3.2 | Context options: `padded` vs `first_sale`; zero rule on or off | |
| 3.3 | Fine-tune the best 1-2 models on pooled windows (own vs pooled) | model-specific recipes, see `model-notes.md` |
| 3.4 | Blend with the LightGBM and the median; choose weights on earlier cuts, confirm on cut 0 | |
| 3.5 | Review with the user | |

## Not in this iteration
Rich-column techniques (iteration 04), the final submission (iteration 05).
