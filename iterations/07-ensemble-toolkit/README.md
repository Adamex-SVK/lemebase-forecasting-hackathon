# Iteration 07: reusable ensemble toolkit + the current best recipe

A self-contained package of the ensembling work (iteration 05 + the final recipe D40) so anyone on
the team can (1) reproduce the current best submission, (2) test whether a **new base model**
improves the blend on the honest 4-cut yardstick, and (3) build a new submission if it does.

## Current best recipe (D40) - `ensemble_top3_blend_v1`
1. Average the raw forecasts of **Moirai 2.0-small + TiRex + Chronos-T5-small** (zero-shot, padded context).
2. Set to 0 for articles with no sale in the last 8 weeks (zero rule, D39).
3. Blend `0.3 * LightGBM (lgbm_own_v1) + 0.7 * ensemble`.
4. Round to whole units and clip at 0 (D38).

| Quantity | Value |
|---|---|
| Local 4-cut mean (D26, standard cuts) | **-13.67** |
| Ensemble alone, 4-cut mean | -14.05 |
| Live leaderboard, weeks 1-7 | **-4.8** (best of our uploads) |
| Submission | `submissions/ensemble_top3_blend_v1.csv` |

Reference points (from `iterations/`/`HANDOVER.md`): LightGBM rounded about -15.1, median-of-13-weeks
about -16.5, AutoARIMA about -34 on the local cuts. Live: LightGBM -7.6, Moirai 2.0-alone -7.3,
AutoARIMA -33.7, Naive -64.5.

## How to add a better model (the workflow this folder exists for)
1. **Produce zero-shot forecasts** for the new model on the GPU machine and copy them into the repo:
   ```bash
   # practice cuts (needed for evaluation)
   .venv-<env>/bin/python src/foundation/run_model.py <model> test --cut-index 0   # ... 1 2 3
   # real future weeks (needed to build a submission)
   .venv-<env>/bin/python src/foundation/run_model.py <model> test --final
   ```
   Files land in `iterations/03-foundation-model/forecasts/` and `.../forecasts_final/`. If it is a
   new model family, add it to `SPECS` in `src/foundation/models.py` first (see
   `iterations/03-foundation-model/gpu-machine.md` for environments).
2. **Evaluate** it against the current recipe on the same 4 cuts:
   ```bash
   .venv/bin/python src/meta/ensemble_recipe.py evaluate \
       --models moirai-2.0-small,tirex,chronos-t5-small,<new-model> [--lgbm-weight 0.3]
   ```
   It prints per-cut scores, the mean, the gain, how many cuts improved, and the verdict.
   **Adoption bar (TEAM-RULES.md §2): mean gain >= +0.5 AND better on >= 3 of 4 cuts.** Anything
   smaller is noise (seed noise alone is about 0.8 on one cut).
3. **Build** only if it passes:
   ```bash
   .venv/bin/python src/meta/ensemble_recipe.py build \
       --models moirai-2.0-small,tirex,chronos-t5-small,<new-model> --lgbm-weight 0.3 \
       --out submissions/<descriptive-name>.csv
   ```
   The CSV is validated. A **person** uploads it and logs it in `submissions/UPLOADS.md`.
4. **Optional: per-article meta-learner (FFORMA).** If you want to re-try learned weights with the
   new models, first produce overlapping-cut forecasts (`--cut-weeks N`, N = 0,4,...,48) into
   `forecasts_cuts/`, then:
   ```bash
   .venv/bin/python src/meta/fforma.py --models <list> --ff-bases lgbm \
       --out iterations/05-meta-ensemble/results_meta_fforma_<tag>.csv
   ```
   FFORMA was tested and **not adopted** (best +0.23 mean, 3/4 on the overlapping test cuts, below
   the bar); see `iterations/05-meta-ensemble/log.md` and `results/` here.

## What is in this folder
| Path | What |
|---|---|
| `README.md` | this guide |
| `recipe.json` | machine-readable current recipe, scores and adoption bar |
| `log.md` | what was done when packaging this iteration |
| `results/` | snapshot of the FFORMA evaluation CSVs (canonical copies in `iterations/05-meta-ensemble/`) |
| `meta-ensemble-plan.md` | the original design (purged protocol, meta-learners) |

Reusable code (canonical, in `src/`):
| Path | What |
|---|---|
| `src/meta/ensemble_recipe.py` | `evaluate` a candidate model set on the 4 cuts; `build` a validated submission |
| `src/meta/fforma.py` | FFORMA-linear meta-learner on the 13 overlapping cuts (purged), simple-blend reference |
| `src/meta/eval_global_weights_4cut.py` | compares candidate global weightings with D40 on the 4 standard cuts |
| `src/meta/base_cuts_lgbm.py` | LightGBM + median13 base forecasts at the 13 overlapping cuts |

Data used by the tools:
- `iterations/03-foundation-model/forecasts/{model}__test__cut{0..3}__padded.parquet` (practice cuts)
- `iterations/03-foundation-model/forecasts_final/{model}__test__final__padded.parquet` (real weeks)
- `iterations/03-foundation-model/forecasts_cuts/{model}__test__w{N}__padded.parquet` (N = 0,4,...,48)
- `iterations/02-robust-eval/forecasts/base_cut{k}_seed0.npy` (LightGBM at the 4 cuts)
- `submissions/lgbm_own_v1.csv` (LightGBM for the real weeks)

## Protocol and rules (do not skip)
- **Yardstick:** the 4 standard rolling cuts of the test shop with the D26 metric
  (`src/validate.py`, `src/rolling.py`); judge on the mean over cuts + cuts won + a paired interval.
  One cut or one live score is not evidence.
- **No leakage:** any feature at origin `o` uses only weeks before `o`; the meta-learner trains only
  on cuts whose whole holdout is before the test cut (`W >= W_t + 13`).
- **Post-processing is part of the recipe:** zero rule (D39), round to whole units, clip at 0 (D38).
- **Uploads:** a person uploads, at most a handful of meaningfully different files per day; assistants
  never upload. Log every upload in `submissions/UPLOADS.md`.
- **Licences (D34):** Moirai cc-by-nc-4.0, TiRex NXAI community licence, TabPFN v2 Prior Labs licence.
  Fine for the hackathon; check before any commercial use.
- Run `bash scripts/check.sh` before proposing a change.

## Reproducibility (verified 2026-11-21)
- `bash scripts/check.sh` passes.
- `python src/meta/ensemble_recipe.py evaluate --models moirai-2.0-small,tirex,chronos-t5-small`
  reports **-13.67** on the 4 standard cuts (matches the recorded local score).
- `python src/meta/ensemble_recipe.py build --models moirai-2.0-small,tirex,chronos-t5-small
  --lgbm-weight 0.3 --out submissions/ensemble_top3_blend_v1.csv` reproduces the uploaded file
  **byte-identically**.
- `python src/prepare.py` rebuilds the weekly tables; raw data is downloaded per `README.md` (CC BY 4.0).

## Caveats
- **TabPFN (D36) was still running on the GPU machine when this was packaged** (5 jobs, about 7 h CPU
  each). When its forecasts exist, add it with step 2 above; on a 200-series subset at cut 0 it looked
  strong, so it is the first thing to test.
- The live board scores weeks 1-7; the final score uses the hidden weeks 8-13. Do not tune on live
  differences of about 2 points or less.
- Window difficulty varies by about 45 points between neighbouring months, so compare models within a
  cut, never across cuts.
