# HANDOVER: start here

Written 2026-11-21 (afternoon). Purpose: a person or an AI model with no memory of this project can pick it up and keep going. Read this file first, then `TEAM-RULES.md`, `decisions-to-make.md` and `next-steps.md`. Everything below was verified by running code unless it says *unverified*.

## 1. The task in one minute
- Forecast **weekly units sold** for **2,904 articles** of a test shop, **13 weeks ahead** (weeks starting Mon 2023-04-24 to 2023-07-17). Brief and rules: https://lemebase.com/hackathon (Brief, Data, Submit, Leaderboard tabs).
- **Score = (1 - MASE) x 100**, higher is better. Per article: mean over the 13 weeks of |actual - forecast| divided by the article's mean absolute week-to-week change in its history; averaged over articles. 0 = as good as a one-week jump; 100 is unreachable (counts are noisy).
- **Live leaderboard** scores **weeks 1-7** only; the **final** score uses **weeks 8-13** (hidden). Submission = CSV `article_id,w1..w13`, 2,904 rows. Limit 100 uploads per team per day. **A person uploads; assistants never upload.**
- Data: four training shops (Sets A-D, `hackathon-retail-sets/set-a ... set-d`) and the test shop's own history (`test-shop/history.parquet`: `article_id, week_start, units, on_hand`, 143 weeks to 2023-04-17). Raw data is **not in git** (download command in `README.md`, CC BY 4.0).
- Brief's extra credit: for each shop, train on its own data vs with the other shops, report both and explain when borrowing helps or hurts. See `EXTRA-CREDIT.md` (draft).

## 2. Where we stand
**Live scores (weeks 1-7; `submissions/UPLOADS.md`):**

| File | What | Live |
|---|---|---|
| `ensemble_top3_blend_v1.csv` | **BEST.** Moirai 2.0-small + TiRex + Chronos-T5-small averaged, zero rule, 30% LightGBM, rounded | **-4.8** |
| `moirai2_small_v1.csv` | Moirai 2.0-small alone | -7.3 |
| `lgbm_own_v1.csv` | global LightGBM on all history | -7.6 |
| `lgbm_d33_3seed_blend_v2.csv` | LightGBM variant blended with the median | -8.4 |
| `dry_run_median13.csv` | median of the last 13 weeks | -9.8 |
| (references on the board) | AutoARIMA -33.7, Naive -64.5, leader -5.4 when last seen | |

**Local yardstick** (test shop, 4 non-overlapping practice cuts, mean score; details in `iterations/03-foundation-model/log.md` and `results_zero_shot.csv`): LightGBM raw -15.52 (rounded -15.12); median13 -16.54; zero-shot after rounding + zero rule: Moirai 2.0-small **-14.72**, TiRex -15.96, Chronos-T5-small -17.94, T5-base -18.13, Moirai 1.1-base -18.28, TimesFM 2.5 -18.73, Chronos-2 -22.96, Moirai 1.0-large -28.58, Chronos-Bolt about -56. Ensemble of three zero-shot models: -14.05 alone, **-13.67 blended 30% with the LightGBM** (4 of 4 cuts better than the rounded LightGBM; the live gain was +2.8). The local numbers predicted the live ranking and gap well.

## 3. The best recipe and how to rebuild the best file
Recipe: average the zero-shot forecasts of the chosen models -> set to 0 for articles with **no sale in the last 8 weeks** -> blend `0.3 x LightGBM + 0.7 x ensemble` -> **round to whole units** -> clip at 0.
```bash
# laptop, repo root (needs data/interim from src/prepare.py and the forecasts below)
.venv/bin/python src/make_lgbm_final.py                 # reproduces submissions/lgbm_own_v1.csv exactly (checked: max difference 0)
.venv/bin/python src/make_submission_foundation.py      # writes moirai2_small_v1.csv and ensemble_top3_blend_v1.csv
```
The foundation forecasts of the real weeks come from `iterations/03-foundation-model/forecasts_final/*.parquet` (committed; made on the GPU machine with `run_model.py MODEL test --final`).

## 4. The yardstick and the rules (do not skip)
- **D26 metric convention** (`src/validate.py`): series = store x article (test shop: article), history zero-filled from the shop's first week, scale = mean absolute week-to-week change of that history, constant-history series excluded. It reproduces the organizers' published baselines within 1.6 points (`iterations/00-baselines`).
- **Judge every change on the 4 rolling cuts** (`src/rolling.py`): mean over cuts, cuts won, paired interval. One cut or one live score is not evidence (seed noise alone is about 0.8 on one cut; live differences of about 2 points or less are noise). Adoption bar: **>= 0.5 points on the mean and better on >= 3 of 4 cuts** (`TEAM-RULES.md` section 2).
- **Cut naming:** `cutK` = last 13*K weeks of the history removed (K = 0..3, the standard four); `wN` = last N weeks removed (overlapping cuts every 4 weeks, iteration 05). The holdout is always the next 13 weeks. **Window difficulty varies enormously** (LightGBM rounded across 13 cuts: from -37.9 to +7.3), so compare models within a cut, never across cuts.
- **No leakage:** features at origin `o` use only weeks before `o` (`tests/test_features.py` checks it).
- Run `bash scripts/check.sh` before proposing a change. Core files (`src/validate.py`, `src/prepare.py`, `src/features.py`) change only with a decision entry and tests; keep defaults, add options.
- Logs are append-only; decisions live in `decisions-to-make.md` (any collaborator can be the decision owner; record who and when).

## 4b. What worked, what did not (so you do not repeat it)
**Worked:** (1) **rounding forecasts to whole units** (+0.4 for the LightGBM, +1.4 to +2.5 for foundation models: the metric rewards integer medians); (2) the **8-week zero rule**; (3) **zero-shot foundation models**, especially the *small* ones (Moirai 2.0-small beats the trained LightGBM; bigger Moirai models are worse); (4) **averaging several models** and blending a small LightGBM share (the gain is diversification: every foundation model adds 0.5-1.2 points when blended rounded, even weak ones); (5) the median of the last 13 weeks is a surprisingly strong baseline (-16.5 vs AutoARIMA -35).
**Did not work:** pooling the other shops into the LightGBM (own -4.83, pooled -5.84 / -4.95, others-only -23.07 on cut 0); seasonality (seasonal naive and MSTL are poor; the test shop's yearly peaks move 50 weeks apart and are driven by different articles, so no calendar features: F6); the `on_hand` regime fix D33 (about +0.0); blending the LightGBM with the *median* (live -8.4); extra seeds (noise); generating synthetic B data (parked, D23); article-level scoring (does not match the organizers).
**Data quirks** (all handled in `src/prepare.py`, see `decisions-to-make.md` D1-D7, D21, D24): partial first/last weeks dropped; holdout weeks are inside the A-D files; `is_return = -1` in Set C is a no-sale row; Set D has 47% of holdout units in series without history; the test shop's `on_hand` changes scale in early 2021; the live board's Naive/AutoARIMA scores sit close to our local cut-0 values (the practice windows are a fair proxy).

## 5. Repository map
| Path | What |
|---|---|
| `HANDOVER.md`, `README.md`, `TEAM-RULES.md`, `CLAUDE.md`, `WORK-BOARD.md` | orientation, cooperation rules, rules for AI assistants, task board |
| `decisions-to-make.md` | every decision with evidence (open / tested / decided) |
| `next-steps.md`, `process-rules.md`, `pipeline-architecture.md` | current plan, how we work, pipeline stages |
| `src/` | `prepare.py` (weekly tables), `validate.py` (metric + cuts), `baselines.py`, `features.py`, `train_gbm.py`, `run_arms.py`, `rolling.py`, `experiments_02.py`, `make_lgbm_final.py`, `make_submission.py`, `make_submission_foundation.py`, `foundation/` (model adapters, runner, scoring, blends), `meta/` (overlapping-cut LightGBM), `probes/` |
| `iterations/00-baselines` | metric reproduction, baselines |
| `iterations/01-global-gbm` | LightGBM, own/pooled/others-only arms (partial: step 1.3 unfinished) |
| `iterations/02-robust-eval` | 4-cut evaluation, F6, quick experiments; also a loss/NN workstream by another session (section 9) |
| `iterations/03-foundation-model` | zero-shot models: `log.md` (full story), `model-notes.md`, `gpu-runbook.md`, `gpu-machine.md`, `results_zero_shot.csv`, `ensemble_eval_output.txt`, `blend_eval_output.txt`, forecasts (`forecasts/` practice cuts, `forecasts_final/` real weeks, `forecasts_cuts/` overlapping cuts) |
| `iterations/04-rich-columns` | rich-column techniques (another session, section 9) |
| `iterations/05-meta-ensemble` | FFORMA-style meta-ensemble: `plan.md`, `log.md`, `base_forecasts/` (LightGBM at 13 cuts) |
| `submissions/` | every submission CSV and `UPLOADS.md` |
| `env/`, `scripts/` | GPU environment files; `check.sh`, `make_gpu_bundle.sh`, `gpu_ssh.sh`, `gpu_scp.sh` |
| `tests/` | metric, feature (incl. leakage) tests |

## 6. Compute
- **Laptop:** Apple M1, 16 GB, `.venv` on **Python 3.13** (`requirements.txt`; Python 3.10 cannot load scipy on this macOS). Run everything with `.venv/bin/python`.
- **GPU machine:** RTX 3090 24 GB, 96 cores, 62 GB RAM, about 40 GB disk free, torch 2.5.1+cu121 preinstalled. Full operating notes in `iterations/03-foundation-model/gpu-machine.md` (access, layout, environments, jobs, pitfalls). **Credentials and host names are not in the repo**: ask the repository owner. Assistants cannot type passwords; key-based SSH was set up by the owner (cleanup in section 10).

## 7. Jobs in flight at handover (14:20; verify with the commands in `gpu-machine.md`)
1. **TabPFN-TS** (`tabpfn-ts-v2`, local mode, ungated v2 weights): 5 processes on the GPU machine (cuts 0-3 and `--final`), about 4h30 of CPU time each, none finished; CPU-bound (feature building), GPU idle. On a 200-series subset at cut 0 it was the best zero-shot model (7.0 vs TiRex -0.5, TimesFM 3.7; median 9.0, LightGBM 10.2). When done: copy `forecasts/tabpfn-ts-v2__test__cut?__padded.parquet` and `forecasts_final/tabpfn-ts-v2__test__final__padded.parquet`, run `score_all.py` and `ensemble_eval.py` (add it to a set), rebuild the submission.
2. **Overlapping cuts for the meta-ensemble:** GPU streams (`run_cuts.sh`) writing `forecasts_cuts/{model}__test__w{N}__padded.parquet` for N = 0,4,...,48 and models moirai-2.0-small, moirai-1.1-base, tirex (done), chronos-t5-small/base. **TimesFM was dropped** (stalled at about 25% CPU on its first cut). The LightGBM part is **finished** (`iterations/05-meta-ensemble/base_forecasts/lgbm_w{N}.npy`, 13 cuts).

## 8. Next steps, in order
1. Collect the finished jobs (above); copy forecasts to the repo (`tar` + `scripts/gpu_scp.sh`), commit them.
2. **Write and run the meta-learner** per `iterations/05-meta-ensemble/plan.md` (code **not written yet**: `src/meta/fforma.py` should assemble base forecasts, actuals, scales and features at each cut, train per-article softmax weights on the exact scaled-absolute-error objective, evaluate with the purged protocol, compare with the simple blend on the same test cuts, apply the adoption bar). Decision D37.
3. Evaluate **TabPFN** in the ensemble (D36) and decide the final recipe; build the final file with `src/make_submission_foundation.py` (extend its model lists).
4. A person uploads at most a handful of meaningfully different files; log each in `submissions/UPLOADS.md`. Final choice by the 4-cut evidence plus the live check, not by live differences of about 2 points or less.
5. Finish the **extra-credit** write-up (`EXTRA-CREDIT.md`); step 1.3 (other shops x arms) is not finished: `python src/run_arms.py` (45-60 min).
6. Open decisions: D34 (licences: Moirai cc-by-nc-4.0, TimesFM 3.0 and TabPFN 2.5+ non-commercial, TiRex community licence: ask the organizers if non-commercial weights are acceptable for the final submission), D36 (TabPFN weights), D37 (meta-ensemble).

## 9. Other workstreams by other sessions in the same working tree (NOT verified by the author of this file)
Another session (same GitHub handle, interleaved commits on this branch) works on: **`iterations/04-rich-columns`** (rich-column techniques, D18: shared rich schema `src/rich.py`, `src/run_rich.py`, `tests/test_rich.py`, design and plan documents; reference result at cut 0: own -4.83, pooled without rich columns -5.69) and a **loss/NN plan in `iterations/02-robust-eval`** (`notes_adamex_loss-and-nn-plan.md`, `src/experiments_loss.py` zero gate, `src/experiments_catboost.py`, results `results_adamex_loss.csv`). Read their own notes before touching those files; do not stage their uncommitted work in your commits (`git add` explicit paths, never `git add -A`). To avoid trampling each other use `git worktree` or separate clones.

## 10. Gotchas and safety
- **Never** enter passwords or tokens on behalf of the user; never upload to the leaderboard; never commit credentials, host names, raw data, environments, files over 5 MB. TabPFN's hosted API would send the shop's data to a third party: **local mode only**.
- Cleanup when finished: delete the last line of `~/.ssh/authorized_keys` on the GPU machine (key comment `hackathon-gpu`), and `~/.ssh/hackathon_gpu*` on the laptop.
- Mixed concurrent jobs on the GPU machine oversubscribe the CPU (thread storms): limit threads (`OMP_NUM_THREADS=4`) or pause a job with `kill -STOP` and resume with `kill -CONT` (done for TabPFN while Moirai ran).
- `pkill -f 'pattern'` over SSH kills your own SSH shell when the pattern appears in the command: select by PID with `ps` + `awk`.

## Iteration 08 (improvements) outcome 2026-11-21
Ten parallel improvement tasks (T1-T10) were run against the current recipe D40 and all were evaluated on the standard 4 cuts; **none passed the adoption bar** (mean gain >= +0.5 and better on >= 3 of 4 cuts), so **D40 is unchanged** (local 4-cut mean **-13.67**: -3.72 / -25.91 / -11.01 / -14.02; live -4.8). The best task was +0.08 (T3 trimmed mean) and the rest were at or below +0.07 or actively worse; the decision is recorded as **D41** in `decisions-to-make.md`, and the consolidated table is in `iterations/08-improvements/README.md`. T5 (TabPFN-TS in the ensemble) is **still blocked**: its GPU jobs stayed CPU-bound with no forecasts for about 3 h and were left running, so D36 remains unresolved. The task branches are merged on `it08/adamex-svk/improvements` and their code (`src/improve/t1_*.py` ... `t10_*.py`) and notes/results (`iterations/08-improvements/t1` ... `t10`) are kept.

New material now in the repo (forecasts under `iterations/03-foundation-model/forecasts/` and `forecasts_final/`, all new model names, D40 files untouched):
- **Intermittent-demand classics (T1):** `sf-ses`, `sf-tsb`, `sf-croston`, `sf-croston-sba`, `sf-croston-opt`, `sf-adida`, `sf-imapa` (padded context, cuts 0-3 + final).
- **Quantile / sample-count variants (T6):** `moirai-2.0-small-q40`, `moirai-2.0-small-q45`, `tirex-q40`, `chronos-t5-small-q40`, `moirai-1.1-base-s200` (SPECS are additive; defaults unchanged).
- **New zero-shot families (T10):** `chronos-2-small`, `toto-open-base`, `sundial-base-128m` (new SPECS + adapters; Toto and Sundial need their own remote venvs).
- **Fine-tuned model (T9):** `moirai-1.1-small-ft` (5 leakage-safe per-cut checkpoints).
- **Context variants (T7):** `first_sale` and `last52` forecast files for `moirai-2.0-small`, `tirex` and `chronos-t5-small`; `run_model.py` gained `--context last52` and `ensemble_recipe.py` gained `--context` (both default to `padded`, byte-identical D40).

Integration re-verified: `bash scripts/check.sh` passes, `ensemble_recipe.py evaluate --models moirai-2.0-small,tirex,chronos-t5-small` prints **-13.67**, and `build` reproduces `submissions/ensemble_top3_blend_v1.csv` byte-identically.
