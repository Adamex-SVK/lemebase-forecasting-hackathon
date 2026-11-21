# Iteration 08: action plan for parallel improvement tasks

**For the orchestrator.** Spin up **one worker per task (T1-T10)**. Give each worker the whole of section A (shared brief) **plus** its own task section. Tasks are independent unless a task says *Depends on*. Local tasks can run in parallel. **GPU tasks (T6, T7, T9, T10) must run one at a time** on the single GPU machine, after checking nothing else heavy is running there. Afterwards, run **T0 (integration)** yourself or with one worker.

Priority order if capacity is limited: **T2, T1, T3, T4, T5** (cheap, local, attack what is known to work), then T8, T6, T7, T10, T9.

---

## A. Shared brief (paste into every worker)

### A1. The project in five lines
- Hackathon: forecast **weekly units** for **2,904 articles** of a test shop, **13 weeks ahead** (weeks 2023-04-24 to 2023-07-17). Score = **(1 - MASE) x 100**, higher is better; per-article MASE with the article's own scale (mean absolute week-to-week change of its zero-padded history). Live leaderboard = weeks 1-7, final = weeks 8-13.
- Repo: `/Users/adomi/Documents/hackathlon`, branch `it03/adamex-svk/foundation-models`. **Read `HANDOVER.md` first**, then `TEAM-RULES.md` and `iterations/07-ensemble-toolkit/README.md`.
- **Current best recipe (D40):** average the raw forecasts of **moirai-2.0-small + tirex + chronos-t5-small** -> 0 for articles with no sale in the last 8 weeks (zero rule) -> `0.3 x LightGBM + 0.7 x ensemble` -> round to whole units -> clip at 0. **Local 4-cut mean -13.67** (cuts 0/1/2/3: -3.72 / -25.91 / -11.01 / -14.02). Live **-4.8** (LightGBM alone -7.6, Moirai 2.0-small alone -7.3, median of last 13 weeks -9.8).
- Known: rounding and the zero rule help a lot; averaging diverse models helps; learned per-article weights (FFORMA) did **not** beat one global weight; pooling other shops into the LightGBM did not help; seasonality features did not help; window difficulty varies hugely between cuts.

### A2. The yardstick (non-negotiable)
- Score only on the **4 standard cuts** of the test shop (cut K = last 13*K weeks removed, K = 0..3; holdout = the next 13 weeks), metric D26 (`src/validate.py`: `split`, `score`; helpers in `src/experiments_02.py`: `cut_frame`, `score_F`; `src/foundation/score_all.py`: `prepare_cut`, zero rule `variant`).
- **Reference = D40 on the same cuts: -13.67.** To compare a model set with D40 use the existing tool:
  ```bash
  .venv/bin/python src/meta/ensemble_recipe.py evaluate --models moirai-2.0-small,tirex,chronos-t5-small[,new-model] [--lgbm-weight 0.3]
  ```
  It reads practice-cut forecasts from `iterations/03-foundation-model/forecasts/{model}__test__cut{K}__padded.parquet` (columns `store_id, article_id, week, forecast`; test shop `store_id = "T"`) and the LightGBM from `iterations/02-robust-eval/forecasts/base_cut{K}_seed0.npy`.
- **Adoption bar:** mean gain over D40 **>= +0.5** AND better on **>= 3 of 4 cuts**. Below that it is noise (seed noise alone is about 0.8 on one cut).
- **No tuning on the scored cuts.** Any parameter you choose (threshold, weight, offset) must be chosen **leave-one-cut-out**: for each cut K, pick the parameter on the other three cuts, score cut K with it, and report the mean of those four honest scores. Also report the in-sample optimum for information, clearly labelled.
- **No leakage:** anything computed for cut K may use only weeks before that cut's first holdout week.

### A3. Working setup (so parallel workers do not trample each other)
```bash
cd /Users/adomi/Documents/hackathlon
git worktree add ../hack-<task-id> -b it08/<your-handle>/<task-id> it03/adamex-svk/foundation-models
cd ../hack-<task-id>
ln -s /Users/adomi/Documents/hackathlon/.venv .venv                                  # Python 3.13 env (use .venv/bin/python)
ln -s /Users/adomi/Documents/hackathlon/hackathon-retail-sets hackathon-retail-sets  # raw data (read only)
ln -s /Users/adomi/Documents/hackathlon/data data                                    # weekly tables (read only)
bash scripts/check.sh                                                                # tests must pass before you start
```
- Put your code in `src/improve/<task-id>_*.py`, your outputs in `iterations/08-improvements/<task-id>/` (`results_<handle>_<task-id>.csv`, `notes_<handle>_<task-id>.md`). New forecast files for new models go to `iterations/03-foundation-model/forecasts/` and `forecasts_final/` with **new model names** (never overwrite existing files).
- **Do not modify** `src/validate.py`, `src/prepare.py`, `src/features.py`, `HANDOVER.md`, `decisions-to-make.md`, any `log.md`, or other people's files. You may *add options* to `src/meta/ensemble_recipe.py` or `src/foundation/*.py` if the default behaviour stays byte-identical (prove it: `ensemble_recipe.py evaluate` still prints -13.67 for D40, and `build` still reproduces `submissions/ensemble_top3_blend_v1.csv` exactly).
- Commit only your own files with explicit paths (never `git add -A`); push your branch. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or your model's line).
- **Never upload to the leaderboard**, never enter passwords or tokens, never commit credentials, host names, raw data or files over 5 MB. If a step needs a person (login, licence acceptance), stop and report.

### A4. GPU machine (only for GPU tasks)
RTX 3090, 96 cores, torch 2.5.1 preinstalled; one environment per model family in `~/hack` (`.venv-chronos`, `.venv-timesfm`, `.venv-tirex`, `.venv-moirai`, `.venv-tabpfn`). Read `iterations/03-foundation-model/gpu-machine.md` (access, layout, pitfalls) and `model-notes.md`. Access: `GPU_SSH_HOST=<ask the owner> bash scripts/gpu_ssh.sh '<cmd>'` and `scripts/gpu_scp.sh` (key `~/.ssh/hackathon_gpu`, key-only). **Work in your own remote folder** so you never change files another job uses: `cp -r ~/hack/src ~/hack-<task-id>/src`, symlink `~/hack/data` and the `~/hack/.venv-*` folders into it, run from there. Set `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`. Check `ps -eo pid,etime,pcpu,args | grep run_model` first; **do not kill other people's processes**. In non-interactive SSH use `/opt/conda/bin/python3` (plain `python3` is not on the PATH). Install packages only with `-c env/constraints_gpu.txt` (keeps torch 2.5.1, `transformers<5`, `pandas<3`).

### A5. Report back (to the orchestrator)
1. A table: variant | cut 0 | cut 1 | cut 2 | cut 3 | mean | gain vs D40 | cuts won | verdict (adopt / reject), honest (leave-one-cut-out) numbers where parameters were chosen.
2. If adopted: the exact recipe change and, if final-week forecasts exist, a submission CSV built with `src/make_submission.py` validation under `submissions/<task-id>_v1.csv` (**not uploaded**).
3. Branch name, commit hash, files added, runtime, anything that failed or looked suspicious.

---

## T1. Intermittent-demand classics as blend members (local, low cost, high priority)
**Goal.** Add forecasts that are built for zero-heavy demand and are very different from the current members.
**Steps.**
1. Write `src/improve/t1_classics.py`. For each cut K = 0..3: history = `cut_frame(full, K)`, then `train, hold, cut = validate.split(df)` (zero-padded history, D26); for `--final`: the full history padded with `validate.pad_to_shop_start(full, keys)`. Fit with `statsforecast` (installed, 2.1.1): `CrostonClassic`, `CrostonSBA`, `CrostonOptimized`, `TSB(alpha_d=0.1, alpha_p=0.1)`, `ADIDA`, `IMAPA`, and `SimpleExponentialSmoothingOptimized` as a non-intermittent reference; horizon 13, weekly freq `W-MON`, `n_jobs=-1`.
2. Save each as a pseudo-model in the standard format: `iterations/03-foundation-model/forecasts/<name>__test__cut{K}__padded.parquet` and `forecasts_final/<name>__test__final__padded.parquet` with names `sf-croston`, `sf-croston-sba`, `sf-croston-opt`, `sf-tsb`, `sf-adida`, `sf-imapa`, `sf-ses` (2,904 articles x 13 weeks; series with an all-zero history get 0).
3. Score each alone: `.venv/bin/python src/foundation/score_all.py` (reports raw / round / zero8 / round+zero8).
4. Blend tests with `ensemble_recipe.py evaluate`: D40 + each one as a 4th member; the best two together; and **replacing** chronos-t5-small by the best classic.
**Accept** per A2. **Time:** under 1 hour.

## T2. Post-processing tuning (local, very low cost, high priority)
**Goal.** Squeeze more out of the step that already gave the biggest gains (rounding, zero rule).
**Steps.** In `src/improve/t2_postproc.py` rebuild the **un-rounded D40 blend** per cut (mean of the three models' raw forecasts -> zero rule -> 0.3 LightGBM + 0.7 ensemble), then apply and score each variant (leave-one-cut-out where a parameter is chosen):
- rounding offset: `floor(x + c)` for c in 0.30 ... 0.70 (step 0.05) (D40 = 0.5);
- shrink before rounding: `round(a * x)` for a in 0.80 ... 1.05 (step 0.05);
- zero-rule window: no sale in the last W weeks -> 0, W in {4, 6, 8, 10, 13, 26} (D40 = 8);
- small-value cut: forecasts below t -> 0 for t in {0.5, 0.75, 1.0} after the blend (before rounding);
- per-segment offset: c chosen separately for articles with mean of the last 13 weeks < 1, 1-5, > 5 (three parameters, leave-one-cut-out).
Report the honest table and the in-sample optimum. **Accept** per A2 (honest numbers only). **Time:** 1 hour.

## T3. Robust combination of the members (local, very low cost)
**Goal.** Replace the plain average of the foundation models by something less sensitive to one model going wrong on an article.
**Steps.** In `src/improve/t3_combine.py`, per cut, combine the raw forecasts of the members with: mean (D40), **median**, trimmed mean (drop min and max when >= 4 members), geometric-style mean of `log1p`, and "median of 5" with moirai-2.0-small, tirex, chronos-t5-small, chronos-t5-base, moirai-1.1-base (all available on cuts 0-3 and final). Keep zero rule, the 0.3 LightGBM weight and rounding fixed (no tuning). Also test the same combinations **including** the LightGBM as a member instead of the fixed 0.3 blend.
**Accept** per A2. **Time:** under 1 hour.

## T4. Flatten forecasts over the horizon (local, very low cost)
**Goal.** Our errors do not grow with the horizon and Moirai's week-to-week wiggle is sampling noise; a flat forecast per article may be better.
**Steps.** In `src/improve/t4_flatten.py`, per cut, take the un-rounded D40 blend and replace each article's 13 values by (a) their median, (b) their mean, (c) the median of the foundation part only (keep the LightGBM part as is), then round. No parameters, so no tuning.
**Accept** per A2. **Time:** 30 minutes.

## T5. Add TabPFN-TS to the ensemble (local once the forecasts exist; low cost)
**Depends on** the TabPFN runs started on the GPU machine (5 processes: cuts 0-3 and `--final`, model `tabpfn-ts-v2`, local mode, ungated v2 weights; see `iterations/03-foundation-model/log.md`). **Do not restart or kill them**; if they are still running, report their CPU time and stop.
**Steps.** When finished, copy `~/hack/iterations/03-foundation-model/forecasts/tabpfn-ts-v2__test__cut{0..3}__padded.parquet` and `forecasts_final/tabpfn-ts-v2__test__final__padded.parquet` into the repo (`scripts/gpu_scp.sh down`), check completeness (2,904 articles x 13 weeks each), score alone with `score_all.py`, then `ensemble_recipe.py evaluate` with tabpfn-ts-v2 added and as a replacement for each of the three members. Note: the adapter forecasts 0 for articles without a sale in 8 weeks (by design).
**Accept** per A2. **Time:** 30 minutes after the forecasts exist.

## T6. Moirai and TiRex: more samples and a lower quantile (GPU, low cost)
**Goal.** Reduce sampling noise in Moirai's median and test whether a slightly lower quantile suits right-skewed counts.
**Steps.** In your remote folder, add **new SPECS names** in `src/foundation/models.py` (default behaviour unchanged): `moirai-2.0-small-q45`, `moirai-2.0-small-q40` (use `forecast.quantile(q)`), `tirex-q40` (TiRex quantiles are 0.1 ... 0.9: index 3 = 0.4), `chronos-t5-small-q40` (`predict_quantiles(..., quantile_levels=[0.4])`), and `moirai-1.1-base-s200` (`num_samples=200`). Run cuts 0-3 and `--final` for each, copy back, then `ensemble_recipe.py evaluate` with the q-variants **replacing** their median versions (all three at q40, all three at q45, one at a time).
**Accept** per A2. **Time:** 1-2 hours.

## T7. Context variants (GPU, low cost)
**Goal.** Test whether the long zero padding in the history hurts the zero-shot models.
**Steps.** `run_model.py` already supports `--context first_sale`; add `--context last52` (keep the last 52 weeks only) as a new option (default unchanged). For moirai-2.0-small, tirex, chronos-t5-small: run cuts 0-3 and `--final` with `first_sale` and with `last52`. `ensemble_recipe.py` reads `__padded` files only: add a `--context` option to it (default `padded`, behaviour unchanged) and evaluate the three-member blend for each context, and mixed (each model with its best context, chosen leave-one-cut-out).
**Accept** per A2. **Time:** 1-2 hours.

## T8. Other tree-model members (local, medium cost)
**Goal.** The 0.3 LightGBM slot is the only trained model; a different trained model may blend better.
**Inputs.** Another session produced, **in the main checkout only (git-ignored)**, `iterations/02-robust-eval/forecasts/cb_F_cut{K}_seed{s}.npy` (CatBoost, MAE), `loss2_F_{tweedie1.2,tweedie1.5,poisson,huber,fair,l2}_cut{K}_*.npy` and `loss_F/loss_P_*` (zero gate); read their notes `iterations/02-robust-eval/notes_adamex_loss-and-nn-plan.md` and `results_adamex_*.csv` first. Rows are in `ShopMatrix` key order of the cut's training part, same as `base_cut{K}_seed0.npy`. Use them read-only from `/Users/adomi/Documents/hackathlon/iterations/02-robust-eval/forecasts/`.
**Steps.** In `src/improve/t8_trees.py` evaluate the D40 recipe with the LightGBM slot replaced by: CatBoost; the average of LightGBM and CatBoost; each `loss2_*` variant available on all 4 cuts; and the 0.3 slot split over two tree models. Fixed weights, no tuning. If one passes the bar, also produce its **final-week** forecast (train on all history, like `src/make_lgbm_final.py`) and a submission CSV.
**Accept** per A2. **Time:** 1-2 hours.

## T9. Fine-tune a small foundation model (GPU, high cost, uncertain)
**Goal.** Adapt the strongest model family to this shop instead of zero-shot use.
**Rules.** Leakage is the main risk: for cut K the model may be fine-tuned **only on data before cut K's holdout** (so four separate fine-tunes plus one on the full history for `--final`). Training data: the test shop's padded weekly series (optionally plus the other shops' series from `data/interim/weekly_{A,B,C,D}.parquet`, all weeks, as in D27).
**Steps.** Pick one route after reading `model-notes.md` (recipes there): (a) Chronos-Bolt-small through AutoGluon `TimeSeriesPredictor` with `fine_tune=True`, or (b) Moirai 1.1-small through the uni2ts CLI. Short schedules (a few hundred steps), fixed hyper-parameters chosen **before** looking at the scores. Save forecasts as a new model name (e.g. `chronos-bolt-small-ft`), then `ensemble_recipe.py evaluate` alone and as a 4th member / replacement.
**Accept** per A2. **Time:** 3-4 hours; stop and report if one fine-tune takes over 30 minutes.

## T10. New zero-shot models (GPU, medium cost)
**Goal.** More diversity from model families we have not tried.
**Steps.** Check availability, licence and Python/torch compatibility (torch 2.5.1 on the machine; separate venv if needed) for candidates such as Datadog **Toto**, THUML **Sundial**, Chronos-2-small (`autogluon/chronos-2-small`), and others you find; **skip anything that needs an account, licence acceptance or a token** (report it instead, e.g. TabPFN 2.5+ weights). For each feasible one: add a SPECS entry and adapter (median forecast, lazy imports, `VERIFY` comments), `smoke.py`, cuts 0-3 and `--final`, then `ensemble_recipe.py evaluate` (alone, as 4th member, as replacement). Record licences in your notes.
**Accept** per A2. **Time:** 2-3 hours.

---

## T0. Integration (orchestrator, after the tasks)
1. Collect the reports. List every change that passed the bar.
2. Gains do not simply add up: **stack the adopted changes and re-evaluate them together** on the 4 cuts (and leave-one-cut-out for chosen parameters). Keep the stack only if it still passes the bar against D40.
3. Build at most **two** candidate submission files (the stack, and the best single change) with validation; a **person** uploads them and logs the scores in `submissions/UPLOADS.md`. Read live differences of about 2 points or less as noise.
4. Record the outcome as a decision in `decisions-to-make.md` (who decided, when, evidence) and update `HANDOVER.md` and `iterations/07-ensemble-toolkit/README.md` with the new recipe if it changed.
5. Remove the worktrees (`git worktree remove ../hack-<task-id>`) after their branches are pushed.
