# Plan: better loss / tree variants (laptop) and neural forecasters (GPU)

Status: PLAN ONLY, nothing run. Written 2026-11-21 by adamex (with an AI assistant). Decisions at the end need a collaborator's yes.

## 0. Starting point (from the logs and code)
- Reference = global LightGBM `own`, mean over the 4 cuts **-15.5**; median13 -16.5 (`iterations/02-robust-eval/log.md`).
- Bar to adopt (TEAM-RULES section 2): mean over cuts +0.5 or more AND 3 of 4 cuts won (or paired interval excludes 0). Seed noise is about 0.8 on one cut, so every config is run with 3 seeds and judged on the mean over cuts.
- **Correction to what was said in chat:** the LightGBM already uses `objective="l1"` on `units / MASE scale` (D29, `src/train_gbm.py`). Since the metric is MASE-like, L1 on the scaled target is the exact optimum for it. So "quantile 0.5 / L1" is already done, and a Tweedie/Poisson objective is *mismatched* to the metric (it predicts the mean; with 60% zeros the median is lower). Tweedie is therefore a **blend-member / diagnostic** candidate, not an expected winner.
- Already direct multi-horizon (feature `h`), so "direct vs recursive" is not an experiment.

## Part 2 (laptop, CPU): loss and tree variants
One new script `src/experiments_loss.py` (not a core file; `g.fit(..., params=)` already takes params). Same windows, features, weights, early-stopping protocol and 4 cuts as `src/experiments_02.py`. Output `results_adamex_loss.csv`, forecasts to `iterations/02-robust-eval/forecasts/loss_<variant>_cut<k>_seed<s>.npy`.

| ID | Variant | Why / expectation |
|---|---|---|
| 2.0 | Reproduce reference (l1, 3 seeds) | Anchor; must match -15.5. |
| 2.1 | LightGBM `huber` (alpha about 1 on the scaled target), `l2`, `fair` | Check if L1's flat gradient hurts tree fitting. Expect within noise. |
| 2.2 | LightGBM `tweedie` (p=1.2, 1.5) and `poisson` on scaled target, then a **median correction** (multiply by a factor fitted on the training windows so the L1 error is minimal) | Diagnostic and blend member. Expect worse alone. |
| 2.3 | **Zero gate (hurdle-lite):** classifier P(any sale in week o+h-1), forecast = L1 model output, set to 0 where P < 0.5 | Generalises the hand rule D20 (zero if no sale in last 8 weeks); principled optimum for L1 on zero-inflated data. Compare with D20 and with no gate. Most promising item of part 2. |
| 2.4 | **CatBoost** `loss_function=MAE` (or `Quantile:alpha=0.5`), same features and weights, depth 6-8, 3 seeds | Different tree algorithm, often better with noisy data. Row count may be large: first time it on one cut; if over about 10 min per fit, subsample windows (origin step 4) or run `task_type=GPU` on the GPU machine. |
| 2.5 | XGBoost `reg:absoluteerror` | Only as a blend member; skip if 2.4 shows little diversity. |
| 2.6 | Blends with **fixed equal weights** (no tuning on the cuts): LGBM+CatBoost, LGBM+CatBoost+median13, plus the best foundation model from `blend_eval.py` | Weights tuned on 4 cuts overfit (the v2 upload with a 70/30 blend scored worse live). Report weights chosen on cuts 1-3 and tested on cut 0, as in iteration 02. |

Order: 2.0, 2.3, 2.4, 2.1, 2.2, 2.6 (2.5 optional). Cost estimate: LightGBM `own` about 35 s per fit, so 4 cuts x 3 seeds x about 6 configs is roughly 1.5 h on the laptop; CatBoost is the unknown.

Leakage: no new features in part 2, so `tests/test_features.py` is untouched. The gate classifier uses the same features at the same origins.

Decision to take after part 2: adopt a variant only if it clears the bar; otherwise log it as a negative result.

## Part 3 (GPU, RTX 3090): neural forecasters trained on our data
Library: `neuralforecast` (own env `.venv-nf` with `--system-site-packages` to reuse torch 2.5.1; add `env/requirements_nf.txt` and a `nf` branch in `env/setup_gpu.sh`). Adapter `src/neural/run_nf.py` (own folder, needs only pandas/numpy/pyarrow plus the library; scoring on the laptop).

Design choices that make it comparable:
1. **Target scaling = our metric.** Divide each series by its D26 MASE scale (zero-padded history from the shop's first week), train with **MAE loss**, multiply back. That is the same objective as the LightGBM (D29).
2. **Cuts:** train from scratch for each of the 4 cuts using only weeks before the cut (reuse `contexts_for` from `src/foundation/run_model.py`); output the same parquet format into `iterations/05-neural/forecasts/{model}__test__cut{k}__padded.parquet` so `score_all.py` and `blend_eval.py` work unchanged.
3. **Dead series:** series with all-zero context get 0 (not trained on); apply the D20 zero rule as a post-processing variant, as for the foundation models.
4. **Data:** test shop alone first (own); then pooled with the other shops (D27, same own-vs-pooled question as for the GBM), only if own is competitive.
5. **Seeds:** 3 per model; average the seeds (neural nets are noisy), report single-seed and averaged.

| ID | Model | Settings to start | Notes |
|---|---|---|---|
| 3.1 | **NHITS** | input_size 52 (then 104), h=13, MAE on scaled target, max_steps 1000-3000, batch of 1024 windows | Main candidate; fast. |
| 3.2 | NBEATS | same | Sanity partner of 3.1. |
| 3.3 | **TFT** | input 52, hist exog: on_hand (log), out-of-stock flag; static: log_scale | Tests whether stock features help in a neural model. Only if 3.1 is competitive. |
| 3.4 | DeepAR / NegBin (optional) | point forecast = median of samples | Only if time; designed for counts, but the sample median is noisy. |

Time budget guess: 3.1 per cut and seed a few minutes on the 3090 (2,904 series x about 200 weeks is small). Smoke test first with `--limit 200` and 100 steps, as in `gpu-runbook.md`.

Success test: same bar as above against the LightGBM reference, and separately the blend test (neural + LightGBM, equal weights), since the likely outcome (as with the foundation models) is "worse alone, useful in a blend".

## Decisions needed (not yet in `decisions-to-make.md`)
- **D38 (proposed): add a neural-forecaster family** (iteration 05, files under `src/neural/` and `iterations/05-neural/`). Evidence: foundation models lost, the GBM leads by only about 1 point over the median, and the blend with rounded foundation forecasts helped. Rec.: yes, 3.1 and 3.2 first.
- **D39 (proposed): CatBoost and the zero gate as options** next to the LightGBM (new optional code, defaults unchanged). Rec.: yes.
- Who owns part 3 on the GPU machine (one person logs in; the assistant never types the password) and how long the GPU is available (open question from `next-steps.md` section C).

## Not planned, with reasons
Tweedie as the main model (mismatched to the metric), ExtraTrees/RandomForest (weaker than boosting; maybe a blend member later), PatchTST/TSMixer (rarely beat GBMs on retail), Croston/SBA/TSB (cheap extra blend members for dead articles; add if 2.3 shows weakness there).
