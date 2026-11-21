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

## Results: items 2.0 and 2.3 (run 2026-11-21, `src/experiments_loss.py`, log `logs/loss_2_0_2_3.txt`, data `results_adamex_loss.csv`)
Test shop, 4 cuts, D26 score, 3 seeds. Single seeds = mean over seeds; "avg" = average of the 3 seeds' forecasts (gate probabilities averaged too).

| Variant | cut0 | cut1 | cut2 | cut3 | mean |
|---|---|---|---|---|---|
| l1 (reference, seeds) | -4.54 | -28.73 | -13.03 | -15.50 | -15.45 |
| l1 + D20 rule | -4.49 | -28.71 | -13.03 | -15.50 | -15.43 |
| l1 + gate (P<0.5 -> 0) | -3.48 | -28.50 | -12.95 | -15.33 | **-15.06** |
| l1 seed average | -4.47 | -28.68 | -13.00 | -15.45 | -15.40 |
| l1 + gate, seed average | -3.36 | -28.49 | -12.89 | -15.28 | **-15.01** |

- Gate gain over reference: +0.39 on the mean (seed average +0.39), wins 4/4 cuts, but about 2/3 of the gain is cut 0 (+1.0); cuts 1-3 gain +0.1 to +0.2. **Below the +0.5 bar of TEAM-RULES section 2, so not adopted.** Paired article intervals not yet computed.
- D20 hand rule adds only +0.02; the gate adds nothing on top of it (identical numbers), so the gate subsumes D20.
- The reference reproduces the logged cut-0 score (-4.83 for seed 0) and the 4-cut mean (-15.5).
- Not done yet: gate threshold sensitivity, paired intervals, CatBoost (2.4), 2.1, 2.2, blends 2.6.

## Results: items 2.1 and 2.2, the remaining loss checks (run 2026-11-21 by kilo; `src/experiments_loss2.py`, log `logs/loss2_2_1_2_2.txt`, rows appended to `results_adamex_loss.csv`)
Same windows/features/weights/early-stopping as item 2.0, target = units / MASE scale, every objective early-stops on the validation **L1** (not on its own default metric), 4 cuts, seeds averaged. `tweedie`/`poisson` are multiplied by a median-correction factor fitted on the training windows (weighted median of y/yhat, which minimises the L1 error of the corrected forecast).

| Variant | cut0 | cut1 | cut2 | cut3 | mean | gain vs l1 | cuts won |
|---|---|---|---|---|---|---|---|
| l1 (reference, seeds) | -4.54 | -28.73 | -13.03 | -15.50 | **-15.45** | - | - |
| huber (alpha 1) | -9.99 | -34.19 | -17.45 | -20.12 | -20.44 | -4.98 | 0/4 |
| fair | -10.38 | -32.84 | -17.35 | -19.44 | -20.00 | -4.55 | 0/4 |
| l2 | -52.46 | -70.60 | -50.21 | -46.89 | -55.04 | -39.59 | 0/4 |
| tweedie 1.2 + median | -27.85 | -46.17 | -27.45 | -25.98 | -31.86 | -16.41 | 0/4 |
| tweedie 1.5 + median | -25.22 | -43.42 | -23.37 | -23.00 | -28.75 | -13.30 | 0/4 |
| poisson + median | -31.12 | -49.82 | -32.24 | -28.91 | -35.52 | -20.07 | 0/4 |

- **Every alternative is much worse, on every cut.** The reason is the one stated in Part 1: the metric is L1 on `units / scale`, and L1 is already the exact optimum; the squared-type losses chase the heavy tail of the scaled target (l2 -40 points), and tweedie/poisson predict the mean of a 60%-zero target (the median correction is not enough).
- **Seed coverage:** huber/fair/l2 have 3 seeds on all 4 cuts. tweedie/poisson have 3 seeds on cuts 0-1 and **1 seed on cuts 2-3**, because they run to the 3000-round cap at 4-10 min per fit; the 13-25 point gap dwarfs the ~0.8 seed noise, so extra seeds cannot change the conclusion. This is the only deviation from "4 cuts x 3 seeds".
- **Conclusion: negative result, as predicted. Do not change `src/train_gbm.py` defaults.** The l1 + zero-gate result (item 2.3, -15.06 mean) is still the best alternative, and still below the +0.5 bar. Full write-up of the loss and neural work: `iterations/06-neural/notes_kilo_neural.md`.
