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

## Results: item 2.4 CatBoost MAE and equal-weight blends (`src/experiments_catboost.py`, log `logs/catboost_2_4.txt`, data `results_adamex_catboost.csv`)
CatBoost: MAE loss, depth 8, lr 0.08, same windows/features/weights/early stopping as the LightGBM; 87-236 trees, 35-190 s per fit on the laptop (no GPU needed). Scores are the mean over 3 seeds per cut; blends use fixed equal weights (not tuned); gate = the LightGBM classifier from 2.3 (P<0.5 -> 0).

| Variant | cut0 | cut1 | cut2 | cut3 | mean |
|---|---|---|---|---|---|
| lgbm (reference) | -4.54 | -28.73 | -13.03 | -15.50 | -15.45 |
| catboost alone | -5.31 | -30.30 | -13.14 | -15.62 | -16.09 |
| lgbm + gate | -3.48 | -28.50 | -12.95 | -15.33 | -15.06 |
| lgbm + catboost (1/2 each) | -4.76 | -29.42 | -13.03 | -15.47 | -15.67 |
| lgbm + catboost + median13 (1/3 each) | -4.91 | -29.46 | -12.40 | -15.54 | -15.58 |
| lgbm + catboost + gate | -3.88 | -28.85 | -12.89 | -15.20 | -15.21 |
| lgbm + catboost + median13 + gate | -3.77 | -28.47 | -12.18 | -15.21 | **-14.91** |

- CatBoost is worse than LightGBM on every cut (-0.6 on the mean) and adds nothing in a plain 50/50 blend (-15.67). It is not a better model here.
- Best combination (lgbm + catboost + median13, then gate): +0.54 over the reference, 4/4 cuts won, but only +0.15 over lgbm + gate alone. The gain beyond the gate comes mainly from the median13 component (cut 2: +0.8); CatBoost's own share is unclear (the lgbm+median13+gate blend was not run).
- Caveat: this is the best of 8 combinations looked at on the same 4 cuts, so the +0.54 is optimistic; it needs the paired intervals and a check against lgbm+median13+gate before any adoption.

## Results: new bases inside the winning foundation blend (`src/foundation/ensemble_base_swap.py`)
Blend = w * base + (1 - w) * mean(moirai-2.0-small, tirex, chronos-t5-small) with 8-week zero rule, rounded; w chosen on cuts 1-3 (0.3 for every base). Mean over 4 cuts:

| Base | Mean | Gain vs submitted recipe | Cuts won |
|---|---|---|---|
| lgbm seed 0 (submitted, reproduces -13.67) | -13.67 | - | - |
| lgbm 3-seed | -13.65 | +0.01 | 2/4 |
| lgbm + gate | -13.59 | +0.07 | 2/4 |
| lgbm + median13 + gate (1/2) | -13.61 | +0.05 | 2/4 |
| 3-seed lgbm, gate applied to the FINAL blend | **-13.36** | +0.31 | 3/4 (cut 0 +1.14, cut 2 -0.25) |

Swapping the base changes nothing (all within noise): the foundation ensemble already supplies what the median13 and the gate add to the LightGBM. The gate applied to the final blend helps (+0.31; the ensemble alone with the gate: -14.05 -> -13.60), mostly on cut 0; below the +0.5 bar.

### Paired intervals for the gate on the final blend (`src/foundation/ensemble_gate_paired.py`)
Gate vs no gate on the same 0.3/0.7 blend, score points [95% bootstrap over articles], cuts 0-3: +1.07 [+0.30,+2.22], +0.11 [-0.27,+0.46], -0.28 [-0.92,+0.23], +0.27 [-0.81,+1.62]. Interval excludes 0 only on cut 0. Threshold sensitivity (info only, not tuned): mean 0.3: -13.42, 0.4: -13.35, 0.5: -13.36, 0.6: -13.54, 0.7: -15.23 (a flat plateau up to 0.6, then a sharp loss). Conclusion: not adopted; evidence is one cut.

## Submission file built: `submissions/ensemble_top3_gate_v1.csv` (`src/make_submission_gated.py`, not uploaded)
Recipe: round(gate(0.3 * LightGBM 3-seed + 0.7 * top3 foundation average with 8-week zero rule)), gate = LightGBM classifier P(sale) < 0.5 -> 0. LightGBM and classifier trained on all test-shop history (reg 208-361 trees, clf 129-161). Passes `make_submission.validate_submission`. Differs from `ensemble_top3_blend_v1.csv` (live -4.8) by 0.17 units per article-week (corr of article means 0.999); 3.1% of cells are gated to 0. Local expectation: about +0.3 on the 4-cut mean, only cut 0 clear (see paired intervals above), so the live change is expected to be small and could be zero or negative. Upload and `UPLOADS.md` entry are for a person.

### Live result of `ensemble_top3_gate_v1.csv` (reported by the user, 2026-11-21): **-5.6** (weeks 1-7), versus -4.8 for `ensemble_top3_blend_v1.csv`: 0.8 points WORSE. This matches the warning from the local evidence (gain on the gate was one cut, cut 0, not significant elsewhere): the gate on the final blend is not adopted, the live recipe stays `ensemble_top3_blend_v1` (-4.8). Lesson: a gain that rests on one of four cuts did not transfer; same pattern as the median blend v2 (-8.4). Not yet logged in `submissions/UPLOADS.md` (a person does that).

## Part 3: NHITS (run and scored by the other session, `results_kilo_neural.csv`) and NHITS inside D40
NHITS own arm, 3 seeds: 4-cut mean -15.30 (LightGBM -15.12, median13 -16.54). In D40 with half the LightGBM weight (0.15 L + 0.15 NHITS + 0.7 FM, fixed in advance, `src/neural/nhits_in_d40.py`): -13.62 vs D40 -13.67, paired vs D40 +0.26 [+0.02,+0.48], -0.08, -0.06, +0.04: below the bar. Equal thirds -14.14 (worse).
Submission file built for a live check at the user's request: `submissions/ensemble_top3_nhits_v1.csv` (`src/neural/make_submission_nhits.py`, NHITS on all history, 3 seeds, 1000 steps; final forecasts in `iterations/06-neural/forecasts_final/`). Passes validation; differs from `ensemble_top3_blend_v1` (-4.8) by 0.14 units per article-week, 8.9% of cells changed; mean forecast 7.75 vs 7.76. Not uploaded by the assistant.
Live result of `ensemble_top3_nhits_v1.csv` (reported by the user): **-5.2** (weeks 1-7), 0.4 worse than D40 (-4.8). Second small D40 variant in a row that lost live (gate -5.6). Conclusion: local gains under about 0.3 do not transfer; stop tuning D40 variants.

## Results: items 2.1 and 2.2, the remaining loss checks (run 2026-11-21 by kilo; `src/experiments_loss2.py`, log `logs/loss2_2_1_2_2.txt`, rows appended to `results_adamex_loss.csv`)
Same windows/features/weights/early-stopping as item 2.0, target = units / MASE scale, every objective early-stops on the validation **L1** (not on its own default metric), 4 cuts, seeds averaged. `tweedie`/`poisson` are multiplied by a median-correction factor fitted on the training windows (weighted median of y/ŷ, which minimises the L1 error of the corrected forecast).

| Variant | cut0 | cut1 | cut2 | cut3 | mean | gain vs l1 | cuts won |
|---|---|---|---|---|---|---|---|
| l1 (reference, seeds) | -4.54 | -28.73 | -13.03 | -15.50 | **-15.45** | - | - |
| huber (alpha 1) | -9.99 | -34.19 | -17.45 | -20.12 | -20.44 | -4.98 | 0/4 |
| fair | -10.38 | -32.84 | -17.35 | -19.44 | -20.00 | -4.55 | 0/4 |
| l2 | -52.46 | -70.60 | -50.21 | -46.89 | -55.04 | -39.59 | 0/4 |
| tweedie 1.2 + median | -27.85 | -46.17 | -27.45 | -25.98 | -31.86 | -16.41 | 0/4 |
| tweedie 1.5 + median | -25.22 | -43.42 | -23.37 | -23.00 | -28.75 | -13.30 | 0/4 |
| poisson + median | -31.12 | -49.82 | -32.24 | -28.91 | -35.52 | -20.07 | 0/4 |

- **Every alternative is much worse, on every cut.** The reason is the one stated in Part 1: the metric is L1 on `units / scale`, and L1 is already the exact optimum; the squared-type losses chase the heavy tail of the scaled target (l2 -40 points), and tweedie/poisson predict the mean of a 60%-zero target (median correction not enough).
- **Seed coverage:** huber/fair/l2 have 3 seeds on all 4 cuts. tweedie/poisson have 3 seeds on cuts 0-1 and **1 seed on cuts 2-3**, because they run to the 3000-round cap at 4-10 min per fit; the 13-25 point gap dwarfs the ~0.8 seed noise, so extra seeds cannot change the conclusion. This is the only deviation from "4 cuts x 3 seeds".
- **Conclusion: negative result, as predicted. Do not change `src/train_gbm.py` defaults.** The l1 + zero-gate result (item 2.3, -15.06 mean) is still the best alternative, and still below the +0.5 bar.
