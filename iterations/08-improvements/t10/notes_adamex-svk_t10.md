# T10 - New zero-shot models (Datadog Toto, THUML Sundial, Chronos-2-small)

Date: 2026-11-21. Branch `it08/adamex-svk/t10`. Verdict: **REJECT all three** (no recipe change).
D40 still -13.67 (-3.72 / -25.91 / -11.01 / -14.02). All new forecasts (cuts 0-3 + final, 2,904 x 13)
are saved under new model names and are ready if a later task wants them.

## 1. Candidate inventory

Already present in `src/foundation/models.py` before this task (not duplicated): `chronos-t5-{small,base,large}`,
`chronos-bolt-{small,base}`, `chronos-2` (`amazon/chronos-2`, 120M), `timesfm-2.5`,
`moirai-{1.0-large,1.1-base,1.1-large,moe-small,moe-base,2.0-small}`, `tirex`, `tabpfn-ts-v2`, the two dummies.
The plan named Chronos-2-small, which is a **different 28M checkpoint** (`autogluon/chronos-2-small`), so it is a new entry.

| Candidate | Status | Licence | Why |
|---|---|---|---|
| `autogluon/chronos-2-small` (28M) | **ADDED** | Apache-2.0 | same `chronos-forecasting 2.3.2` env as Chronos-2; no token; 28M vs the 120M already used |
| `Datadog/Toto-Open-Base-1.0` (151M) | **ADDED** | Apache-2.0 (weights and `toto-ts` code) | own venv; package pins torch 2.7.0 but runs on the machine's 2.5.1 (see env notes) |
| `thuml/sundial-base-128m` (Timer 3.0, 128M) | **ADDED** | Apache-2.0 (weights and remote code) | own venv with `transformers==4.40.1` as the model card requires |
| TabPFN 2.5 / 3.5 weights | SKIPPED | Prior Labs licence | needs an account, licence acceptance and a token - same blocker as iteration 03 (only ungated v2 is in the repo) |
| Nixtla TimeGPT | SKIPPED | proprietary hosted API | needs an API token, sends data to a third party |
| TimesFM 3.0 (`timesfm3`) | SKIPPED | weights **non-commercial** | licence not acceptable for a shop forecast; TimesFM 2.5 already covers the family |
| `ibm-granite/granite-timeseries-ttm-r2` | SKIPPED (not attempted) | Apache-2.0, ungated | `tsfm_public` is git-only (not on PyPI) and r2 zero-shot is per-frequency revision; not worth the env risk for the expected gain in the time budget |
| `time-series-foundation-models/Lag-Llama` | SKIPPED (not attempted) | Apache-2.0, ungated | old GluonTS inference path and a 32-step practical context; the added diversity was unlikely to pay for the environment work |
| Moirai / Moirai-MoE other sizes | NOT NEW | cc-by-nc-4.0 (research only) | same families already in SPECS |

No candidate needed an account, licence acceptance, password or token; nothing was uploaded anywhere.

## 2. Code changes (`src/foundation/models.py`, additive only)

New SPECS: `chronos-2-small` (family chronos, kind c2), `toto-open-base` (toto, samples=256, chunk=64),
`sundial-base-128m` (sundial, samples=32, chunk=64). New loaders/forecasters use lazy imports and are
registered in `_LOADERS` / `_FORECASTERS`; `CHUNK` gained `toto`/`sundial`. Existing entries and defaults
are untouched.

* **Toto:** `Toto.from_pretrained(...).to(device)`, `TotoForecaster(model.model).forecast(MaskedTimeseries(...),
  prediction_length=13, num_samples=256)`; output = `forecast.median` over 256 sample paths (their published
  point forecast). One variate per series, left padding masked, real POSIX weekly timestamps (interval 604800 s)
  and `id_mask = 0`. `samples_per_batch` is set so it divides `num_samples` (the package asserts this).
* **Sundial:** `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`, `generate(X, max_new_tokens=13,
  num_samples=32)` -> `[batch, samples, horizon]`, median over samples; contexts are grouped by length because
  batches must be rectangular (Sundial left-pads to a multiple of its 16-step patch internally).
* Both are clipped at 0 by the shared `forecast()` wrapper.

## 3. Environments on the GPU machine (in `~/hack-t10`, own copies/symlinks)

```
~/hack-t10/src                       # copy of this worktree's src (not ~/hack/src)
~/hack-t10/data -> ~/hack/data       # symlink
~/hack-t10/.venv-{chronos,sundial,toto} -> ~/hack/.venv-{...}
~/hack/.venv-toto    : python3 -m venv --system-site-packages
                       pip install -c env/constraints_gpu.txt --no-deps toto-ts==0.2.0
                       pip install -c env/constraints_gpu.txt einops jaxtyping rotary-embedding-torch beartype "gluonts[torch]==0.16.2" safetensors orjson pyarrow "huggingface_hub==0.25.2"
~/hack/.venv-sundial : python3 -m venv --system-site-packages
                       pip install -c env/constraints_gpu.txt "transformers==4.40.1" "huggingface_hub==0.25.2" "tokenizers<0.20" "pandas<3" pyarrow
```
Notes: `toto-ts` 0.2.0 pins `torch==2.7.0, numpy==1.26.4, pandas==2.2.3, huggingface_hub` unpinned; installing it
with `--no-deps` was the only way to keep the machine's torch 2.5.1 (`-c constraints_gpu.txt` would make the pin
conflict). It runs on torch 2.5.1 + numpy 2.1.2 + pandas 2.3.3. Its `ModelHubMixin` breaks with
`huggingface_hub` 2.x (`_from_pretrained() missing proxies/resume_download`), so hub is pinned to 0.25.2 in that venv.
Sundial's remote code fails on `transformers` 4.57 (`DynamicCache.seen_tokens` removed), hence the 4.40.1 venv from
the model card. No pip output was hidden; the five TabPFN processes were never touched.

## 4. Runs (GPU, `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`)

Five runs per model (cuts 0-3 + `--final`), 3 model streams in parallel, one process at a time per stream:

| model | load s (5 runs) | forecast s (5 runs) | notes |
|---|---|---|---|
| chronos-2-small | 33.6 | 24.7 | chronos env |
| sundial-base-128m | 22.3 | 125.0 | ~25 s/cut |
| toto-open-base | 33.5 | 404.1 | ~55-95 s/cut with 256 samples |

Wall ~20 min total (15:30-15:50 UTC). Every file: 2,904 articles x 13 weeks, no NaN, no negative values
(the wrapper clips; Toto's raw medians had some -1e-4). Series with an all-zero history were not sent to
the model by `run_model.py` and got 0. No leakage: all cuts use `run_model.py --cut-index`, which only uses
weeks before the holdout.

## 5. Results on the 4 standard cuts (same-metric as D40)

`src/foundation/score_all.py`, post-processing `round+zero8` (best of raw/round/zero8/round+zero8 for each):

| model alone (round+zero8) | cut0 | cut1 | cut2 | cut3 | mean |
|---|---|---|---|---|---|
| sundial-base-128m | -4.48 | -29.83 | -14.43 | -17.25 | **-16.50** |
| chronos-2-small | -8.14 | -36.63 | -18.94 | -30.29 | -23.50 |
| toto-open-base | -12.38 | -41.29 | -31.76 | -18.61 | -26.01 |
| median-of-13 reference | -6.68 | -30.53 | -12.59 | -16.34 | -16.54 |

`ensemble_recipe.py evaluate` (0.3 x LightGBM + 0.7 x models, zero rule, round, clip):

| model | variant | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 | cuts won | verdict |
|---|---|---|---|---|---|---|---|---|---|
| chronos-2-small | alone | -5.23 | -30.47 | -14.70 | -21.13 | -17.88 | -4.22 | 0/4 | reject |
| chronos-2-small | 4th member | -3.80 | -26.05 | -11.19 | -14.04 | -13.77 | -0.10 | 0/4 | reject |
| chronos-2-small | replace moirai-2.0-small | -4.26 | -26.85 | -11.75 | -14.68 | -14.39 | -0.72 | 0/4 | reject |
| chronos-2-small | replace tirex | -4.10 | -26.35 | -11.09 | -14.62 | -14.04 | -0.37 | 0/4 | reject |
| chronos-2-small | replace chronos-t5-small | -3.88 | -26.41 | -11.33 | -14.95 | -14.14 | -0.47 | 0/4 | reject |
| sundial-base-128m | alone | -3.32 | -28.42 | -12.38 | -15.76 | -14.97 | -1.30 | 1/4 | reject |
| sundial-base-128m | 4th member | -3.42 | -26.31 | -11.11 | -14.31 | -13.79 | -0.12 | 1/4 | reject |
| sundial-base-128m | replace moirai-2.0-small | -3.75 | -26.19 | -11.22 | -14.05 | -13.80 | -0.14 | 0/4 | reject |
| sundial-base-128m | replace tirex | -3.64 | -26.70 | -11.64 | -14.37 | -14.09 | -0.42 | 1/4 | reject |
| sundial-base-128m | replace chronos-t5-small | -3.28 | -26.72 | -11.24 | -14.32 | -13.89 | -0.22 | 1/4 | reject |
| toto-open-base | alone | -11.07 | -39.64 | -28.33 | -17.15 | -24.05 | -10.38 | 0/4 | reject |
| toto-open-base | 4th member | -3.40 | -26.15 | -11.89 | -14.11 | -13.89 | -0.22 | 1/4 | reject |
| toto-open-base | replace moirai-2.0-small | -3.32 | -26.35 | -12.41 | -14.06 | -14.04 | -0.37 | 1/4 | reject |
| toto-open-base | replace tirex | -3.32 | -26.67 | -12.34 | -14.18 | -14.13 | -0.46 | 1/4 | reject |
| toto-open-base | replace chronos-t5-small | -2.97 | -26.57 | -12.20 | -14.40 | -14.04 | -0.37 | 1/4 | reject |
| all three new models + D40 | 6 members | -3.47 | -26.45 | -11.86 | -14.23 | -14.00 | -0.34 | 1/4 | reject |
| sundial + chronos-2-small + D40 | 5 members | -3.72 | -26.37 | -11.18 | -14.12 | -13.85 | -0.18 | 1/4 | reject |

Reference D40 reproduced four times during these runs: -3.72 / -25.91 / -11.01 / -14.02, mean -13.67.
No parameter was tuned, so no leave-one-cut-out table is needed.

## 6. Honest verdict and recommendation

* Nothing reaches the +0.5 / 3-of-4 bar. Best single change is the 4th-member variants at **-0.10 / -0.12**
  (noise, and 0-1 cuts won). Replacements are all negative. Stacking all three is worse (-0.34).
* Sundial is the only interesting new member: best solo score of the three and closest to neutral inside the blend.
  Toto is trained on observability metrics (high-frequency, continuous) and is badly calibrated for 60-96%
  zero weekly counts; Chronos-2-small is dominated by its 120M sibling already in the repo.
* **Recipe change: none.** D40 stays as is. New forecasts are saved for reuse if a future task wants
  a different weighting or a per-series selection; do not add these models to the fixed average.

## 7. Files, licence record, runtime

Added forecast files (new names only):
`iterations/03-foundation-model/forecasts/{chronos-2-small,sundial-base-128m,toto-open-base}__test__cut{0,1,2,3}__padded.{parquet,json}`
and `iterations/03-foundation-model/forecasts_final/{...}__test__final__padded.{parquet,json}` (30 files, each < 300 KB).
Code: `src/foundation/models.py` (additive SPECS/adapters only).
Outputs: `iterations/08-improvements/t10/results_adamex-svk_t10.csv`, this file.
Licences of the used models: Chronos-2-small Apache-2.0; Toto-Open-Base-1.0 + `toto-ts` Apache-2.0;
Sundial-base-128m Apache-2.0. Libraries: transformers 4.40.1 Apache-2.0, `toto-ts` 0.2.0 Apache-2.0.
Runtime: GPU wall ~20 min, local scoring/evaluation ~3 min, environment setup ~15 min.
No submission was built (nothing passed; per TEAM-RULES a person uploads, and we do not upload).

## 8. Failures / suspicious things

* Toto: the `toto-ts` dependency pins (torch 2.7.0, numpy 1.26.4, pandas 2.2.3) conflict with the machine;
  `--no-deps` + constrained manual deps works and the runs are numerically sane. Re-check before reusing.
* Sundial: unpatched on transformers >=4.5x (DynamicCache API change) - own venv required. The remote
  code auto-downloads from the HF repo (not pinned to a revision); we pin nothing, so a future repo update
  could change results. Licence and repo are unchanged today.
* SSH launches with `setsid nohup ... &` sometimes kept the SSH channel open until the client timeout even
  though the remote job had started; verified each run through `ps` and the logs, no duplicate jobs were left.
* At the end of the task no `run_model` processes were running; the long-lived TabPFN processes were gone
  before our last check and were not stopped by us.
