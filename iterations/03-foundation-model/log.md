# Iteration 03 log (foundation models on the GPU)

Append-only.

## 2026-11-21: GPU access received, new branch `it03/adamex-svk/foundation-models`
- The organizers provided an NVIDIA **RTX 3090** (24 GB) through a JupyterLab page with a terminal and Python (a test notebook saw the GPU), plus an optional SSH route through a Cloudflare tunnel. The URL and the password were given privately in chat; **they are intentionally not recorded in the repo**.
- An AI assistant cannot type the password, so the workflow is: a person logs in, uploads `dist/gpu_bundle.tar.gz` (built by `scripts/make_gpu_bundle.sh`), runs the commands in `gpu-runbook.md` and pastes the output back; forecasts are downloaded and scored locally with `src/foundation/score_all.py`. An optional key-based SSH setup would let an assistant run the jobs directly (see the runbook).
- RTX 3090 capacity (24 GB) is enough for every model on the list (the largest weights, Moirai-MoE-base and Chronos-T5-large, are below about 4 GB in fp32).
- Open: whether the GPU machine can reach Hugging Face and PyPI (first step of the runbook), which Python versions it has (Moirai needs 3.10/3.11), how long we may use it.
- Adapters in `src/foundation/models.py` are still untested; the smoke tests (runbook Part A, step 5) come first.

## 2026-11-21: first look at the GPU machine (output pasted by the user)
- GPU: NVIDIA GeForce RTX 3090, 24,576 MiB, idle; driver 580.178.04, CUDA 13.0. Python **3.11.10** (`python3`; there is no `/usr/bin/python3*`, so it lives elsewhere, probably a conda or image install). RAM 62 GB total (38 GB available). Disk **51 GB free of 1.8 TB (98% used)**: tight, four separate torch environments would take about 25 GB, so `env/setup_gpu.sh` now uses `uv` (shared packages through hard links) when available.
- **Internet access is unknown and probably missing:** the two `curl -sI` lines for huggingface.co and pypi.org printed nothing (`-s` hides errors). The runbook step 4 now runs a proper check (`curl -sS -m 10 -I`, proxy variables, installed packages) and has a "no internet" fallback (wheels and weights uploaded from the laptop, split into parts of about 90 MB because of the likely Cloudflare upload limit).
- Python 3.11 satisfies all model families including Moirai (needs 3.10 or 3.11), so no separate Python is needed; `PY_MOIRAI` was dropped from the setup script.

## 2026-11-21: network and software check on the GPU machine (output pasted by the user)
- **Internet works:** both `https://huggingface.co` and `https://pypi.org/simple/` answered `HTTP/2 200` (the earlier empty `curl -sI` output was a formatting problem of the pasted command, not a blocked network). The bundle was uploaded and extracted to `~/hack`.
- **Software:** conda at `/opt/conda` (python3, pip, conda; no `uv`), **torch 2.5.1+cu121 installed with CUDA available**, torchvision 0.20.1, torchaudio 2.5.1, numpy 2.1.2, JupyterLab 4.6.6; pandas, transformers, gluonts and the model packages are not installed.
- **Consequences:** Chronos, TimesFM and TiRex reuse the machine's torch through `--system-site-packages` virtual environments (no second torch, saves about 10 GB of the tight disk). **Moirai needs its own full environment** (uni2ts pins torch < 2.5 and numpy ~= 1.26). `env/setup_gpu.sh` was rewritten accordingly. TiRex additionally needs `nvcc` to compile its CUDA kernels (to be checked). The fallback for no internet is kept in the runbook but not needed.
- Next: create the `chronos` and `timesfm` environments and run the first smoke tests (runbook Part A, steps 4-5).

## 2026-11-21: environment problems on the GPU machine and the fix
- The `chronos` environment first broke: the unconstrained install pulled **torch 2.15.0** into the venv (newer `accelerate` / `transformers` want a recent torch), which clashed with the machine's **torchvision 0.20.1** (`operator torchvision::nms does not exist`). A repair with `pip install torchvision` failed halfway (OSError on a missing file in the venv) and left the venv with the machine's torch 2.5.1, which `transformers` 5.23 refuses ("PyTorch >= 2.6 is required").
- **Fix:** keep the machine's torch stack and `transformers` on 4.x. `env/constraints_gpu.txt` (`torch==2.5.1`, `torchvision==0.20.1`, `transformers<5`) is applied with `pip install -c`. A dry-run resolution for Linux / Python 3.11 (done on the laptop, nothing installed) picks transformers 4.57.6, accelerate 1.15.0, chronos-forecasting 2.3.2, torch 2.5.1; TimesFM 3.0.2 and tirex-ts 1.4.2 resolve under the same constraints. `env/setup_gpu.sh` now uses the constraints (except for Moirai, which has its own full environment) and shows pip's output instead of hiding it (`-q` had hidden the earlier failure).
- The `timesfm` environment installed fine and reports `torch 2.5.1+cu121 True`.
- Lesson: unpinned installs on a machine with a preinstalled torch stack drift to the newest torch; always constrain.

## 2026-11-21: first zero-shot results on the test shop, 4 cuts (run on the GPU machine, scored there with `score_all.py`)
- Smoke tests: all five adapters ran on the first try (chronos-bolt-small, chronos-bolt-base, chronos-2, chronos-t5-small, timesfm-2.5); forecasts on 3 toy series were sensible. Then `run_model.py` on the test shop, cuts 0-3, padded context (D26). **chronos-t5-small is missing from the scored table: its real runs probably failed; error not yet seen.**
- **Zero-shot scores (D26, raw median forecast clipped at 0), cuts 0 / 1 / 2 / 3 and mean:**

  | model | 0 | 1 | 2 | 3 | mean |
  |---|---|---|---|---|---|
  | LightGBM own (iteration 01/02, for reference) | -4.83 | -28.45 | -13.35 | -15.44 | -15.52 |
  | median of the last 13 weeks | -6.68 | -30.53 | -12.59 | -16.34 | -16.54 |
  | timesfm-2.5 | -11.21 | -34.40 | -22.18 | -17.67 | -21.37 |
  | chronos-2 | -10.84 | -36.83 | -22.88 | -27.15 | -24.43 |
  | chronos-bolt-base | -39.96 | -86.40 | -69.51 | -40.48 | -59.09 |
  | chronos-bolt-small | -38.73 | -83.93 | -68.59 | -47.42 | -59.67 |

- Reading: zero-shot TimesFM 2.5 and Chronos-2 are usable but 5-8 points behind the median on average; the Bolt models are about 40 points behind. A likely cause is the zero-heavy data (the metric punishes any non-zero forecast for an article that truly sells 0, because its scale is tiny); two cheap post-processing variants were added to test this: rounding to whole units, and 0 for articles with no sale in the last 8 weeks (`score_all.py`, columns `post`). The main hope for these models is as blend members with the LightGBM (`src/foundation/blend_eval.py`, same protocol as iteration 02: weight chosen on cuts 1-3).
- Plumbing check (laptop, dummy model): `blend_eval.py` reproduces the iteration 02 blend of LightGBM and median13 exactly (-4.96, -28.79, -12.53, -15.46; mean -15.44).
- Environment notes: Chronos and TimesFM environments work with the machine's torch 2.5.1 (see the entry above); TiRex and Moirai environments not yet created.

## 2026-11-21: post-processing results; rounding also helps the LightGBM; SSH access set up
- **Post-processing of the zero-shot forecasts (mean over the 4 test-shop cuts, D26):**

  | model | raw | zero8 | round | round + zero8 |
  |---|---|---|---|---|
  | timesfm-2.5 | -21.36 | -21.07 | -18.81 | **-18.73** |
  | chronos-2 | -24.42 | -24.03 | -23.06 | **-22.96** |
  | chronos-bolt-base | -59.09 | -58.29 | -56.56 | -56.46 |
  | chronos-bolt-small | -59.67 | -58.50 | -56.98 | -56.79 |

  (median13 -16.54, LightGBM -15.52.) Rounding to whole units is the useful part (+2.5 for TimesFM, +1.4 for Chronos-2); the zero rule adds about 0.1 on top. The Bolt models stay about 40 points behind, so rounding is not what is wrong with them. Best zero-shot model so far: TimesFM 2.5 with round + zero8, about 2.2 points behind the median, 3.2 behind the LightGBM.
- **Rounding applied to our own LightGBM forecasts (saved 4-cut forecasts, mean over cuts):** raw -15.52, **rounded -15.12** (+0.4, better on 4 of 4 cuts), `floor(x + 0.4)` -15.22, `floor(x + 0.3)` -15.36, rounded and blended 70/30 with the median -14.92 (+0.6, 4 of 4). The D33 variant behaves the same (raw -15.51, rounded -15.18). Rounding is principled for this metric (the best constant forecast of integer counts is an integer) and is a candidate for the next submission file; the blend with the median did NOT help on the live board before (v2, -8.4), so rounding alone is the cleaner test.
- **Chronos-T5-small** ran out of GPU memory (2.56 GiB allocation on a 24 GB card) because 1,024 series were processed per batch and T5-Chronos samples many paths per series. Fix: `chunk=48` in its spec (`src/foundation/models.py`, environment override `FM_CHUNK`).
- **SSH:** a dedicated key (`~/.ssh/hackathon_gpu`, no passphrase) was created on the laptop and its public key was added to the GPU machine's `authorized_keys` by the user; `cloudflared` was installed with Homebrew. The assistant connects with the key only (password authentication disabled in every command). The host name is not recorded in the repo. **When the work is done: delete the last line of `~/.ssh/authorized_keys` on the GPU machine.**

## 2026-11-21: zero-shot scores of 8 more models, and the blend test against the LightGBM (SSH runs by the assistant)
- All remaining adapters passed their smoke tests on the first try (TiRex, Moirai 2.0-small, 1.1-large, MoE-small). The full run on the test shop (4 cuts) completed for the 8 non-Moirai models; **every Moirai run failed first** with `ImportError: Unable to find a usable engine ... pyarrow`: `uni2ts` pins numpy 1.26 and `pyarrow` 26 needs numpy 2. Fix: `pyarrow<18` in the Moirai environment (added to `env/requirements_moirai.txt`), Moirai runs relaunched.
- **Zero-shot score, mean over the 4 cuts (D26), raw and after rounding + the 8-week zero rule:** tirex -18.75 -> **-15.96**; chronos-t5-small -19.50 -> -17.94; chronos-t5-base -19.73 -> -18.13; timesfm-2.5 -21.36 -> -18.73; chronos-t5-large -21.68 -> -19.58; chronos-2 -24.42 -> -22.96; chronos-bolt-base -59.09 -> -56.46; chronos-bolt-small -59.67 -> -56.79. References: median13 -16.54, LightGBM -15.52 (rounded -15.12). **TiRex with rounding and the zero rule is better than the median and close to the LightGBM** (cut 1 -27.33 and cut 3 -14.67 beat the LightGBM's -28.45 and -15.44). Moirai-2.0-small cut 0: -4.60 (round + zero8), better than the LightGBM's -4.83 (only one cut had finished).
- **Blend with the LightGBM (`src/foundation/blend_eval.py`, weight on the LightGBM chosen on cuts 1-3):**
  - raw foundation forecasts, raw blend: almost no gain (chronos-2 +0.23, all others about 0).
  - **foundation forecasts rounded + zero rule, final blend rounded, versus the rounded LightGBM alone (-15.12):** tirex (w = 0.6) **-13.91, +1.21, 4/4 cuts**; chronos-t5-small (0.6) -14.12 (+1.00, 4/4); chronos-t5-base (0.6) -14.24 (+0.88, 4/4); timesfm-2.5 (0.6) -14.31 (+0.80, 4/4); chronos-2 (0.8) -14.44 (+0.68, 4/4); chronos-t5-large (0.6) -14.62 (+0.50, 3/4); chronos-bolt-small (0.9) -14.57 (+0.55, 4/4); chronos-bolt-base (0.9) -14.66 (+0.46, 4/4). Every model gives a gain once everything is rounded, including the weak Bolt models, so the effect is diversification plus whole-unit forecasts, not one good model. All but one clear the adoption bar of TEAM-RULES.md (>= 0.5 points mean and >= 3 of 4 cuts).
  - Caveats: the weight is picked on 3 of the 4 cuts and the best model is picked among 8 (small optimism); the live board did not reward the earlier blend of the LightGBM with the median (v2, -8.4), so a live check is needed before trusting it.
- Added `--final` to `src/foundation/run_model.py` (forecast the real 13 weeks after 2023-04-17 from the full history, output in `forecasts_final/`); checked locally with the dummy model (2,904 articles, 2023-04-24 to 2023-07-17). Final-mode runs for tirex, timesfm-2.5, chronos-2 and chronos-t5-small/base/large started on the GPU machine, next to the relaunched Moirai runs.

## 2026-11-21: TabPFN-TS added; Moirai status
- **TabPFN-TS (local mode, ungated v2 weights, telemetry off)** installed in `.venv-tabpfn` (reusing torch 2.5.1; `tabpfn` 10.2.0, `tabpfn-time-series` 1.4.0), smoke test passed on the first try. Speed: **about 1 second per series** (200 series: 199 s). On the same 200 active series at cut 0 (non-random subset: first 200 active articles by id; score on this subset only): tabpfn-ts-v2 raw 6.97 (round + zero8 7.07), tirex -0.51 (0.68), timesfm-2.5 3.72 (5.15), chronos-t5-base 3.02 (3.29), chronos-2 -12.78 (-11.90), median13 8.97, LightGBM 10.22 (10.37). So TabPFN is the best zero-shot model on this subset, close to the median and the LightGBM; a promising sign, not a result.
- To make it affordable the adapter does not call the model for series with no sale in the last 8 weeks (forecast 0, the baselines' zero rule). Full runs for cuts 0-3 and the real-future (`--final`) forecast started in parallel (5 processes; the machine has 96 cores and 62 GB RAM, GPU memory use is small).
- Moirai status at 11:24: 4 of 6 models finished on all 4 cuts (2.0-small, 1.1-base, 1.1-large, MoE-small), MoE-base running, then 1.0-large. Real-future forecasts finished for tirex, timesfm-2.5, chronos-2, chronos-t5-small/base/large (`forecasts_final/`).

## 2026-11-21 (afternoon): Moirai results, ensemble evaluation, first foundation-model submission files
- **All six Moirai models finished on 4 cuts.** Moirai 2.0-small (11M parameters) is the best single zero-shot model: round + zero8 mean **-14.72** (cuts -4.60 / -27.68 / -11.76 / -14.83), better than the LightGBM on all four cuts (raw -15.52, rounded -15.12). The bigger Moirai models are worse (1.1-base -18.28, MoE-small -19.45, 1.1-large -21.63, MoE-base -22.51, 1.0-large -28.58). The TabPFN jobs were paused (SIGSTOP) while Moirai ran, because both competed for the CPU, and resumed automatically at 12:24.
- **Ensembles (`src/foundation/ensemble_eval.py`; model sets fixed in advance; average of raw forecasts, 8-week zero rule, rounding; blend weight on the LightGBM chosen on cuts 1-3):** moirai-2.0 alone -14.72 (blend w = 0.5: -14.48); top2 moirai-2.0 + tirex -14.16 (blend w = 0.2: -13.72, +1.40 over the rounded LightGBM, 4/4 cuts); **top3 + chronos-t5-small -14.05 (blend w = 0.3: -13.67, +1.45, 4/4)**; top5 -14.47 (-13.68, +1.44, 4/4); diverse5 (moirai-2.0, tirex, timesfm, t5-base, chronos-2) -14.67 (-13.74, +1.38, 4/4); 9 models -14.32 (-13.86, +1.25, 3/4). The result does not depend on the particular models chosen.
- **Which cut resembles the live board:** AutoARIMA live -33.7 vs cuts -28.8 / -51.4 / -30.8 / -31.0 and naive live -64.5 vs -58.3 / -97.6 / -70.4 / -46.6: cuts 0, 2 and 3 are similar in difficulty to the live weeks, cut 1 (Oct-Jan, includes the 2022 peak) is an outlier. The ensemble gain is positive on every cut; on cuts 0, 2, 3 alone it is +1.2.
- **Submission files built** (`src/make_submission_foundation.py`; recipe: average raw forecasts -> 0 for the 1,073 articles with no sale in the last 8 weeks of history -> optional blend with `lgbm_own_v1` -> round -> clip at 0; both pass `src/make_submission.py` validation): `submissions/moirai2_small_v1.csv` (Moirai 2.0-small alone) and `submissions/ensemble_top3_blend_v1.csv` (moirai-2.0-small + tirex + chronos-t5-small averaged, 30% LightGBM). Differences from the uploaded `lgbm_own_v1` (-7.6 live): mean absolute difference 1.10 units (Moirai alone) and 0.76 units (blend) per article-week. Not uploaded yet (a person uploads; log in `submissions/UPLOADS.md`).
- Still running: TabPFN full runs (5 processes, about 3h20 of CPU time each, none finished) and the real-future forecast of moirai-1.1-base.

## 2026-11-21: LIVE SCORE of the ensemble: -4.8 (reported by the user)
- `submissions/ensemble_top3_blend_v1.csv` (moirai-2.0-small + tirex + chronos-t5-small averaged, 8-week zero rule, 30% LightGBM, rounded): **live -4.8** (weeks 1-7), versus -7.6 for `lgbm_own_v1`, -9.8 for the median and the leader's -5.4 seen earlier. Gain over our previous best: **+2.8 points**; the local 4-cut evaluation had predicted about +1.5 (about +1.9 over the unrounded LightGBM that produced the -7.6), so the live gain is larger than expected. The recipe (foundation-model average, zero rule, rounding, small LightGBM weight) transferred from the practice cuts to the real weeks.
- Caveats: the live score covers weeks 1-7 only (the final score uses weeks 8-13), one window can be lucky, and the board keeps each team's best score; expect some regression toward the local estimate (-13.7 mean over cuts, which is not on the live scale). `moirai2_small_v1.csv` (Moirai 2.0-small alone) live score: not yet reported.
