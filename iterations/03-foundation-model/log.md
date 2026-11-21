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
