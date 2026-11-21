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
