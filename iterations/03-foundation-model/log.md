# Iteration 03 log (foundation models on the GPU)

Append-only.

## 2026-11-21: GPU access received, new branch `it03/adamex-svk/foundation-models`
- The organizers provided an NVIDIA **RTX 3090** (24 GB) through a JupyterLab page with a terminal and Python (a test notebook saw the GPU), plus an optional SSH route through a Cloudflare tunnel. The URL and the password were given privately in chat; **they are intentionally not recorded in the repo**.
- An AI assistant cannot type the password, so the workflow is: a person logs in, uploads `dist/gpu_bundle.tar.gz` (built by `scripts/make_gpu_bundle.sh`), runs the commands in `gpu-runbook.md` and pastes the output back; forecasts are downloaded and scored locally with `src/foundation/score_all.py`. An optional key-based SSH setup would let an assistant run the jobs directly (see the runbook).
- RTX 3090 capacity (24 GB) is enough for every model on the list (the largest weights, Moirai-MoE-base and Chronos-T5-large, are below about 4 GB in fp32).
- Open: whether the GPU machine can reach Hugging Face and PyPI (first step of the runbook), which Python versions it has (Moirai needs 3.10/3.11), how long we may use it.
- Adapters in `src/foundation/models.py` are still untested; the smoke tests (runbook Part A, step 5) come first.
