# GPU machine: operating notes (no credentials in this file)

Facts verified 2026-11-21. Host name and password were given privately by the organizers to the repository owner; ask them. Nothing secret is stored in the repo.

## Hardware and software
NVIDIA RTX 3090 (24 GB), driver 580.178.04 / CUDA 13.0, **96 cores, 62 GB RAM, about 40 GB disk free (98% used overall)**, Linux user `hack`, home `/home/hack`, working copy `~/hack`. Conda at `/opt/conda` (Python 3.11.10), **torch 2.5.1+cu121 preinstalled** and working, numpy 2.1.2, JupyterLab 4.6.6. Internet to PyPI and Hugging Face works. **No `nvcc`** (not needed). In a non-interactive SSH session `python3` is **not** on the PATH: use `/opt/conda/bin/python3` or the venv's own `python`.

## Access
1. **Browser:** JupyterLab through a Cloudflare tunnel (URL + password from the organizers; a person logs in). Files uploaded through the Upload button land in `/workspace` (the project bundle is `/workspace/gpu_bundle.tar.gz`, unpacked into `~/hack`).
2. **SSH (key-based, set up by the owner):** `cloudflared` on the laptop (`brew install cloudflared`), a dedicated key `~/.ssh/hackathon_gpu` whose public half is in the machine's `~/.ssh/authorized_keys`. Helpers: `scripts/gpu_ssh.sh '<command>'` and `scripts/gpu_scp.sh up|down <src> <dst>` (set `GPU_SSH_HOST`). They disable password authentication, so an assistant cannot fall back to a password. The tunnel host changes if the organizers restart it.
3. Bundle for a fresh machine: `bash scripts/make_gpu_bundle.sh` -> `dist/gpu_bundle.tar.gz` (code, env files, the five small weekly tables); then `bash env/setup_gpu.sh [chronos timesfm tirex moirai]` (TabPFN env: see below).

## Layout of `~/hack`
- Virtual environments (one per model family because pins conflict): `.venv-chronos`, `.venv-timesfm`, `.venv-tirex`, `.venv-tabpfn` (all `--system-site-packages`, reusing torch 2.5.1, installed with `-c env/constraints_gpu.txt`), `.venv-moirai` (own full env: torch 2.4.1, numpy 1.26, `pyarrow<18`).
- Code `src/`, data `data/interim/weekly_*.parquet`, outputs `iterations/03-foundation-model/{forecasts,forecasts_final,forecasts_cuts}/`, logs `logs/`, driver scripts `run_all_test.sh`, `run_moirai.sh`, `run_final.sh`, `run_cuts.sh`.
- Model/env map: chronos-t5-{small,base,large}, chronos-bolt-{small,base}, chronos-2 -> `.venv-chronos`; timesfm-2.5 -> `.venv-timesfm`; tirex -> `.venv-tirex`; moirai-* -> `.venv-moirai`; tabpfn-ts-v2 -> `.venv-tabpfn`.

## Commands
```bash
# one cut of one model (cut K = last 13*K weeks removed; or --cut-weeks N for any offset; --final = the real future weeks)
.venv-tirex/bin/python src/foundation/run_model.py tirex test --cut-index 0
.venv-tirex/bin/python src/foundation/run_model.py tirex test --cut-weeks 8
.venv-tirex/bin/python src/foundation/run_model.py tirex test --final
.venv-chronos/bin/python src/foundation/smoke.py chronos-bolt-small        # adapter smoke test on 3 toy series
.venv-chronos/bin/python src/foundation/score_all.py                       # score every forecast in forecasts/ (post-processing variants)
# bring results back (laptop): tar on the machine, then scripts/gpu_scp.sh down ...
```
Background jobs: `setsid nohup ./script.sh > logs/x.txt 2>&1 < /dev/null &`; check with `ps -eo pid,etime,time,pcpu,args`.

## Pitfalls learned (do not repeat)
- **Unconstrained installs drift to the newest torch** and clash with the machine's torchvision. Always `pip install -c env/constraints_gpu.txt ...` (torch 2.5.1, torchvision 0.20.1, `transformers<5`, `pandas<3`); transformers 5.x refuses torch < 2.6. Do not hide pip output (`-q` hid a failure).
- **Moirai (`uni2ts`)** pins torch < 2.5 and numpy 1.26: own environment, and `pyarrow<18` (pyarrow 26 needs numpy 2).
- **Chronos-T5** ran out of GPU memory with 1,024 series per batch: `chunk=48` in `src/foundation/models.py` (override with `FM_CHUNK`).
- **TabPFN-TS:** local mode and the ungated v2 weights only (the hosted mode would send data to a third party; 2.5+ weights need a Prior Labs account and licence acceptance by a person). About 1 s of CPU-heavy work per series; the GPU stays idle. 5 parallel jobs used about 2.6 cores each. Dead series (no sale in 8 weeks) are skipped (forecast 0). Telemetry off: `TABPFN_DISABLE_TELEMETRY=1`.
- **CPU oversubscription:** parallel jobs each start up to 96 threads; set `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`, or pause a job with `kill -STOP <pids>` and resume with `kill -CONT <pids>`.
- **TimesFM** stalled on the first overlapping cut (about 25% CPU for 11 minutes while other jobs ran) although it ran fine earlier; it was dropped from the overlapping-cut set. Cause unknown.
- `pkill -f` with a pattern that also occurs in your own SSH command line kills your session; select PIDs with `ps` + `awk`.

## State at handover (14:20)
TabPFN: five processes running (PIDs in `~/hack/logs/tabpfn_pids.txt`). Overlapping-cut streams: moirai2, moirai11, chronos running, tirex done, timesfm killed. Check `ls iterations/03-foundation-model/forecasts_cuts | wc -l` (target 65 without TimesFM: 5 models x 13 cuts) and `logs/cuts_*.txt` (`STREAM_DONE` lines).

## Cleanup when finished
Remove the last line of `~/.ssh/authorized_keys` on the machine (comment `hackathon-gpu`); delete `~/.ssh/hackathon_gpu*` on the laptop; stop background jobs.
