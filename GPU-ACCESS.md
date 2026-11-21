# GPU machine: how to access it and run jobs (for people and AI agents)

Read this before using the GPU machine. Details of the foundation-model workflow are in `iterations/03-foundation-model/gpu-runbook.md`; this file is the short, current entry point.

## What the machine is
- An NVIDIA RTX 3090 (24 GB) host provided by the organizers, reached through a JupyterLab page with a terminal, and optionally over SSH through a Cloudflare tunnel.
- About 96 CPU cores and 62 GB RAM (the laptop has 8 cores and 16 GB, shared with other sessions). Conda Python 3.11 at `/opt/conda`; torch 2.5.1+cu121 preinstalled and sees the GPU; internet to PyPI and Hugging Face works; about 51 GB of disk free (tight); no `uv`.
- The project lives in `~/hack` on that machine (unpacked from a bundle, not a git checkout). It has no git access to this repository.
- **The GPU is used only by the neural / foundation models (Chronos, TimesFM, TiRex, Moirai, TabPFN).** LightGBM here is the standard CPU build: running it there helps because of the cores and RAM, not the GPU.

## Secrets: never in the repo
The JupyterLab URL, the password and the SSH host name were shared privately by the organizers and are **not stored anywhere in this repository and must never be written into any file, commit, log or message you produce**. A person logs in and types the password; an AI agent never types a password. Ask the person for the SSH host name if you need it, use it only in the command line, and do not save it.

## Two ways to work
### A. A person drives the browser terminal (always works)
1. Person opens the JupyterLab URL, logs in, opens a Terminal.
2. On the laptop build the bundle: `bash scripts/make_gpu_bundle.sh` (creates `dist/gpu_bundle.tar.gz`: `src`, `env`, `requirements.txt`, the weekly tables; keep it under about 100 MB, the tunnel upload limit; if it needs more files, add them to that script first).
3. Person uploads it with the JupyterLab Upload button, then in the terminal: `mkdir -p ~/hack && tar xzf gpu_bundle.tar.gz -C ~/hack && cd ~/hack` (unpacking overwrites `src/` and `env/` but not the `.venv-*` environments).
4. Person runs the commands you give them and pastes the output back; for results, `tar czf results.tgz <files>` and download it from the JupyterLab file browser, then unpack it in the repo on the laptop.

### B. The agent runs commands over SSH (saves copy and paste; needs a one-time key setup by a person)
Key-based login only (password authentication is never used):
1. Laptop: `brew install cloudflared`; the dedicated key `~/.ssh/hackathon_gpu` (no passphrase) already exists on the laptop if it was set up earlier (`ls ~/.ssh/hackathon_gpu*`); otherwise `ssh-keygen -t ed25519 -f ~/.ssh/hackathon_gpu -N ""`.
2. In the JupyterLab terminal, a person appends the public key once: `mkdir -p ~/.ssh && echo "<contents of ~/.ssh/hackathon_gpu.pub>" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys`.
3. Connect (the host comes from the person, not from this repo):
   ```bash
   ssh -i ~/.ssh/hackathon_gpu -o PasswordAuthentication=no -o ProxyCommand="cloudflared access ssh --hostname %h" hack@<ssh-host> "<command>"
   ```
   Copy files with `scp` using the same `-i` and `-o ProxyCommand` options. Always run long jobs with `nohup ... &` (see below) because tunnel connections drop.
4. Only do this if the person has set it up or tells you to. **When the work is done, the person should delete the last line of `~/.ssh/authorized_keys` on the machine** (the key added for assistant access): remind them.

## Environments on the machine (in `~/hack`)
| Environment | For | Notes |
|---|---|---|
| `.venv-chronos` | Chronos 1.0 (T5), Chronos-Bolt, Chronos-2 | `--system-site-packages` (reuses the machine's torch 2.5.1) |
| `.venv-timesfm` | TimesFM 2.5 | same |
| `.venv-tirex` | TiRex | needs `nvcc` for its CUDA kernels |
| `.venv-moirai` | Moirai 1.x / MoE / 2.0 | own full environment (needs torch < 2.5, numpy 1.26, `pyarrow<18`) |
| `.venv-tabpfn` | TabPFN-TS (local mode, v2 weights) | telemetry off: `TABPFN_DISABLE_TELEMETRY=1` |
| `.venv-gbm` | LightGBM / pandas runs (`src/run_rich.py` etc.) | may not exist yet: create it with `python3 -m venv .venv-gbm && .venv-gbm/bin/pip install -r requirements.txt`, WITHOUT the torch constraints file |
Create the foundation-model environments with `bash env/setup_gpu.sh chronos timesfm tirex moirai` (it applies `env/constraints_gpu.txt` so pip cannot replace the machine's torch; an unconstrained install breaks it). Always use `-c env/constraints_gpu.txt` for anything that should reuse the machine's torch.

## Typical jobs
```bash
cd ~/hack
# smoke test of a model family (prints whether the adapter works)
.venv-chronos/bin/python src/foundation/smoke.py chronos-bolt-small
# zero-shot forecasts for the test shop, cut 0..3 (4 rolling cuts); --final forecasts the real future
.venv-chronos/bin/python src/foundation/run_model.py chronos-bolt-base test --cut-index 0
# LightGBM / rich-column runs on the CPU cores (LightGBM threads are hardcoded to 8 in src/train_gbm.py; add a
# --threads option to the runner first, see iterations/04-rich-columns/prompt_evaluate_rich_features.md)
.venv-gbm/bin/python src/run_rich.py pooled_norich --seeds 0,1 --topic reference_gpu
```
Forecasts land in `iterations/03-foundation-model/forecasts/` (and `forecasts_final/`); scoring happens back on the laptop (`src/foundation/score_all.py`), because the scoring environment lives there.

## Rules for running long jobs
1. **Check the machine first:** `nvidia-smi; nproc; free -g; uptime; df -h ~ | tail -1`. Other people and sessions may be using it.
2. **Run long jobs in the background** so they survive a dropped terminal or tunnel: `nohup <command> > logs/<name>.txt 2>&1 &` then `echo $! > logs/<name>.pid`. Check with `tail -n 5 logs/<name>.txt` and `ps -p $(cat logs/<name>.pid)`. Do not poll in a tight loop.
3. **Do not start two heavy jobs at the same time unless memory allows** (`free -g`); a pooled LightGBM arm needs roughly 10-15 GB. Out-of-memory kills are silent: a job that stops with no error in its log was probably killed; rerun it (our runners replace rows by arm / cut / seed, so a rerun only fills gaps).
4. **Use separate topic / output names** for runs on the machine (for example `reference_gpu`) so results do not collide with the laptop's files.
5. **Disk is tight (about 51 GB):** do not download large models you do not need; clean up only files you created.
6. Judge results by the 4 rolling cuts of the test shop (`TEAM-RULES.md`), not by one run.
7. No leakage and no raw data: the bundle contains only the small weekly tables and code. Never upload the raw dataset, tokens or keys.

## If something breaks
- `operator torchvision::nms does not exist` or "PyTorch >= 2.6 is required": pip replaced the machine's torch; recreate that venv and install with `-c env/constraints_gpu.txt` (`transformers<5`).
- Moirai `ImportError ... pyarrow`: `pip install "pyarrow<18"` in `.venv-moirai`.
- Chronos-T5 out of GPU memory: lower the chunk size (`FM_CHUNK=48` environment variable).
- No internet on the machine (not seen so far): fallback in `iterations/03-foundation-model/gpu-runbook.md` (wheels and weights uploaded in parts of about 90 MB).
