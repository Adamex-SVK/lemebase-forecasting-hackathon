# GPU runbook (iteration 03)

**Machine (from the organizers, 2026-11-21):** an NVIDIA RTX 3090 (24 GB) reached through a JupyterLab page with a terminal; SSH is optional. The URL and the password were shared privately. **They are not stored in this repo and must never be committed or pasted into files.** Logging in is done by a person; an AI assistant does not type the password.

Why this works without git access on the GPU machine: the models only need the five small weekly tables (about 2 MB) and our code, so we upload one bundle and download the forecasts. Scoring happens back on the laptop.

## Part A: first 15 minutes (a person, in the browser)
1. Open the JupyterLab URL, enter the password, open a **Terminal**.
2. Check the machine and paste the output into the chat:
   ```bash
   nvidia-smi; python3 --version; ls /usr/bin/python3* ; df -h ~ | tail -1; free -g | head -2
   curl -sI https://huggingface.co | head -1; curl -sI https://pypi.org | head -1
   ```
   (The two `curl` lines show whether the machine can download model weights and packages. If they fail, tell us: we then need another route for the weights.)
3. On the laptop build the bundle: `bash scripts/make_gpu_bundle.sh` (creates `dist/gpu_bundle.tar.gz`, about 2-3 MB). In JupyterLab use the **Upload** button, then in the terminal:
   ```bash
   mkdir -p ~/hack && tar xzf gpu_bundle.tar.gz -C ~/hack && cd ~/hack
   ```
4. **Environments.** Checked on the machine (2026-11-21): internet works (Hugging Face and PyPI answer), Python 3.11.10 under `/opt/conda`, **torch 2.5.1+cu121 is already installed and sees the GPU**, numpy 2.1.2, no `uv`. So Chronos, TimesFM and TiRex reuse that torch (`--system-site-packages`); only Moirai (needs torch < 2.5) gets its own full environment. In the terminal, from `~/hack`:
   ```bash
   which nvcc || echo "no nvcc"          # TiRex compiles CUDA kernels and needs nvcc; Chronos and TimesFM do not
   bash env/setup_gpu.sh chronos timesfm  # a couple of minutes each; start with these two
   ```
   `tirex` and `moirai` come after the first smoke tests (`bash env/setup_gpu.sh tirex moirai`).
5. **Smoke tests** (adapters were written from documentation and never run): the output tells us which adapter to fix. Paste the full output.
   ```bash
   .venv-chronos/bin/python src/foundation/smoke.py chronos-bolt-small chronos-bolt-base chronos-2 chronos-t5-small   # first batch
   .venv-timesfm/bin/python src/foundation/smoke.py timesfm-2.5
   # later: chronos-t5-base chronos-t5-large (2.8 GB download)
   .venv-tirex/bin/python  src/foundation/smoke.py tirex
   .venv-moirai/bin/python src/foundation/smoke.py moirai-1.1-large moirai-moe-small moirai-2.0-small
   ```

## Part B: zero-shot runs (after the smoke tests pass)
Start small, then scale. `--limit 200` sends only the first 200 non-empty series (a few seconds), to confirm the output format and timing.
```bash
.venv-chronos/bin/python src/foundation/run_model.py chronos-bolt-base test --cut-index 0 --limit 200
.venv-chronos/bin/python src/foundation/run_model.py chronos-bolt-base test --cut-index 0          # all 2,904 test articles
# the standard evaluation is the 4 rolling cuts, cut-index 0..3:
for k in 0 1 2 3; do .venv-chronos/bin/python src/foundation/run_model.py chronos-bolt-base test --cut-index $k; done
```
Order of the models (cheapest first): `chronos-bolt-small`, `chronos-bolt-base`, `chronos-2`, `timesfm-2.5`, `tirex`, `chronos-t5-small/base/large`, `moirai-2.0-small`, `moirai-1.1-large`, `moirai-moe-small/base`, `moirai-1.0-large`. Do the test shop first (it is the only shop that is submitted); A, B, C and D are for the extra-credit comparison and for fine-tuning later. Set D (50k series) is the slowest: do it last.

Forecasts land in `iterations/03-foundation-model/forecasts/`.

## Part C: bring the results back and score them
```bash
cd ~/hack && tar czf forecasts.tgz iterations/03-foundation-model/forecasts      # then download it from the JupyterLab file browser
```
On the laptop unpack it into the repo and run:
```bash
.venv/bin/python src/foundation/score_all.py        # D26 score per model, shop and cut, next to the median-of-13-weeks reference
```
Judge models by the **mean over the 4 cuts, cuts won, and a paired interval against the LightGBM and the median** (TEAM-RULES.md section 2). Then blends with the LightGBM (our reference: live -7.6).

## Option: let the assistant run jobs directly over SSH (saves copy and paste)
Password login cannot be done by the assistant. Key-based login can be set up by a person once:
1. Laptop: `brew install cloudflared` and create a key: `ssh-keygen -t ed25519 -f ~/.ssh/hackathon_gpu -N ""`.
2. In the JupyterLab terminal: `mkdir -p ~/.ssh && echo "<contents of ~/.ssh/hackathon_gpu.pub>" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys`.
3. Test: `ssh -i ~/.ssh/hackathon_gpu -o ProxyCommand="cloudflared access ssh --hostname %h" hack@<ssh-host-from-the-organizers>`. If it logs in without asking for the password, an assistant can run the commands of Parts B and C directly. Only do this if you are comfortable with the assistant having shell access to that machine.

## Fine-tuning (later in this iteration)
Only after zero-shot results: recipes per model are in `model-notes.md` (Chronos-Bolt through AutoGluon, Chronos 1.0 training script, Moirai through the uni2ts CLI, TimesFM LoRA example; TiRex has none). An RTX 3090 with 24 GB is enough for all of the listed models.

## If the machine ever loses internet access (fallback, not needed so far)
- Check first whether `torch` and `transformers` are already installed (output above). If so, the lightweight families may work with `--system-site-packages` environments, and only the missing small packages (`chronos-forecasting`, `timesfm`, `tirex-ts`, `uni2ts`) need to come in as wheel files.
- Download wheels and model weights on the laptop (`pip download ...`, `huggingface-cli download amazon/chronos-bolt-base ...`) and upload them through JupyterLab. Cloudflare tunnels often limit one upload request to about 100 MB, so large files must be split (`split -b 90m`) and joined (`cat`) on the machine. Approximate weight sizes in fp32: Chronos-Bolt-base 0.8 GB, Chronos-2 0.5 GB, TimesFM 2.5 0.8 GB, TiRex 0.14 GB, Chronos-T5-large 2.8 GB, Moirai-MoE-base 3.6 GB.
- Start with the smallest useful set: `chronos-bolt-small`, `chronos-bolt-base`, `chronos-2`, `timesfm-2.5`, `tirex`.
