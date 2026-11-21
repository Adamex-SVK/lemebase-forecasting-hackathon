# Prompt: evaluate whether the rich columns improve the test-shop forecast

Copy everything below the line into another model/agent that has shell access to this repository.

---

You are working in the repository at `/Users/adomi/Documents/hackathlon` (a retail demand-forecasting hackathon: forecast weekly units per article, 13 weeks ahead, for a "test shop"; four other shops A-D are available as training data). Use `.venv/bin/python` (Python 3.13) and run from the repo root. Read `CLAUDE.md`, `TEAM-RULES.md` (section 2 is the adoption bar) and `iterations/04-rich-columns/design.md` first.

## Question to answer
Does a LightGBM trained on the test shop PLUS the donor shops, with extra "rich" columns (price, discount share, promotion flag, markdown level) from the donors and feature dropout, forecast the test shop better than our current best? The test shop has NO rich columns, so at forecast time they are always NaN; feature dropout trains the model to cope with that by masking the rich features on a random share of donor training windows.

## What already exists (do not rebuild)
- Rich tables: `data/interim/rich_B.parquet`, `rich_C.parquet`, `rich_D.parquet` (built by `src/rich.py`; definitions in `iterations/04-rich-columns/notes_adamex-svk_rich_schema.md`). A and the test shop have no rich table (NaN plus `rich_missing = 1`).
- Features and dropout: `src/features.py` (`feature_names`, `features_at(..., rich=True)`, `build_windows(..., rich=True, rich_drop=p)`), model plumbing in `src/train_gbm.py`.
- Runner: `src/run_rich.py`. CLI: `.venv/bin/python src/run_rich.py ARM[,ARM...] --seeds 0,1 [--cuts 0,1,2,3] [--topic NAME]`. Arms: `own`, `pooled_norich`, `pooled_rich_nodrop`, `pooled_rich_drop50`, `pooled_rich_drop80`. Results go to `iterations/04-rich-columns/results_adamex-svk_<topic>.csv` (columns arm, cut_index, seed, score, n_scored, trees, seconds), saved after every arm-cut.
- Metric and cuts: D26 score `(1 - MASE) x 100` from `src/validate.py`; 4 rolling cuts of the test shop (cut k drops the last 13*k weeks), same convention as `src/rolling.py`. Known references for `own` (seed 0): cuts 0-3 = -4.83, -28.45, -13.35, -15.44 (mean -15.52).
- Tests: `bash scripts/check.sh` must pass before you propose anything.

## Where to run it: the laptop, or the GPU machine (recommended for the pooled arms)
LightGBM here is the standard CPU build, so **the GPU itself is not used**. The reason to use the GPU machine is its CPU and memory: it has about 96 cores and 62 GB RAM (RTX 3090 host), against 8 cores and 16 GB on the laptop. The pooled arms (7-9M rows, about 100 minutes per seed on the laptop) should be several times faster there and no longer risk running out of memory. The same machine and workflow as the foundation-model runs apply (`iterations/03-foundation-model/gpu-runbook.md` is the reference). Facts: JupyterLab with a terminal; conda Python 3.11 at `/opt/conda`; internet to PyPI works; about 51 GB disk free; the project lives in `~/hack` from the earlier bundle. The URL, password and SSH host name are NOT in the repo and must never be written into any file or typed by you: a person logs in; you only use the key-based SSH route if it is already set up (see the runbook, section "Option: let the assistant run jobs directly over SSH": key `~/.ssh/hackathon_gpu`, `ssh -i ~/.ssh/hackathon_gpu -o ProxyCommand="cloudflared access ssh --hostname %h" hack@<host>`, password authentication disabled; ask the person for the host if you do not have it). Only use the machine if the person has set that up or does the steps below for you; otherwise run on the laptop.

Steps for the GPU machine (adapt if a step already exists there):
1. **Code changes first, on the laptop (additive, defaults unchanged):**
   - `scripts/make_gpu_bundle.sh`: add `data/interim/rich_B.parquet data/interim/rich_C.parquet data/interim/rich_D.parquet`, `tests`, `scripts/check.sh` and `iterations/04-rich-columns` (the notes and design only, not the results CSVs) to the tar list, so the rich tables and the new code travel with the bundle (the current bundle only has the five weekly tables and `src`).
   - `src/run_rich.py`: add `--threads N` (default 8 = today's behaviour) that sets `num_threads` for LightGBM (`src/train_gbm.py` hardcodes `num_threads=8` in `PARAMS`; override through the params the runner already passes, do not change the default in `train_gbm.py`). On the GPU machine use `--threads 64` (leave cores for other users; check `nproc` and `uptime` first).
   - Rebuild: `bash scripts/make_gpu_bundle.sh` (creates `dist/gpu_bundle.tar.gz`, a few MB; the rich parquet files add a few more MB: keep the bundle well under 100 MB, the upload limit of the tunnel).
2. **Upload and unpack** (a person in JupyterLab: Upload button; or `scp` over the key-based route): `mkdir -p ~/hack && tar xzf gpu_bundle.tar.gz -C ~/hack && cd ~/hack` (this overwrites `src/` and `env/`; it does not touch the model virtual environments `.venv-*`).
3. **Environment for LightGBM** (new, once; the foundation-model environments do not have it): 
   ```bash
   cd ~/hack && python3 -m venv .venv-gbm && .venv-gbm/bin/pip install -q --upgrade pip \
     && .venv-gbm/bin/pip install -r requirements.txt && .venv-gbm/bin/python -c "import lightgbm, pandas, numpy; print(lightgbm.__version__, pandas.__version__)"
   ```
   No `-c env/constraints_gpu.txt` here: this environment must NOT use the machine's torch (not needed). `requirements.txt` pins what the laptop uses (pandas below 3, numpy, lightgbm, pyarrow, scipy...). If it fails on Python 3.11, report the error instead of guessing versions. If `libgomp` is missing, report it.
4. **Tests and smoke on the machine before the long run:**
   ```bash
   cd ~/hack && PY=.venv-gbm/bin/python bash scripts/check.sh
   .venv-gbm/bin/python src/run_rich.py own --seeds 0 --cuts 0 --topic smoke_gpu --threads 64
   ```
   Expected: tests pass; `own` cut 0 seed 0 score about -4.83 (same code and data: tolerate 0.8 of seed noise and note that a different thread count can change LightGBM results slightly). If the score differs by more, stop and report: it means the environment or the data differs from the laptop.
5. **Long runs in the background** (survive a closed terminal; JupyterLab terminals can disconnect):
   ```bash
   mkdir -p logs && nohup .venv-gbm/bin/python src/run_rich.py own,pooled_norich --seeds 0,1 --topic reference_gpu --threads 64 > logs/rich_reference_gpu.txt 2>&1 &
   echo $! > logs/rich_reference_gpu.pid
   ```
   then the rich arms the same way (topic `dropout_gpu`, log `logs/rich_dropout_gpu.txt`). Run ONE job at a time unless `free -g` shows memory for two (a pooled arm needs roughly 10-15 GB; check `free -g`, `nproc`, `uptime` before launching). Watch with `tail -n 5 logs/rich_*_gpu.txt` and `ps -p $(cat logs/rich_reference_gpu.pid)`; do not poll in a tight loop.
6. **Bring results back** (the per-article detail files from the protocol below are the only thing the analysis needs besides the CSVs):
   ```bash
   cd ~/hack && tar czf rich_results.tgz iterations/04-rich-columns/results_*gpu*.csv iterations/04-rich-columns/detail logs/rich_*_gpu.txt
   ```
   then download it from the JupyterLab file browser (or `scp -i ~/.ssh/hackathon_gpu -o ProxyCommand=... hack@<host>:hack/rich_results.tgz .`), unpack in the repo on the laptop, and do the analysis there with `.venv/bin/python` (same environment as everything else; the analysis is light).
7. **Clean up:** when the work is done, remind the person to delete the last line of `~/.ssh/authorized_keys` on the GPU machine (the key added for assistant access) and tell them which large files (`.venv-gbm`, logs, results tarball) can be removed to free disk. Do not delete anything yourself on that machine except files you created.

Do not run the pooled arms on the laptop and the GPU machine at the same time for the same arm/seed (duplicate work and conflicting CSV files): the topic names `reference_gpu` and `dropout_gpu` are separate from the laptop's `reference` and `dropout`.

## Protocol (do exactly this; judge only on the 4 cuts, never on one cut or on a live leaderboard score)
1. Check the machine first (laptop: 8 cores, 16 GB RAM, shared; GPU machine: about 96 cores, 62 GB, see the section above). A pooled arm trains on 7-9M rows and takes about 100 minutes per seed on the laptop. Run ONE pooled job at a time (running two can run out of memory). Check `ps` and `logs/` for an already running `run_rich.py` job (on the laptop: reference arms `own,pooled_norich`, topic `reference`) and wait for it rather than starting a second one. Where the commands below say `.venv/bin/python` and `--topic dropout`, use `.venv-gbm/bin/python`, `--threads 64` and `--topic dropout_gpu` on the GPU machine.
2. Reference: make sure `own` and `pooled_norich` exist for seeds 0 and 1 on all 4 cuts (`results_adamex-svk_reference.csv`).
3. Before launching the rich arms, add per-article output to `src/run_rich.py` so paired intervals are possible: for every arm, cut and seed save the per-article MASE (the `detail` returned by `validate.score(train, hold, fc, detail=True)`, indexed by store_id, article_id) to `iterations/04-rich-columns/detail/<arm>_cut<k>_seed<s>.parquet`. This is an additive change; defaults and existing behaviour must stay identical; the pooled_norich and own reference runs need to be rerun with it if their detail files are missing.
4. Run the rich arms: `.venv/bin/python src/run_rich.py pooled_rich_nodrop,pooled_rich_drop50,pooled_rich_drop80 --seeds 0,1 --topic dropout > logs/rich_dropout.txt 2>&1` (in the background; do not poll in a tight loop).
5. Analyse, in `iterations/04-rich-columns/notes_<yourhandle>_dropout.md` and `results_<yourhandle>_dropout.csv` (use your own handle, never edit `log.md` or `changes.md`):
   - Table of score by cut and seed for every arm; mean over the 4 cuts and over the 2 seeds; number of cuts won (mean over seeds per cut).
   - Two comparisons for each rich arm: (a) versus `pooled_norich` (does the rich information itself help?) and (b) versus `own` (does it beat our current best?).
   - Paired per-article 95% bootstrap interval (1,000 resamples) of the score difference for each comparison, per cut, from the detail files.
   - Feature importance (gain) of the rich features for the best rich arm (refit once if needed, or save it from the runner).
   - Seed noise: report the spread between seeds per arm and cut; differences below it are noise (seed spread is about 0.5-0.8 on one cut).
6. Decision rule (TEAM-RULES.md section 2): an arm is adopted over a reference only if the mean over the 4 cuts improves by at least 0.5 points AND it wins at least 3 of 4 cuts (or the paired interval excludes 0 on the cuts that matter). Report a null result as plainly as a positive one.

## Rules
- You are not the decision maker: present evidence, 2-3 options and a recommendation; the collaborator decides (record nothing in `decisions-to-make.md` unless asked).
- No leakage: features at origin `o` use only weeks before `o`. Donor shops may use all their weeks (decision D27); the test shop's training windows must stop before each cut's holdout.
- Never commit raw data, environments, files over 5 MB, tokens or keys; never upload to the leaderboard or push to `main`; do not commit unless asked. Other sessions edit this repo concurrently: touch only `src/run_rich.py`, the files under `iterations/04-rich-columns/` and `logs/`.
- Delete the throwaway files `iterations/04-rich-columns/results_adamex-svk_smoke.csv` and `results_adamex-svk_smokerich.csv`.
- Report outcomes faithfully, including failures, crashes (for example out of memory) and anything not verified.

## Deliverable
A short report: the table, the two comparisons per arm with intervals, whether any arm clears the adoption bar, what it implies for the next techniques (cleaning, auxiliary tasks, teacher-student) and your recommendation.
