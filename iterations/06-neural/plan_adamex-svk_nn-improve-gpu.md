# Plan for an AI agent: make the neural forecaster clearly better or clearly different (GPU machine)

Written 2026-11-21 for adamex-svk, to hand to another AI agent. Self-contained: read this file first, then
`CLAUDE.md`, `TEAM-RULES.md` (section 2, the adoption bar) and `GPU-ACCESS.md`.

## 1. Why this exists (the starting point)
An untuned NHITS (own arm, input 52, 1000 steps, lr 1e-3, 3 seeds) was trained and scored on 2026-11-21
(`iterations/06-neural/notes_adamex-svk_nhits-baseline.md`):

| Variant (4 standard cuts, D26) | cut0 | cut1 | cut2 | cut3 | mean |
|---|---|---|---|---|---|
| D40, the current best recipe | -3.72 | -25.91 | -11.01 | -14.02 | **-13.67** |
| LightGBM alone | -4.03 | -28.06 | -13.08 | -15.31 | -15.12 |
| NHITS alone (3-seed mean) | -3.77 | -29.63 | -12.67 | -15.14 | -15.30 |
| best NHITS blend (0.15 LGBM + 0.15 NHITS + 0.7 FMs) | -3.47 | -25.99 | -11.07 | -13.98 | -13.62 (+0.04, 2/4) |

NHITS is as good as the LightGBM but carries the same information, so it does not improve D40. Every fit
ran to `max_steps = 1000` without early stopping, so it is probably undertrained; cut 1 is its weak spot.
**Goal:** find a neural variant that is either clearly better alone (target: 4-cut mean of -14.5 or better)
or clearly different, and then test whether it lifts D40 past the bar.

## 2. Rules you must follow (non-negotiable)
1. **Judge only on the 4 standard rolling cuts** (D26 metric). Adoption bar: mean gain over D40 >= +0.5
   **AND** better on >= 3 of 4 cuts. Smaller is noise (seed noise is about 0.5-0.8 points).
2. **No tuning on the scored cuts.** Choose every setting **leave-one-cut-out (LOCO)**: for each cut K, pick
   the config with the best mean on the other three cuts, score cut K with it; report the mean of those four
   honest scores. You may also print the in-sample best, labelled "in-sample, for information".
3. **No leakage:** a forecast for a cut uses only weeks before that cut. Covariates are historical only
   (no future on-hand stock).
4. **Secrets:** the JupyterLab URL, password and SSH host are never written to any file, commit, log or
   message. You never type a password. A person gives you the SSH host for the command line only.
5. **You decide nothing.** Run, score, report. A collaborator decides adoption (D41 is still a proposal).
   Do not edit `log.md`, `changes.md`, `WORK-BOARD.md` or `decisions-to-make.md`. Never upload to the
   leaderboard, never push to `main`.
6. Write results only to `iterations/06-neural/results_<your-handle>_<topic>.csv` and notes to
   `iterations/06-neural/notes_<your-handle>_<topic>.md`. Report failures as failures.
7. Core files (`src/validate.py`, `src/prepare.py`, `src/features.py`) are not touched. All changes go to
   `src/neural/` and keep the current defaults.

## 3. Where things run
- **GPU machine** (RTX 3090, ~96 cores, details in `GPU-ACCESS.md`): training only. The project is in
  `~/hack`, unpacked from a bundle. It is not a git checkout.
- **Laptop** (this repo, `.venv/bin/python`): scoring, because it needs the LightGBM and foundation-model
  forecasts that are not in the bundle (`iterations/02-robust-eval/forecasts/base_cut*_seed0.npy`,
  `iterations/03-foundation-model/forecasts/*__test__cut*__padded.parquet`).
- Round trip: laptop `bash scripts/make_gpu_bundle.sh` -> upload/unpack on GPU (see `GPU-ACCESS.md` A or
  B) -> run -> `tar czf nn_results.tgz iterations/06-neural/forecasts` -> copy back -> unpack in the repo
  -> score on the laptop.

## 4. Step 0: code changes on the laptop, before the first bundle (about 1 hour)
All in `src/neural/`; defaults unchanged so the baseline files stay valid.

**0.1 `run_nf.py --tag TAG`** (required). Today every output file is named only by the model
(`NHITS__test__cut0__padded.parquet`) and **an existing file is reused instead of retrained**, so a grid would
silently re-score the baseline. Add `--tag` (default empty): the file stem becomes `{model}{tag}`, e.g.
`--tag _s3000_lr3e-4_in104` -> `NHITS_s3000_lr3e-4_in104__test__cut0__padded.parquet`. Then
`src/neural/eval_nf_d40.py --model NHITS_s3000_lr3e-4_in104` works unchanged. Write all run settings into the
`.json` next to the parquet (it already stores most of them).

**0.2 `run_nf.py` model knobs** (only the ones in the grid below): `--dropout` (NHITS `dropout_prob_theta`,
default 0.0) and `--models` gains `TFT` only if you do stage 4. Keep `--val-size 13` and `--patience`.

**0.3 `src/neural/select_loco.py`** (new). Input: a list of model stems (configs). For each stem it reads the
four cut forecasts and scores them with the D40 code path, reusing `src/neural/eval_nf_d40.py` /
`src/meta/ensemble_recipe.py` (`cut_contexts`, `score_cut`, the zero rule). For two targets:
(a) neural alone, (b) the blend that matters, `0.3 * neural + 0.7 * FM3` and `0.15 * LGBM + 0.15 * neural + 0.7 * FM3`.
It prints the LOCO choice per held-out cut, the honest LOCO mean, and the in-sample table. Writes
`results_<handle>_loco.csv`.

**0.4 `run_nf.py --final`** (only needed if something passes in step 5): train on all 143 weeks of the
test shop and forecast the 13 weeks after the last week, written to `iterations/06-neural/forecasts_final/`.
Mirror `src/foundation/run_model.py --final`.

**0.5 Tests and check:** extend `tests/test_neural.py` (for example: `--tag` changes only the file stem; the
pooled panel uses no week >= cut for the test shop; covariates for cut K end before the cut). Run
`bash scripts/check.sh`, which must pass before you make the bundle. Install pytest into `.venv-nf` if you
need it there (`.venv-nf/bin/pip install pytest`).

## 5. On the GPU machine: setup and smoke test (15 minutes)
```bash
cd ~/hack
nvidia-smi; nproc; free -g; uptime; df -h ~ | tail -1         # someone else may be using it
[ -d .venv-nf ] || bash env/setup_gpu.sh nf                     # reuses the machine's torch 2.5.1
.venv-nf/bin/python -c "import neuralforecast, torch; print(neuralforecast.__version__, torch.cuda.is_available())"
# expect 3.0.0 True
mkdir -p logs
.venv-nf/bin/python src/neural/run_nf.py --cuts 0 --models NHITS --seeds 0 --limit 200 --max-steps 100 \
    --accelerator gpu --tag _smoke
```
If torch breaks (`torchvision::nms`), recreate the venv with `-c env/constraints_gpu.txt` (`GPU-ACCESS.md`).
Delete the `_smoke` files afterwards. Run every real job with `nohup ... > logs/<name>.txt 2>&1 &`.

## 6. Experiments (in this order; stop early if a stage clearly fails)
Each config = 4 cuts x 3 seeds (seeds 0 1 2), `--accelerator gpu`. The laptop CPU took about 2 min per fit,
so expect well under that on the 3090. Run configs one after another in one `nohup` script, not in parallel
(GPU memory, other users).

### Stage 1: train properly (fixes the undertraining) - 7 new configs
Grid over the own arm, NHITS:

| knob | values |
|---|---|
| `--max-steps` | 1000 (baseline exists), 3000 |
| `--lr` | 1e-3, 3e-4 |
| `--input-size` | 52, 104 (104 needs `--start-padding`: cut 3 has only 91 weeks of history) |

Keep `--patience 300`, and check in the logs whether early stopping now triggers. Tag example:
`--tag _s3000_lr3e-4_in104`. The baseline is the stem `NHITS` (s1000, lr 1e-3, in52).
**Then add one knob** on the LOCO-best config: `--dropout 0.1`.

### Stage 2: pooled arm (D27): different information, the most likely source of "different"
Best stage-1 config with `--arm pooled --start-padding` (shops B and D have only 42 and 46 weeks, D has
50k series). Check in the log that all five shops appear in `n_series`. If D makes it too slow,
run once with shops A and C only (add a `--pool-shops` option) and say so in the notes.

### Stage 3: NBEATS (cheap second architecture)
`--models NBEATS` with the best stage-1 settings, own arm (and pooled if stage 2 helped). Also score the
NHITS+NBEATS average as one member.

### Stage 4 (optional, more code): stock covariates
The weekly tables have `on_hand`, `on_hand_missing`, `on_hand_was_negative`, `out_of_stock_flag`. Add them as
**historical** exogenous inputs (`hist_exog_list`, `log1p(on_hand)` and `out_of_stock_flag`) to NHITS, and/or
TFT (`--models TFT`). Never future exogenous. Add a leakage test in `tests/test_neural.py`. Out-of-stock weeks
explain zeros the LightGBM cannot see in the same way, so this is the best bet for "clearly different".
Do it only if stages 1-3 leave time.

## 7. Scoring and the decision gate (on the laptop)
For every config whose 4 cuts exist:
```bash
.venv/bin/python src/neural/eval_nf_d40.py --model <stem> --topic <stem>   # alone + fixed-weight D40 blends
.venv/bin/python src/neural/select_loco.py --models NHITS <stem1> <stem2> ...  # honest choice across configs
```
Report against these gates, using LOCO numbers:
1. **Better alone:** LOCO 4-cut mean of the neural model alone is -14.5 or better (the LightGBM is -15.12),
   and cut 1 is no longer much worse than the LightGBM's -28.06.
2. **Useful in D40:** some fixed-weight blend (NHITS as 4th member / replaces LGBM / split slot / equal5)
   has a mean gain >= +0.5 over D40 (-13.67) AND wins >= 3 of 4 cuts, with the paired per-article intervals
   printed by `eval_nf_d40.py`.
If gate 2 fails, the result is a **negative result**: write it up and stop. Do not build a submission.

If gate 2 passes, run `--final` with the chosen config on the GPU, build the candidate CSV on the laptop
with the same blend, and validate it (`src/meta/ensemble_recipe.py build` shows how the D40 file is built and
validated; add the neural member next to it). **A person decides and uploads.** You only report.

## 8. What to hand back
- `iterations/06-neural/notes_<handle>_nn-improve.md`: what ran (configs, seeds, run time, early-stopping
  step), the LOCO table, the D40-blend table, gate verdicts, failures and anything skipped.
- `iterations/06-neural/results_<handle>_*.csv` from the scripts.
- The forecasts stay in `iterations/06-neural/forecasts/` (git-ignored if large; never commit files over 5 MB).
- A 5-line summary for the person: best config, its LOCO mean alone, its best blend vs D40 (gain, cuts won),
  verdict, and what you would try next.
- Remind the person to remove the assistant's SSH key from the GPU machine when the GPU work is done.

## 9. Time budget (rough)
Step 0 code + tests: 1 h. Stage 1: 8 configs x 12 fits on the GPU, about 1-2 h. Stage 2: about 1 h (pooled is
bigger). Stage 3: 30 min. Stage 4: 2-3 h including code. Scoring each config on the laptop: about 1 min.
