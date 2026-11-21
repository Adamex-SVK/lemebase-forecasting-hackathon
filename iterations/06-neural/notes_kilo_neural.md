# Neural forecasters and the remaining loss checks — kilo, 2026-11-21

Plan: `~/.local/share/kilo/plans/1795268567240-neural-forecasters-and-loss-plan.md` (Phase A laptop,
Phase B GPU, Phase C evaluation). Branch `it03/adamex-svk/foundation-models`, base commit `4c892ae`.
Nothing here is adopted: every result below is a negative result, and the current recipe (D40) stays.

Division of work: the loss checks (Phase A) and the neural tooling (`src/neural/run_nf.py`,
`score_nf.py`, `env/requirements_nf.txt`, `tests/test_neural.py`) are from this session. The first NHITS
baseline on top of that tooling was run and scored by adamex-svk
(`iterations/06-neural/notes_adamex-svk_nhits-baseline.md`); this file cross-references it and does not
repeat it.

---

## Phase A — the remaining loss checks (items 2.1 and 2.2)

Code `src/experiments_loss2.py` (sibling of `experiments_loss.py`, which is unchanged so its
`variants(P, F, Pr, thr)` contract still holds). Log `logs/loss2_2_1_2_2.txt`; rows appended to
`iterations/02-robust-eval/results_adamex_loss.csv` with the existing schema (`cut, seed, variant, score`);
the full table and caveats are in `iterations/02-robust-eval/notes_adamex-loss-and-nn-plan.md`.

```
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python src/experiments_loss2.py
```

Same windows, features, weights and early-stopping protocol as the l1 reference; target = units / MASE
scale; every objective early-stops on the validation **L1**; tweedie/poisson get the median correction
(weighted median of y/ŷ on the training windows, the L1-optimal scalar factor).

| Variant | cut0 | cut1 | cut2 | cut3 | mean | vs l1 |
|---|---|---|---|---|---|---|
| l1 (reference) | -4.54 | -28.73 | -13.03 | -15.50 | **-15.45** | - |
| huber (alpha 1) | -9.99 | -34.19 | -17.45 | -20.12 | -20.44 | -4.98 |
| fair | -10.38 | -32.84 | -17.35 | -19.44 | -20.00 | -4.55 |
| l2 | -52.46 | -70.60 | -50.21 | -46.89 | -55.04 | -39.59 |
| tweedie 1.2 + median | -27.85 | -46.17 | -27.45 | -25.98 | -31.86 | -16.41 |
| tweedie 1.5 + median | -25.22 | -43.42 | -23.37 | -23.00 | -28.75 | -13.30 |
| poisson + median | -31.12 | -49.82 | -32.24 | -28.91 | -35.52 | -20.07 |

L1 on the scaled target is the metric's own loss (D29) and nothing else comes close; all 0/4 cuts won.
Seed coverage: huber/fair/l2 3 seeds on all cuts; tweedie/poisson 3 seeds on cuts 0-1 and 1 seed on cuts
2-3 (they hit the 3000-round cap, 4-10 min per fit; the gap is 13-25 points, far beyond the ~0.8 seed
noise). `src/train_gbm.py` defaults unchanged.

---

## Phase B — environment and adapter (GPU-ready, validated on the laptop CPU)

* `env/requirements_nf.txt`: `neuralforecast==3.0.0` (last 3.x line that accepts torch>=2.0; 3.1+ needs
  torch>=2.9 and would replace the machine's 2.5.1), `pytorch-lightning==2.4.0`, `ray[tune]==2.40.0`.
* `env/setup_gpu.sh nf`: `.venv-nf` with `--system-site-packages` and `-c env/constraints_gpu.txt`, so the
  machine's torch 2.5.1 is reused. Installed and checked on the laptop: torch 2.5.1, neuralforecast 3.0.0,
  pytorch-lightning 2.4.0, pandas 2.3.3.
* `src/neural/run_nf.py` (own arm and pooled arm):
  * target `units / scale`, where `scale` is the D26 metric scale of the shop's zero-padded history —
    `tests/test_neural.py` asserts it equals `features_at(M, M.W)` and that the rows are in `ShopMatrix` key
    order;
  * MAE loss, `scaler_type="identity"`, `val_size=13` (the last 13 weeks of the history, never the holdout),
    `max_steps` default 1000, 1024 windows per batch, 3 seeds averaged;
  * scale-0 (constant-history) series are excluded from training and get forecast 0, but keep their row;
  * outputs both formats: `{model}{tag}__test__{cut|wN}__padded.parquet` (seed mean, plus per-seed files) and
    `.npy` in `ShopMatrix` key order, in `iterations/06-neural/forecasts/` (`forecasts_cuts/` for
    `--cut-weeks`);
  * `--tag` is required for grid runs because an existing file is reused instead of retrained; `--dropout`,
    `--pool-shops`, `--start-padding`, `--limit`, `--accelerator gpu`.
  * Pooled arm (D27): all weeks of A-D plus the test shop before the cut, per-shop scale; `--start-padding`
    keeps B/D (42/46 weeks) trainable with `input_size=52`.
* `src/neural/score_nf.py`: D26 scoring of the neural parquet with rounding + zero rule, blends against the
  LightGBM and the D40 foundation ensemble, weights chosen on cuts 1-3. Validated by feeding it the
  median13 forecast and by reproducing D40: -13.70 vs the logged -13.67.
* Local smoke runs (CPU, exact pinned versions): `--limit 20 --max-steps 4` for the own and pooled arms,
  plus `--cut-weeks 8`; every output had 36,752 rows (2,904 x 13) and 20 non-zero rows, and `--tag` changed
  only the file stem. `bash scripts/check.sh` passes (now includes `tests/test_neural.py`).

Commands for the GPU machine (blocked here: `GPU_SSH_HOST` is not set, and `GPU-ACCESS.md` forbids storing it):

```bash
bash env/setup_gpu.sh nf
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-nf/bin/python src/neural/run_nf.py \
    --cuts 0 --models NHITS --seeds 0 --limit 200 --max-steps 100 --accelerator gpu --tag _smoke
.venv-nf/bin/python src/neural/run_nf.py --arm own --cuts 0 1 2 3 --models NHITS NBEATS --seeds 0 1 2 --accelerator gpu
.venv-nf/bin/python src/neural/run_nf.py --arm pooled --start-padding --cuts 0 1 2 3 --models NHITS --seeds 0 1 2 \
    --accelerator gpu --tag _pooled
```

## Phase C — NHITS result (run by adamex-svk on top of this adapter)

Own arm, 3 seeds, 1000 steps, rounded + zero rule: **-15.29** (cut0 -3.77, cut1 -29.62, cut2 -12.65,
cut3 -15.14) vs LightGBM -15.12 and median13 -16.54. Every fit stopped at `max_steps` (no early stopping),
so it is probably undertrained. In D40 it is interchangeable with the LightGBM, not additive: the best
variant (0.15 LGBM + 0.15 NHITS + 0.7 FMs) is -13.62 vs D40 -13.67 (+0.04, 2/4 cuts), below the bar.
Full table: `notes_adamex-svk_nhits-baseline.md`. NBEATS, the pooled arm and stock covariates are in
`plan_adamex-svk_nn-improve-gpu.md` and have not been run.

---

## Attempt to break D40 in the blend (`src/neural/blend_probe.py`)

Question: is there any honest blend that beats the current winner (D40 = 0.3 LightGBM + 0.7 mean of
moirai-2.0-small, tirex, chronos-t5-small) by the adoption bar? Members tried: LightGBM (base, D33,
CatBoost), NHITS, median13, NHITS zero-gate, and foundation-model sets of 3, 5 and 14 models (all with 4
cuts available). Protocol: weights chosen on cuts 1-3, reported on cut 0 and the 4-cut mean; every grid also
gets a **leave-one-cut-out** mean (choose on 3 cuts, score the held-out one), which is the honest estimate of
a searched blend. D40 is reproduced exactly (-13.67). Results: `results_kilo_blend-probe.csv`.

```
.venv/bin/python src/neural/blend_probe.py           # 39 attempts + 5 weight grids
.venv/bin/python src/neural/blend_probe.py --build   # also writes the best buildable submission CSVs
```

| Attempt | weights | cut0 | cut1 | cut2 | cut3 | mean | vs D40 | cuts won |
|---|---|---|---|---|---|---|---|---|
| gate 0.3 FL + 0.7 E3 | 0.3 FLgated + 0.7 FM3 | -3.53 | -25.88 | -10.96 | -14.05 | **-13.61** | +0.06 | 3/4 |
| grid FL+FN+E3, chosen on 1-3 | 0.2 LightGBM + 0.1 NHITS + 0.7 FM3 | -3.55 | -25.86 | -11.03 | -14.01 | -13.61 | +0.05 | 3/4 |
| triple | 0.15 + 0.15 + 0.7 | -3.47 | -25.99 | -11.07 | -13.98 | -13.62 | +0.04 | 2/4 |
| grid FL+FN+E5, chosen on 1-3 | 0.2 L + 0.2 NHITS + 0.6 FM5 | -3.35 | -25.60 | -11.58 | -14.00 | -13.63 | +0.03 | 3/4 |
| **D40 reference** | 0.3 L + 0.7 FM3 | -3.72 | -25.91 | -11.01 | -14.02 | **-13.67** | 0 | - |
| grid FL+FN+E3, **LOCO** | per held-out cut | -3.55 | -26.03 | -11.03 | -14.01 | -13.66 | +0.01 | 2/4 |
| 14 foundation models, w=0.3 | | -6.08 | -25.87 | -13.04 | -14.26 | -14.81 | -1.15 | 1/4 |

* Best honest gain: **+0.06** (gate) and **+0.05** (3-way grid). LOCO of that grid: **+0.01**. The bar is
  +0.5 and 3/4 cuts; nothing is close. **The winner cannot be broken in the blend with the models we have.**
* NHITS does not add information: it is interchangeable with the LightGBM (the triple and the 3-way grid
  agree to 0.01-0.02), and putting it into the foundation-model average makes it worse (0.2-0.4 points).
* Bigger foundation-model sets are **worse**: 5 models -13.83 at best, all 14 -14.81 at w=0.3 (chronos-bolt,
  moirai-1.0, moirai-moe, timesfm drag the mean up). The small D40 set is not an accident, it is better.
* median13, CatBoost and the zero gate add nothing in the blend (median13 hurts: 4-way LOCO -13.92).
  The gate's +0.06 is within noise and its live test was -5.6 vs -4.8, already rejected.

### Built submission CSVs (format-validated, **not uploaded**)

| file | weights | local 4-cut | vs D40 | mean forecast | diff vs current submission |
|---|---|---|---|---|---|
| `submissions/blend_probe_1_grid_FL+FC+FN+E3_chosen_on_cuts_1-3.csv` | 0.2 L + 0.1 NHITS + 0.7 FM3 | -13.61 | +0.05 | 7.755 | 0.09 units, 6.9% cells, corr 0.9999 |
| `submissions/blend_probe_2_triple_0.15-0.15-0.7.csv` | 0.15 L + 0.15 NHITS + 0.7 FM3 | -13.62 | +0.04 | 7.751 | 0.14 units, 9.0% cells, corr 0.9998 |
| `submissions/blend_probe_3_FL+FMed+E3_0.2-0.1-0.7.csv` | 0.2 L + 0.1 median13 + 0.7 FM3 | -13.66 | +0.00 | 7.841 | 0.15 units, 9.0% cells, corr 0.9998 |

All three pass `src/make_submission.py` validation and are nearly identical to `ensemble_top3_blend_v1.csv`
(D40). An upload would be a noise test, not a real improvement; TEAM-RULES section 6 limits uploads to
meaningfully different files, so none was uploaded.

## Status / next
* Done: Phase A, the GPU-ready adapter + env + tests, D26 scoring/blends, the NHITS baseline (by
  adamex-svk), the blend probe, and the negative-result documentation.
* Blocked: NBEATS, pooled arm, stock covariates, the full 4-cut x 3-seed GPU grid — all need
  `GPU_SSH_HOST` (a person must supply it; never stored).
* D41 (neural family) remains a proposal in `decisions-to-make.md`; nothing was adopted, `log.md` and
  `changes.md` are for the iteration owner.

## Live checks of the blend-probe candidates (reported by the user, 2026-11-21)
The candidates did not improve the board: every one scored worse than the -4.8 reference (worse than -5),
so the local +0.05 (+0.06 for the unbuildable gate) was noise. Together with the earlier gate upload (-5.6)
and NHITS-in-D40 (-5.2), the evidence is consistent: local gains below about 0.3 do not transfer to the
live weeks, and weeks 8-13 are scored separately from the weeks the leaderboard shows. D40
(`ensemble_top3_blend_v1.csv`, -4.8) stays the reference; no candidate is adopted. Exact per-file live
scores belong in `submissions/UPLOADS.md` (a person logs uploads).
