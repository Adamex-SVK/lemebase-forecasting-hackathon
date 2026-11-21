# T9: fine-tune a small foundation model (Moirai 1.1-small) -- REJECT, D40 unchanged

Worker `adamex-svk`, iteration 08, branch `it08/adamex-svk/t9`. GPU work in `~/hack-t9` on the shared
RTX 3090; the 5 stalled TabPFN processes were left untouched.

**Verdict: no recipe change.** The fine-tuned model is a real but small improvement over its own
zero-shot base (+0.91 solo, 3 of 4 cuts) and is *not* strong enough to improve D40 (-13.67): best
variant is a replacement of moirai-2.0-small at -13.79 (gain -0.13, better on 1/4 cuts).

## 1. Route, hyper-parameters, training data (fixed before any score was seen)

**Route (b): Moirai 1.1-small fine-tune** (`Salesforce/moirai-1.1-R-small`, 13.8M params,
uni2ts 2.0.0, torch 2.4.1 in `~/hack/.venv-moirai`). Chosen over Chronos-Bolt/AutoGluon because:
the Moirai family is the strongest member of D40; the fine-tuning classes ship with the installed
package (no install, no env change); the AutoGluon route would have downgraded the shared
`.venv-chronos` (pip dry-run: gluonts 0.16.3, transformers 4.49, ray/catboost/... -- a large, risky
change to an environment other forecast files were produced with).

**Deviation from the brief's wording:** uni2ts 2.0.0 no longer ships the `cli` package
(`python -m cli.train` -> ModuleNotFoundError), so `src/improve/t9_moirai_ft.py` drives the same
classes the CLI used: `MoiraiModule.from_pretrained`, `MoiraiFinetune` (PyTorch Lightning),
`FinetuneDataset`, `PadCollate`, then `MoiraiForecast` for inference. No package was installed.

| hyper-parameter | value | note |
|---|---|---|
| context_length | 65 weeks | cut3 has only 91 padded weeks (65+13 must fit) |
| prediction_length | 13 | horizon |
| window stride (distance) | 1 | uses every origin; 53/40/27/14/66 windows per cut0/1/2/3/final |
| patch_size | 16 | one of the module's patch sizes [8,16,32,64,128] |
| steps | 400 | optimiser steps, batch 32 series |
| lr / warmup / wd | 1e-4 / 40 / 0.01 | constant schedule (MoiraiFinetune default optimiser) |
| finetune pattern | full | all 13.8M params trainable |
| mask ratios / min patches | 0.15-0.5 / 2 | MoiraiFinetune defaults for finetuning |
| forecast quantile | median of 50 samples | same as the zero-shot adapter |
| seed | 0 | |

**Training data per run:** the test shop's own padded weekly series only (D26 padding to the shop
start), *active series only* (2839 / 2762 / 2680 / 2599 / 2904 at cut0/1/2/3/final). Other shops
A-D were **not** used on purpose: their files run to 2024-06-03, i.e. past every cut's holdout, so
"all weeks" would leak future time information even if it is another shop. Keeping to the test
shop's pre-cut weeks makes the leakage argument trivial.

**Leakage check (recorded per run in the output JSON):**

| cut | train last week | holdout first week | separate checkpoint |
|---|---|---|---|
| 0 | 2023-01-16 | 2023-01-23 | runs/cut0/module.pt |
| 1 | 2022-10-17 | 2022-10-24 | runs/cut1/module.pt |
| 2 | 2022-07-18 | 2022-07-25 | runs/cut2/module.pt |
| 3 | 2022-04-18 | 2022-04-25 | runs/cut3/module.pt |
| final | 2023-04-17 | (none) | runs/final/module.pt |

Each cut is trained **and** forecast in one process from its own checkpoint; no checkpoint is
reused across cuts. `contexts_for` -> `validate.split` guarantees train ends the week before the
holdout. Nothing was tuned on the scored cuts (400 steps etc. fixed before the first score).

## 2. Results (D40 = mean(moirai-2.0-small, tirex, chronos-t5-small) -> zero8 rule -> 0.3*LGBM -> round -> clip 0)

| variant | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 | cuts won | verdict |
|---|---|---|---|---|---|---|---|---|
| D40 reference | -3.72 | -25.91 | -11.01 | -14.02 | **-13.67** | 0.00 | 0 | reference |
| ft alone (classic recipe incl. 0.3 LGBM) | -4.62 | -28.47 | -11.91 | -15.20 | -15.05 | -1.39 | 0/4 | REJECT |
| ft as 4th member | -3.74 | -26.12 | -11.32 | -14.07 | -13.81 | -0.14 | 0/4 | REJECT |
| ft replaces moirai-2.0-small | -3.57 | -26.01 | -11.46 | -14.13 | -13.79 | -0.13 | 1/4 | REJECT |
| ft replaces tirex | -3.86 | -26.83 | -11.54 | -14.26 | -14.12 | -0.46 | 0/4 | REJECT |
| ft replaces chronos-t5-small | -3.71 | -26.46 | -11.34 | -14.49 | -14.00 | -0.33 | 1/4 | REJECT |

Solo scores (`src/foundation/score_all.py`, raw model output on the 4 cuts):

| post-processing | cut0 | cut1 | cut2 | cut3 | mean |
|---|---|---|---|---|---|
| ft raw | -6.21 | -29.55 | -12.90 | -16.22 | -16.22 |
| ft round | -5.44 | -29.02 | -12.54 | -15.53 | -15.63 |
| ft zero8 | -6.04 | -29.42 | -12.72 | -16.01 | -16.05 |
| **ft round+zero8** | -5.42 | -29.01 | -12.48 | -15.48 | **-15.60** |
| zero-shot control round+zero8 (same code, `--no-fit`) | -5.36 | -30.07 | -14.45 | -16.15 | -16.51 |

For context, zero-shot medians in the same table: moirai-2.0-small -14.72, tirex -15.96,
chronos-t5-small -17.94, moirai-1.1-base -18.28 (round+zero8). So the fine-tune genuinely lifts
its base (+0.91, better on cut1 +1.06, cut2 +1.97, cut3 +0.67, cut0 -0.06) but Moirai 1.1-small
starts ~1.8 points behind Moirai 2.0-small and the fine-tune does not close that gap.

## 3. Recipe change

None. D40 is unchanged and was re-verified after this work: `ensemble_recipe.py evaluate --models
moirai-2.0-small,tirex,chronos-t5-small` prints -13.67 (-3.72 / -25.91 / -11.01 / -14.02).
`src/foundation/models.py` and `src/meta/ensemble_recipe.py` were **not** modified.

## 4. Runtime, files, failures

* Fine-tune runtime (fit incl. model load + dataset build): **cut0 49.4s, cut1 49.2s, cut2 46.8s,
  cut3 49.4s, final 49.3s**; forecast 25-27s each. All 5 ran in parallel, wall ~2.5 min.
  **The 30-minute stop was never approached** (400 steps x 32 series is tiny for a 13.8M model).
* GPU: 3.7 GiB, 99% util while running; OMP/MKL threads 4 per job; TabPFN PIDs untouched; no
  other worker's process touched.
* New files (model name `moirai-1.1-small-ft`, reused by nothing else):
  `iterations/03-foundation-model/forecasts/moirai-1.1-small-ft__test__cut{0..3}__padded.{parquet,json}`
  and `forecasts_final/moirai-1.1-small-ft__test__final__padded.{parquet,json}`. All verified
  2,904 articles x 13 weeks, 0 NaN, 0 negative. The `--final` file exists but is unused (nothing
  was adopted).
* The zero-shot control run (`moirai-1.1-small-0shot`) was scored on the GPU machine and **not**
  copied into the repo (it is not an ensemble candidate; the numbers are in the table above).
* Code: `src/improve/t9_moirai_ft.py` (self-contained; `run --cut K|--final [--steps N] [--no-fit]
  [--name] --workdir DIR`). Runs only with the Moirai venv.
* Nothing failed after the pipeline was debugged (first smoke run needed `num_warmup_steps <=
  num_training_steps`; fixed by clamping warmup to steps).
* Suspicious / worth knowing for T0: fine-tuning this small a model on ~2.6-2.9k sparse series for
  400 steps is cheap (~50s), so more steps or a bigger base (moirai-1.1-base, 91M) are affordable
  -- but a bigger base would have to beat moirai-2.0-small zero-shot to matter, and the 30-minute
  budget and the fixed-hyper-parameter rule were respected here. Cross-shop training data was
  skipped for leakage safety, not because it was measured to be harmful.

## 5. Reproduce

```bash
# remote (own folder; Moirai venv)
cd ~/hack-t9 && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-moirai/bin/python \
  src/improve/t9_moirai_ft.py run --cut 0 --workdir runs/cut0     # and --cut 1|2|3, --final
# local
.venv/bin/python src/meta/ensemble_recipe.py evaluate --models moirai-2.0-small,tirex,chronos-t5-small,moirai-1.1-small-ft
.venv/bin/python src/foundation/score_all.py   # solo raw/round/zero8/round+zero8
```
