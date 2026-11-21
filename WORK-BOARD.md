# Work board

Claim a task by putting your handle in **Owner** and your branch in **Branch** (pull request). One owner per task. Update **Status**: `open`, `in progress`, `review`, `done`.

| Task | Iteration | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|
| Restart step 1.3 (all shops x arms, ~45-60 min) and analyse own vs pooled vs others-only (step 1.4) | 01 | | | open | `python src/run_arms.py`; first attempt: test shop own -4.83, pooled equalised -5.84, unweighted -4.95, others only -23.07 |
| GPU access: ask the organizers (machine, access, Python, internet, time) | 03 | | | open | questions in `next-steps.md` section C |
| Foundation models: Chronos 1.0 / Bolt / 2.0 | 03 | | | open | `iterations/03-foundation-model/model-notes.md`, run `smoke.py` first |
| Foundation models: TimesFM 2.5 | 03 | | | open | |
| Foundation models: Moirai 1.x / MoE / 2.0 (needs its own environment) | 03 | | | open | licence cc-by-nc-4.0 (D34) |
| Foundation models: TiRex | 03 | | | open | CUDA kernels need compute capability >= 8.0 |
| Rich-column techniques (D18): feature dropout -> cleaning -> auxiliary tasks -> teacher-student | 04 | adamex-svk | it04/adamex-svk/rich-columns | done (closed, D43) | feature dropout tested, NOT adopted (best rich arm -0.80 vs own, 0/4 cuts); cleaning, auxiliary tasks and teacher-student not run; `iterations/04-rich-columns/notes_adamex-svk_dropout.md` |
| LightGBM hyperparameter search on the 4 cuts | 02 | | | open | seed noise about 0.8 on one cut: use the mean over cuts |
| Combine several robust level estimators (medians over different windows) | 02 | | | open | |
| TabPFN-TS (local mode, v2 weights): smoke test, 4 cuts, blend test | 03 | | it03/adamex-svk/foundation-models | in progress | D36; environment being installed on the GPU machine |
| Meta-learning ensemble (FFORMA style): write `src/meta/fforma.py`, run the purged evaluation | 05 | it05 | it03/adamex-svk/foundation-models | done | D37 tested, NOT adopted (best +0.23, 3/4; keep D40); results `iterations/05-meta-ensemble/results_meta_fforma_*.csv`; TabPFN (D36) still pending |
| Collect finished GPU jobs (TabPFN, overlapping cuts), commit the forecasts, rebuild the submission | 03 | it05 | it03/adamex-svk/foundation-models | in progress | overlapping cuts (5 models x 13) collected, submission rebuilt (byte-identical); TabPFN full runs still running (D36) |
| Finish step 1.3 (other shops x arms) and the extra-credit table | 01 | | | open | `EXTRA-CREDIT.md`, `python src/run_arms.py` |
| Clean up: remove the SSH key from the GPU machine and the laptop | - | | | open | when all GPU work is done |
| Finish the remaining loss checks (huber/fair/l2, tweedie/poisson + median correction) and the neural forecasters (neuralforecast NHITS/NBEATS, own then pooled) | 02 + 06 | kilo | it06/kilo/neural-forecasters | in progress | plan `1795268567240-neural-forecasters-and-loss-plan.md`; results in `iterations/02-robust-eval/results_adamex_loss.csv` and `iterations/06-neural/notes_kilo_neural.md`; D41 proposed |
