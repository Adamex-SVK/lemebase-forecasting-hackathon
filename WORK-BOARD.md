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
| Rich-column techniques (D18): feature dropout -> cleaning -> auxiliary tasks -> teacher-student | 04 | | | open | needs side tables joined (D7) |
| LightGBM hyperparameter search on the 4 cuts | 02 | | | open | seed noise about 0.8 on one cut: use the mean over cuts |
| Combine several robust level estimators (medians over different windows) | 02 | | | open | |
