# Team rules

How we work together on this repo. Short on purpose; the reasons are in `process-rules.md` (iterations, decision gates) and `decisions-to-make.md`.

## 1. The five rules that matter most
1. **Everything runs locally and reproducibly.** Raw data is never committed; `python src/prepare.py` rebuilds the weekly tables in seconds. Seeds are fixed, versions are pinned in `requirements.txt`.
2. **One yardstick.** Models are judged on the **4 rolling cuts** of the test shop with the D26 metric (`src/validate.py`, `src/rolling.py`), by the **mean over the 4 cuts**, the **cuts won**, and a **paired per-article interval** against a reference. A single cut or a single live score is not evidence (seed noise alone moved one cut by 0.8 points).
3. **No leakage.** A feature at forecast origin `o` uses only weeks before `o`. Any new feature extends the leakage test in `tests/test_features.py`.
4. **Decisions are visible.** Anything that changes what the data means, the splits, the metric, the features or the model family goes in `decisions-to-make.md` with evidence, options and a recommendation, and is settled by the decision owner (the repository owner unless the team agrees otherwise). Do not decide silently in code.
5. **Logs are append-only.** Add to `log.md`; never rewrite someone else's entry. Failures and negative results are logged like successes.

## 2. What counts as an improvement
A change is adopted only if, against the current reference (now: the global LightGBM `own` arm, live -7.6):
- the mean over the 4 cuts improves by **at least 0.5 points**, and
- it wins **at least 3 of 4 cuts** (or the paired 95% interval over articles excludes 0 on the cuts where it matters).

Smaller differences are noise by our own measurements (seeds, blend weights, the `on_hand` fix were all within 0.1 point on the mean).

## 3. Git workflow
- `main` always works: tests pass, results in the logs can be reproduced from the commit. Do not force-push `main`.
- Work on a branch: `it<NN>/<your-handle>/<topic>` (for example `it03/ana/chronos-bolt-zero-shot`). Open a pull request; one teammate (or the decision owner) reviews. Doc-only and results-only PRs may be merged without review.
- Commit small and often. Message format: `it03: Chronos-Bolt zero-shot on test shop, 4 cuts`. If an AI assistant helped, keep its `Co-Authored-By` trailer.
- Before every PR: `bash scripts/check.sh` (tests) and a log entry. The PR template lists the rest.
- Never commit: raw data, `.venv*/`, files over 5 MB, tokens, keys or passwords, personal data. `.gitignore` already covers data, environments and regenerable outputs.

## 4. How an iteration is run locally by everyone
1. `git pull --rebase`, then read the **RESUME HERE** section of `next-steps.md` and `WORK-BOARD.md`.
2. **Claim a task** in `WORK-BOARD.md` (owner, branch). One owner per task; two people never edit the same file at the same time.
3. Create your branch. Set up once: download the data (README), create `.venv`, run `python src/prepare.py` and `bash scripts/check.sh`.
4. Run your experiment with the standard tools. Write results to **your own files** so merges never conflict:
   `iterations/NN-name/results_<handle>_<topic>.csv` and notes in `iterations/NN-name/notes_<handle>_<topic>.md`
   (what you ran, command, commit hash from `git rev-parse --short HEAD`, the table over the 4 cuts, what you conclude).
5. Open a PR. The **iteration owner** merges and folds the notes into the iteration's `log.md` and `changes.md` (only the owner edits those two files).
6. At the end of your session update the **RESUME HERE** section of `next-steps.md` (state, running jobs, open questions) in a small PR.

## 5. Core files: extra care
`src/validate.py` (metric and cuts), `src/prepare.py` (weekly tables), `src/features.py` (features and windows) define what every number means. Changes to them need: a decision entry if the meaning changes, updated tests, and proof that the baselines still reproduce the organizers' published scores (`src/baselines.py`, `iterations/00-baselines/log.md`). Add new behaviour behind an option and keep the default unchanged (as done for `d33`).

## 6. Leaderboard uploads
- Uploads are a **sanity and calibration check**, not the way to choose models. Choose by the 4-cut yardstick.
- The board keeps each team's **best** live score, and the final score uses different weeks (8-13): uploading many near-identical variants inflates the live number without improving the final one. Rule of thumb: at most **5 uploads per person per day**, only for files that differ meaningfully.
- Log every upload in `submissions/UPLOADS.md` (date, who, file, commit, live score). Registering the team and uploading is done by a person, not by a script or an assistant.
- Submission files are made with `src/make_submission.py` (it validates the format).

## 7. Environments and compute
- Everyday work: Python 3.13 and `requirements.txt` (`.venv`). The scipy build for Python 3.10 does not load on macOS 27.
- GPU work (iteration 03): one environment per model family with `env/setup_gpu.sh`; run `src/foundation/smoke.py` first. Record the machine, CUDA and package versions in your notes.
- Model licences: Moirai weights (cc-by-nc-4.0), TimesFM 3.0 and TiRex have restrictions (D34). Using them in the final submission needs the decision owner's agreement.

## 6b. Working with an AI assistant (Claude Code or similar)
Point it at `CLAUDE.md`. It must follow the same rules: propose decisions with evidence and options and wait for the person's go; run the tests; never push to `main` directly unless you ask; never print or commit secrets.

## 8. Splitting the work
Work in parallel only on **different files**. Good splits: one person per foundation-model family; one person on the rich-column techniques (D18); one on hyperparameters and ensembling; one on analysing the pooled arms (step 1.3). `WORK-BOARD.md` is the list.
