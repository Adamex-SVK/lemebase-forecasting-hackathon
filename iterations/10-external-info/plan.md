# Iteration 10: bring in outside information (plan for ideas 2, 3, 4)

Goal: get below the current best (D40, live -4.8) by adding information the shop's own history does not contain. Re-weighting the same forecasts is exhausted (about twenty variants, all rejected; see `iterations/08-improvements/ens-search/`). Stretch target: live -3.5 to -4; the -2 aim is a long shot.

Three independent tasks, none needs the GPU. Each can run in its own worktree and session. Task IDs: **X2** (idea 2, original sources), **X3** (idea 3, calendar and holidays), **X4** (idea 4, bias correction learned on sibling shops). Idea 1 (fine-tune on sibling data) and idea 5 (synthetic data) are NOT in this plan; X2 decides whether idea 1 gets more data.

## Shared brief (read first)

**State.** D40 = 0.3 x LightGBM (`lgbm_own_v1`) + 0.7 x mean of the zero-shot forecasts of moirai-2.0-small, tirex, chronos-t5-small; 0 if no sale in the last 8 weeks; round to whole units, clip at 0. Local 4-cut mean -13.67 (cuts -3.72 / -25.91 / -11.01 / -14.02), live -4.8. Details: `HANDOVER.md`.

**Yardstick (non-negotiable).** D26 metric, `src/validate.py`; 4 rolling cuts (`src/rolling.py`, cut K removes the last 13*K weeks, K = 0..3). Adopt only if mean gain vs D40 >= +0.5 AND better on >= 3 of 4 cuts. Any tuned parameter: leave-one-cut-out (choose on 3 cuts, score the 4th). Report all variants tried, not only the best.

**No leakage.** Features at origin `o` use only weeks before `o`. Anything learned from sibling shops (Sets A, C, D) is learned on those shops only. Extend `tests/test_features.py` for every new feature. Run `bash scripts/check.sh` before proposing a PR.

**What we already know about borrowing (do not repeat it).** Pooling the other shops' rows into the LightGBM hurt (-5.84 / -4.95 vs -4.83 own-only); training on other shops only gave -23.07 (about zero forecast). Set B is synthetic with its own calendar; imposing another shop's calendar hurt. Rich columns such as category were tested in it04 and not adopted (D43).

**Rules from the repo.**
- Decisions are made by a collaborator, not by the agent: present evidence, 2-3 options and a recommendation; record in `decisions-to-make.md` only after the collaborator answers.
- **Downloads need explicit approval from the person in chat**: state the filename, source and size first. Check licences (D34 is still open).
- Never commit raw data, environments, files over 5 MB, tokens or keys. Never upload to the leaderboard. Never push to `main`.
- Logs are append-only. Write only to your own `results_<handle>_<topic>.csv` and `notes_<handle>_<topic>.md` in your task folder; only the iteration owner edits `log.md` and `changes.md`.
- Use `.venv/bin/python` (Python 3.13). Commit with explicit paths, never `git add -A`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Worktree setup per task** (as in `iterations/08-improvements/action-plan.md`):
```bash
git worktree add ../hack-x2 -b it10/<handle>/x2 it08/adamex-svk/improvements
ln -s "$PWD/.venv" ../hack-x2/.venv; ln -s "$PWD/data" ../hack-x2/data; ln -s "$PWD/hackathon-retail-sets" ../hack-x2/hackathon-retail-sets
```
Verify first: reproducing D40 on the 4 cuts must give -13.67.

**Report format** (end of each task): what was tried, a table (per-cut scores, mean, gain vs D40, cuts won, paired interval), verdict ADOPT or reject, and what it changes for the other tasks.

---

## X2: check the original public sources (about 1 hour)

**Why.** The sibling sets come from public datasets: Set A VN2 Inventory Planning Challenge, Set B Mancino Market (synthetic, GitHub), Set C Zenodo Retail Transactions and Stocks, Set D Mendeley Retail Transactions and Stocks. The test shop has no identified source. The hackathon extracts may be smaller than the originals. More real intermittent retail series would feed idea 1 (fine-tuning a foundation model on real retail demand). This task only answers whether that is worth doing.

**Steps.**
1. Read the dataset pages (no download yet): size, number of shops/articles, period covered, granularity, licence, whether weekly aggregation is possible. Use the links on the hackathon page and `README.md`.
2. Compare each with what we already hold in `hackathon-retail-sets/` (rows, articles, weeks). Record how much extra real history the originals would add.
3. Check licences and whether using the originals is allowed by the hackathon rules (`https://lemebase.com/hackathon`); log it as an open item for D34.
4. Check whether any original contains the test period or the test articles (it must not be used if so; flag it to the person immediately).
5. If a download is worthwhile, **ask the person first** with filename, source and size. Do not download before the yes.

**Deliverable.** `iterations/10-external-info/x2/notes_<handle>_sources.md`: a table (source, size, period, licence, extra over what we hold, verdict) and a recommendation: go or no-go for idea 1 with this data.

**Done when.** Each of the four sources has a row and a clear go/no-go; no download happened without approval.

---

## X3: holiday and calendar features (about half a day)

**Why.** The LightGBM sees only the shop's own lags and rolling features. A holiday and calendar signal is information not in the history, and it is cheap to test. Category columns did not help (D43), but holidays are a different kind of signal.

**Steps.**
1. **Infer the country** from the data: peak weeks of the test shop (Christmas, Easter, summer, back-to-school) and the week-of-year profile; also `src/` EDA outputs from iteration 00. Record the evidence and confidence; if it is not clear, say so and stop at a short list of candidates for the person to pick (a decision, not yours).
2. Build features at origin `o` and for each target week: holiday flag and distance to the nearest holiday, Easter week, Christmas run-up, week of year, with a source for the holiday dates (the `holidays` Python package or a table written by hand; check it is installed in `.venv`, ask before installing).
3. Add them to the LightGBM as an option (defaults unchanged; core `src/features.py` changes only with a decision entry and passing tests). The future calendar is known at the origin, so it is not leakage, but add tests that no feature uses sales after `o`.
4. Evaluate on the 4 cuts, own-arm LightGBM alone and inside the D40 blend (the 0.3 slot). Same seed and settings as `lgbm_own_v1`.
5. Remember the 13-week test horizon is 2023-04-24..2023-07-17: it contains Easter Monday and early-summer holidays. Cuts 0-3 hold out different seasons, so check the per-cut pattern, not only the mean.

**Deliverable.** `iterations/10-external-info/x3/`: results csv, notes, tests; verdict ADOPT or reject with the paired interval.

**Done when.** Country evidence recorded, features tested for leakage, 4-cut table for LightGBM alone and inside D40.

---

## X4: bias correction learned on sibling shops (2-3 hours)

**Why.** The metric is MAE, so the best forecast is the median; for sparse articles rounding and shrinkage matter a lot (rounding alone was adopted, D38). A small correction might be learned from the zero-heavy sibling shops (Sets C and D are 94% zero weeks) and checked on the test shop. Expected gain is small (a few tenths of a point), so keep it light.

**Steps.**
1. On Sets A, C, D (never the test shop), run D40-style forecasts on their own rolling cuts (LightGBM own arm plus a cheap proxy for the foundation models, such as the median of the last 13 weeks if the foundation models are too slow; say which).
2. Fit a one-or-two-parameter correction, for example a multiplier on the forecast by segment (mean of last 13 weeks: below 1, 1-5, above 5) and a rounding threshold other than 0.5. Choose the parameters on the sibling shops only.
3. Apply the fixed correction to D40 on the test shop's 4 cuts. No tuning on the test shop. Report with and without the correction.
4. Also report the leave-one-cut-out version, where the parameters are re-chosen on the test shop's other 3 cuts, as a ceiling for how much a correction could give.

**Deliverable.** `iterations/10-external-info/x4/`: results csv, notes; verdict ADOPT or reject.

**Done when.** The correction is fixed before it sees the test shop; the 4-cut table with and without it; the LOCO ceiling.

---

## Order and integration

- X2, X3, X4 are independent; run them in parallel. X3 and X4 touch different code; if both touch `src/features.py`, X3 owns it and X4 stays in its own folder.
- **Integration (after all three report).** If more than one is adopted, combine them and re-run the 4-cut check of the combination (gains do not add by default). Then rebuild the submission with `src/make_submission_foundation.py`, verify it against the cut results, and give the file to the person to upload (the agent never uploads). Add a row to `submissions/UPLOADS.md` once the live score is known.
- If X2 says go, the next plan is idea 1 (fine-tune on real sibling retail data, GPU, about a day); see `GPU-ACCESS.md` for the GPU rules.

## Time estimates (rough)

| Task | Estimate | Needs |
|---|---|---|
| X2 | about 1 hour | web reading, approval before any download |
| X3 | about half a day | CPU only |
| X4 | 2-3 hours | CPU only |
