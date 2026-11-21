# Instructions for AI assistants working in this repo

Read first: **`docs/HANDOVER.md`** (current state, best recipe, jobs in flight, next steps), then `docs/TEAM-RULES.md`, `docs/process-rules.md`, `docs/next-steps.md` (RESUME HERE), `decisions-to-make.md`.

Non-negotiable:
- **Decisions are made by a collaborator, not by you.** Any collaborator can be the decision owner (usually the person you work for). Present evidence, 2-3 options and a recommendation; wait for their answer. Record who decided and when in `decisions-to-make.md`.
- **Judge models only on the 4 rolling cuts** (`src/rolling.py`, D26 metric in `src/validate.py`): mean over cuts, cuts won, paired interval. Not on one cut, not on a live score. Adopt a change only if it clears the bar in `docs/TEAM-RULES.md` section 2.
- **No leakage.** Features at origin `o` use only weeks before `o`; extend `tests/test_features.py` for any new feature.
- **Run `bash scripts/check.sh` before proposing a PR.** Core files (`src/validate.py`, `src/prepare.py`, `src/features.py`) change only with a decision entry and passing tests; keep defaults unchanged and add options.
- **Logs are append-only.** Write your results to your own `results_<handle>_<topic>.csv` and `notes_<handle>_<topic>.md` files; only the iteration owner edits `log.md` and `changes.md`.
- **Never** commit raw data, environments, files over 5 MB, tokens or keys; never upload to the leaderboard or push to `main` unless the person asks; never print credentials.
- Use `.venv/bin/python` (Python 3.13). Report outcomes faithfully, including failures.
