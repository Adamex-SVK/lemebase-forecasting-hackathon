# Iteration 09 — Kimi status

Date: 2026-11-21. Branch `it09/kimi/different-angles` (worktree `../hack-kimi`), based on
`it03/adamex-svk/foundation-models`. The source plan is copied verbatim to `plan_kimi.md`.

## Delivered

- `src/experiments_different_angles_kimi.py` — `audit` and `pooled-level` subcommands; the
  `analog` subcommand is not implemented yet.
- `results_kimi_audit.csv`, `notes_kimi_audit.md` — Step 0 reproduction, oracle diagnostics, and
  the A selector.
- `results_kimi_pooled-level.csv`, `notes_kimi_pooled-level.md` — angle B.
- `notes_kimi_analog.md` — explicit NOT RUN marker for C.
- `tests/test_different_angles_kimi.py` — fast action/leakage tests for the shared helpers.

## Decisions

| angle | result | decision |
|---|---|---|
| Step 0 | Canonical D40 −13.67 reproduced; O_sel11 −3.95; D2/D4/D5 reproduced | Trust the cached evidence |
| A — selector | No stump improved training loss; gains +0.01/−0.44/−0.51/−0.53; previous-cut-best worse | **Close** |
| B — pooled L1 level | −1.35 mean, 0/4 cuts, both horizons negative | **Close** |
| C — analog donors | Not run | Open, last ranked new-information test |

The permuted oracle null mean is −5.54 versus observed −3.95, so only ≈+1.6 of the +8.8 oracle gap
survives destroying article-specific alignment. This supports treating most of the quoted oracle
headroom as hindsight/winner's-curse, not predictable signal.

## Honest remaining chance

Conditional on A and B failing, the chance that C both passes its donor sanity test and clears the
canonical +0.5 / 3-of-4 / both-horizon bar is roughly **10–15%**. The default recommendation is to
keep D40, finish robustness/extra-credit/licence work, and not relax the adoption bar after a third
failed predeclared experiment.

No submission was built or uploaded. D40 and the core files are unchanged.
