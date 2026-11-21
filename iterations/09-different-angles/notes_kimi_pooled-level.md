# Iteration 09 — angle B: horizon-pooled L1 level model (Kimi)

Date: 2026-11-21. Worktree: `../hack-kimi`, branch `it09/kimi/different-angles`.
Command:

```bash
.venv/bin/python src/experiments_different_angles_kimi.py pooled-level
```

Results: `results_kimi_pooled-level.csv`. One predeclared configuration, seed 0, no sweep.

## Method

- Reuse the existing origin features and LightGBM capacity (`train_gbm.PARAMS`), but remove the
  `h` column and pool all 13 targets per article-origin as `units / scale` with weight `1/13`.
  The L1 fit therefore estimates the conditional **pooled weekly median**, not 13 horizon-specific
  medians. The action is flat: `s_i * prediction` repeated over the 13 weeks, then the zero rule,
  rounding, and clipping.
- Training origins, the last-7-origin validation split, early stopping (`lgb.early_stopping(50)`),
  and the `best_iter * 1.1 + 1` refit mirror the frozen LightGBM protocol
  (`src/train_gbm.py`, `src/experiments_02.py`). Only history before each cut origin is used.
- Reference: canonical D40 with the frozen convention (zero rule on the foundation ensemble before
  the 0.3 LightGBM blend, round, clip: mean −13.67).

## Results (canonical four cuts)

| cut | candidate | D40 | gain | best_iter |
|---|---|---|---|---|
| 0 | −3.96 | −3.72 | −0.23 | 285 |
| 1 | −28.07 | −25.91 | −2.16 | 213 |
| 2 | −12.82 | −11.01 | −1.81 | 52 |
| 3 | −15.22 | −14.02 | −1.20 | 91 |
| mean | | | **−1.35** | |

- Cuts won: **0/4**. Paired article bootstrap 95% interval for the mean gain:
  **[−2.47, −0.19]**, i.e. the loss is unlikely to be noise in the expected direction.
- Balanced-objective check: w1–7 mean gain **−1.96** [−3.41, −0.58]; w8–13 mean gain
  **−0.64** [−1.69, +0.57]. Neither horizon improves.
- Isolated level check against the flattened D40 level (same zero rule/rounding): mean
  **−0.96** score points. The failure is in the level estimate itself, not only in the
  flat-vs-shaped action.

The pooled model is systematically worse than D40, with its early stopping typically far earlier
than the horizon-specific LightGBM. This matches the repo's prior that L1 and loss variants do not
help and that D40's level already carries the useful information.

**Verdict: keep D40. The horizon/loss reframing is rejected under the predeclared +0.5 / 3-of-4 /
both-horizon bar.**

## Scope notes

- No blend, shrink, seed, or feature sweep was run.
- Results condition on the four canonical windows; the paired interval resamples article IDs once
  across cuts to preserve within-article cross-cut dependence.
