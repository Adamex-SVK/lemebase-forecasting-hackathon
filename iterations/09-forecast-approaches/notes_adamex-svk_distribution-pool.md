# Iteration 09 / task 3 — distribution pooling (adamex-svk)

**Status: BLOCKED — no quantile exports could be produced here.**

## What exists

The additive code path is implemented and tested, without changing any default:

- `src/foundation/models.py`: `QUANTILE_LEVELS` (0.1..0.9) and `forecast_quantiles(...)`, with
  family-specific paths for Chronos (T5/Bolt/2), Moirai (1.x/MoE/2.0), TiRex (native 0.1..0.9 grid) and
  TimesFM. `forecast()` and every `SPECS` default are unchanged.
- `src/foundation/run_model.py`: opt-in `--emit-quantiles`. It writes the usual point forecast (the
  interpolated median of the exported quantiles) plus
  `iterations/09-forecast-approaches/quantiles/{model}__test__{cutK|final}__padded.npz`
  (`quantiles [n,13,Q]`, `levels`, `store_id`, `article_id`, `week`).
- `src/improve/i09_distribution_pool.py`: equal-weight linear pool of the three CDFs, mixture median by
  monotone inverse-CDF interpolation, monotonicity and 80%-coverage checks, tail-dependence flag,
  comparison with D40, and saving the point forecasts as pseudo-model `mixture-median-top3`.
- `tests/test_improve_i09.py`: pooling monotonicity/bracketing checks.

## Why it is blocked

The pool needs quantile forecasts from Moirai 2.0-small, TiRex and Chronos-T5-small. Producing them
requires the GPU machine's model environments. No SSH host or credential is available to this session
(and must not be stored), and the local environments contain none of `chronos`, `uni2ts` or `tirex`.
`results_adamex-svk_distribution-pool.csv` therefore contains one `BLOCKED` row and no scores.

The script exits gracefully and prints the exact GPU commands to run when access is available:

```bash
python src/foundation/run_model.py moirai-2.0-small test --cut-index 0 --emit-quantiles
python src/foundation/run_model.py tirex            test --cut-index 0 --emit-quantiles
python src/foundation/run_model.py chronos-t5-small test --cut-index 0 --emit-quantiles
# ... cuts 0-3 and --final, then:
python src/improve/i09_distribution_pool.py
```

## Decision

No comparison and no file were produced, so the `+0.25` kill / `+0.5` adoption gates were never reached.
This branch is unresolved rather than rejected. It does not block the local results and must not be
counted as evidence for or against distribution pooling.
