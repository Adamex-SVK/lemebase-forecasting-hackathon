# Test a fixed 10% XGBoost addition

User requested testing the current foundation-model ensemble with 30% LightGBM
and 10% XGBoost locally. Interpret weights as 60% equal foundation average,
30% original LightGBM, 10% XGBoost (sum 100%). Announced this interpretation
before execution. No weight search or change to the existing submission.

Consulted Senior Data Scientist skill. Use all four existing target-shop 13-week
cuts, exact project MASE, and a fixed configuration. All cuts are development
evidence already used previously, not independent confirmation.

## Fixed protocol

- Reuse immutable Moirai 2.0-small, TiRex and Chronos-T5-small predictions from
  donor commit `71b82f33f0fe5647a13d3786cd17d11b4c4e5493`; no foundation inference
  or GPU required. Verify original LightGBM array order against keyed forecasts.
- Train XGBoost 3.3.0 locally in `.venv-autogluon`. Use the existing lag/rolling/
  stock/horizon features and origin-specific MASE-normalized targets.
- Absolute-error objective, CPU histogram trees, seed 0, 2 threads, depth 5,
  minimum child weight 50, learning rate 0.05, 80% row/column subsampling,
  lambda 1, 128 histogram bins. No hyperparameter search.
- Last seven training origins are inner validation. Earlier training targets
  finish before the first validation origin (13-week purge). Early stop after
  30 nonimproving rounds, maximum 400. Refit on all pre-cut windows with
  `int(best_rounds*1.1)+1` trees, matching the reference's refit convention.
- The 8-week zero rule applies to the foundation average before blending.
  Blend raw component forecasts, round once at the end, clip at zero.
- Compare original 70/30 ensemble, candidate 60/30/10 ensemble, and rounded
  XGBoost alone. Report all 13 weeks plus weeks 1–7 and 8–13, exclusions, paired
  product-bootstrap gain intervals (500 draws; exploratory, no multiplicity or
  temporal correction), mean gain, and cuts won.
- Save model binaries, predictions, config, package versions and fingerprints
  in ignored `runs/`. Preserve small numerical evidence separately for review.

```bash
.venv-autogluon/bin/python iterations/08-xgboost-blend/run.py
```

Choose a fresh `--output PATH` to rerun. Target-shop-only results do not prove
cross-shop transfer. No final submission generation, upload, commit, or push.

Objective reference: [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).
