# Experiment log

2026-11-21: user approved proposed three-parameter configuration. Prepared isolated
runner from iteration 13 paired-control evaluation; removed recency reweighting
and set candidate-only parameters. Historical inputs and incumbent preserved.
Command: `.venv/bin/python iterations/19-conservative-lightgbm/run.py`.
Seed 0; observed versions and cut dates will be saved in results metadata/CSVs.

Completed paired fits on all six periods. Primary ensemble mean incumbent
-13.665947 versus candidate -13.758754 (gain -0.092807);
1/4 primary wins. Weeks 8–13 mean gain -0.095389.
Supplementary offsets 16 and 32 both regress. Fails prespecified review gate;
retain original ensemble and original LightGBM settings. Stronger regularization
did not improve this experiment; this does not establish that the incumbent is
free of overfitting. All periods have already been used in development.

Own-shop training only; no A-D equal-average, pooled or unseen-shop claim.
Per-cut dates, excluded counts and scored coverage: results/scores.csv.
Original raw data, saved forecasts, submission and src/train_gbm.py unchanged.
Models and forecasts preserved under results/runs. No final CSV generated/uploaded.
Runtime: Python 3.12.4, LightGBM 4.7.0, NumPy 2.5.5,
pandas 2.3.3. Seed 0. Hashes and settings: results/metadata.json.
Research screening from earlier request saved in research-notes.txt; no extra
experiments started. Review checkpoint reached before any further stage.

Verification command passed: `.venv/bin/python iterations/19-conservative-lightgbm/verify.py`. All six controls exactly reproduce saved predictions; input/code hashes match, group partitions reconcile to MASE, and saved-model reload reproduces predictions exactly. See results/verification.json.
