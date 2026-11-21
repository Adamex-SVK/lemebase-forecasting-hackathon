# Conservative LightGBM — approved fixed configuration

2026-11-21: user explicitly approved num_leaves=31, min_data_in_leaf=500,
lambda_l2=5. Consulted senior-data-scientist skill in this session; apply temporal
validation and feature timing guidance with the project MASE evaluator.

Hypothesis: stronger regularization reduces variance; possible cost is underfitting
recent activations. One candidate, no hyperparameter search. Retain all original
features, scale-normalized L1 targets, learning_rate=.05, feature_fraction=.8,
bagging_fraction=.8, bagging_freq=1, seed=0, threads=8. Original purged inner
stopping (13-week separation, seven validation origins, cap 3000, patience 50),
final int(best*1.1)+1 refit. No recency weighting or activation augmentation.

Fresh control and candidate on primary offsets 0,13,26,39 and supplementary
16,32. Use pinned foundation predictions, 70/30 blend, foundation-only zero rule,
one rounding and clipping unchanged. Compare exact control predictions to saved
incumbent. Save models, forecasts, versions, cut dates, hashes and group results.
Score full 13 weeks and early/late subsets using unchanged project evaluator.

Review gate: mean gain >=.5, >=3/4 primary wins, nonnegative mean weeks 8–13
gain, no regression on supplementary periods. Reused development periods are
not independent confirmation. Offset 65 remains closed. This stage is target-shop
own-only; no pooling or A-D training is implied by approval of these three
parameters. No cross-shop generalization claim; a separate A-D evaluation must
report own/pooled, equal-weight A-D mean, and D exclusions/scored coverage.
No source-reconstructed future labels. No submission generation or upload.
Stop at review checkpoint after evaluation, before promotion or broader research.

Command: `.venv/bin/python iterations/19-conservative-lightgbm/run.py`
Verification: `.venv/bin/python iterations/19-conservative-lightgbm/verify.py`
