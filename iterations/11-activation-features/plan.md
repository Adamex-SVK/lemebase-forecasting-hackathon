# Phase 2: activation-aware LightGBM ablation

Use the senior-data-scientist advisor's temporal validation and leakage guidance.
One prespecified feature group: positive-sale frequency divided by min(13, weeks since
first observed positive sale), and median positive sales in the previous 13 weeks
normalized by the origin-specific MASE scale. Never-sold frequency is zero; missing
positive median stays NaN. Existing series_age already measures observed activation.

Original feature builder remains unchanged. Train seed-0 reference and candidate
with identical original parameters, 2-week training origins, seven inner validation
origins, 13-week label purge, 3000-tree cap and 50-round early stopping. Refit on all
pre-cut windows with int(best_iteration * 1.1) + 1 trees. The tree count is selected
separately inside each arm's inner validation. No tuning on outer holdouts.

Freeze foundation forecasts, dead-product processing, 70/30 weights and rounding.
Primary offsets 0,13,26,39; supplementary forward checks 0,16,32. Use pinned source
batches from the Phase 1 contract, and compare against both saved incumbent and
retrained control. These periods were previously examined and are development data.
Target-shop own-only training; no A-D or unseen-shop generalization claim.

Review threshold: >=0.5 primary mean score points, >=3/4 wins, no mean late-horizon
regression. Also inspect control drift, activation/activity/stock groups and forward
checks. No final submission generation, upload, weight search or Phase 3 work.

Run: `.venv/bin/python iterations/11-activation-features/run.py`
Use `--output <new-directory>` for independent reproduction. Model binaries and
per-fit forecast arrays remain local under results/runs; CSV evidence is retained.

Artifact verification: `.venv/bin/python iterations/11-activation-features/verify.py`.
This checks locked inputs/code hashes, exact saved-control reproduction, group
partition totals and one serialized candidate model's prediction round trip.
Results and the decision are recorded in log.md; the candidate was not promoted.
