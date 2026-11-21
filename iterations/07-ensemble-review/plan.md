# Review the current winning ensemble

User authorized examining saved component forecasts and blend weights while
preserving the current ensemble. User confirmed that -4.8 is a leaderboard score.
Consulted the Senior Data Scientist skill for saved-prediction comparisons,
temporal validation, paired errors, and avoiding claims from repeated selection.

1. Fetch remote references without merging; inspect the foundation-model branch.
2. Pin its immutable commit and read prediction blobs, not execute donor code.
3. Verify keyed forecasts and LightGBM array ordering; reproduce current scores.
4. Compare individual models, leave-one-out blends, rounding, and zero-sales rule.
5. Check a fixed 0.0–1.0 LightGBM weight grid in increments of 0.1; earlier cuts
   are development data and cut 0 is already used, not untouched confirmation.
6. Reconstruct the final submission in memory; require exact agreement with the
   saved winning CSV. Preserve that original CSV unchanged on Maya's branch.

No training, new submission recipe, merge, push, or upload. No A-D transfer claim:
the available component forecasts are for the target shop only.

Reproduce from repository root:

```bash
git fetch origin
.venv/bin/python iterations/07-ensemble-review/evaluate.py
```

The default output must not exist. For a rerun choose a fresh directory with
`--output /tmp/ensemble-review-rerun`. The immutable source commit is recorded in
the script and results metadata. Fetching is necessary because component blobs
belong to the friend's branch and were not merged into this branch.
