# Recency experiment log (append-only)

## 2026-11-21 — Fixed experiment authorized

User requested modest recency weighting and senior engineer advice. Searched local
user/repo/plugin skills; senior engineer unavailable. Used senior-data-scientist
skill and workflow guidance. Fixed half-life 104 weeks, normalized training weights;
unweighted inner validation and official outer scoring. No other model change.

`.venv/bin/python -m unittest discover -s tests -p 'test_recency_weighting.py' -v`:
3 passed (half-life, normalization, row consistency, origin translation, nonmutation).
Training and scoring command: `.venv/bin/python iterations/13-recency-weighting/run.py`.

## Completed — no promotion

| Offset | Incumbent | Recency | Gain |
|---|---:|---:|---:|
| 0 | -3.724944 | -3.623149 | +0.101795 |
| 13 | -25.907431 | -25.918742 | -0.011311 |
| 16 | -36.258865 | -36.168913 | +0.089952 |
| 26 | -11.010614 | -10.915901 | +0.094713 |
| 32 | -17.715976 | -17.858503 | -0.142527 |
| 39 | -14.020799 | -14.063396 | -0.042597 |

Primary mean: -13.665947 -> -13.630297; gain +0.035650. Wins 2/4.
Primary weeks 8–13 gain +0.021709. Additional checks consistent: False. Promotion review passed: False.
Forward-check mean (0/16/32): -19.233262 -> -19.216855.

The gain is small and inconsistent: below +0.5 threshold, only 2/4 primary wins, and offset 32 worsens. No further decay rate was tested. Keep the fixed original ensemble. No final CSV, deployment, commit, push or upload. Experiment remains isolated from production features/trainer.

All results are own-only target-shop development checks, not A-D, pooled or unseen-shop evaluation. Previously examined periods are not independent confirmation. Constant-scale exclusions and scored-unit coverage match incumbent for every variant; reserved offset 65 stayed unopened. Full coverage, group diagnostics, pinned inputs/code hashes and runtime versions are saved under results/. Exact training origins/model hashes are in training_manifest.json.

### Final artifact verification

`.venv/bin/python iterations/13-recency-weighting/verify.py`: passed. Six controls
match original arrays exactly; all group partitions conserve score counts/error;
locked data, code and winning submission hashes match. Reloaded largest candidate
model reproduces its forecast array with maximum absolute difference 0.0.
`git diff --check`: passed. Successful incumbent remains untouched.
