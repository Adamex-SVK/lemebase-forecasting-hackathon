# Experiment log (append-only)

## 2026-11-21 — Phase 2 authorized

User requested Phase 2 implementation. Consulted senior-data-scientist SKILL.md
and ds-workflows.md. Implemented a separate opt-in feature adapter; original model
and final submission remain unchanged. Prespecified seed 0 and original LightGBM
parameters. Six outer cuts, two paired model fits each, with purged inner stopping.
No external dependencies installed and no remote donor code executed.

### Pre-run verification

Commands:
- `.venv/bin/python -m unittest discover -s tests -p 'test_activation_features.py' -v`: 3 passed (hand-computed definitions, future invariance, row alignment).
- `.venv/bin/python -m unittest discover -s tests -p 'test_benchmark_diagnosis.py' -v`: 3 passed.
- `.venv/bin/python tests/test_features.py`: passed.
- `.venv/bin/python tests/test_validate.py`: passed.
- `git diff --check`: passed.

Existing tests emit NumPy timedelta deprecation warnings; assertions passed.
Runtime: Python 3.12.4, LightGBM 4.7.0, NumPy 2.5.5, pandas 2.3.3;
additional runtime versions recorded in requirements-runtime.txt. No dependencies changed.
Execution: `.venv/bin/python iterations/11-activation-features/run.py`.

### Completed comparison — no promotion

Twelve outer-cut model fits completed (each with inner early stopping and final
refit). All six retrained controls exactly reproduce the saved LightGBM arrays:
maximum absolute difference zero. Foundation forecasts and incumbent inputs are
pinned and hashed in results/metadata.json. Model/forecast hashes and exact origin
indices are retained in results/training_manifest.json; local binaries are ignored.

| Offset | Incumbent score | Activation score | Gain |
|---|---:|---:|---:|
| 0 | -3.724944 | -3.521996 | +0.202948 |
| 13 | -25.907431 | -25.994059 | -0.086629 |
| 26 | -11.010614 | -10.916705 | +0.093909 |
| 39 | -14.020799 | -14.040675 | -0.019876 |
| **Primary mean** | **-13.665947** | **-13.618359** | **+0.047588** |
| 16 (additional) | -36.258867 | -36.353266 | -0.094399 |
| 32 (additional) | -17.715974 | -17.743568 | -0.027594 |

Full-precision values in results/scores.csv and paired_gains.csv are authoritative.
Primary weeks 8–13 score improves -15.267328 -> -15.210306 (+0.057022).
However, primary gain is below the required +0.5 and only 2/4 primary periods
improve (required 3/4). Both genuinely additional historical offsets regress.
The 0/16/32 forward-check mean improves only +0.026985 because the reused offset-0
gain outweighs those two losses. This is weak, inconsistent development evidence.

Recent first-sale <=13-week products improve their equal-period mean group score
by approximately +1.317, but their scale-weighted contribution to total MASE barely
moves; large errors remain. Rare sellers improve slightly; recent-inactive group
is unchanged; older products and zero-stock products slightly regress. See
results/primary_groups.csv and groups.csv. Groups use only pre-cut information,
and stock/activation groups overlap. No causal or unseen-shop inference is made.

Coverage is identical across variants: 2,904 products per cut; primary scored counts
2,839 / 2,762 / 2,680 / 2,599, excluded 65 / 142 / 224 / 305. Scored holdout-unit
coverage is 99.138% / 98.460% / 98.726% / 99.602%. This remains own-only target-shop
evaluation, not A-D or pooled-transfer evaluation. No untouched future confirmation
was available; the reserved older offset-65 stress period was not opened.

Decision: preserve the candidate as a reproducible negative/weak experiment; keep
the original ensemble as champion. No final CSV generated or changed, no upload,
no commit/push, and no Phase 3 started. Checkpoint for user review.

### Verification and transcription correction

The first artifact verifier run hit a pandas/NumPy assertion API mismatch
(comparing Series with assert_equal). Changed that assertion to compare arrays;
no model, prediction, metric or feature changed.

Correction to rounded table transcription above: offset 16 incumbent -36.258865,
candidate -36.353267, gain -0.094402; offset 32 incumbent -17.715976, candidate
-17.743567, gain -0.027591; offset 39 gain rounds to -0.019875. Machine-generated
scores.csv and paired_gains.csv contain the correct values throughout.

Final verification: `.venv/bin/python iterations/11-activation-features/verify.py`
passed. Locked data/submission unchanged; code hashes match; all six controls
exact; 18 score rows and all group partitions checked. Reloaded offset-0 candidate
model reproduces saved predictions with maximum absolute difference 0.0.
Re-ran the three activation tests and `git diff --check`: passed.
