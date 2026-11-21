# Experiment log

2026-11-21: user approved hurdle-lognormal target-shop first. Scope frozen in
plan.md before scoring. Historical target support: one fractional positive value,
90.3 for A203032 on 2020-09-28; retain unchanged. Each of six inner training/validation
splits has 14-week origin separation for 13-week labels. No outer labels used for
spread fitting or stopping. All commands, versions, dates and hashes preserved.

Implementation checks passed before training: `.venv/bin/python iterations/20-hurdle-lognormal/test_hurdle.py` verifies inverse mixture CDF, boundary p=.5 and sigma=0, invalid probabilities, agreement with 300,000 Monte Carlo draws (seed 0), and feature invariance to changes at/after origin. Simulation is a unit check, not the forecast inference method.

Training command: `.venv/bin/python iterations/20-hurdle-lognormal/run.py`. One candidate; original parameters, seed 0; no outer score-dependent changes.

Completed all six paired periods. Primary incumbent mean -13.665947,
hurdle ensemble -13.753767; mean gain -0.087820.
Primary wins 3/4, but weeks 8–13 mean gain -0.396271.
Both supplementary periods regress. Candidate fails prespecified promotion gate;
recommend retaining original champion. No automatic model promotion.

The three primary gains do not outweigh the offset-26 loss. No claim of proving
or removing overfitting: these are repeatedly examined development periods.
Standalone rounded hurdle score is saved as diagnostic in scores.csv and does
not determine blend selection. Constant lognormal spread and distribution shape
remain modeling limitations; no causal attribution for the regression established.

Commands completed successfully:
`.venv/bin/python iterations/20-hurdle-lognormal/verify.py`
`.venv/bin/python iterations/20-hurdle-lognormal/diagnostics.py`
All six fresh controls reproduce saved forecasts exactly. Input/code hashes match,
group totals reconcile to official MASE, model reload differences are zero, and
all six saved distribution forecasts reconstruct exactly. Probability/Brier/spread
diagnostics in results/distribution_diagnostics.csv; not used to tune candidate.

Python 3.12.4, LightGBM 4.7.0, NumPy 2.5.5, pandas 2.3.3;
SciPy version separately in diagnostics_versions.json. Seed 0. Dates, exclusions,
scored-unit coverage and standalone scores: results/scores.csv; paired comparison:
results/paired.csv. Artifact hashes: results/artifact_hashes.json.

Target-shop own-only evaluation as explicitly approved; no A-D/pooled experiment
or unseen-shop claim. Raw data and incumbent CSV preserved. No final submission,
upload, new tuning, Monte Carlo inference, or perturbation experiment performed.
Results review checkpoint reached; next scope remains user's decision.

User subsequently requested an uploadable CSV. Generated full-history challenger
with `.venv/bin/python iterations/20-hurdle-lognormal/make_final.py`, unchanged
hurdle parameters and 70/30 recipe. Existing submission validator passes: 2,904
articles, 13 weeks; 4,009 cells differ from incumbent. Output:
submissions/ensemble_top3_hurdle_lognormal30_v1.csv. Final models, distribution,
source/code/data/output hashes and tree counts in final/. Incumbent reconstruction
matches exactly and its file hash remains unchanged. Challenger still failed
historical review; CSV generation is not promotion. No upload performed.
