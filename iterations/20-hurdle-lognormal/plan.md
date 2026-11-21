# Hurdle-lognormal — fixed target-shop experiment

2026-11-21: user explicitly approved hurdle-lognormal on target shop first.
Consulted senior-data-scientist skill. One fractional target observation (90.3)
precludes assuming strictly integer counts without changes; preserve every target.

Hypothesis: explicit zero probability plus a positive distribution yields better
weekly medians for intermittent sales than L1 point regression. Not a reproduction
of DeepAR/renewal paper architectures; a small distributional challenger inspired
by their occurrence/size distinction. No arbitrary perturbation or Monte Carlo.

P(Y=0)=1-p. Conditional positive Y/scale is lognormal(mu,sigma). Fit binary
LightGBM and L2 LightGBM to log(positive normalized units), respectively, with
original feature list including h; original seed-0 parameters (63 leaves, min leaf
200, L2 penalty 1, learning rate .05, fractions .8, threads 8). Seven inner
validation origins with >=13-week target purge, cap 3000, patience 50. Select
rounds independently then final refit int(best*1.1)+1 on all eligible origins.
Estimate one shared sigma from positive inner-validation log residual RMS before
refitting; validation used for stopping and spread estimation, not outer labels.
This is an approximation; constant sigma, lognormal shape and uncalibrated binary
probabilities may misfit count-like demand. No calibration or parameter search.

Weekly median: zero if p<=.5; otherwise exp(mu + sigma*Phi^-1(1-.5/p))*scale.
No future stock or simulated future covariates; direct marginal horizons 1..13,
not a joint path model. Zero-scale histories default to zero and are excluded by
unchanged evaluator. Preserve original training rows/sample weights and data.

Replace only the 30% LightGBM component with hurdle medians; retain pinned 70%
foundation average, foundation-only zero rule, rounding once and clipping. Also
report rounded standalone medians as a diagnostic, not an extra tuned candidate.
Primary offsets 0,13,26,39, supplementary 16,32. Paired freshly trained L1 controls
must reproduce saved incumbent. Save models, distribution parameters, keys and
predictions. Full/early/late MASE, coverage, exclusions, and history-defined groups.
Gate: >=.5 mean primary gain, >=3/4 wins, nonnegative late mean gain and no losses
on either supplementary period. Known development periods, not independent tests.

Scope is explicitly target-shop own-only first. A-D/pooled deferred to next user
checkpoint; no cross-shop claim or D coverage claim from this experiment. Offset
65 closed; source continuations excluded; champion preserved. No final submission,
upload, perturbation, extra models or search. Stop at results review.

Commands: .venv/bin/python iterations/20-hurdle-lognormal/test_hurdle.py
.venv/bin/python iterations/20-hurdle-lognormal/run.py
.venv/bin/python iterations/20-hurdle-lognormal/verify.py
