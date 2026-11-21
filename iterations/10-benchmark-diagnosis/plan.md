# Phase 1: diagnose the incumbent and freeze validation

User authorized implementing phase 1 of the three-phase improvement plan.
Consulted the Senior Data Scientist skill and workflow reference. Scope:
read-only evaluation, reproducible diagnostics, inventory additional predictions,
write the validation contract, and prioritize one hypothesis for review.

Preserve the exact incumbent and submission; train no replacement model, tune no
weights, and create/upload no new final CSV. All diagnostics use the project's
existing MASE convention. History-defined groups use no holdout actuals.

1. Verify original four benchmark scores and error decomposition.
2. Analyse activity, sparsity, recent level shifts, positive sale variability,
   last stock, stock missingness, forecast disagreement, horizon and direction.
3. Quantify post-hoc error concentration without using outcome-selected IDs as rules.
4. Inventory the latest fetched donor commit and use complete additional forecasts
   as overlapping development evidence. Do not describe them as independent tests.
5. Freeze a feasible purged forward-validation protocol and explicitly identify
   the absence of untouched confirmation. Reserve feasible older windows for
   unscored stress tests, not falsely label them prospective future validation.
6. Select one evidence-backed phase-2 hypothesis; no phase-2 implementation yet.

```bash
git fetch origin
.venv/bin/python iterations/10-benchmark-diagnosis/diagnose.py
```

Script pins source commits and refuses to overwrite results. Use a fresh
`--output` for another run. Detailed product diagnostics are regenerable/local;
commit compact scores, provenance, group summaries, plot and conclusions.
