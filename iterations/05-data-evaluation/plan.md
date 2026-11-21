# Data evaluation before model selection

Authorized scope: evaluate the existing local A-D and target-shop datasets with
one agent; report findings and required user actions. No training, submission,
raw-data changes, or changes to preprocessing and validation conventions.

Consulted `senior-data-scientist/SKILL.md` and `references/ds-workflows.md`.
Use temporal inspection and the existing MASE evaluator, rather than the skill's
generic regression scorer or random splits.

1. Inventory raw Parquet files and column quality; fingerprint inputs.
2. Check prepared keys, weekly continuity, numeric validity, and reconcile sales
   against raw aggregation under the existing preparation rules.
3. Measure sparsity, activity, stock availability, aggregate patterns, and shop differences.
4. Enumerate four proposed nonoverlapping 13-week holdouts; distinguish a valid
   forecast context from enough target-shop windows to train the current model.
5. Re-score saved LightGBM predictions, compare median13, and diagnose error groups.
6. Save evidence, explain limitations, and recommend the next decision checkpoint.

Completion: reproducible audit scripts, numerical outputs, an inspected plot,
and a findings log. Recommendations remain proposals pending user review.
