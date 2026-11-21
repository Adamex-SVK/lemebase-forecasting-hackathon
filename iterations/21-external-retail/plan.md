# External sales history for the winning ensemble

The user selected additional real sales beyond packaged Sets A–D and requested a distinct-record check before training. The forecast of interest starts on 2023-04-24. The reconstructed outcomes are outside the training data.

First evaluate Daqing Chen's UCI Online Retail dataset (2015, DOI 10.24432/C5BW33, CC BY 4.0). It has dated item-level invoices from a real UK gift retailer, a different domain from the pharmacy target. Compare the release identity and observed week keys with each packaged shop. Count exact repeated invoice lines, isolate merchandise, and trim partial weeks. Only aggregate weekly quantities are needed by the model.

At the source review, report how many complete weeks and distinct products are usable and any material interpretation choices. The proposed model experiment changes only the original LightGBM training data, pooling target history and the verified donor with the existing per-row weighting. Evaluate the fixed 70/30 ensemble on the existing target-shop holdouts under VALIDATION.md. Adding this donor is a hypothesis about the original target shop. The four packaged shops serve as overlap references here.

## Approved experiment

On 2026-11-21 the user approved the UK-retail comparison. Use the 48 consecutive donor weeks starting 2011-01-03, donor origins every four weeks, target origins every two weeks and unit row weights. Donor stock is unavailable. Use the original LightGBM features, L1 objective, seed 0, and target-only purged inner validation for early stopping. Pool the UK and eligible target rows for both fitting steps. Score against the pinned own-only ensemble on primary offsets 0, 13, 26 and 39 and supplementary offsets 16 and 32. The review requires at least +0.5 primary mean gain, at least three primary wins, nonnegative late mean gain, and nonnegative gains on both supplementary periods. Review after historical evaluation.
