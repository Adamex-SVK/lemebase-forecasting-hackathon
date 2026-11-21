# Sale probability gate, 2026-11-21

Run with `.venv/bin/python iterations/15-sale-probability-gate/run.py --output NEW_DIRECTORY`. Source commits and file hashes are in results/metadata.json. The probability arrays were unavailable locally. We reproduced the binary classifier from the donor's recipe using seed 0 and the original purged early stopping. Training used pre-cut target-shop data only.

All four standalone gated and ungated primary scores reproduced the published seed-0 results to 1e-7. The six ungated ensemble scores reproduced the frozen benchmark to 1e-8. The group counts and MASE contributions reconcile with the canonical scorer.

Four-period standalone gain was +0.4197 points. Ensemble gain was +0.0609, with wins on three of four primary periods. Mean late gain was +0.0431. Additional gains were +0.0777 and -0.0261. Keep the champion. The four periods have been used previously and the experiment covers only the target shop. No leaderboard evaluation was performed.

The user later reported a public leaderboard score of **−4.9** for `ensemble_top3_lgbm30_sale_gate_v1.csv`. The original ensemble scored **−4.8**, a 0.1-point lead on the public weeks. Keep the original ensemble as the preferred submission.
