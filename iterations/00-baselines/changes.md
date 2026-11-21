# Iteration 00: changes versus the previous iteration

This is the first iteration, so there is no previous one. It establishes the reference everything else is compared with. Later iterations list only what differs from the one before.

| What | Result |
|---|---|
| Weekly tables for A, B, C, D, test (`data/interim`) | Built with D1-D7, D21, D24; integrity checks pass |
| Local metric and cuts (`src/validate.py`, D14, D15, D26) | Reproduces the brief's 8 published baselines within 1.6 points (A exact) |
| Baselines (`src/baselines.py`, `results.csv`) | Best simple model on the test-shop cut: median of the last 13 weeks, -6.7 (AutoARIMA -28.8, naive -58.3) |

Reference scores for iteration 01 (local cuts, series level, D26): see the table in `log.md`.
