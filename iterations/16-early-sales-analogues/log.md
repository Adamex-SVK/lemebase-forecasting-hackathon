# Historical peer forecasts, 2026-11-21

The fixed peer rule scored −13.8178 across the four primary periods. The original ensemble scored −13.6659. The newest period improved by 0.9228 points. The other three lost 0.0928, 0.7705 and 0.6670 points. Mean weeks 8–13 improved by 0.1795 points, and the additional periods split (+0.6449 and −0.9060). Retain the original ensemble.

Across 337 recent-product observations in the four primary periods, 333 met the peer-count requirement and 93 received at least one different weekly forecast. These are observations per forecast origin. Multiple observations can represent the same article. The recent-product mean MASE rose from 12.1387 to 12.1698. A short prefix can have many exact ties, with a median of 19 donors used across recent-product observations. The evidence covers historical periods from the target shop that were previously used in development.

Run from the project root with the project virtual environment:

    .venv/bin/python tests/test_early_sales_analogues.py
    .venv/bin/python iterations/16-early-sales-analogues/backtest.py --output /tmp/early-sales-analogue-reproduction

Use a new output directory. The method is deterministic. Two synthetic tests verified week alignment, the last permissible donor label, fallback, boundary ties, and matching based on selling frequency and weekly sales shape. All six champion scores reproduced the frozen references. Scored group counts and weighted errors reconciled to the project MASE evaluator. See results/paired.csv for date-offset comparisons and results/scores.csv for dates, excluded series and unit coverage. Results/metadata.json records code, data and saved component hashes.
