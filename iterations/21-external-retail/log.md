# Iteration 21 log

## 2026-11-21 — External source review

The user chose real sales outside the packaged shops. Downloaded Daqing Chen's UCI Online Retail (CC BY 4.0), DOI 10.24432/C5BW33. This is a UK online gift retailer with some wholesale customers. The target shop is a pharmacy. The public workbook has invoice dates and product quantities, with no stock ledger or line identifier.

Commands used to download and reproduce the audit in the project environment:

    curl -fL 'https://archive.ics.uci.edu/static/public/352/online+retail.zip' -o /tmp/lemebase-uci-352-online-retail.zip
    .venv/bin/python -c 'import sys, runpy; sys.path.append("/Users/mayasjain/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages"); sys.argv = ["audit.py", "/tmp/lemebase-uci-352-online-retail.zip"]; runpy.run_path("iterations/21-external-retail/audit.py", run_name="__main__")'

The first network attempt failed at DNS inside the workspace sandbox. The approved download completed and the archive integrity check passed. Audit deterministic, no seed. Versions, input and derived table hashes in results/source_audit.json. The release dates and retailer provenance distinguish these records from A–D and the target. Every packaged shop's actual weekly date set was compared, with zero overlap (results/date_overlap.csv).

Of 541,909 workbook lines, 5,268 are exact repeats on all original fields. The conservative preparation counts each exact line once since the workbook has no unique line ID. Positive-priced, positive-quantity merchandise sales use five-digit product codes with an optional letter suffix. Excluded cancellations, non-product codes and the two partial boundary weeks. This measures gross merchandise units. Output contains 495,525 eligible lines aggregated to 96,034 product-weeks.

The week of 2010-12-27 has no eligible sales anywhere in the shop. Proposed training slice 2011-01-03 through 2011-11-28 contains 48 consecutive weeks with observed sales, 3,709 products, 90,083 observed product-weeks and 4,959,043 units. The existing four-week donor origin spacing gives 251,654 supervised horizon rows. Source review checkpoint reached. The source's stock level is unknown. A UK gift retailer can have different bulk buying, seasonality and returns from the target shop. Measure transfer on the target-shop holdouts before any model change.

## 2026-11-21 — Approved paired training

The user approved the UK retailer at the source checkpoint. Command run from the repository root:

    .venv/bin/python iterations/21-external-retail/backtest.py

All target training weeks precede each holdout. The 251,654 UK horizon rows precede the first target-shop observation. Only target validation windows select the tree count, with a 13-week purge from target training origins. UK stock level is missing throughout, with no positive stockout flags available. Frozen target-only LightGBM controls and all three foundation predictions come from the pinned commits recorded in the validation contract. Seed 0, row weight 1, eight CPU threads. Source, runner fingerprints and library versions are recorded in results/backtest/summary.json. Each candidate fit and its keyed historical forecasts are saved in results/backtest/fits.

The six fits completed. The mean full-horizon score across the four primary periods moved from -13.665947 to -13.733408 (gain -0.067461, two wins). Mean MASE moved from 1.136659 to 1.137334. Weeks 8–13 lost 0.062452 points. Supplementary periods at offsets 16 and 32 gained 0.216929 and lost 0.124165 respectively. The pooled tree alone also scored lower on the primary mean (-15.65652 versus -15.52038). Recommend the original ensemble for the competition horizon.

On the primary periods, mean within-group score changes included -0.605 for products first sold in the previous 13 weeks, +0.004 for products with one to three sale weeks in the previous 13, zero for inactive products, and -0.226 for products whose last stock observation was missing. These are descriptive results on reused development periods. Paired scores, scored counts, excluded counts and scored-unit coverage are in results/backtest/paired.csv. This stage evaluates only the target pharmacy. A–D served as source-overlap checks, with no A–D scores or Set D coverage estimate for this experiment.

Verification command:

    .venv/bin/python iterations/21-external-retail/verify.py

All six saved models reproduced the stored tree and ensemble predictions exactly, the project scorer reproduced the recorded comparison and the review gate failed. Local results checkpoint complete.
