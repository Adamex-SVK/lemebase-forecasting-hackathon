2026-11-22. Read the published target history only. Found 2,904 products × 143 weeks. The target provides one observation per product-week. The weekly stock ledger makes it possible to measure changes in stock alongside units sold. A gain can represent a delivery, correction, transfer, or other adjustment.

Ran .venv/bin/python iterations/32-rethink/history/inventory.py. On the four standard target windows, high-flow products with sales in 1–3 of the prior 13 weeks sold in 3.2–9.8 percentage points more future weeks than matched low-flow products. The matched strata had 74–91 effective pairs per cut. Rows in stock_matched.csv compare products with exactly the same count of recent selling weeks and similar recent volume and stock cover. For the 4–8-week group the corresponding difference was 0.8–5.9 points, with 39–47 effective pairs per cut. Products can recur over dates; these are descriptive, reused development folds. Historical demand, assortment, seasonality and other unmeasured factors can affect both stock decisions and future demand.

After the review checkpoint Maya authorized the controlled stock-flow experiment. Rules were recorded in plan.md before the fits. Ran the following from the repository root:

    .venv/bin/python iterations/32-rethink/history/train.py target
    .venv/bin/python iterations/32-rethink/history/train.py shops
    .venv/bin/python iterations/32-rethink/history/check.py features

The independent training-sample check reproduced all incumbent feature values and all binary labels on 98,088 examples. Corrupting later ledger values did not change the as-of-origin features. Own-shop A and B reproduced the existing CDF forecast in every cell as expected from their empty numeric inventory histories. Original data, winning forecast and earlier experiment outputs were preserved.

All 15 standard keyed files passed the equality-of-keys, finiteness, and nonnegativity checks in check.py results. Loading the full target model reproduced all 37,752 raw tree forecasts. Seed 0, Python 3.12.4, NumPy 2.5.5, pandas 2.3.3, LightGBM 4.7.0. Per-cut dates and training row counts appear in results/*training.log; keyed validation counts in checks.csv.

After the root froze model selection, ran .venv/bin/python iterations/32-rethink/history/stress.py. The two separate raw-tree fits used history through 2021-10-18 for the 2021-10-25 origin. The saved stress/incumbent.parquet and stress/history.parquet were checked for expected row count, one-to-one keys and an exact parquet read-back. No stress outcome values were used for fitting or scoring by this agent.
