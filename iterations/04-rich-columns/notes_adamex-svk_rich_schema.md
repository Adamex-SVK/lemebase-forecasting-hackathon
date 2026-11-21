# Rich schema notes (iteration 04, step 4.0)

Evidence: `.venv/bin/python src/probes/rich_codes.py` (throwaway probe). Mapping below is what `src/rich.py` (Tasks 2-3) implements. Interpretations are inferred from the data, not confirmed by the data owner (see F3 below).

## Set C (`set-c/daily.parquet`)
Numeric columns are strings/decimals: cast `units`, `sales_amount`, `unit_price`, `discount_pct` with `pd.to_numeric`.

Observed codes:
- `sales_type`: R 444045, C 45713, PROMO 1291, R_RETURN 696, MD 65, PROMO_RETURN 40, MD_RETURN 7. Suffix `_RETURN` = return row (`is_return` is 1 on returns, 0 on sales, -1 on stock-only rows with `units` 0 (all 462376 of them, checked by the probe)). Base type = `sales_type` with `_RETURN` removed.
- `stock_status`: R 442908, C 48949. Crosstab: PROMO, MD, C sales types occur only with `stock_status` C; R sales types are almost all status R (a few R rows with status C).
- `discount_pct`: positive on only 0.95% of rows. PROMO rows: median discount 0.25, 97% have `discount_pct` > 0. MD rows: median 0.25, 77% > 0. `C` and `R` sales types: ~0% discounted.
- Price: median `unit_price` is 9.98 for status C vs 34.95 for R. Price ratio to the article's own median list price is 1.00 in both statuses (MD rows 0.86), so status C is not a price cut of the same article: it is a separate low-price / .98-ending article group (70% of C prices end .98/.99 vs 32% for R). 78 of 338 articles switch status over time.
- `is_phasing_out`: 24% of rows; 0% discounted, ~0 net units sold (-43 total), median `days_since_receipt` 469 vs 249. It is not linked to `stock_status` (C share 6.5% among phasing-out vs 11% otherwise).

Decisions:
- `markdown_level`: **NaN for all C rows** (provisional). C's `stock_status == 'C'` is not supported as clearance: price ratio 1.0, zero discount for sales_type C, and unrelated to `is_phasing_out`. `MD` is a markdown code but has only 65 rows and no tier information, so it is not mapped to a level either. Alternative (needs an owner decision): MD -> level 1; with 65 rows the effect on the model would be negligible.
- `promo_flag` = 1 when the series-week has any row with base `sales_type == 'PROMO'` (incl. `PROMO_RETURN`: ignore returns for the flag, use non-return rows only).
- `disc_share` = units-weighted share of the week's net sales rows (non-return) with `discount_pct` > 0 (covers PROMO and MD).
- No separate `stock_status` column: C's `R`/`C` code is not mapped to a rich column (see Deviation).
- `price` = units-weighted mean `unit_price` of non-return rows in the week times (1 - `discount_pct`) i.e. realised price; if no sale in the week, last `unit_price` forward-filled (list price).

## Set D (`set-d/sales.parquet`, `set-d/stock.parquet`)
- `sales_type` counts: FULL_PRICE 93929, PROMO 23651, MARKDOWN_TIER_1 4384, MARKDOWN_TIER_2 2434, CLEARANCE 1353. Returns are rows with `is_return == 1` (same `sales_type` codes).
- `stock.stock_status`: FULL_PRICE 261899, MARKDOWN_TIER_1 12091, MARKDOWN_TIER_2 6937, CLEARANCE 3798, PROMO 30 (negligible).
- `end_date` can be `9999-12-31` (open ended): clip to a late date before `pd.to_datetime` (pandas overflows otherwise), then clip to the last week of the panel.
- Crosstab sales_type x stock tier (92591 of 125751 sales rows matched to a stock range by store, article, date): FULL_PRICE -> stock FULL_PRICE; MARKDOWN_TIER_1 -> stock TIER_1; MARKDOWN_TIER_2 -> stock TIER_2 (17 rows in TIER_1); CLEARANCE -> stock CLEARANCE (56 in TIER_1, 21 in TIER_2); PROMO -> stock FULL_PRICE (16866) with a few in markdown tiers. So tiers in sales are consistent with tiers in stock, and PROMO is a sales-side overlay on a FULL_PRICE (mostly) article.
- These crosstab counts rest on the D7 grain checks in the probe: every matched sales row hits exactly 1 stock range (92591 rows, overlap count {1: 92591}, 0 rows with conflicting tiers, 0 duplicate (store, article, start_date) stock rows), so the merge does not multiply rows; `articles.article_id` of B is unique (1500). The 33160 unmatched sales rows (26%) are: 22898 sold after the article-store's last stock range (the stock table ends 2026-04-23, sales run to 2026-04-24; stock ranges end when the article has no stock, so the sale is outside any recorded range), 6479 on a date in a gap between ranges, 2717 for store-article pairs absent from stock, 1066 before the first range. They are spread over all sales types (FULL_PRICE 23751, PROMO 6457, TIER_1 1677, TIER_2 824, CLEARANCE 451; 1.1% returns), so the crosstab is not biased toward one tier. For these rows use `sales_type` only (no stock tier).
- Realised price vs the matched stock range `unit_price`: PROMO rows sell below it in 86% of rows (median ratio 0.67). FULL_PRICE, TIER_1, TIER_2 and CLEARANCE rows have median ratio 1.00 (12-19% below, rounding/mixed lines). The tier cut is in the stock `unit_price` itself: stock price relative to the article's median stock price is median 1.0 for FULL_PRICE, about 0.50 for CLEARANCE (quartiles 0.40-0.50), and for TIER_1/TIER_2 25th percentile 0.56/0.47 but median 1.0 (only some tier rows are cut). Sales in a tier are therefore at the tier's stock price, so `sales_amount / units` already carries the tier cut, and PROMO is a further discount on top.

Decisions:
- `markdown_level`: FULL_PRICE 0, MARKDOWN_TIER_1 1, MARKDOWN_TIER_2 2, CLEARANCE 3. For a series-week with sales, take the maximum level among the week's non-return `sales_type` codes, excluding PROMO (PROMO rows contribute the level of the stock range covering the date, else 0). For weeks without sales, take the level from the stock range covering the week's Monday (expanded date ranges); stock status `PROMO` (30 rows) is treated as level 0 and also sets `promo_flag`.
- `promo_flag` = 1 if any non-return row in the week has `sales_type == 'PROMO'` (or the covering stock range is PROMO).
- `disc_share` = units-weighted share of non-return units with `sales_type != 'FULL_PRICE'` (design). Note this counts markdown tiers as discounted although their sale prices equal the stock tier price (not below it); the tier cut shows in `price`. Keep the design definition; document it.
- `price` = realised `sales_amount / units` (non-return rows, units-weighted) when sold; otherwise stock `unit_price` of the covering range.

## Set B (`set-b`)
- Promotions: 264, all 6 days long (start to end inclusive). Mechanic: COUPON 97, MANAGER_SPECIAL 80, BUNDLE 60, BOGO 27. Scope: SKU 125 (all have `promotion_articles` rows), CATEGORY 139 (none have article rows; use `promotions.category` matched to `articles.category`).
- Expansion: SKU -> 3099 promotion-article pairs; CATEGORY -> 13869 pairs; total 16968. After expanding to Monday weeks: per article-week 26094 have 1 promotion, 3921 have 2, **maximum 2** (so aggregate with `nunique`/any: no row multiplication once collapsed to a flag). 6 to 12 promotions are active per week (mean 11.7).
- Store scope: `promotion_stores` has exactly one store per promotion, 149 of 264 promotions have a row (all 5 stores appear); the other 115 have none. Rule: promotion with store rows applies to that store only; promotion without store rows applies to **all stores** (assumption, flag it).
- A 6-day promotion starting on a Sunday/other weekday can span two Monday weeks: expand every date in [start_date, end_date] to its Monday week and flag both weeks.

Decisions:
- `promo_flag` = 1 when the (store, article, week) has at least one expanded promotion (article from SKU list or category match; store from rule above). Optionally also `promo_n` (count, 0-2) for B only; not part of the shared schema.
- `markdown_level`: NaN (B has no tiers). No separate `stock_status` column.
- `disc_share` = units-weighted share of lines with `line_discount` > 0; `price` = units-weighted mean `unit_price` (lines table has no store: join `baskets.store_id` via `transaction_id`).
- Grain check before joining: collapse promotions to one row per (store, article, week) before merging onto weekly sales (assert row count unchanged).

## Deviation from the design (Review Focus 5)
Design says C `markdown_level` comes "from `stock_status` / `sales_type`". Evidence does not support it: C `stock_status == 'C'` is not a clearance code, so C `markdown_level` is NaN (provisional). D's tier mapping follows the design. Therefore only D has a non-NaN `markdown_level`; the technique runs with markdown_level observed for one donor shop out of three.

`stock_status` is merged into `markdown_level` (D's stock tiers already feed it; C's `R`/`C` is not a markdown code; B has none). Tasks 2-3 therefore build exactly 4 rich columns: `price`, `disc_share`, `promo_flag`, `markdown_level`, not a fifth `stock_status` column (this supersedes the design table's fifth row).

## Open (F3)
- Confirm in the HF Community tab (or from the data owner) what `R` and `C` mean in C (`stock_status`, `sales_type`) and whether `MD` is a markdown with tier. Until confirmed, C `markdown_level` stays NaN and the codes are flagged as inferred.
- B promotion store scope default (no store rows = all stores) is an assumption.
- D disc_share counts markdown tiers although their sale price equals the stock tier price; decision owner may prefer PROMO-only.
