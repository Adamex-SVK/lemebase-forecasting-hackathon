# Iteration 04: rich-column techniques, design (draft for approval, 2026-11-21)

Status: DRAFT, not approved. Nothing is built until the decision owner approves this document (then `writing-plans`).
Authority: D18 (the four techniques), D19 (order: feature dropout, cleaning, auxiliary tasks, teacher-student), D7 (check the grain before every join), D21, D26, D27, `TEAM-RULES.md` section 2 (adoption bar).

## Intent and why
The test shop gives only four columns (`article_id`, week, `units`, `on_hand`). The four training shops carry much more: prices, discounts, promotions, markdown tiers, stock status. The goal of this iteration, stated by the user, is **a better test-shop score**: use the donor shops' rich columns to make a model that, at forecast time, sees only the test shop's four columns but has learned better dynamics. Extra credit (own vs pooled per shop) is a by-product, not the target.

## Goals
- G1. One **shared rich schema**, built once, joined at the weekly grain for the donor shops, reusable by all four techniques.
- G2. Technique 1 (**feature dropout**) run and scored on the 4 rolling test-shop cuts (`src/rolling.py`).
- G3. Techniques 2-4 (cleaning, auxiliary tasks, teacher-student) run in D19 order on the same schema, each with its own decision gate.

## Non-goals and anti-goals
- Not a change to the metric, the cuts, or `src/validate.py` / `src/prepare.py` defaults (core files change only with a decision entry and tests; add options, keep defaults).
- Not generating synthetic data (D23). Not touching the foundation-model work (iteration 03 files).
- Anti-goal: a gain that appears only on cut 0, or from tuning on the scoring cuts. Anti-goal: any feature that uses weeks at or after the origin (leakage).
- Honest risk: the pooled arm has so far not beaten own data (cut 0: own -4.83, pooled -5.84, others only -23.07), so donor data may add little. This iteration tests that; a null result is a valid, reportable outcome.

## Constraints
- Test-shop rich inputs are always missing at scoring: every rich feature is NaN there. The model must work with them missing.
- Past-only features at origin `o`; `tests/test_features.py` extended for each new feature (including a leakage test).
- D26 metric, mean over the 4 cuts, cuts won, paired interval; adopt only if it clears the bar (at least +0.5 mean and at least 3 of 4 cuts, see `TEAM-RULES.md`). Seed noise is about 0.8 on one cut, so use several seeds and the mean over cuts.
- Own result files only: `results_<handle>_<topic>.csv`, `notes_<handle>_<topic>.md`; `log.md` and `changes.md` belong to the iteration owner. Run `bash scripts/check.sh` before proposing a PR. No raw data or large files committed.

## Approaches considered (decided by the user: shared schema)
1. **One shared rich schema, weekly grain (chosen).** Build `data/interim/rich_<shop>.parquet` (key store, article, week) with common columns; join C and D first, then B; skip A (its only extra column, `in_stock`, is already used). All techniques read the same table. Cost: more upfront work; benefit: one definition of "promo", "markdown" and "price" for every technique.
2. C and D only. Faster first result, but leaves B's promotion data untested.
3. Per-technique tables. Simpler per step, but repeated joins and definitions that can drift apart.

## Design
### Rich schema (builder to finalise definitions from the evidence; NaN where a shop lacks the column)
Per series-week, from the donor shops' raw files, aggregated to Monday weeks like the weekly tables (D2/D6):
| column | meaning | C | D | B |
|---|---|---|---|---|
| `price` | mean realised unit price in the week (list price if no sale) | `unit_price` | `unit_price` (stock) / `sales_amount/units` | `unit_price` from `sales_lines` |
| `disc_share` | share of the week's units sold with a discount | `discount_pct` > 0 | `sales_type` != `FULL_PRICE` | `line_discount` > 0 |
| `promo_flag` | a promotion is active for the series | `sales_type == PROMO` | `sales_type == PROMO` | `promotion_articles` x `promotions` dates (article scope; CATEGORY scope via `articles.category`) |
| `markdown_level` | 0 full, 1 tier 1, 2 tier 2, 3 clearance | from `stock_status` / `sales_type` (codes `R`, `C`: confirm meaning, D4 / F3) | `MARKDOWN_TIER_1/2`, `CLEARANCE` | none (NaN) |
| `stock_status` | regular vs clearance/other | `stock_status` | `stock.stock_status` (date ranges, expanded to weeks) | derived from stock events if useful, else NaN |
Grain checks (D7) before every join: B promotions multiply rows (an article in several promotions, store-level scope); D stock is a date-range table; C and D rows are daily and per store. Row counts before and after each join are asserted and logged. Returns are netted as in D3.

### Rich features (model inputs)
Past-only, computed on `[o-W, o)` windows, scale-free so shops are comparable: price ratio (last 4 weeks vs the 26-week median), `disc_share` and `promo_flag` shares over 4 and 13 weeks, current `markdown_level`, weeks since last promo/markdown, plus a `rich_missing` indicator. They are added next to the existing `FEATURES` as an option (`features_at(..., rich=True)`); default behaviour unchanged.

### Technique 1: feature dropout
Train the pooled global LightGBM (D27, D28, D29 as in iteration 01/02) on all shops' windows with the rich features, masking them (set to NaN) on a random share of windows (grid for the share, for example 0.5 and 0.8, plus whole-group masking), so the trees learn rules that work when the columns are missing. Test-shop windows have the rich features NaN by construction. Compare on the 4 cuts against: (a) own arm, (b) pooled WITHOUT rich features on all 4 cuts (new reference, pooled was only run on cut 0), (c) pooled with rich features and no dropout.

### Techniques 2-4 (outline; each gets its own plan and gate after technique 1)
- Cleaning: mark promo/markdown weeks in the donor shops so the lags and means used as inputs and targets do not read promo spikes as ordinary demand (cleaned lags vs raw lags as an option).
- Auxiliary tasks: heads (or stacked models) that predict promo-week and markdown level from the four columns, whose output becomes a feature available at the test shop.
- Teacher-student: a teacher with all columns on the donor shops; a student with only the four columns trained to match the teacher's forecasts (alone and blended with the true target).

## Steps and gates
| # | Step | Output |
|---|---|---|
| 4.0 | Inspect and decide codes (C `R`/`C`, markdown mapping), grain checks, B promotion expansion | `notes_<handle>_rich_schema.md`; decisions logged |
| 4.1 | Build `rich_<shop>.parquet` for C, D, B + tests (grain, no future info) | tables + `tests/` |
| 4.2 | Reference runs: pooled no-rich on 4 cuts; own arm | `results_<handle>_reference.csv` |
| 4.3 | Rich features + feature dropout on 4 cuts, several seeds | `results_<handle>_dropout.csv`, decision gate |
| 4.4-4.6 | Cleaning, auxiliary tasks, teacher-student (each its own plan/gate) | |

## Left to the builder (marked as the builder's calls)
Exact column definitions above, window lengths, the dropout share grid, whether pooled training uses the D28 weighted sampling unchanged (default: yes), compute budget (pooled arms are slow: 7-9M rows; D31 origin step 4 for other shops applies).

## Open for the decision owner
- Approve this design (then the implementation plan is written).
- Whether the markdown/clearance meaning of C's `R`/`C` codes should be confirmed in the HF Community tab (F3) or inferred and flagged.
