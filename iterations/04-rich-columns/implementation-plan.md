# Rich columns and feature dropout: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one shared weekly "rich" table for the donor shops C, D and B, add rich features with feature dropout to the pooled LightGBM, and score it on the 4 rolling test-shop cuts against pooled-without-rich and own-data references (iteration 04, steps 4.0-4.3).

**Architecture:** `src/rich.py` turns raw donor files into `data/interim/rich_<shop>.parquet`, aligned row for row with `weekly_<shop>.parquet`. `src/features.py` gets an opt-in `rich=True` path (matrices in `ShopMatrix`, extra features, per-window dropout in `build_windows`); defaults stay unchanged. `src/run_rich.py` runs the arms on the test shop's 4 cuts (donors use all weeks, D27) and writes `results_<handle>_*.csv`.

**Tech Stack:** Python 3.13 (`.venv/bin/python`), pandas, numpy, LightGBM; tests are plain scripts like `tests/test_features.py`.

**Spec:** `iterations/04-rich-columns/design.md` (approved by the user with "Continue", 2026-11-21).

## Global Constraints
- Decisions D7 (check grain before every join), D11/D29 (target = units / MASE scale, L1), D26 (metric), D27 (all donor weeks), D28 (weighted sampling for D), D31 (donor origin step 4) are fixed.
- Rich features at origin `o` use only weeks before `o`; the test shop has all rich inputs NaN.
- Core files `src/features.py` and `src/train_gbm.py` change only by adding options; default outputs must be bit-identical (a test pins this). `src/validate.py` and `src/prepare.py` are not touched.
- Adoption bar: mean over 4 cuts at least +0.5 and at least 3 of 4 cuts won, against the stated reference. Several seeds; judge the mean over cuts.
- Own result files only: `iterations/04-rich-columns/results_adamex-svk_<topic>.csv` and `notes_adamex-svk_<topic>.md`. Do not edit `log.md` or `changes.md`. Never commit raw data, files over 5 MB, secrets.
- Commit steps below run only when the user asks for a commit; branch name `it04/adamex-svk/rich-columns`. Commit messages end with the Co-Authored-By trailer from the session. Run `bash scripts/check.sh` before proposing a PR.

## Review Focus
1. B CATEGORY-scope promotions expand to many articles and stores: the row count of `rich_B` must still equal `weekly_B` (no duplicated keys), asserted and tested.
2. A series-week with zero units must not get a NaN or inflated `price`/`disc_share` from a division by zero (`disc_share` = 0, `price` falls back to the list price).
3. Rich features of a series with no rich table at all (A, test shop, masked windows) are all NaN with `rich_missing` = 1, and the model still predicts (no crash, finite forecasts).
4. Dropout probability 0 reproduces the no-dropout windows exactly; probability 1 yields all-NaN rich columns; the same seed gives the same mask.
5. The deviation from the design (see Task 1) is stated in the notes: D's `stock_status` carries the same tier codes as `sales_type`, so `stock_status` is merged into `markdown_level` instead of being its own column.

---

### Task 1: Inspect the codes and write the rich-schema notes (step 4.0)

**Files:**
- Create: `iterations/04-rich-columns/notes_adamex-svk_rich_schema.md`
- Create (throwaway probe): `src/probes/rich_codes.py`

**Interfaces:**
- Produces: the final mapping tables used by Tasks 2-3 (written in the notes file): C `stock_status`/`sales_type` code meaning; D tier mapping; B promotion expansion rule.

- [ ] **Step 1: Write `src/probes/rich_codes.py`** printing, for C: counts of `sales_type` and `stock_status`, and median `unit_price` and `discount_pct` per `stock_status` and per `is_phasing_out`; for D: crosstab `sales_type` x `stock.stock_status`, and the share of rows with `sales_amount / units` below the list price per tier; for B: number of promotions per article-week after expanding SKU and CATEGORY scope, and the maximum.
- [ ] **Step 2: Run it:** `.venv/bin/python src/probes/rich_codes.py`. Expected: prints without error.
- [ ] **Step 3: Write the notes file** with the mapping decided from the output: `markdown_level` 0 full price, 1 tier 1, 2 tier 2, 3 clearance (D tiers `FULL_PRICE`, `MARKDOWN_TIER_1`, `MARKDOWN_TIER_2`, `CLEARANCE`; C `stock_status == 'C'` provisionally clearance only if C's median price ratio or `is_phasing_out` supports it, otherwise leave C's `markdown_level` NaN and say so); `promo_flag` = 1 when the series has a promotion that week (C and D `sales_type == 'PROMO'`; B via promotion dates and `promotion_articles`, CATEGORY scope via `articles.category`, store scope via `promotion_stores`). State the deviation from the design (Review Focus 5) and any F3 question left open.
- [ ] **Step 4: Commit** (only when asked): `git add src/probes/rich_codes.py iterations/04-rich-columns/notes_adamex-svk_rich_schema.md`

### Task 2: Rich table builders for C and D

**Files:**
- Create: `src/rich.py`
- Create: `tests/test_rich.py`

**Interfaces:**
- Produces in `src/rich.py`:
  - `RICH_COLS = ["price", "disc_share", "promo_flag", "markdown_level"]`
  - `rich_c(daily: pd.DataFrame, weekly_keys: pd.DataFrame) -> pd.DataFrame`
  - `rich_d(sales: pd.DataFrame, stock: pd.DataFrame, weekly_keys: pd.DataFrame) -> pd.DataFrame`
  - `load_rich(shop: str) -> pd.DataFrame | None` (reads `data/interim/rich_<shop>.parquet`, `None` for A and test)
  - `weekly_keys` has columns `store_id, article_id, week` (the shop's weekly table keys). Every builder returns exactly one row per key, same order, columns `store_id, article_id, week` + `RICH_COLS` (float32), and asserts `len(out) == len(weekly_keys)` and no duplicated keys (D7).

- [ ] **Step 1: Write failing tests in `tests/test_rich.py`:** `test_rich_c_grain_and_values` with a hand-built daily frame (2 series, 3 weeks, one promo day, one zero-unit week) asserting: output row count equals `weekly_keys`, `disc_share == 0` and `price` equals the list price in the zero-unit week, `promo_flag == 1` only in the promo week, values in range (`price > 0`, `disc_share` in [0, 1], `markdown_level` in {0,1,2,3}); `test_rich_d_tiers` with a hand-built sales/stock frame asserting `MARKDOWN_TIER_2` rows give `markdown_level == 2` and a week with two tiers takes the maximum; `test_rich_duplicate_keys_rejected` asserting `weekly_keys` with a duplicate raises `AssertionError`.
- [ ] **Step 2: Run:** `.venv/bin/python tests/test_rich.py`. Expected: FAIL (`ModuleNotFoundError: rich`).
- [ ] **Step 3: Implement `rich_c`, `rich_d`, `load_rich`** following the mapping in the Task 1 notes. Use `prepare.monday` for weeks. `price` = unit-weighted mean realised unit price of the week, else the mean list price of that week's rows, else NaN. `disc_share` = discounted units / units (0 when no units). `promo_flag` and `markdown_level` = max over the week's days. Left-join onto `weekly_keys` with `validate="one_to_one"`.
- [ ] **Step 4: Run again.** Expected: all tests print OK.
- [ ] **Step 5: Add a `build_all()` function** that reads the raw C and D files and `data/interim/weekly_<shop>.parquet`, writes `data/interim/rich_C.parquet` and `rich_D.parquet`, and prints row counts and the NaN share per column. Run `.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import rich; rich.build_all()"`. Expected: row counts equal the row counts of `weekly_C.parquet` and `weekly_D.parquet`, no assertion error.
- [ ] **Step 6: Commit** (only when asked): `src/rich.py tests/test_rich.py`.

### Task 3: Rich table builder for B

**Files:**
- Modify: `src/rich.py`, `tests/test_rich.py`

**Interfaces:**
- Consumes: `RICH_COLS`, the weekly-keys contract and `build_all()` from Task 2.
- Produces: `rich_b(sales_lines, baskets, promotions, promotion_articles, promotion_stores, articles, weekly_keys) -> pd.DataFrame`; `build_all()` also writes `data/interim/rich_B.parquet`. B has no `markdown_level` (NaN).

- [ ] **Step 1: Write failing tests:** `test_rich_b_promo_expansion` with 2 promotions (one SKU scope, one CATEGORY scope covering two articles, one store-restricted) over 2 stores and 3 weeks, asserting `promo_flag` per (store, article, week) exactly as hand-computed and that an article in two overlapping promotions still gives one row; `test_rich_b_disc_share` asserting `disc_share = discounted units / units` from `line_discount > 0`; `test_rich_b_rowcount` asserting output length equals `weekly_keys`.
- [ ] **Step 2: Run:** `.venv/bin/python tests/test_rich.py`. Expected: new tests FAIL.
- [ ] **Step 3: Implement `rich_b`** per the Task 1 notes: a promotion is active in week `w` if its `[start_date, end_date]` overlaps `[w, w+6]`; reduce to unique (store, article, week) keys before merging (D7). `price` from `sales_lines.unit_price`, list price fallback from `articles.price`.
- [ ] **Step 4: Run tests; then `build_all()`.** Expected: OK and `rich_B` rows equal `weekly_B` rows.
- [ ] **Step 5: Commit** (only when asked).

### Task 4: Rich features in `features.py` with leakage and default-unchanged tests

**Files:**
- Modify: `src/features.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: `rich.load_rich(shop)`, `rich.RICH_COLS`.
- Produces in `src/features.py`:
  - `RICH_FEATURES = ["price_ratio4", "disc4", "disc13", "promo4", "promo13", "markdown_last", "markdown_max4", "weeks_since_promo", "rich_missing"]`
  - `feature_names(rich: bool = False) -> list[str]`: `FEATURES[:-1] + RICH_FEATURES + ["h"]` when `rich`, else `FEATURES`.
  - `ShopMatrix(df, mask_oh_before=None, rich=None)`: optional `rich` frame (columns `store_id, article_id, week` + `RICH_COLS`) fills `M.R`, a dict `col -> n x W float32` array, NaN where missing; `M.has_rich` is a bool.
  - `features_at(M, o, d33=False, rich=False)` returns X with the rich columns inserted before `h` when `rich=True`. Definitions: `price_ratio4` = mean of `price` over weeks `o-4..o-1` divided by the median over `o-26..o-1`; `disc4/13`, `promo4/13` = mean over the last 4/13 weeks; `markdown_last` = value at `o-1`; `markdown_max4` = max over 4 weeks; `weeks_since_promo` = weeks since the last `promo_flag == 1`, capped at `DEAD_CAP`; `rich_missing` = 1 for all series when `M.has_rich` is False, else 1 only where the series' rich values in `o-13..o-1` are all NaN. All NaN (except `rich_missing`) when `M.has_rich` is False.

- [ ] **Step 1: Write failing tests** in `tests/test_features.py` using a synthetic rich frame: (a) leakage: with `rich=True`, overwriting `M.R[...][:, o:]` with random values leaves `features_at(M, o, rich=True)` unchanged, for `o` in (13, 30, 57); (b) default unchanged: `features_at(M, o)` and `build_windows(M, origins)` with defaults equal the outputs of the pre-change code (store the expected hash of `X` for the synthetic data, computed before the edit in Step 2); (c) `ShopMatrix(df)` without `rich` gives all-NaN rich features and `rich_missing == 1`; (d) hand-computed `promo4` and `weeks_since_promo` for one series.
- [ ] **Step 2: Before editing, compute and record** the SHA-256 of `features_at(M, 30)[0].tobytes()` and of `build_windows(M, origins)["X"].tobytes()` on the synthetic data and put them in the test as constants. Run: `.venv/bin/python tests/test_features.py`. Expected: new tests FAIL (missing symbols), old tests pass.
- [ ] **Step 3: Implement** the signatures above. Keep `FEATURES`, `FEATURES_D33` and the default path untouched.
- [ ] **Step 4: Run** `.venv/bin/python tests/test_features.py`. Expected: all pass, including the old leakage test.
- [ ] **Step 5: Commit** (only when asked).

### Task 5: Window-level feature dropout and model plumbing

**Files:**
- Modify: `src/features.py` (`build_windows`, `inference_features`), `src/train_gbm.py` (`fit`, `forecast_matrix`)
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: `feature_names`, `features_at(..., rich=True)`.
- Produces:
  - `build_windows(M, origins, ..., rich=False, rich_drop=0.0)`: with `rich=True`, a fraction `rich_drop` of the windows (chosen with `rng`, one decision per series-origin, all 13 `h` rows alike) get every `RICH_FEATURES` entry except `rich_missing` set to NaN and `rich_missing` set to 1.
  - `inference_features(M, horizon=H, d33=False, rich=False)`.
  - `train_gbm.fit(win_train, win_val=None, rounds=None, params=PARAMS, names=None)` uses `names or f.FEATURES`; `forecast_matrix(model, M, weeks, rich=False)`.

- [ ] **Step 1: Write failing tests:** `test_dropout_zero_is_identity` (`rich_drop=0.0` equals windows built without dropout, same X); `test_dropout_one_masks_all` (all rich columns NaN, `rich_missing == 1`, other features unchanged); `test_dropout_seeded` (same `rng` seed gives the same mask); `test_dropout_share` (observed masked share within 0.05 of 0.5 on 40 series x many origins); `test_forecast_finite_without_rich` (a tiny fit with `names=feature_names(True)` on a synthetic shop without rich table predicts finite non-negative values).
- [ ] **Step 2: Run:** `.venv/bin/python tests/test_features.py`. Expected: FAIL.
- [ ] **Step 3: Implement** the signatures. Defaults leave the Task 4(b) hashes unchanged.
- [ ] **Step 4: Run** `.venv/bin/python tests/test_features.py` and `bash scripts/check.sh` (add `tests/test_rich.py` to `scripts/check.sh` first). Expected: all checks passed.
- [ ] **Step 5: Commit** (only when asked).

### Task 6: Reference runs on the 4 cuts (step 4.2)

**Files:**
- Create: `src/run_rich.py`
- Create: `iterations/04-rich-columns/results_adamex-svk_reference.csv`, `notes_adamex-svk_reference.md`

**Interfaces:**
- Consumes: `build_windows`, `fit`, `forecast_matrix`, `run_arms.merge` (reuse, do not copy), `rolling` cut convention (cut `k` drops the last `13*k` weeks of the test shop), `rich.load_rich`.
- Produces: `run_rich.run(arms: list[str], seeds: list[int], cuts=range(4)) -> pd.DataFrame` with columns `arm, cut_index, seed, score, n_scored, trees, seconds`, saved after every arm-cut; CLI `.venv/bin/python src/run_rich.py ARM[,ARM...] --seeds 0,1 [--cuts 0,1,2,3] > logs/rich_<arm>.txt 2>&1`. Arms: `own`, `pooled_norich`, `pooled_rich_nodrop`, `pooled_rich_drop50`, `pooled_rich_drop80`. Donor windows: all weeks, origin step 4 (D31), `sample_inactive` for D, shop weights `equal` (D30); test-shop windows step 2; fixed 300 trees for pooled arms and the early-stopped count for `own` (the builder may instead reuse the own arm's best iteration; state it in the notes).

- [ ] **Step 1: Write `run_rich.py`** with the `run` function and CLI. Each arm builds donor windows once per run (cache), builds the test-shop windows per cut, fits, forecasts, scores with `validate.score` on that cut's holdout, and appends a row.
- [ ] **Step 2: Smoke test:** `.venv/bin/python src/run_rich.py own --seeds 0 --cuts 0`. Expected: cut 0 score about -4.8 (matches `iterations/02-robust-eval/results_rolling_test.csv`, tolerance seed noise 0.8).
- [ ] **Step 3: Run the references in the background:** `.venv/bin/python src/run_rich.py own,pooled_norich --seeds 0,1 > logs/rich_reference.txt 2>&1` (pooled arms train on 7-9M rows; expect up to about 100 minutes per seed). Poll the log file only when the job ends.
- [ ] **Step 4: Write the notes:** table of score by cut, mean, cuts won, paired interval for `pooled_norich` minus `own`; whether pooling helps on 4 cuts (it decides how much to expect from dropout).
- [ ] **Step 5: Commit** (only when asked).

### Task 7: Rich features and feature dropout (step 4.3) and decision gate

**Files:**
- Modify: `src/run_rich.py` (only if the arms need adjusting)
- Create: `iterations/04-rich-columns/results_adamex-svk_dropout.csv`, `notes_adamex-svk_dropout.md`

**Interfaces:**
- Consumes: Task 6 outputs and the arms `pooled_rich_nodrop`, `pooled_rich_drop50`, `pooled_rich_drop80`.

- [ ] **Step 1: Run:** `.venv/bin/python src/run_rich.py pooled_rich_nodrop,pooled_rich_drop50,pooled_rich_drop80 --seeds 0,1 > logs/rich_dropout.txt 2>&1` (background).
- [ ] **Step 2: Analyse** with the same table as Task 6: each arm versus `pooled_norich` and versus `own`, mean over cuts and seeds, cuts won, paired article-level interval, plus feature importance of the rich features in the best arm.
- [ ] **Step 3: Write `notes_adamex-svk_dropout.md`:** results, whether any arm clears the adoption bar, what it means for techniques 4.4-4.6, and a short "decision for the owner" with evidence, options and a recommendation (CLAUDE.md: the collaborator decides). Failures and null results are reported as they are.
- [ ] **Step 3b: Add a decision entry proposal** for `decisions-to-make.md` only if an arm clears the bar or the owner asks; otherwise just report.
- [ ] **Step 4: Run `bash scripts/check.sh`.** Expected: all checks passed.
- [ ] **Step 5: Commit** (only when asked) and stop at the decision gate: cleaning, auxiliary tasks and teacher-student get their own plans.

---

## Self-review (done)
- **Spec coverage:** shared schema (Tasks 1-3), rich features and leakage (4), dropout (5), references (6), dropout runs and gate (7), D7 grain checks (2-3), adoption bar and seeds (Global Constraints, 6-7). Techniques 2-4 are outside this plan by design.
- **Deviation from the design:** `stock_status` is merged into `markdown_level` (four rich columns, not five); flagged in Task 1 and Review Focus 5.
- **Types:** `RICH_COLS`, `RICH_FEATURES`, `feature_names`, `rich_drop`, `names` are defined once and used with the same names in later tasks.
