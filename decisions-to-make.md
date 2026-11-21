# Decisions to make

Early-sales peer forecasts (2026-11-21) scored −13.8178 against the fixed ensemble’s −13.6659 on the four primary holdouts, with one win. The two additional periods split. The user reported −4.8 for the 50/50 peer blend on the public leaderboard, tying the original at displayed precision. Retain the original ensemble. See `iterations/16-early-sales-analogues/log.md`.

Sale-probability gate (2026-11-21): keep the fixed ensemble. Applying the seed-0 gate only to its 30% LightGBM component improved the primary historical mean by 0.0609 points (3/4 wins), below the 0.5 threshold. One of two additional periods regressed. The user reported −4.9 on the public leaderboard, 0.1 points below the original ensemble. See `iterations/15-sale-probability-gate/log.md`.

Living list for the team. Settled items are under **Decided** with date and choice. Last updated: 2026-11-21.

## Maya's branch: AutoGluon integration (2026-11-21)

User authorized adding AutoGluon as an optional challenger. Isolate dependencies,
retain current preprocessing and D26 scoring, use prior internal validation
windows, and save outer-cut predictions for independent comparison. Initial
integration is own-shop/units-only; pooling, final model selection, and submission
remain separate steps. See `iterations/06-autogluon/plan.md` and `log.md`.

## Maya's branch: fixed XGBoost blend (2026-11-21)

User requested local testing with 30% LightGBM and 10% XGBoost; announced and
tested 60% foundation average / 30% LightGBM / 10% XGBoost. Fixed settings and
four existing historical cuts; no weight search. Candidate mean -13.8501 versus
incumbent -13.6659; worse on 4/4 cuts. No promotion or submission change.
Evidence and limits: `iterations/08-xgboost-blend/log.md`.

## Maya's branch: phase-1 diagnosis checkpoint (2026-11-21)

Phase-1 checkpoint (2026-11-21): benchmark retained; validation contract recorded
in `VALIDATION.md` and `iterations/10-benchmark-diagnosis/validation.json`.
Recent observed activation is the prioritized hypothesis for the next experiment:
within-activation sale frequency and conditional positive-sale size features for
LightGBM, with all other ensemble settings fixed. This is a proposal, not a
trained/promoted change. Details: `iterations/10-benchmark-diagnosis/log.md`.

## Maya's branch: phase-2 activation experiment (2026-11-21)

User authorized implementation of Phase 2. Test exactly two activation-aware
features with seed 0 and unchanged original LightGBM training settings. Keep
the foundation forecasts, 70/30 blend, zero rule and rounding fixed. Run an
otherwise identical retrained control to measure environment drift. Use primary
offsets 0,13,26,39 and supplementary checks 0,16,32 under the Phase-1 contract.
No outer-fold tuning, pooling, weight changes or final submission generation.
Evidence: `iterations/11-activation-features/plan.md` and append-only `log.md`.

Completed: primary score -13.665947 -> -13.618359 (+0.047588), wins 2/4,
late-horizon gain +0.057022. Both additional offsets (16 and 32) regress.
All six retrained controls exactly match saved forecasts. Candidate fails
the prespecified promotion threshold; retain incumbent. No Phase 3 started.

## Maya's branch: phase-3 learned global blend (2026-11-21)

Rejected: purged global learned weights scored -14.3972 versus incumbent
-13.6659 (-0.7312); all three fitted primary periods and both additional checks
regressed. Oldest period used an unchanged fallback. User requested removing
the unsuccessful changes. Challenger code, tests and full evidence were moved
to ignored local `logs/archived-12-learned-blend/`; they are excluded from GitHub.
Keep the fixed 70/30 ensemble and original submission unchanged.

## Maya's branch: modest recency weighting (2026-11-21)

User authorized one fixed training-recency test. Use 104-week half-life on window
origins; normalize to preserve original total training weight. Validation stays
uniform; original features, loss, seed, parameters, purged stopping, 70/30 blend,
foundation forecasts and rounding remain unchanged. No half-life search.
Senior engineer skill was not found in local/repo/plugin skills; consulted the
available senior-data-scientist skill instead. Original champion remains active.

Completed: primary mean -13.665947 -> -13.630297 (+0.035650), only 2/4 wins.
Additional offset 16 improves, offset 32 regresses. Fails promotion thresholds;
do not apply recency weighting to the main model or submission. Reproducible
isolated experiment and evidence: `iterations/13-recency-weighting/`.

## 1. Task facts (from the brief, not decisions)
- We submit ONE forecast: the **test shop**, 2,904 articles x 13 weeks (w1 = Mon 2023-04-24 ... w13 = Mon 2023-07-17), as a CSV `article_id, w1..w13`.
- Target = **units** (pieces), not money. The test shop has no price column.
- Live leaderboard scores w1-w7 only; the final score uses w8-w13 (hidden). Limit: 100 uploads per team per day. Deadline: not stated; the team wants results by tomorrow morning.
- Score = (1 - MASE) x 100. Per the page: each week's error divided by how much the article usually moves week to week in its history; averaged per article, then over all articles (every article counts equally). Live-board references: Naive -64.5, AutoARIMA -33.7, best team so far -7.1.
- Sets A-D are training / validation data only (no shared stores or products between any sets or the test shop). Extra credit: per shop, own-data-only vs pooled with the other sets, with both scores and an explanation.
- **Test-shop inputs are only 4 columns:** `article_id, week_start, units, on_hand`. It is a real shop; its dates were shifted by whole weeks and article IDs randomized (HF card). `on_hand` is noisy (ledger does not reconcile on ~56% of rows, 7 negatives, 12% missing overall but ~1% in the last 13 weeks; 43% missing in the first 10 weeks). Future `on_hand` is unknown.
- Compute: tonight a laptop (Apple M1, 16 GB, 8 cores; project venv `.venv` on Python 3.13); GPUs from the company tomorrow.

### Data profile after preparation (step 0.4, complete weeks, D2-D5 applied)
Zero-week % = share of series-weeks with 0 units in our weekly tables (each series zero-filled from its start week).

| Shop | Series | Complete weeks | Zero-week % | `on_hand` missing | Brief's daily zero % (raw days) |
|---|---|---|---|---|---|
| A | 599 | 165 | **43.2%** | 100% (flag only) | n/a (weekly) |
| B | 7,500 | 42 | **23.5%** | 100% (STOCKOUT flag only) | n/a |
| C | 851 | 120 | **79.0%** | 28.2% | 96.0% |
| D | 50,443 | 46 | **93.9%** | 39.7% | 99.3% |
| test | 2,904 | 143 | **64.7%** | 14.2% | n/a (weekly) |

The weekly zero rates are lower than the brief's daily ones because a week has seven chances to sell. Before preparation (raw, full weekly grid from the shop's first week) the zero shares were A 43.2%, B 23.8%, C 79.9%, D 95.6%, test 64.7%.

## 2. Key findings to discuss with the team
1. **Only the 4 test-shop columns reach the final model**, so price/promotions/hierarchy/supplier from A-D can only help indirectly (D18: feature dropout, teacher-student, auxiliary tasks, cleaning).
2. **Stock-outs explain many zeros:** in the test shop, weeks with `on_hand = 0` average 1.0 units vs 15.8 when stock > 0.
3. **The score is probably decided by slow and dead articles.** Median article sells ~0.5 units/week; 37% of articles (1,073) sold nothing in the last 8 weeks but are only 4.2% of last-year units. MASE divides by each article's own typical weekly move (10th percentile only 0.04 units), so a small error on a slow article costs a lot. Hypothesis until our local metric reproduces the brief's baselines.
4. **Set D validation trap:** after preparation (complete weeks, clipping), 9,613 of 50,443 series have no units before the last 13 weeks and are **46.8% of holdout units** (D8: report with and without). My earlier raw count was 9,751 and 45.6%.
5. **B and D have under a year of complete history** (42 and 46 weeks): no yearly cycle can be learned from them alone. The brief's own experiment: borrowing Set C's season helped D (-52.5 -> -15.8) and hurt B (6.9 -> -129.8).
6. **Test shop trend and season (file's own dates):** weekly total falls across years (35k in 2020 -> 27k in 2023), peaks in Nov (34.9k), dips in June (26.7k).
7. **The holdout weeks are inside the A-D files** (confirmed in step 0.2), and **partial weeks exist at both ends** of B, C, D. A's `stock_snapshot` (2024-04-08) lies inside A's holdout window: leakage if used.
8. **Set C codes (inferred):** `is_return = -1` = a no-sale day row (462k rows, zero units) that carries the stock level; `stock_status` C = reduced-price status, R = regular.
9. **Weekly stock per shop:** real for test, C, D (D with gaps); only an out-of-stock flag for A and B. B's stock cannot be rebuilt exactly (replenished - sold = end stock in 0.4% of series).
10. **The test shop's yearly peak is not on a fixed date** (2020-11-16, 2021-11-01, 2022-10-17), unlike A's, so "same week last year" may mislead (F6).
11. **Many C articles disappear** (485 of 851 stop appearing before the end), which drives C's falling total (F7).

## 3. Settled at gate 0.7: D26 (evidence kept here)

- [x] **D26. DECIDED (a) on 2026-11-21: adopt this local score convention (resolves F1 and F8).** Evidence from step 0.6 (details in `iterations/00-baselines/log.md`): the score reproduces the brief's published A-D baselines best with **series level** (store x article), **scale = full history, with every series zero-filled from the shop's first week (leading zeros)**, and series whose history is constant or empty excluded.

  | Brief baseline | Brief | Ours (series level, series-start grid) | Ours (series level, leading-zero grid) |
  |---|---|---|---|
  | A AutoARIMA / Top-down | 8.7 / 10.2 | 8.7 / 10.2 | 8.7 / 10.2 (A has no leading zeros) |
  | B AutoARIMA / Top-down | 20.4 / 20.4 | 20.2 / 20.2 | 20.2 / 20.2 (same) |
  | C AutoARIMA / Top-down | 41.8 / 37.7 | 44.1 / 42.7 | 40.2 / 38.7 |
  | D Top-down (AutoARIMA not rerun) | 44.9 | 63.3 | 44.5 |

  **Final rerun under D26 (series level, leading-zero grid):** A 8.7 / 10.2, B 20.2 / 20.2, C 40.2 / 38.7, D 58.6 / 44.5 for AutoARIMA / Top-down, all within 1.6 points of the brief.

  Article level (summed over stores) does not match (D: 7.8 and 20.8 vs the brief's 57.2 and 44.9; with leading zeros D Top-down is -49.4). Also, on the real test shop the live board's Naive -64.5 and AutoARIMA -33.7 sit close to our local-cut -58.3 and -28.8 with the full-history scale, while the "history from first sale" scale would give AutoARIMA +29.5.
  - Options: (a) adopt this convention for all local scoring, which means padding series with leading zeros in the metric (the weekly tables and D5 stay as they are, the padding only affects the history scale); (b) keep the series-start grid and accept up to 5-19 points of gap on C and D; (c) investigate the remaining 0.2-1.6 point gaps first.
  - Rec.: (a), and note (c) as optional. Consequence for D8: under this convention D's cold-start series are excluded by the metric itself (about 47% of holdout units), so "with and without" collapses to one number unless we also report a secondary metric for them.

## 3a. Iteration 01 decisions (all settled 2026-11-21; see `iterations/01-global-gbm/plan.md`)
- [x] **D27. DECIDED: all their weeks; the final submission is trained on all five shops' full histories. Data of the other shops in the `pooled` arm.** Options: all their weeks (the shops share no stores or products and have different calendars, so their last 13 weeks leak nothing about shop X) / only their training weeks. Rec.: all their weeks. The final submission is then trained on all five shops' full histories.
- [x] **D28. DECIDED: the weighted sample. Window sampling for big shops (D: 50k series).** Options: keep every window / keep all windows with a sale in the last 26 weeks plus a random 10% of the rest, weighted x10 / cap series per shop. Rec.: the weighted sample, so zeros stay represented without exploding the data.
- [x] **D29. DECIDED: as recommended (units / MASE scale, L1 loss, skip windows with scale 0). Training target.** Options: raw units / units scaled by the series' MASE scale / log1p units. Rec.: units / MASE scale with an L1 loss, which optimises the leaderboard metric directly (D11, D20); skip windows with scale 0.

## 3a-2. Open decision found in step 1.1
- [ ] **D30. Shop weighting in the pooled arm.** Over all weeks the pooled data is about 7.9M rows, and Set D is 4.16M of them (53%); D is 96% zero targets and is the shop least like the test shop (long history, 65% zeros). Options: (a) every row weight 1 (D dominates); (b) equalise shops (each shop contributes the same total weight); (c) cap D (for example sample D's series to about 1M rows); (d) weight by similarity to the target shop. Rec.: (b) as the first run, compared with (a) in the same iteration; the comparison itself tells us whether D helps.

## 3a-3. Historical open decisions prepared for the next steps
- [x] **D31. DECIDED 2026-11-21 (user: "whatever makes it faster"): option (b), window origins every 4 weeks for the other shops (the target shop's own windows keep step 2); Set D is not capped. The first attempt (origin step 2 everywhere) was stopped after the test shop's four arms and A's own arm; its partial results are kept in `iterations/01-global-gbm/results_1_3_partial_origin_step2.csv`. Speed of step 1.3.** The pooled arms train on 7-9M rows each; at 5+ minutes per arm the run could take 2+ hours. Options: (a) let it run; (b) restart with window origins every 4 weeks for the other shops (adjacent 2-week origins are highly redundant), halving the rows; (c) (b) plus capping Set D at about 1M rows. Rec.: check the first pooled arm's time, then (b) if it is over about 8 minutes. Changes the protocol slightly versus the own arm, so it would be logged.
- [ ] **D32. Rolling-origin evaluation for model selection.** Score the test shop at 4 cuts (last 13, 26, 39, 52 weeks) instead of one, so a 2-point gain is not decided by one window. Rec.: yes, from iteration 02 (cheap: the own arm trains in about 35 s).

## 3a-4. Found on 2026-11-21 (night)
- [ ] **D33 (TESTED 2026-11-21 night: no consistent gain, mean +0.0 over 4 cuts, so the recommendation is now NOT to adopt it; evidence in `iterations/02-robust-eval/log.md`; awaiting the user's confirmation). The test shop's `on_hand` changes regime.** Total stock was about 955k in 2020, 388k in Jan 2021 and about 100k from Apr 2021 on; stock cover (`on_hand` / weekly units) fell from 27 to 3; the share of missing values fell from 62% to 2%; median `on_hand` per article-week 14 (2020), 4 (2021), 5 (2022), 3 (2023). Our feature `oh_last_log` is the raw level, so early windows (2020 to early 2021) carry a different scale than the recent ones the final forecast uses. Options: (a) leave as is; (b) add a relative feature (`on_hand` / mean of the last 13 weeks of units, i.e. weeks of cover) next to the raw one; (c) set `on_hand` to missing before 2021-04-26 in the test shop; (d) both (b) and (c). Rec.: (d), evaluated on the rolling cuts (D32) in iteration 02, with the unchanged features as the reference.
- [ ] **D34. Model licences.** Moirai weights (1.0, 1.1, MoE, 2.0) are cc-by-nc-4.0 (non-commercial, research); TimesFM 3.0 weights are non-commercial (TimesFM 2.0 and 2.5 are Apache-2.0); TiRex is under the NXAI community licence (not read); Chronos is Apache-2.0. Fine for evaluating in the hackathon, but if the company wants to use a winning model commercially, those licences matter. Options: (a) evaluate all models, check licences before the final choice; (b) restrict to permissively licensed models. Rec.: (a), and ask the organizers whether non-commercial weights are acceptable for the final submission.
- [x] **D35. DECIDED 2026-11-21 (user): foundation models to evaluate in iteration 03:** Chronos 1.0 (T5 family), Chronos-Bolt, Chronos-2, TimesFM (2.5 first; 2.0 and 3.0 optional), Moirai 1.0 and 1.1, Moirai-MoE, Moirai 2.0, TiRex. Facts, package pitfalls and unverified items are in `iterations/03-foundation-model/model-notes.md`.

## 3b. Open follow-ups (not blocking)
- [x] **F1. MASE edge cases (part of D15). Resolved by D26: such series are excluded and counted.** Series with no history, or a history of only zeros (scale = 0), have no defined MASE; in `src/validate.py` they are excluded and counted. **Evidence (step 0.5), local cut:** A, B, C: every series is scorable. D (series level): 3,935 series have no training rows and 6,587 have a zero scale, so only 39,921 of 50,443 series are scored, covering **53.1%** of holdout units. Test shop (cut 2023-01-23): 65 articles have a constant (all-zero) history and are excluded from the local score because the metric divides by 0 (not because they are easy): **all 65 sell in the holdout** (3,021 units, 0.86% of holdout units, about 2.9 selling weeks each), so forecasting 0 for them would be wrong. The real test shop has none (no all-zero history). The near-dead articles are the real exposure: in the local cut, of the 898 articles with no sale in the last 13 weeks, 19.5% (175) sell in the holdout; in the real history 918 articles have no sale in the last 13 weeks. This feeds the zero-rule experiment (D20). Which convention the organizers use is found in step 0.7 by reproducing the brief's A-D baseline scores.
- [x] **F8. What is an "article" in A-D? Resolved by D26: a store x article series.** The brief says "every article in each shop". In A-D, which have stores, the score could be computed per store x article series or per article summed over stores (the test shop has no stores, so it is unaffected). **Evidence (step 0.5):** at article level D has 2,326 articles, of which 2,169 are scorable, covering **83.8%** of holdout units (vs 53.1% at series level), so the Set D cold-start problem (D8) shrinks from about 47% to about 16% of units. A, B, C are fully scorable at both levels. Decided by reproducing the brief's baseline scores in step 0.7 (`level` parameter).
- [ ] **F2. Cold-start modelling for D (follow-up to D8).** D8(a) fixes how we report. Whether to also forecast those series from article-level history in other stores and stock records (option c) is for a later iteration.
- [ ] **F3. Optionally confirm the Set C code inferences (D4) in the HF Community tab.**
- [ ] **F4. Define "reliable" for the stock mask (D12)** when we build features (iteration 02).
- [x] **F5. Build-spec assumptions (resolved in step 0.4).** All 7,500 B series have stock events from week 2025-01-06 and no sale precedes its first event, so every B series starts at week 1. D's 6,687 negative `on_hand` values were clipped to 0 and flagged (`on_hand_was_negative`).
- [x] **F6. The test shop's annual peak moves. INVESTIGATED 2026-11-21 (`src/probes/peak_f6.py`, `iterations/02-robust-eval/log.md`): the top weeks are exactly 50 weeks apart; each peak is driven by different articles (the top-50 articles overlap only 25-30% between years); the 13 forecast weeks contain no peak and the next one would fall around October 2023. No calendar feature is needed for this task.** Its top week is 2020-11-16 (53,036), 2021-11-01 (37,396), 2022-10-17 (39,333): about two weeks earlier each year. A's peaks sit on fixed dates (mid-Nov and ~Dec 18-20 every year). So the test shop's peak is probably tied to a moving event (or an unusual date shift), not a fixed week of year. This affects D10 and the calendar features; investigate in iteration 02 before using any week-of-year feature.
- [ ] **F7. C articles that disappear.** 485 of 851 C series have their last daily row before the last complete week (median 2026-03-23, 10th percentile 2024-12-23). Per D5 they are zero-filled to the shop's end, consistent with the brief's 96.0% zero rate over the full grid. C's falling weekly total (about 830 in Jun 2024 to about 130 in spring 2026) is largely this delisting. Treating them as dead series (forecast 0, D20) is part of the zero-rule experiment.

## 4. Decided
All on 2026-11-21 (decision gate 1 and the following round), unless noted. "Rec." items were accepted as recommended.

### Data preparation
- **D1** Holdout weeks are inside the A-D files; we cut the last 13 full weeks ourselves. A's `stock_snapshot` is not used as a feature.
- **D2** Drop both the first and the last partial weeks. Complete-week ranges: A 2021-04-12..2024-06-03 (165 weeks), B 2025-01-06..2025-10-20 (42), C 2024-01-01..2026-04-13 (120), D 2025-06-02..2026-04-13 (46), test 2020-07-27..2023-04-17 (143).
- **D3** Net returns off the units; clip the weekly total at 0.
- **D4** Accept the inferences about the C codes (F3 optional).
- **D5** A series starts at its first stock record where one exists, else at its first sale; after that, missing weeks are zeros to the end of the shop's range.
- **D6** One weekly table per shop in the common schema (Monday weeks); extra columns stay in side tables.
- **D7** Check the grain before every join (a join can multiply rows, e.g. B promotions).
- **D21** Real `on_hand` for test, C, D; out-of-stock flag only for A (`in_stock`) and B (STOCKOUT events); always add an explicit "on_hand missing" indicator so the model works with it missing (fits feature dropout, D18a).
- **D24** Test-shop `on_hand`: leave missing values missing and add a "missing" flag; clip negatives to 0.

### Set D cold start
- **D8** Option (a): score D **with and without** the series that have no sales before the cut, and report both. Evidence from the raw files: 9,751 of 50,447 series, 45.6% of holdout units; 6,592 of them have a stock record before the cut; ~71% are articles that sold in other stores earlier. After step 0.4 (complete weeks, clipping): 9,613 of 50,443 series, 46.8% of holdout units, 6,579 with a pre-cut stock record. (Modelling them: F2.)

### Seasonality and pooling
- **D9** Report both variants for B and D: (a) trained on non-seasonal dynamics only, season taken from the long shops; (b) short windows only.
- **D10** Take the test shop's seasonal shape from its own history; do not force a shared week-of-year across shops without checking.
- **D11** Scale each series by its own history (MASE-style).
- **D12** Stock as a feature, plus masking where reliable (F4). Do not de-censor the target: the test target is also observed units.
- **D20** MAE loss; test the zero rule (no sale in last 8 weeks -> 0) as an explicit experiment.

### Modelling and validation
- **D13** LightGBM global model on CPU tonight; foundation model on the GPU tomorrow; compare on all shops.
- **D14** Cut the last 13 weeks of each of A-D and of the test shop (cut 2023-01-23); evaluate all 13 weeks, not only the first 7. Does not collide with D8 (D8 only governs reporting for series without history).
- **D15** Implement the score as the page states it: per article, the mean over the 13 weeks of |actual - forecast| / (typical week-to-week move in the history), averaged over articles; score = (1 - MASE) x 100. We assume the scale is the mean absolute one-week difference in history, and verify by reproducing the brief's A-D baselines: AutoARIMA A 8.7, B 20.4, C 41.8, D 57.2; Top-down A 10.2, B 20.4, C 37.7, D 44.9 (F1 covers edge cases). The live-board Naive -64.5 / AutoARIMA -33.7 are on the hidden test shop and can only be checked by uploading.
- **D16** Extra-credit design: same model, same cut, per shop: own data only vs own + other sets; both scores logged every iteration.
- **D17** Fractional units allowed; clip forecasts at 0.
- **D19** Order of the rich-column techniques: feature dropout first, then cleaning, then auxiliary tasks, then teacher-student.
- **D22** Laptop tonight: preparation, metric, baselines, LightGBM. GPU tomorrow: foundation-model fine-tuning, neural models, teacher-student. Foundation-model zero-shot on the M1 to be checked when we get there.
- **D25** The live leaderboard is a sanity check and format check only; the user registers and uploads.

### Gate 0.7
- **D26** Local score convention: series level (store x article), history zero-filled from the shop's first week (leading zeros), scale = mean absolute week-to-week change of that history, constant/empty-history series excluded and counted. Implemented in `src/validate.py` (`split(..., pad=True)`); the weekly tables and D5 are unchanged. Evidence table above (section 3).

### Iteration 01 (2026-11-21)
- **D27** The `pooled` arm uses all weeks of the other shops; the final submission model is trained on all five shops' full histories.
- **D28** Window sampling for big shops: keep all windows with a sale in the last 26 weeks plus a random 10% of the rest, weighted x10.
- **D29** Target = units(t+h) / the series' MASE scale at the origin, L1 objective, one model with horizon `h` as a feature; windows with scale 0 are skipped for training.

### Earlier decisions
- **D18** Use all four techniques to pass rich-column knowledge into a model that only sees the 4 test-shop columns: (a) feature dropout, (b) teacher-student, (c) auxiliary tasks, (d) cleaning. Each is an experiment scored on the extra-credit comparison; order in D19.
- **D23** Park generating more synthetic Set B data (a public generator may exist; not checked). Revisit only if B helps as a donor in the extra-credit experiment.
- **Process** Work in iterations with decision gates: `AGENTS.md`, `pipeline-architecture.md`, `iterations/00-baselines/plan.md`.

## Configuration review proposal (2026-11-21; awaiting user decision)

Recommendation: retain the verified fixed 70/30 ensemble. Proposed next controlled
challenger, not authorized/trained: change only original LightGBM
`num_leaves=63 -> 31`, `min_data_in_leaf=200 -> 500`, `lambda_l2=1 -> 5`;
retain L1, learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
bagging_freq=1, existing features/scaling and purged early stopping. Hypothesis:
smaller trees and larger leaves reduce variance; risk: underfitting rare activation
patterns. Keep component forecasts, ensemble weights, zero rule and rounding fixed.
Use VALIDATION.md development cuts and promotion gates; no broad parameter search.

A-D should be evaluated as own-only versus pooled using identical feasible
13-week temporal holdouts, reporting each shop, equal-weight A-D average and
target separately; D zero-history exclusions and scored-unit coverage mandatory.
Existing target evidence does not establish unseen-shop generalization. Earlier
single-cut target results favor own (-4.8342) over equal-pooled (-5.8449),
unweighted pooled (-4.9533), and other-shops-only (-23.0702); these are limited
single-cut comparisons, not a new cross-shop evaluation.

Score -2 implies MASE 1.02; reported public -4.8 implies 1.048, requiring about
2.67% lower scaled error on the same scoring period. No validated configuration
guarantees this.
replacement or upload performed. Review checkpoint: user selects experiment scope.

## Conservative LightGBM authorization (2026-11-21)

User approved the proposed num_leaves=31, min_data_in_leaf=500, lambda_l2=5.
Run one isolated target-shop paired experiment under VALIDATION.md; retain all
other settings and incumbent. Plan: iterations/19-conservative-lightgbm/plan.md.
Review results before promotion, A-D expansion, or submission generation.

Conservative LightGBM experiment completed: primary score -13.665947 ->
-13.758754 (-0.092807), wins 1/4, late-horizon gain
-0.095389; both supplementary periods regress. Fails review gate.
Recommendation: retain original champion; candidate is evaluated, not promoted.
See iterations/19-conservative-lightgbm/log.md and results/paired.csv.

## Research shortlist review (2026-11-21; proposals only)

User renewed request for perturbation/Monte Carlo and research-based 13-week
forecasting methods. Screened primary papers: Bagged ETS (Bergmeir et al. 2016),
why bagging works (Petropoulos et al. 2018), DeepAR, renewal-process intermittent
demand forecasting (Turkmen et al. 2020), FrAug (2023), ADIDA (2011), TFT and
Chronos. Links and initial applicability notes are in iteration 19/research-notes.txt.

Recommended next direction for user review: a small explicit intermittent-demand
probability distribution with horizon-dependent sale probability and conditional
positive demand size; derive weekly marginal medians, compare against incumbent.
This is a complete distributional challenger, not another application of the
failed iteration-15 sale-probability gate. Check units support before choosing a
count likelihood. A compact DeepAR/renewal model is a subsequent alternative.
Monte Carlo is optional numerical inference from a fitted distribution; analytical
medians are preferable when available. More simulations do not remove bias.

Alternative: small prespecified block-bootstrap/bagging stability experiment;
do not independently shuffle weeks or inject arbitrary Gaussian noise into sparse
counts. FrAug/TFT lower priority given transfer uncertainty and complexity.
ADIDA is an inexpensive benchmark but disaggregated mean rates can harm weekly
MASE even if horizon totals improve. No future-stock inputs, source continuation
labels, broad search, new blend-weight search, training or uploads authorized here.
All candidate scopes require the user's stage decision under AGENTS.md.

## Hurdle-lognormal authorization (2026-11-21)

User explicitly selected hurdle-lognormal, target shop first. Freeze one small
distributional model and 70/30 replacement comparison, temporal checks unchanged.
Details: iterations/20-hurdle-lognormal/plan.md. No pooling, simulation, extra
models or uploads; stop at results checkpoint.

Hurdle-lognormal completed: primary ensemble mean -13.665947
-> -13.753767 (-0.087820); 3/4 wins, late gain
-0.396271, both supplementary periods regress. Fails gate; recommendation
is retain incumbent. Six exact control reproductions and artifact verification
passed. Target-shop own-only; no cross-shop conclusion or upload. Review evidence:
iterations/20-hurdle-lognormal/log.md and results/paired.csv.

User requested upload-format hurdle CSV after results review. Generated and
validated submissions/ensemble_top3_hurdle_lognormal30_v1.csv locally using full
history and unchanged 70/30 recipe. Keep incumbent preferred; no promotion/upload.
Reproduction and final metadata in iterations/20-hurdle-lognormal/.

The user selected distinct real sales beyond Sets A–D for the original forecast.
Iteration 21 audited UCI Online Retail and found 48 consecutive eligible weeks
after a holiday data gap, with 3,709 products and zero calendar overlap with any
packaged shop. Recommendation at source review: run one target-history plus
UCI LightGBM against the own-only model inside the fixed 70/30 ensemble.
Use the established per-row weights and donor window spacing, the four primary
target-shop periods and two supplementary checks. The gift-retail source is a
domain-transfer experiment. Review the recommendation before model fitting.

2026-11-21: User approved the 48-week UCI donor and fixed target-shop
ensemble comparison. Scope and predeclared review thresholds in iteration 21.

Iteration 21 completed. The target-only 70/30 ensemble scored -13.665947
against -13.733408 with the UCI donor across the four primary periods.
Gain -0.067461, two wins, late gain -0.062452, supplementary gains +0.216929
and -0.124165. All six saved fits passed read-back verification. The donor
failed the promotion gate. Retain the original forecast for the target shop.
See iterations/21-external-retail/results/backtest/paired.csv.

## Iteration 22 — audited source data and controlled fine-tuning

2026-11-21: User approved the model-only leaderboard plan, two pharmacy timing features and a bounded GPU Chronos-T5-small target-only versus pooled experiment, including A–D evaluation. The audited original A–D sales sources add zero new native sales keys or weeks; use each history once. The pharmacy timing LightGBM candidate failed the predeclared gate (two of four primary wins, both supplementary periods regressed). Retain original LightGBM through the Chronos stage. See iterations/22-extended-data/.

Iteration 22 completed all 30 Chronos evaluations and the six-period LightGBM ablation. The best challenger, pharmacy-only Chronos fine-tuning, gained +0.096983 primary score points, won 3/4 and regressed on both supplementary periods. Pooled Chronos lost 0.033732 versus the pinned pharmacy ensemble. Across A–D, pooled Chronos gained 0.094869 points over own-only on the equal-shop average, with only A improving. Select the original 70/30 forecast under the agreed gate. The full-history reconstruction exactly reproduced the preserved upload-format CSV. Detailed results and the artifact fingerprint are under iterations/22-extended-data/results/.

Iteration 22 export: user requested a local upload-format CSV for each of the four comparison rows. Generate full-history daily-feature LightGBM, pharmacy-only Chronos and pooled Chronos ensemble files alongside the existing original. Reuse the registered settings and historic selection result.

## Bounded Chronos retraining loop — proposed

User reported C-Team's public best of -4.7. The screenshot identifies the team and score; confirm which submitted CSV earned it before attributing the change. Target-only Chronos selected the 500-update ceiling in three of six historical folds and in the full-history fit; the other historical folds chose 100 or 250. Proposed first round: extend the existing checkpoint menu with 750 and 1,000 updates, keep the earlier checkpoints eligible, and hold the training data, seed, ensemble and all other settings fixed. Choose steps on the earlier inner period of each fold. Compare on the six registered outer periods against both the current target-only Chronos run and the pinned original ensemble. Keep the existing promotion criteria and report the incremental change versus target-only. Stop after this bounded comparison for review. Use observed historical sales as training labels.
