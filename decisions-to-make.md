# Decisions to make

Living list for the team. Settled items are under **Decided** with date and choice. Last updated: 2026-11-21.

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

## 3a-3. Open decisions prepared for the next steps (see `next-steps.md`)
- [x] **D31. DECIDED 2026-11-21 (user: "whatever makes it faster"): option (b), window origins every 4 weeks for the other shops (the target shop's own windows keep step 2); Set D is not capped. The first attempt (origin step 2 everywhere) was stopped after the test shop's four arms and A's own arm; its partial results are kept in `iterations/01-global-gbm/results_1_3_partial_origin_step2.csv`. Speed of step 1.3.** The pooled arms train on 7-9M rows each; at 5+ minutes per arm the run could take 2+ hours. Options: (a) let it run; (b) restart with window origins every 4 weeks for the other shops (adjacent 2-week origins are highly redundant), halving the rows; (c) (b) plus capping Set D at about 1M rows. Rec.: check the first pooled arm's time, then (b) if it is over about 8 minutes. Changes the protocol slightly versus the own arm, so it would be logged.
- [ ] **D32. Rolling-origin evaluation for model selection.** Score the test shop at 4 cuts (last 13, 26, 39, 52 weeks) instead of one, so a 2-point gain is not decided by one window. Rec.: yes, from iteration 02 (cheap: the own arm trains in about 35 s).

## 3a-4. Found on 2026-11-21 (night)
- [ ] **D33 (TESTED 2026-11-21 night: no consistent gain, mean +0.0 over 4 cuts, so the recommendation is now NOT to adopt it; evidence in `iterations/02-robust-eval/log.md`; awaiting the user's confirmation). The test shop's `on_hand` changes regime.** Total stock was about 955k in 2020, 388k in Jan 2021 and about 100k from Apr 2021 on; stock cover (`on_hand` / weekly units) fell from 27 to 3; the share of missing values fell from 62% to 2%; median `on_hand` per article-week 14 (2020), 4 (2021), 5 (2022), 3 (2023). Our feature `oh_last_log` is the raw level, so early windows (2020 to early 2021) carry a different scale than the recent ones the final forecast uses. Options: (a) leave as is; (b) add a relative feature (`on_hand` / mean of the last 13 weeks of units, i.e. weeks of cover) next to the raw one; (c) set `on_hand` to missing before 2021-04-26 in the test shop; (d) both (b) and (c). Rec.: (d), evaluated on the rolling cuts (D32) in iteration 02, with the unchanged features as the reference.
- [ ] **D34. Model licences.** Moirai weights (1.0, 1.1, MoE, 2.0) are cc-by-nc-4.0 (non-commercial, research); TimesFM 3.0 weights are non-commercial (TimesFM 2.0 and 2.5 are Apache-2.0); TiRex is under the NXAI community licence (not read); Chronos is Apache-2.0. Fine for evaluating in the hackathon, but if the company wants to use a winning model commercially, those licences matter. Options: (a) evaluate all models, check licences before the final choice; (b) restrict to permissively licensed models. Rec.: (a), and ask the organizers whether non-commercial weights are acceptable for the final submission.
- [x] **D35. DECIDED 2026-11-21 (user): foundation models to evaluate in iteration 03:** Chronos 1.0 (T5 family), Chronos-Bolt, Chronos-2, TimesFM (2.5 first; 2.0 and 3.0 optional), Moirai 1.0 and 1.1, Moirai-MoE, Moirai 2.0, TiRex. Facts, package pitfalls and unverified items are in `iterations/03-foundation-model/model-notes.md`.

## 3a-5. Added 2026-11-21 (late): TabPFN and a meta-learning ensemble
- [ ] **D36. TabPFN-TS (tabular foundation model, suggested by a team contact): how to use it.** Facts from the package source and pages (`iterations/03-foundation-model/model-notes.md`): code Apache-2.0; the pipeline runs LOCALLY by default in the code (the README says cloud: out of date); the cloud mode would send the shop's data to a third party; the default weights (v3.5) are gated, non-commercial and need a Prior Labs account, licence acceptance and a token (only a person can do that); the v2 regression weights are ungated (licence "Prior Labs License", Apache-style with an attribution requirement). Options: (a) local mode with the v2 weights only (no account, no data leaves the machine); (b) also the 3.5 weights, which needs a person to create the account and accept the licence; (c) skip. Rec.: (a) now, (b) only if v2 looks promising and the team accepts the non-commercial licence (D34). Telemetry is disabled with `TABPFN_DISABLE_TELEMETRY=1`.
- [ ] **D37. Meta-learning ensemble (FFORMS / FFORMA style).** Learn, from series features, which forecasters to trust per series: FFORMS selects the single best method, FFORMA learns per-series averaging weights (the M4 runner-up approach). Needs forecasts of every base model at many origins (about 8-12 cuts instead of 4; the foundation models are cheap, the LightGBM needs a retrain per cut of about 35 s), strictly time-ordered training and validation, and strong regularisation (shallow trees, few base methods). Baseline to beat: the simple blend of the LightGBM with the average of the best foundation models, one global weight, rounded. Options: (a) FFORMA-style weights with LightGBM (custom MASE objective); (b) FFORMS-style selection with a classifier; (c) skip. Rec.: (a), after the simple ensemble is evaluated.

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

### Late additions (2026-11-21 afternoon; evidence in `iterations/03-foundation-model/log.md`)
- **D38 (adopted):** round every submitted forecast to whole units and clip at 0. Evidence: LightGBM +0.4 on the 4-cut mean (4 of 4 cuts), foundation models +1.4 to +2.5; the metric's per-article scale is tiny for slow articles, so integer medians win.
- **D39 (adopted):** set the forecast to 0 for articles with no sale in the last 8 weeks (zero rule) for the foundation-model forecasts. Adds about 0.1 on top of rounding (the models already forecast about 0 for them).
- **D40 (current best recipe, confirmed live):** average Moirai 2.0-small + TiRex + Chronos-T5-small zero-shot forecasts, zero rule, blend 0.3 x LightGBM (`lgbm_own_v1`) + 0.7 x ensemble, round: local -13.67 (4 cuts), **live -4.8** (LightGBM alone -7.6, Moirai 2.0-small alone -7.3). Model sets were fixed in advance; every set tried gave -13.7 to -13.9.
- **D36 (applied, unconfirmed by the owner):** TabPFN-TS runs in LOCAL mode with the ungated v2 weights only; the hosted mode would send the shop's data to a third party and the 2.5+ weights need an account and licence acceptance by a person.
- **D32 (de facto adopted):** the 4 rolling cuts are the standard yardstick (`TEAM-RULES.md`); overlapping cuts (every 4 weeks) are used only to train the meta-ensemble, with a purged protocol (iteration 05).
- **D33:** tested, not adopted (about +0.0).
- Still open: **D34** (model licences), **D37** (meta-ensemble, iteration 05), the extra-credit write-up (`EXTRA-CREDIT.md`).

### Earlier decisions
- **D18** Use all four techniques to pass rich-column knowledge into a model that only sees the 4 test-shop columns: (a) feature dropout, (b) teacher-student, (c) auxiliary tasks, (d) cleaning. Each is an experiment scored on the extra-credit comparison; order in D19.
- **D23** Park generating more synthetic Set B data (a public generator may exist; not checked). Revisit only if B helps as a donor in the extra-credit experiment.
- **Process** Work in iterations with decision gates: `process-rules.md`, `pipeline-architecture.md`, `iterations/00-baselines/plan.md`.

### Iteration 08 (improvements) - 2026-11-21, owner `adamex-svk`
- **D41. DECIDED 2026-11-21 (T0 integration, owner `adamex-svk`): keep D40 unchanged.** Ten improvement tasks (T1-T10) were run against D40 (local 4-cut mean **-13.67**: -3.72 / -25.91 / -11.01 / -14.02). **None passed the adoption bar** (mean gain >= +0.5 AND better on >= 3 of 4 cuts); the best task was +0.08 (T3 trimmed mean). Per-task results of record (each branch holds its own results CSV and notes under `iterations/08-improvements/<task>/`):

  | Task | Branch / commit | Best result vs D40 | Verdict |
  |---|---|---|---|
  | T1 intermittent-demand classics | `it08/adamex-svk/t1` (f55b44d) | best blend -14.10 | REJECT |
  | T2 post-processing tuning | `it08/adamex-svk/t2` (83ab4a6, 63ed403) | D40 optimal; zero-rule W=4 +0.03 | REJECT |
  | T3 robust combination | `it08/adamex-svk/t3` (92c6461) | trimmed mean +0.08 | REJECT |
  | T4 flatten forecasts | `it08/adamex-svk/t4` (bc9d35e) | all variants lose every cut | REJECT |
  | T5 TabPFN-TS in the ensemble | `it08/adamex-svk/t5` (ae8aaa8) | no forecasts produced | BLOCKED |
  | T6 quantile / sample-count variants | `it08/adamex-svk/t6` (e1901c6) | best chronos-q40 +0.03 | REJECT |
  | T7 context variants | `it08/adamex-svk/t7` (99b433c) | first_sale -25.91, last52 -14.21 | REJECT |
  | T8 other tree-model members | `it08/adamex-svk/t8` (bd9f473) | best +0.07 | REJECT |
  | T9 fine-tune Moirai 1.1-small | `it08/adamex-svk/t9` (fc3f36c) | best 4th member -0.14 | REJECT |
  | T10 new zero-shot models | `it08/adamex-svk/t10` (9c51907) | best chronos-2-small 4th -0.10 | REJECT |

  **T5 blocked:** all five TabPFN-TS GPU jobs were CPU-bound and had written no full forecasts after about 3 h; not killed per the task rules (D36 remains unresolved, still blocked). None of the new zero-shot models or forecast variants (sf-* classics, q40/q45, s200, chronos-2-small, toto-open-base, sundial-base-128m, moirai-1.1-small-ft, first_sale/last52 contexts) came close to the bar; the branches and files are merged/kept so any can be retried later. Integration verified: `scripts/check.sh` passes, `ensemble_recipe.py evaluate` prints -13.67, and `build` reproduces `submissions/ensemble_top3_blend_v1.csv` byte-identically.

### Iteration 09 (forecasting headroom) - 2026-11-21, owner `adamex-svk`
- **Draft iteration-09 record (superseded by D45 below; the earlier "D42" label was removed because D42 is the it08 FFORMA hedge).** The reference was re-verified at local 4-cut mean **-13.67** (-3.72 / -25.91 / -11.01 / -14.02) and the final build still reproduces `submissions/ensemble_top3_blend_v1.csv` byte-identically. Per-task results of record (details and CSVs under `iterations/09-forecast-approaches/`):

  | Task | Result vs D40 | Verdict |
  |---|---|---|
  | 1. Error/headroom atlas | oracle +8.34 (overlap) / +11.35 (standard), but the purged pre-origin selector beat D40 on only 1/8 development cuts (mean -1.59) | gate closed; no dynamic blending |
  | 2. TabPFN-TS | no complete forecasts produced; it08 verdict stands (rejected, live -5.2) | UNRESOLVED here |
  | 3. Distribution pooling | RAN on GPU: pooled_in_D40 -13.84 (gain -0.18, 2/4); chronos-t5-small miscalibrated (80% coverage ~0.30) | REJECT (below +0.25 kill) |
  | 4. Temporal reconciliation | honest LOCO gain -0.32, 1/4 cuts; best fixed lambda=0.5 -0.82 | REJECT |
  | 5. Analog conditional median | -17.07 alone vs median13 -16.54; best blend -0.13 | REJECT / STOP |
  | 6. Purged recency ensemble | task 1 gate closed | NOT RUN |
  | 7. Moirai 2.0 fine-tune | optional; GPU obtained late (used for the pooling export) | NOT ATTEMPTED |

  The atlas is the substantive output: scaled error is concentrated in low-scale, sparse, recently-active articles (smallest scale quartile 34-45% of error; zero share > 0.75 47-61%; top 10% of articles 59-64%), and the per-article model-choice oracle is large but not predictable enough to beat a fixed robust blend. Padded and last52 contexts are near-duplicates (per-article MASE Spearman 0.992-0.994, last52 better for only 15% of articles), so no context backtests for dynamic blending were generated. A below-bar demonstration file `submissions/ensemble_pooled_median_v1.csv` (pooled median in the D40 recipe, local gain -0.18) was built at the person's request for a live sanity check only; uploading remains a person's action.

### Iteration 09 (forecasting approaches) - 2026-11-21, owner `adamex-svk`
- **D45 (DECIDED by Adamex-SVK, 2026-11-21): iteration 09 forecasting-approach candidates all rejected; D40 remains frozen.** Error/headroom atlas (`iterations/09-forecast-approaches/results_adamex-svk_error-atlas.csv`): D40 -13.67 reproduced; oracle headroom +8.34 overlap / +11.35 standard (non-adoptable); a purged pre-origin model selector gains -1.59 and wins 1/8, so the headroom is not predictable and dynamic blending (plan task 6) is gated OFF. Temporal reconciliation (2-/4-week aggregated LightGBM reconciled with D40, `results_adamex-svk_temporal-reconcile.csv`): best honest LOCO variant -13.98 (gain -0.32, 1/4); all fixed-lambda=0.5 variants worse (-0.82 to -5.07); low-scale articles regress -> rejected. Analog nearest-neighbour conditional median (`results_adamex-svk_analog.csv`): analog alone -17.07 (worse than median13 -16.54); as a 4th D40 member -14.03 (gain -0.36, 0/4); fixed 0.15/0.15/0.7 split -13.79 (gain -0.13, 2/4, below the +0.25 stop gate) -> rejected (task 5 stop conditions met). Distribution pooling (task 3) was then run on the GPU (authorised SSH access): the equal-weight linear pool's median scores -13.84 in the D40 recipe (gain -0.18, 2/4 cuts; pooled alone -14.15) because chronos-t5-small is badly miscalibrated (80% coverage ~0.30) -> rejected below the +0.25 kill gate; a below-bar demonstration submission `ensemble_pooled_median_v1.csv` was built for a live check only. Moirai 2.0-small fine-tune (task 7) was not attempted. TabPFN was already settled in it08 (rejected, live -5.2). Conclusion: no recipe change; `HANDOVER.md` and `iterations/07-ensemble-toolkit/README.md` unchanged; negative results recorded in the iteration notes.
