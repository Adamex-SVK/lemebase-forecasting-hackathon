# Planning brief: find the next real improvement (for a reasoning / planning agent)

**You are Kimi Code**, acting as a senior ML research planner joining a team that is stuck. Your job
is **not** to write code or to produce another ensemble variant. Your job is to **think**,
re-derive what the bottleneck actually is, and propose a small portfolio of genuinely different,
falsifiable experiments. Reason explicitly: show the chain from evidence to hypothesis to
experiment, and say where your confidence is low.

**Repository:** `/Users/adomi/Documents/hackathlon` (read `HANDOVER.md`, `TEAM-RULES.md`,
`decisions-to-make.md`, `iterations/07-ensemble-toolkit/README.md`, and
`iterations/08-improvements/ens-search/notes_adamex-svk_ens-search.md`).
**Current date in the project:** 2026-11-21. The working tree has uncommitted work from other
sessions; use a `git worktree` (see §7) and never stage other people's files.

**Deliverable:** one plan file at `iterations/09-different-angles/plan_kimi.md` plus a short reply
summarising the ranked portfolio. No code changes during planning. Every claim must be either
verified with a command in this repo or explicitly labelled as a hypothesis.

---

## 0. What "good" looks like

1. **At least five candidate angles, of which at least three are materially different from the seeds
   in §5.** Adding a sixth way to weight the same forecasts does not count.
2. For each angle: a one-line thesis, the mechanism, the evidence for and against, a **minimal
   experiment that can run in ≤ 2 hours**, a **falsifier** ("this dies if …"), expected gain with a
   range, cost, and re-tread risk.
3. A ranked portfolio for the next ~6 hours with a decision tree ("if X then Y, else Z"), and the
   single first experiment you would run and why.
4. An explicit statement of the **predictable ceiling**: how much of the theoretical headroom is
   actually predictable from information available at the forecast origin, and how much is
   irreducible luck. If the honest answer is "−4.8 is close to the practical ceiling", say so with
   numbers. Stopping is a legitimate recommendation.
5. A **do-not-run list** of attractive ideas that the evidence already refutes (so the team does not
   spend another day on them).

State your reasoning. Show the arithmetic. Distinguish clearly between:
- a **mean/level signal** and a **decision that minimises mean absolute error**;
- **oracle headroom** (uses the holdout) and **predictable headroom** (uses only origin-time data);
- **bias** and **variance** (window-to-window luck).

---

## 1. The task in one page

- Forecast **weekly units sold** for **2,904 articles** of one test shop, **13 weeks ahead**:
  weeks starting **2023-04-24 … 2023-07-17**.
- Submission CSV: `article_id, w1..w13`, 2,904 rows, non-negative, fractional units allowed.
- **Score = (1 − MASE) × 100**, higher is better. Per article: mean over the 13 weeks of
  `|actual − forecast|` divided by that article's **history scale** = mean absolute week-to-week
  change of its zero-padded history. Then averaged over articles (each article equal weight).
  Constant/empty-history articles are excluded.
- **Live leaderboard = weeks 1–7 only.** The **final** score uses the hidden **weeks 8–13**.
  A person uploads; assistants never upload. Limit 100 uploads/team/day.
- Data: test-shop history `data/interim/weekly_test.parquet` (2020-07-27 … 2023-04-17, 143 complete
  weeks; columns `store_id, article_id, week, units, on_hand, on_hand_missing, out_of_stock_flag`),
  plus four unrelated training shops A–D. Raw data is not in git.
- The metric implementation is `src/validate.py` (`split`, `history_scale`, `score`); the standard
  evaluation cuts are `src/experiments_02.py` (`cut_frame`, `score_F`) and `src/rolling.py`.

**Current best recipe (D40, frozen):**
average the zero-shot forecasts of **moirai-2.0-small + tirex + chronos-t5-small** →
set to 0 for articles with no sale in the last 8 weeks ("zero rule") →
`0.3 × LightGBM + 0.7 × ensemble` → round to whole units → clip at 0.
Local 4-cut mean **−13.67**; live **−4.8**; leader was **−5.4** when last seen.
Per-cut: cut 0 = −3.72, cut 1 = −25.91, cut 2 = −11.01, cut 3 = −14.02.

**The four standard cuts are four different seasons** (holdouts):
- cut 0: 2023-01-23 … 2023-04-17
- cut 1: 2022-10-24 … 2023-01-16
- cut 2: 2022-07-25 … 2022-10-17
- cut 3: 2022-04-25 … 2022-07-18  ← **same calendar window as the real holdout**
The real holdout is 2023-04-24 … 2023-07-17; live = w1–7 (2023-04-24 … 2023-06-11);
final = w8–13 (2023-06-12 … 2023-07-17).

---

## 2. What has already been tried and rejected (do not re-tread)

Read the sources; the numbers below are from the repo.

- **Every global blend weighting / selection scheme** — fixed weight sweep, greedy ensemble selection
  (with and without replacement, cold and warm-started at D40), SLSQP non-negative weights, inverse
  stand-alone error, softmax temperature, per-sales-segment weights, per-article historical
  track-record weights. All below the +0.5 / 3-of-4 bar. On the 13 overlapping cuts, the
  **in-sample** best weight per cut beats D40 by only **+0.33 mean**; learned weights are worse on
  the held-out cut. See `iterations/08-improvements/ens-search/`.
- **FFORMA-style per-article meta-learner** over raw forecasts: best **+0.23**, below the bar; two
  live uploads were **−7.9** and **−6.2** versus −4.8 (`iterations/05-meta-ensemble/`).
- **TabPFN-TS**: adding it −0.13; swapping it for a member −0.37 … −0.46. Live swap −5.2.
- **Pooling the other shops into LightGBM / rich columns / feature dropout**: own −15.49 vs best
  pooled −16.29 on the 4 cuts; 0/4 wins (`iterations/04-rich-columns/`, D43).
- **Post-processing grid**: rounding offsets, shrink factors 0.80–1.05, small-value cut, zero-rule
  window sweep 4–26 weeks. Nothing cleared the bar. (Scaling *up* was not in the grid but is also
  refuted; see §3.)
- **Seasonality / calendar features** for the moving yearly peak: investigated and closed (F6).
- **Stock (`on_hand`) relative-cover features** added to LightGBM (D33): +0.0.
- **CatBoost, NHITS, loss variants, extra seeds**: no gain over the current reference.
- **"New model" route (T9 fine-tuning, T10 new families)** is still open but is a compute bet, not a
  reasoning angle; propose it only if you can explain why it should change the *level* estimate.

The rut is real: roughly fifteen independent ways of **recombining or post-processing the same
forecasts** have failed. More of that family is unlikely to work.

---

## 3. Verified diagnostics that change the picture

These were computed on this repo on 2026-11-21. Reproduce with Appendix A. Treat them as evidence,
not as conclusions; find the errors if you can.

### D1. The score is dominated by a tiny, unstable set of articles
- Top **1%** of articles = **25–33%** of total MASE; top **5%** = **47–53%**; top **10%** = **59–64%**
  (standard cuts).
- The identity of the worst 5% is **not stable**: pairwise overlap across cuts is only **6–37%**.
  A fixed "hard article list" will not work.
- The *type* is stable: worst articles have ~96% zero weeks, history scale often < 0.03, and a few
  units in the holdout. One article can cost ~20 score points (D5).

### D2. Week-to-week shape is almost worthless; the level is everything
- Flattening D40 to each article's own 13-week mean changes the score by **< 1 point** on every
  window tested (sometimes better, sometimes worse).
- Replacing D40's per-article level by the **oracle holdout level** while keeping D40's week-to-week
  shape gives within **0.3–1.1 points** of a flat oracle-level forecast. So the shape contributes
  almost nothing; the entire gap is the level.
- Implication: sequence modelling, horizon structure, and "wiggle" are not where the loss is. Any
  intervention that does not improve the **13-week level per article** is noise.

### D3. The level headroom is large, and simple level estimators already contain it
- Oracle **constant** forecast per article (the holdout median) beats D40 by **+9 to +20 points** on
  the overlapping windows tested (e.g. W=40: −9.92 vs −29.46).
- Oracle **selection among eleven history-only level estimators** (`mean4/8/13/26/52`,
  `median8/13/26`, LightGBM level, ensemble level, D40 level) scores **−3.95 mean** versus D40
  **−12.67** on the 13 overlapping cuts → **+8.7** of oracle headroom.
- Even oracle selection between just **D40 and mean26** scores **−9.23** (+3.4).
- But **no single** simple level estimator is close: averaged over the 13 overlapping cuts, the best
  single one is around **−27**. The value is in knowing *which* level to use per article, not in
  replacing D40 globally.
- The hard question: **how much of that +8.7 is predictable from origin-time data?** This is the
  central question of the whole brief.

### D4. The team may be optimising the wrong window
- On the standard cuts, D40's **w1–7** mean is **−12.29** but **w8–13** is **−15.27**. The final score
  is w8–13 only; the live board is w1–7 only.
- The standard 4-cut mean mixes four seasons. On the **calendar-matched** window (2022-04-18 …
  2022-07-11, W=40) D40 scores **−29.46**; four weeks earlier (W=44, 2022-03-21 … 2022-06-13) it
  scores **+7.51**; the adjacent standard cut 3 (holdout one week later) scores **−14.02**. Window
  difficulty is dominated by a few articles, not by a smooth seasonal effect.
- Implication: recipe ranking can flip between windows. Any recommendation should say which window
  it targets and how it behaves on the **final** (w8–13, Apr–Jul) window, not only on the 4-cut mean.

### D5. The landmine mechanism: dormant articles that revive
- Example: article **A139270** at W=40 had a history scale of **0.0112** (it had essentially never
  moved), then sold **79 units** in the holdout. D40 forecast 0. Its per-article MASE is **540.85**;
  with 2,596 scored articles that one article contributes **~21 points** of the −29.46 score.
- Its `on_hand` went from missing to **15** two weeks before the forecast origin and stayed positive.
  `on_hand` arrivals are a genuine **leading indicator** of launches: at W=40 there were 19 dead-8
  articles with positive `on_hand` at the origin and no positive `on_hand` in the previous 8 weeks;
  they averaged 2.47 units/week in the holdout (versus 0.59 for dead-8 overall).
- **But** forcing 1 unit on those 19 articles made W=40 **worse** (−30.32): for most of them the
  weekly median is still 0, and MAE is minimised by the median. A mean/level signal is not enough;
  you need a calibrated probability that the weekly median is ≥ 1, and a magnitude.
- The same mechanism can be a *drop*: the metric's denominator is fixed from history, so a dormant
  article that revives is catastrophic regardless of how good the rest of the forecast is. Some of
  this is irreducible without assortment/launch data.

### D6. Simple post-processing overrides fail
- Scaling all forecasts by `a > 1` is monotonically worse (a=1.15: −14.31 mean; a=1.30: −15.88).
  D40 underforecasts the mean by ~13% on cut 0, but that is the correct median behaviour under skew.
- Zeroing articles with scale < 0.02 gains +1.2 on cut 0 but loses on cut 2; mean −13.43, below bar.
- Recency anchors (`(1−β)·D40 + β·mean4`) and upward-shift overrides (`max(D40, mean4)` for recent
  surges) make the mean worse. Moving the zero rule from the ensemble to the final blend, or keeping
  the LightGBM for dead articles, is neutral (±0.02). D40 is a robust local optimum in this space.

### D7. The dead segment is already solved; the alive segment is the score
- "dead8" (no sale in the last 8 weeks) = **32–34%** of scored articles, mean MASE **0.39–0.55**,
  and only **11–16%** of total MASE. The zero rule is doing its job.
- "alive" articles are **66–68%** of articles and **84–89%** of MASE. That is where any real gain has
  to come from.
- 20–26% of dead8 articles revive, but they are only 0.6–2.1% of holdout units; their individual
  MASE is high (1.8–2.3) but they are a small share of the total.

### D8. The local metric is an approximation; the live board is the only external check
- D26 reproduces the organisers' published A–D baselines within 1.6 points, but the test-shop
  convention is not directly verified. Eight uploads with live scores exist
  (`submissions/UPLOADS.md`). The live window is w1–7, and the actuals are hidden, so this is a
  weak-identification problem; still, it is the only external data.

---

## 4. The reframing I want you to test

The evidence above points to a single bottleneck:

> **The task is not sequence forecasting; it is per-article 13-week level estimation under a
> per-article scale, where a small unstable tail dominates the average.**

If that is right, then:
- Better combination weights, better post-processing, and better temporal patterns are exhausted.
- The only routes to a real gain are (a) **new information** that predicts the level, (b) **better
  selection** among existing level estimates, (c) **decision rules for the unstable tail**, or
  (d) **a different evaluation target/window** that reveals a real, transferable gain.
- The honest ceiling depends on how much of D3's +8.7 oracle headroom is predictable. Estimate it
  before proposing a model. If only +1–2 is predictable, a +0.5 gain is already most of the
  available prize.

Use this reframing as a hypothesis to attack, not as a conclusion to accept.

---

## 5. Seed directions (evaluate, critique, and go beyond)

These are starting points. For each, state whether you agree with the mechanism, what the evidence
misses, and whether it is worth a slot. Then add **at least two angles that are not on this list**.

### S1. Level-only modelling (umbrella)
**Thesis.** Stop forecasting the path; forecast one number per article (the 13-week level), then
repeat it. **Mechanism.** D2: shape is worth <1 point; the metric is a level-estimation problem.
**Minimal experiment.** Re-score every existing base model by replacing its 13 values with its own
per-article mean; compare level-estimation quality directly (MAE per unit scale). **Falsifier.** If
flattening is neutral, the path is irrelevant and all sequence work is wasted. **Re-tread risk.**
T4 tested flattening as a post-process and it was rejected; this is a diagnostic/reframing, not a
new submission. **Expected gain.** 0 directly; it redirects effort.

### S2. Low-capacity level selection
**Thesis.** A tiny gate (2–3 features, or a shallow tree with 2 leaves) can pick between D40's level
and one or two history-only levels per article; the oracle gap is +8.7, so even 10% capture is +0.9.
**Mechanism.** D3: the information is already in simple estimators; the failure of FFORMA was
variance (many weights over noisy raw forecasts), not the absence of signal. **Features to try.**
Instability (CV of weekly units, number of level shifts), zero share, recent trend, tenure, on_hand
availability/arrival. **Minimal experiment.** Purged/LOCO gate on the 13 overlapping cuts; choose the
gate on 12 cuts, score the 13th; report honest mean and cuts won versus D40. **Falsifier.** Honest
gain < +0.5 or < 3/4 standard cuts, or the gate's chosen levels jump wildly across folds.
**Re-tread risk.** High (FFORMA, per-segment weights); justify why a level-only, 2-feature gate is
different. **Cost.** ~2–3 h local.

### S3. Revival / landmine decision model
**Thesis.** The tail is the score. Build a calibrated model of `P(weekly units ≥ 1)` and the
conditional magnitude for dormant/low-scale articles, and apply the MAE-optimal action (forecast 0
unless the predicted weekly median is ≥ 1). Use `on_hand` arrivals and stock cover as leading
indicators. **Mechanism.** D1/D5; the current zero rule is a hard heuristic, and the models output
point forecasts, not decisions. **Minimal experiment.** On the standard cuts and W=40, fit a small
logistic/quantile model on origin-time features for the dead8 + low-scale segment; compare the
honest score and the tail (95th percentile of per-article loss) versus the zero rule. **Falsifier.**
The calibrated probabilities do not separate revived from non-revived articles (AUC ≈ 0.5), or the
optimal action is 0 for almost all of them (then the loss is irreducible). **Re-tread risk.** Low;
D33 only added features to LightGBM, and the zero rule is applied to the ensemble, not learned.
**Cost.** ~2–3 h local; GPU not needed.

### S4. Calendar-matched and final-window optimisation
**Thesis.** The team has been selecting on a 4-season average while the final is one Apr–Jul window
(w8–13). Re-rank candidates on the final window and on calendar-matched cuts; allow a different
recipe for w1–7 (live) and w8–13 (final) inside the same CSV. **Mechanism.** D4. **Minimal
experiment.** Re-score all existing candidates (D40, members, TabPFN, FFORMA variants, T1–T8 outputs)
on (a) cut 3, (b) W=36/40/44, (c) a new Apr–Jul 2021 cut (W≈92; LightGBM locally, foundation models
on GPU if cheap), and (d) w8–13 only. **Falsifier.** Rankings are unchanged and the calendar-matched
mean does not separate any candidate by ≥ 0.5. **Re-tread risk.** Low; nobody has separated the
final window. **Cost.** 1–2 h local; GPU optional. **Caveat.** Only two Apr–Jul seasons exist in the
history (2021, 2022); do not overfit to W=40.

### S5. Cross-article / hierarchical level
**Thesis.** Articles co-move (promotions, assortment, store traffic). Forecast a cluster total or
the shop total (much less zero-inflated) and allocate to articles by stable/damped shares; or use
the cluster level as an empirical-Bayes prior for sparse articles. **Mechanism.** D3: the level is
the bottleneck; borrowing strength across articles is new information. **Minimal experiment.**
Cluster articles by co-movement on history only; forecast cluster totals with the existing models;
allocate by recent shares with damping; compare per-article. Also test top-down reconciliation of
D40 to a forecast shop total. **Falsifier.** Allocation error exceeds the level gain; cluster
membership is unstable across cuts. **Re-tread risk.** Low (pooling across *shops* failed; this is
within-shop structure). **Cost.** 2–3 h local.

### S6. Measurement: many-cut evaluation and a stack of small gains
**Thesis.** The rut may be a measurement problem: 4 cuts with 45-point swings, seed noise ~0.8, and
a per-change bar of +0.5 hide real small gains. Build a common harness over the 13 overlapping cuts
and test the **stack** of previously rejected small changes (zero voting +0.05, scale-zeroing +0.24,
segment weights, FFORMA hedge +0.26) jointly. **Mechanism.** If the small gains are independent, the
stack may clear +0.5 even when each component does not. **Minimal experiment.** Pair the 13-cut
scores, use a paired bootstrap over articles within cuts, choose all parameters leave-one-cut-out,
and require the stack to beat D40 on ≥ 8/13 cuts and a CI that excludes 0. **Falsifier.** The stack
is not better than the best component, or the gain disappears under LOCO. **Re-tread risk.** Low as
a methodology; high as a source of false positives if the cuts are treated as independent.
**Cost.** 2–3 h local.

### S7. New information: selective pooling / donors / fine-tuning
**Thesis.** Use A–D only where the test shop's own history is uninformative (cold/sparse/revived
articles): as a prior for the *level*, not as extra rows for a global model. Fine-tune a small
foundation model on the test shop (T9) with leakage-safe cuts. **Mechanism.** D3/D5; global pooling
failed because it changed the average, not because all borrowing is useless. **Minimal
experiment.** Match donor series to the target segment by dynamics (zero share, scale, inter-sale
interval); estimate a level prior; test on the segment only. **Falsifier.** Donor levels do not
predict target levels better than the target's own history. **Re-tread risk.** Medium (D43, D27);
be explicit about what is different. **Cost.** 3–4 h+; GPU for fine-tuning.

### S8. Reverse-engineer the live metric
**Thesis.** The local metric is an approximation. Eight uploads with live scores are a calibration
set. Test whether a different scale/window/zero convention explains the live ordering better; if so,
re-optimise for the organisers' convention. **Mechanism.** D8. **Minimal experiment.** For each pair
of uploads, bound the live score difference using `mean|f−g|/scale`; check consistency with the
observed differences; fit a small set of candidate conventions (scale from last 52 weeks, from first
sale, article-level, clipping) and see which predicts the 8 scores. **Falsifier.** The bounds are too
wide (no identification) or all conventions fit equally. **Re-tread risk.** Low; nobody has tried it.
**Cost.** 1–2 h local. **Caveat.** Live is w1–7 and noisy; do not over-interpret.

---

## 6. What to return

Write `iterations/09-different-angles/plan_kimi.md` with:

1. **Diagnosis (≤ 1 page).** What is actually limiting the score, with the numbers you verified.
   Correct me where the diagnostics above are wrong or overreach.
2. **Predictable-ceiling estimate.** A range for how much of D3's +8.7 is predictable, with the
   reasoning and the experiment that would narrow it.
3. **Angle table.** ≥ 5 rows: angle | thesis | new information or new decision? | minimal experiment
   | falsifier | honest expected gain (range) | cost | re-tread risk. Mark which 3 are materially new.
4. **Ranked portfolio.** Top 3 experiments for the next ~6 h, with a decision tree and the single
   first experiment. Include a "stop" branch if the ceiling is low.
5. **Do-not-run list.** The attractive ideas that §2/§3 already refute, in one line each, with the
   number that kills them.
6. **Open questions for the human.** Only questions that change the plan (e.g. deadline, whether the
   final score matters more than the live board, licence constraints, GPU availability).

Then reply with a ≤ 15-line summary: the reframing, the three angles, the first experiment, and the
ceiling estimate.

---

## 7. Rules and yardstick (non-negotiable)

- **Judge on the 4 standard cuts** (`src/experiments_02.py`, `src/rolling.py`) with the D26 metric
  (`src/validate.py`). Use the **mean over cuts, cuts won, and a paired per-article interval**.
  Adoption bar: **≥ +0.5 mean AND better on ≥ 3 of 4 cuts** (TEAM-RULES.md §2). For exploratory
  work, also report the 13 overlapping cuts (W = 0, 4, …, 48) and say they are correlated.
- **No tuning on the scored cuts.** Any parameter chosen must be chosen **leave-one-cut-out** (LOCO):
  pick on the other cuts, score the held-out cut, report the mean of those honest scores. Report the
  in-sample optimum separately and label it.
- **No leakage.** Anything computed for cut K may use only weeks before that cut's first holdout
  week. `tests/test_features.py` enforces this for features; your diagnostics must respect it too.
- **Do not modify** `src/validate.py`, `src/prepare.py`, `src/features.py`, `HANDOVER.md`,
  `decisions-to-make.md`, any `log.md`, or other people's files. Add options only if defaults stay
  byte-identical (prove: `ensemble_recipe.py evaluate` still prints −13.67 for D40).
- **Use a worktree** and explicit paths:
  ```bash
  cd /Users/adomi/Documents/hackathlon
  git worktree add ../hack-kimi -b it09/kimi/different-angles it03/adamex-svk/foundation-models
  cd ../hack-kimi
  ln -s /Users/adomi/Documents/hackathlon/.venv .venv
  ln -s /Users/adomi/Documents/hackathlon/hackathon-retail-sets hackathon-retail-sets
  ln -s /Users/adomi/Documents/hackathlon/data data
  bash scripts/check.sh
  ```
  (If that base branch is not what you want, use the branch the team names; do not edit the main
  checkout.)
- **Never upload to the leaderboard, never enter passwords/tokens, never commit raw data,
  environments, credentials, host names, or files > 5 MB.** Licences: Moirai cc-by-nc-4.0, TiRex
  NXAI community licence, TabPFN v2 Prior Labs licence (D34).
- **GPU:** one job at a time; `OMP_NUM_THREADS=4`; do not kill other people's processes; work in
  your own remote folder (see `iterations/03-foundation-model/gpu-machine.md`).
- **Money/time:** the team wants results by tomorrow morning. Prefer local, ≤ 2 h experiments.
  Expected value = P(works) × gain − cost. Say when an angle is not worth its cost.

---

## 8. Reasoning checklist (use it explicitly)

1. What **decision** is being made per article, and what loss does the metric actually impose?
   (MAE ⇒ the per-week median is optimal; a mean signal is not a reason to forecast nonzero.)
2. Is the proposed change adding **information** at the origin, or only re-weighting existing
   forecasts? (The latter is exhausted.)
3. Is the signal **predictable**, or is it only visible in the holdout (oracle)? Measure the gap.
4. Is the gain **level** or **shape**? (D2 says shape is worth < 1 point.)
5. Is the effect **consistent across windows**, or driven by one landmine article? Report the
   leave-one-article-out and leave-one-cut-out results.
6. Does the change survive **rounding and clipping**? The metric rewards integer medians.
7. Does it target the **alive** segment (84–89% of MASE), or only the dead segment (already solved)?
8. Does it target **w8–13** (final) or only **w1–7** (live)? Say which, and why it transfers.
9. How many **effective degrees of freedom** does the change add? With 4 cuts, a 2-parameter gate is
   already a lot; a 20-parameter meta-learner is hopeless.
10. What would **falsify** it in ≤ 2 h? If nothing can, the experiment is not ready.
11. What is the **irreducible** part? (Dormant-to-active launches without assortment data; a single
    spike on a tiny-scale article is ~20 points.)
12. If the answer is "the ceiling is near −4.8", what is the **evidence** for that, and what would
    the team do with the remaining time (e.g. robustness, final-window choice, extra-credit)?

---

## Appendix A. Reproduce the key diagnostics

All commands use the project venv (`.venv/bin/python`), from the repo root. `fforma.load_cut(W)`
loads the overlapping-cut bases and actuals with the D26 scale; `W` = weeks removed from the history
end before holding out the next 13.

```python
import sys, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "src/meta"); sys.path.insert(0, "src/foundation")
import fforma as ff
ff.MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]

def score(c, G):                       # G already zero-ruled, rounded, clipped
    err = np.abs(c["A"] - G).mean(1)
    ok = c["valid"]
    return (1 - np.mean(err[ok] / c["s"][ok])) * 100

for W in [0, 4, 8, 12, 36, 40, 44, 48]:
    c = ff.load_cut(W)
    raw = ff.simple_blend(c, 0.3)
    base = ff.post(raw, c)
    flat = np.repeat(raw.mean(1)[:, None], 13, axis=1)
    med = np.median(c["A"], axis=1)
    oracle_level = med[:, None] + (raw - raw.mean(1)[:, None])
    print(W, c["weeks"][0].date(),
          "D40", round(score(c, base), 2),
          "flat", round(score(c, ff.post(flat, c)), 2),
          "oracle level+D40 shape", round(score(c, ff.post(oracle_level, c)), 2),
          "oracle flat", round(score(c, np.clip(np.round(np.repeat(med[:, None], 13, 1)), 0, None)), 2))
```

For the oracle selection among simple levels, build the candidate matrix from `M.U`:
`mean4/8/13/26/52`, `median8/13/26`, the LightGBM level (`simple_blend(c, 1.0).mean(1)`), the
ensemble level (`simple_blend(c, 0.0).mean(1)`), and D40's level; set dead8 rows to 0; take the
per-article minimum absolute error against the holdout (oracle) and score.

For the standard 4-cut reference:

```bash
.venv/bin/python src/meta/ensemble_recipe.py evaluate \
  --models moirai-2.0-small,tirex,chronos-t5-small
# reference D40: -3.72 -25.91 -11.01 -14.02 mean -13.67
```

**Caveat:** `forecasts_cuts/…w{W}` and `forecasts/…cut{K}` are separate GPU runs, so W=0 does not
reproduce cut 0 exactly (Moirai is stochastic; e.g. −4.18 vs −3.72). Use the overlapping cuts for
*relative* comparisons, never as a substitute for the canonical 4-cut score.

## Appendix B. Numbers you can cite

| Quantity | Value |
|---|---|
| D40, 4 standard cuts | −3.72 / −25.91 / −11.01 / −14.02; mean −13.67 |
| D40 live (w1–7) | −4.8; leader −5.4 at last check |
| D40 w1–7 vs w8–13, mean over 4 cuts | −12.29 vs −15.27 |
| Flattening D40 to its own per-article mean | < 1 point change |
| Oracle level + D40 shape vs oracle flat | within 0.3–1.1 points |
| Oracle constant median vs D40 | +9 to +20 points (window-dependent) |
| Oracle selection among 11 history-only levels | −3.95 vs D40 −12.67 (13 overlapping cuts) |
| In-sample best global blend weight | +0.33 mean over D40 (13 cuts) |
| Top 1% / 5% / 10% of articles, share of MASE | 25–33% / 47–53% / 59–64% |
| Worst-5% set overlap across standard cuts | 6–37% |
| dead8: share of articles / share of MASE | 32–34% / 11–16% |
| dead8 mean MASE | 0.39–0.55 |
| A139270 at W=40 | scale 0.0112, 79 units, forecast 0, MASE 540.85 (~21 points) |
| Forcing 1 unit on 19 stock-arrival articles at W=40 | −30.32 vs −29.46 (worse) |
| Scaling a=1.15 / 1.30 | −14.31 / −15.88 mean (worse) |
| Scale-zeroing q=0.02 | −13.43 mean (below bar) |
| W=40 / W=44 / cut3 | −29.46 / +7.51 / −14.02 |
| Live uploads available | 8 (`submissions/UPLOADS.md`) |
| Overlapping cuts available | W = 0, 4, …, 48 for 5 foundation models + LightGBM |
