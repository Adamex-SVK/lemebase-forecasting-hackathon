# Iteration 09: different angles — evidence-first plan

> **Delivery note:** the requested destination was `iterations/09-different-angles/plan_kimi.md`, but plan-mode filesystem permissions allow writes only under designated plan directories. An implementation-capable agent can copy this document to that requested path; no source work is required for that copy.

## Evidence status and decision frame

- **Objective resolved with the human:** balance live (w1–7) and final (w8–13). A candidate must not trade one horizon for the other. Do **not** build different w1–7/w8–13 recipes.
- **[V] Verified in repo artifacts on 2026-11-21:** D40 = −3.72 / −25.91 / −11.01 / −14.02, mean **−13.67**; FFORMA best **+0.23**, with live **−7.9 / −6.2** versus D40 **−4.8**; rich pooled best **−16.29** versus own **−15.49**; TabPFN and the tested global/per-article weightings failed. Commands used included `git status --short`, `git branch --list it03/adamex-svk/foundation-models`, and recursive `grep` over the cited Markdown/CSV artifacts. `git diff` also shows that some supporting notes are uncommitted.
- **Key artifacts:** `iterations/08-improvements/ens-search/notes_adamex-svk_ens-search.md`, `iterations/05-meta-ensemble/notes_meta_fforma_hedge.md`, `iterations/04-rich-columns/notes_adamex-svk_dropout.md`, `iterations/02-robust-eval/notes_adamex_loss-and-nn-plan.md`, `decisions-to-make.md`, and `submissions/UPLOADS.md`.
- **[B] Unverified brief-reported hypothesis, not independently recomputed:** D1–D8, including the +8.7 selector oracle, level/shape results, tail shares, W=40 landmine, and horizon scores. Python execution was denied by the current read-only command policy, so every `[B]` statement is explicitly a hypothesis to test, not a claim of independent reproduction. Step 0 below reproduces them before any modelling.
- **[H] Hypothesis/prior:** every expected gain, probability, and predictable-ceiling range below. “Cost” means implementation/compute burden, not a runtime estimate. Each minimal experiment has a **2 h hard cutoff**; an unfinished experiment is inconclusive.

## 1. Diagnosis — what actually limits the score

1. **The loss is a decision problem, not a mean-forecast problem.** For scored article `i`, with origin-fixed scale `s_i`, the variable part of the score is
   `-100 / (13 N s_i) * sum_h |y_ih - f_ih|`.
   Thus each article-week's Bayes action is a conditional **median**. For a flat 13-week action it is the median of the pooled future-week distribution. A positive conditional mean, or evidence that revivals average 2.47 units/week, does not justify a nonzero forecast when the median remains zero. Integer rounding matters because the actions are integer-valued counts.
2. **[B] D40's shape is weak; this does not prove that all temporal information is useless.** Flattening D40 changes its score by <1 and oracle-level+D40-shape is close to oracle-flat. That bounds the value of **D40's existing wiggles on those windows**. It does not rule out value from genuine state changes such as launches, delistings, or stockouts, especially when a weekly median crosses 0/1. The useful unit of analysis is nevertheless the article level or active state, not a smoother seasonal path.
3. **[B] The +8.7 is candidate coverage, not predictable headroom.** Selecting the best of 11 levels after seeing the holdout scores −3.95 versus D40 −12.67. The candidates are origin-legal; the **selector is not**. With thousands of articles and 11 choices, hindsight selection alone can create a large winner's-curse gap. It also says nothing about new candidates not in that set.
4. **[V+B] Recombination has low remaining EV.** Repo results refute global weights, performance weights, FFORMA, TabPFN, pooling, and small gates/blends. [B] Even the in-sample best global weight adds only +0.33. This is evidence against another combiner, not proof that a discrete level selector or a new-information model cannot work.
5. **[B] Variance is concentrated and window-specific.** One A139270 example implies `100 * 540.85 / 2596 = 20.83` score points; `540.85 * 0.0112 * 13 = 78.75`, consistent with about 79 missed units. The 6–37% worst-set overlap rejects a fixed blacklist but does not establish whether dynamic origin features predict risk. The 4-cut D40 mean is `(-3.72 -25.91 -11.01 -14.02)/4 = -13.665`, so a single landmine can dominate both a cut and its paired interval.
6. **Two overreach corrections.** [B] Dead8's 11–16% average MASE share makes revival modelling lower EV, not “solved”; a rare revival can still dominate one window. Conversely, that landmine does not make revival the top priority: the zero rule is already sensible and [B] forcing one unit on stock-arrival rows was worse (−30.32 versus −29.46). Also, [B] w1–7 −12.29 versus w8–13 −15.27 is a 2.98-point descriptive gap on four practice windows, not a prediction that the real final will be 2.98 points worse.

**Bias versus variance:** bias means a repeatable, origin-predictable signed residual across origins (tested by E), not simply a mean underforecast; [B] D40's roughly 13% mean underforecast on cut 0 can be correct median behaviour under skew. Variance is window/article-specific error, including landmines. Averaging four windows does not convert the latter into learnable bias.

**Working diagnosis [H]:** the most plausible remaining gain is a better origin-time **level/active-state decision for alive articles**, or genuinely new cross-article information. The unstable tail mostly sets variance and stopping uncertainty. More weights, shapes, global scaling, and small post-processing changes are not worth another search.

## 2. Predictable-ceiling estimate

**Planning prior [H, low confidence]: expect only 5–20% of the fixed-candidate +8.7 to be predictable with current origin information: +0.44 to +1.74 points.** Arithmetic: `8.7 * .05 = .435`; `8.7 * .20 = 1.74`, moving the [B] 13-cut reference from −12.67 to roughly **−12.23…−10.93** if captured. The +0.5 adoption bar is only `0.5/8.7 = 5.7%` of that oracle gap.

The low prior comes from [V] FFORMA +0.23, per-article track-record weights ≈0, learned-weight instability, and [B] unstable worst-article identities. The upper end is deliberately nonzero because those failures do not test a two-level discrete gate, a horizon-pooled target, or within-shop analogues.

This is **not** an estimate that 80–95% is mathematically irreducible. One realised holdout per article-origin cannot identify the Bayes conditional distribution, and the 11-candidate oracle is not total theoretical headroom. It means 80–95% is currently **unproven and likely luck/selection hindsight under the existing information channels**. Nor can we infer that live −4.8 is near the practical ceiling: live and local scores cover different windows, and the local w1–7/final gap is unstable.

Keep four different oracles separate: `O_path=100` uses every future actual; `O_const` is the best flat holdout median; `O_sel11` is the 11-level hindsight selector; `O_sel2` is hindsight choice between D40 and mean26. On unrestricted flat forecasts, `O_const >= O_sel11 >= O_sel2`. The +8.7 belongs only to `O_sel11` and is a lower bound on same-cut constant-level hindsight headroom, not its upper bound.

[B] At W=40, D40 −29.46 and oracle-flat −9.92 imply level headroom `100*(1.2946-1.0992)=19.54` points. Perfect-foresight path headroom is another `100*1.0992=109.92` points. The latter is mostly unidentifiable timing/luck, not a practical modelling target; this arithmetic prevents “level is everything” from being confused with total theoretical headroom.

Measure capture on exactly the same cuts, with denominators matched to the action:

- A: `C2 = gain_A / (O_sel2-D40)` **and** `C11 = gain_A / (O_sel11-D40)`; [B] the former denominator is about +3.4, so 10% capture is +0.34, not +0.87.
- B and flat C: `C_const = gain / (O_const-D40)`. If C adds a new candidate, also report expanded-set selection separately.

Use ratios of mean gains, report per-cut denominators, and never average unstable per-cut ratios. As a winner's-curse diagnostic, independently permute each non-D40 candidate error column within origin-defined dead8/scale strata, keep D40 fixed, and recompute `O_sel11` 200 times. This heuristic null preserves marginal errors but breaks article-specific alignment; if the observed oracle lies inside its 95% interval, discount the oracle denominator heavily. A real forward-purged gate gain still counts, but a null-only oracle plus a negative gate closes selection.

Failure of one gate bounds that gate family only. If the predeclared selector and both follow-on experiments fail, update the practical expected gain to **0…+0.5 [H]** and stop modelling; that is an EV decision, not proof of a universal ceiling.

## 3. Non-negotiable experiment protocol

1. **Step 0 — reproduce before trusting:** run `bash scripts/check.sh`, the canonical D40 `ensemble_recipe.py evaluate` command, and Appendix A. Because Appendix A flattens to D40's **mean**, repeat flattening with its **median**. Rebuild the two- and 11-candidate oracles, with and without forced dead8 zero, plus w1–7/w8–13 splits. Reproduce D1 overlap on the intersection of scored article IDs and D5 before any tail experiment. Investigate material discrepancies before ranking any angle.
2. **Adoption:** canonical four standard cuts, D26, after the same zero rule, rounding, and clipping; require **≥+0.5 mean and ≥3/4 cuts won**, plus a paired per-article interval. Under the balanced objective, also require non-negative mean change on **both** w1–7 and w8–13. No horizon-specific recipe.
3. **Origin legality:** every feature, donor outcome, residual, calibration fit, and early-stopping choice for target origin `o` must end before `o`; training holdouts must satisfy `W_train >= W_test + 13`. LOCO selection is nested **within eligible past cuts**, never permission to train on a later holdout for an earlier origin.
4. **Exploration versus adoption:** A/E use only W=0/4/8/12 for comparability with FFORMA and to avoid fits with one or two training cuts; W=16…32 is technically possible but not used. These correlated cuts overlap canonical actuals, and cached bases stop at W=48 whereas canonical cut 3 needs W≥52 for purged training. Therefore A/E are **diagnostic-only in this portfolio**: a positive screen cannot lead to an unbiased canonical adoption claim, even after discarding the fits. Any future gate requires a separately approved design whose screen, features, thresholds, and configuration are chosen exclusively on newly generated pre-canonical W≥52 bases before one canonical evaluation. Do not generate those GPU bases on the strength of this diagnostic alone. History-only B/C can run the canonical protocol directly with predeclared settings.
5. **Uncertainty:** pair article-level loss differences against D40. For the four-cut summary, resample article IDs once across all cuts to preserve within-article cross-cut dependence. These intervals condition on the four windows; they do not establish season-to-season significance. Treat W=0,4,…,48 as correlated and never as 13 independent replicates.
6. **Landmine audit:** report cut wins, selected/switched counts, leave-one-cut-out stability, and gain after removing the largest positive article contributor. Do not adopt a one-article gain unless the same predeclared mechanism repeats at other origins. Include a feature/label shuffle negative control for any learned gate.
7. **Decision metrics before AUC:** report calibration, precision among rows switched to zero/nonzero, number of switches, and exact rounded score. AUC can look useful while every probability remains below the MAE action threshold.
8. **One predeclared configuration per angle; no sweep.** Record the full command, commit hash, all cuts (including failures), and in-sample optima separately. D40 remains frozen unless every adoption condition passes.

## 4. Angle portfolio

Signed gains and pass probabilities are subjective **[H]** versus D40; negative realised results mean keep D40. All experiments below use cached outputs or CPU only and have the same hard cutoff.

| Angle | Thesis and mechanism; evidence for/against | New information or decision? | Minimal experiment | Falsifier | Honest expected gain / pass prior | Cost | Re-tread risk |
|---|---|---|---|---|---|---|---|
| **A. Level-information audit + one-stump selector (S1/S2; diagnostic only)** | Test whether any of the [B] +8.7 is stable. Only D40 versus mean26, flat levels, at most one split from trend/zero-share/CV. Against: [V] FFORMA +0.23 and track-record ≈0. | New **decision**, no new information | Reproduce oracle; forward-purged stump and previous-cut-best diagnostic on W=0/4/8/12; shuffle control; report descriptive capture only, never an adoption claim | <+0.5 or <3/4 exploratory wins; unstable split/sign; shuffle performs similarly; then close selection | −0.2…+0.8 descriptive; 10–20% signal prior | Low | **High**, but discrete level choice is not another raw-forecast weighting |
| **B. Horizon-pooled L1 level model (extends S1)** | [B] Flattening <1 supports a level target. Remove horizon and pool all 13 targets as `y_ih/s_i` with identical origin features and weight `1/13`; multiply the output by `s_i`. L1 then estimates the conditional pooled weekly median, unlike 13 horizon-specific medians or fitting noisy sample medians. Against: other loss variants failed. | New **target/decision**, no new information | One LightGBM with existing features/capacity, flat output; canonical forward evaluation; compare level loss and exact score with D40 | No level-loss gain, or exact score <+0.5 / <3/4, or either horizon declines | −0.3…+1.0; 15–25% pass | Medium | Medium: this changes horizon aggregation, not the loss shape |
| **C. Within-shop analog level donors (S5/S7-adjacent; not counted as new)** | Similar article-origin trajectories may add level-change information. Pooling donor **weekly** ratios directly targets the flat-action loss. For: this exact within-shop transfer is untested; against: cross-shop pooling failed and match stability is unknown. | **New information channel** from other test-shop articles | Eligible donors have mean26>0, complete 13-week outcomes before the target origin, and no shared article ID. Match `log1p(mean4/13/26)`, zero share, recency, CV, tenure and stock arrival. Robust-scale features using eligible-donor median/IQR; Euclidean distance; fixed 50 neighbours with `(distance, article_id, origin)` tie-break. Predict target mean26 times the unweighted median of all 650 donor weekly `y[h]/donor_mean26` ratios; D40 fallback otherwise; no blend | Analog cannot beat mean26 on eligible past cuts, matches are unstable, or canonical exact score fails adoption | −0.8…+1.2; 10–20% pass | Medium | Medium: a new estimator, but the same borrowing family as S5/S7 |
| **D. Active-to-dormant lifecycle decision — non-seed mechanism, gate-family variant** | [B] Alive articles carry 84–89% of MASE. Estimate pooled future-week zero probability `q = mean_h P(Y_h=0)`; `q >= .5` makes zero the optimal flat action. Against: [V] the existing sale-probability gate (.5→0) gained only +0.31 on the final blend and lost live (−5.6). | New **pooled-level target**, but an adjacent zero decision | Sales-only pooled logistic on 13 binary outcomes/article with identical origin features and just recency plus active-run length; fixed threshold .5; compare flat zero versus flat D40 level **and the existing gate**; report switched-row precision | Poor calibration, almost no switches, no lift over the existing gate, false-zero loss, or exact score fails | −0.5…+0.5; 5–15% pass | Low–medium | **High**; only the pooled target, flat action, and alive-only scope are distinct |
| **E. Purged per-article residual-bias correction — non-seed, low re-tread** | Signed D40 residuals may persist even when model rankings do not; that persistence is untested. Against: [V] track-record weights ≈0 and [B] global scaling/recency failed. Define `r_W=(median(A_W)-D40_level_W)/s_W`; correct raw level by `s_t * n/(n+2) * median(r)` over ≥3 fully prior residuals. | New use of past **scaled forecast errors**, not weights | First test lagged residual sign/rank persistence; only if positive, evaluate the fixed correction on W=0/4/8/12 | No persistence, correction shrinks to zero/no switches, or honest gain <+0.5 | −0.3…+0.6; 5–15% pass | Low | Low–medium: residual bias is not a model ranking or global override |
| **F. Calibrated revival decision (S3)** | [B] Stock arrivals average 2.47 versus 0.59 units/week, but forcing one unit was worse. Only a calibrated predictive median justifies nonzero output. Against: D44's learned final-blend gate is adjacent negative evidence. | New **probability/magnitude decision** | On dead8+low-scale only, fit a tiny pooled hurdle model for `q=P(Y=0)` and positive magnitudes with stock arrival. Output 0 if `q>=.5`; otherwise use positive-side quantile `(.5-q)/(1-q)` | No calibrated actionable cohort, poor magnitude calibration, or score fails; AUC alone cannot rescue it | −1.0…+0.4 mean; 5–10% pass | Medium | **Medium**, not “low” |
| **G. Calendar/horizon robustness (S4)** | [B] W=40/W=44 swings show window variance, not a reliable seasonal correction. Re-ranking can expose a fragile candidate; it creates no forecast information by itself. | New **measurement**, no direct gain | Score cached candidates on cut3, W=36/40/44 and both horizons. At W≈92 evaluate history-only candidates only; no exact D40 foundation forecast exists there | Rankings do not change, or a calendar choice wins only one matched window | 0 direct; 0…+0.5 selection value [H] | Low | Low, but only two Apr–Jul seasons |
| **H. Cluster empirical-Bayes level (S5)** | Borrow relative levels within stable within-shop groups. Against: [V] share-based `topdown_mstl` already scores −36.3 versus median13 −16.54, and MASE weights every article by `1/s`, not by total units. Only a damped cluster median remains distinct. | New cross-article **information/prior** | Origin-only co-movement clusters; damp each article level toward the cluster median; fixed damping; exact per-article score | Cluster instability or allocation error removes the gain | −1.0…+0.6; 5–10% pass | Medium | Medium–high; C is the cleaner test of within-shop borrowing |
| **J. Stock-availability transition/censoring — non-seed mechanism, D33/gate variant** | [V] Stock-zero weeks average 1.0 units versus 15.8 in-stock, but ledger noise is ~56%, future stock is unknown, and D33 gained ≈0. For **alive** articles, stock transitions may separate temporary availability zeros from demand zeros; this is not S3's dormant-launch problem. | New availability-state **ablation**, using existing information differently | Add only current stock state and weeks since last stock transition to D's sales-only pooled zero model. Compare against D, the existing sale gate, and D33's negative evidence on the same forward cuts; fixed .5 threshold | No lift over D/existing gate, missing/noisy transitions destroy calibration, or exact score fails | −0.5…+0.5; 5–10% pass | Medium | **High**; only the pooled-median transition mechanism is distinct |

**Novelty accounting:** D, E, and J have mechanisms outside S1–S8 (active attrition, signed-residual calibration, and alive-stock censoring), but only E is low-re-tread; D/J must beat the existing gate/D33 comparisons rather than merely D40. C is intentionally ranked for its new information channel but is **not** counted toward novelty because it belongs to the S5/S7 borrowing family.

### Seed triage outside the ranked three

- **S1:** agree as a diagnostic; flattening alone is not a submission. B tests a genuinely different fitting target.
- **S2:** allow exactly A, not another gate/meta-learner search. **S3:** reserve F; its prompt underestimates the learned-gate evidence.
- **S4:** adopt G as measurement only; the human's balanced choice rules out final-only optimisation and split-horizon recipes.
- **S5:** [V] naive share-based top-down/MSTL is already refuted (−36.3 versus median13 −16.54), so prefer C; H is a fallback only if C demonstrates within-shop transfer.
- **S6:** the measurement mechanism is valid; the stack thesis lacks evidence that component errors are independent. Adopt the many-cut, paired, negative-control harness, not a stack slot. If a human insists, the only minimal test is one fixed stack under nested LOCO; it dies if it does not beat its best origin-legal component. **[H]** −0.2…+0.3, low compute, high false-positive/re-tread risk.
- **S7:** global pooling failure does **not** refute a sparse-segment dynamics prior, but disjoint products/stores rule out same-item donor information. Park A–D donors unless C first proves within-shop transfer. The minimal follow-up is one fixed dynamics-matched donor prior on eligible past cuts; it dies if it cannot beat target mean26 on later cuts. **[H]** −0.5…+0.5, medium cost, high re-tread risk. Fine-tuning needs a separate level hypothesis, GPU approval, and licence decision; it is not in this portfolio.
- **S8:** metric mismatch is possible, but at least ten rounded aggregate live scores (eight tabulated plus gate/NHITS in notes) may not identify a convention and cannot reveal the final window. Do not spend the first pass on it unless Step 0 finds a concrete contradiction. The minimal test is pairwise score-difference bounds from existing forecast CSVs followed by the prompt's small convention set; it dies if bounds are wide or all conventions fit equally. **[H]** zero direct forecast gain, low cost, low re-tread risk; it is a diagnostic, not a submission angle.

## 5. Ranked next-pass portfolio and decision tree

### 1) A — level-information audit and single-stump gate (single first experiment)

This is first because it tests the central interpretation of the largest quoted number with cached data and can close an entire family. Predeclare D40/mean26, the three candidate features, one split, and the shuffle control. For each W_t, choose feature/threshold only on W≥W_t+13 using the exact scaled-absolute-loss difference, with thresholds fixed at training-feature quantiles 0.1,…,0.9; never use classification accuracy. The previous-cut-best diagnostic chooses the lower-loss level at the smallest eligible prior W where that article is scored, falling back to D40 if absent. Deliver reproducibility, `C2`/`C11`, oracle-null results, selected counts, forward-purged scores, and both horizons. Exploratory +0.5/3-of-4 is only a correlated continue/kill screen, not adoption evidence. **Why start here:** a cheap negative closes flexible selection; a positive is still descriptive only.

### 2) B — horizon-pooled L1 level model

This is the cleanest direct test of the level reframing and can be evaluated on all canonical cuts using history only. It changes the training distribution and action, not D40's blend weights. First compare exact per-article flat-level loss with D40; only then compute full score. No blend or shrink sweep.

### 3) C — within-shop analog donors

This is the only ranked source of a genuinely new article-level information channel. Keep it alive-focused and purged. Donor origins use the fixed W=0,4,…,104 grid, require W_donor≥W_target+13 and at least 26 origin-history weeks; compute exact Euclidean neighbour queries in target chunks of 256 rather than materialising an all-target-by-all-donor matrix. Treat missing stock arrival as false and an all-constant donor feature's IQR as 1. The pooled-weekly donor-median sanity test precedes any full D40 comparison; failure closes both C and the cross-shop donor extension. Do not turn a weak analogue into another ensemble member.

```text
Reproduce diagnostics
├─ D40 or the central level/oracle result materially disagrees → stop model work; audit metric/data; keep D40.
└─ Reproduced
   └─ A: origin-legal selector diagnostic
      ├─ ≥+0.5 and ≥3/4 exploratory cuts, stable across shuffle/horizon checks
      │  → report descriptive capture; make no gate submission; proceed to B.
      │    Any later gate needs separate human approval and wholly pre-canonical development.
      └─ otherwise → close A; test E only if its cheap residual-persistence screen is positive.
         └─ B: pooled-L1 level
            ├─ canonical + both-horizon bar → adopt only after robustness/rebuild checks.
            └─ fail → stop horizon/loss variants in this portfolio.
               └─ C: analog donor sanity test
                  ├─ donor lift and canonical bar → robustness/rebuild; H is optional confirmation.
                  └─ no donor lift → STOP modelling; keep D40; finish robustness, extra credit, licences.
```

Run D+J or F only in place of C if Step 0 finds a fixed, origin-only cohort covering at least 1% of articles whose relevant pooled zero/nonzero prevalence exceeds 50% on every eligible past cut; J is always an ablation against D's sales-only model, not another standalone search. Otherwise their expected value does not justify displacing a new-information test. After three failed predeclared experiments, stop rather than relax the bar.

## 6. Do-not-run list

| Attractive idea | Evidence that kills or blocks it |
|---|---|
| Another global/greedy/SLSQP/inverse-error/softmax/segment/track-record weighting | [V] honest gains ≤+0.05 for zero voting and ≤0 for the other cited schemes; [B] even in-sample per-cut global optimum only +0.33 |
| FFORMA or a hedge on the same bases | [V] +0.23/+0.26 locally, below bar; live −7.9/−6.2 versus −4.8 |
| TabPFN addition/member swap | [V] addition −0.13; Chronos/TiRex swaps −0.37/−0.46; live swap −5.2 |
| More pooled rich columns/feature dropout/global A–D LightGBM | [V] own −15.49 versus best pooled −16.29, 0/4 wins |
| Global upward scaling or another shrink/offset/zero-threshold grid | [B] scale 1.15/1.30 = −14.31/−15.88 versus −13.67; scale-zeroing −13.43; D44 froze small variants |
| Fixed “worst articles” blacklist | [B] worst-5% overlap only 6–37%; holdout-derived membership would also leak |
| Forcing nonzero forecasts on dormant stock arrivals | [B] W=40 −30.32 versus −29.46; mean revival lift is not an MAE action |
| More calendar/seasonal-shape features | [V] F6: moving peak and only 25–30% top-50 article overlap; calendar **evaluation** remains allowed |
| Naive top-down/MSTL reconciliation | [V] `topdown_mstl` −28.6/−54.6/−31.5/−30.4, mean −36.3, versus median13 −16.54 |
| Fair/Huber/L2/Poisson/Tweedie loss search | [V] L1 −15.45; fair/Huber −20.00/−20.44, L2 −55.04, Poisson −35.52, Tweedie1.2/1.5 −31.86/−28.75 |
| More static relative-stock-cover features | [V] D33 mean ≈0; cut 1 paired effect −0.54, so J must test a different pooled-median mechanism |
| Median13/LightGBM blend retry | [V] local gain only ≈+0.08; submitted blend live −8.4 versus LightGBM −7.6 |
| CatBoost/NHITS/extra-seed/small-gate variants of D40 | [V] NHITS −15.30 alone; D40 NHITS +0.04 from unrounded means and live −5.2; gate +0.31 and live −5.6; CatBoost alone −16.09 |
| A fifth/sixth blend member or larger zero-shot family | [V] Chronos-T5-base addition −0.01 (2/4); TimesFM −18.73, Chronos-2 −22.96, Moirai 1.0-large −28.58, Chronos-Bolt ≈−56 |
| Stacking rejected micro-gains | [V] components are only +0.05/+0.26 and several lost live; no evidence of independent gains, and joint selection would add another multiple-comparison search |
| Final-only or split w1–7/w8–13 recipe | Not a numeric refutation: it violates the human's balanced-objective decision |

## 7. Implementation boundaries and human questions

- `git status --short` confirms extensive unrelated work. Implementation must never copy or stage unrelated uncommitted files. Set up the requested isolated worktree from the main repo:
  ```bash
  git worktree add ../hack-kimi -b it09/kimi/different-angles it03/adamex-svk/foundation-models
  ln -s /Users/adomi/Documents/hackathlon/.venv ../hack-kimi/.venv
  ln -s /Users/adomi/Documents/hackathlon/hackathon-retail-sets ../hack-kimi/hackathon-retail-sets
  ln -s /Users/adomi/Documents/hackathlon/data ../hack-kimi/data
  ```
  Run all subsequent commands from `../hack-kimi`, starting with `bash scripts/check.sh` and the canonical D40 evaluation.
- Use one Kimi-owned module, `src/experiments_different_angles_kimi.py`, with `audit`, `pooled-level`, and `analog` subcommands; add `residual`, `state`, and `robustness` only if their decision-tree branches trigger. Add leakage/action tests in `tests/test_different_angles_kimi.py`. Write only `iterations/09-different-angles/results_kimi_<angle>.csv` and `notes_kimi_<angle>.md`, plus the requested final plan copy. Keep all feature/donor construction in that module rather than changing core features.
- Do not change `src/validate.py`, `src/prepare.py`, `src/features.py`, handover/decision documents, or others' logs. Prove unchanged D40 evaluation before and after any new experiment module. No leaderboard upload by an assistant.
- **Open questions that change this plan:** none blocking. Live/final priority is resolved as balanced. Deadline changes only when the stop branch triggers. GPU fine-tuning and new model licences are explicitly outside this CPU portfolio; D34 and any actual upload remain human decisions.
