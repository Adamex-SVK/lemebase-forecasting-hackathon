# Iteration 08: parallel improvements to D40 (consolidated)

Ten improvement tasks (T1-T10) were run in parallel against the current best recipe **D40**, each on
its own branch/worktree. This file is the T0 integration summary: the verdicts, the evidence and the
outcome. Per-task code and notes live in `src/improve/<task>_*.py` and
`iterations/08-improvements/<task>/`.

## Reference recipe and adoption bar
- **D40:** mean of the zero-shot forecasts of `moirai-2.0-small` + `tirex` + `chronos-t5-small`,
  zero rule (no sale in the last 8 weeks -> 0), `0.3 x LightGBM (lgbm_own_v1) + 0.7 x ensemble`,
  round and clip at 0.
- **Reference score:** local 4-cut mean **-13.67** (cut0 -3.72, cut1 -25.91, cut2 -11.01, cut3 -14.02);
  live -4.8.
- **Adoption bar (TEAM-RULES.md section 2):** mean gain over D40 **>= +0.5** AND better on
  **>= 3 of 4 cuts**. Below that it is noise (seed noise alone is about 0.8 on one cut).

## Verdicts

| Task | Branch / commit | Best result vs D40 | Verdict |
|---|---|---|---|
| T1 intermittent-demand classics | `it08/adamex-svk/t1` (f55b44d) | best blend -14.10 | REJECT |
| T2 post-processing tuning | `it08/adamex-svk/t2` (83ab4a6, 63ed403) | D40 optimal; zero-rule W=4 +0.03 | REJECT |
| T3 robust combination | `it08/adamex-svk/t3` (92c6461) | trimmed mean +0.08 (3/4 cuts) | REJECT |
| T4 flatten forecasts | `it08/adamex-svk/t4` (bc9d35e) | all variants lose every cut | REJECT |
| T5 TabPFN-TS in the ensemble | `it08/adamex-svk/t5` (ae8aaa8) | no forecasts (jobs stalled) | BLOCKED |
| T6 quantile / sample-count variants | `it08/adamex-svk/t6` (e1901c6) | best chronos-q40 +0.03 | REJECT |
| T7 context variants | `it08/adamex-svk/t7` (99b433c) | first_sale -25.91, last52 -14.21 | REJECT |
| T8 other tree-model members | `it08/adamex-svk/t8` (bd9f473) | best +0.07 | REJECT |
| T9 fine-tune Moirai 1.1-small | `it08/adamex-svk/t9` (fc3f36c) | best 4th member -0.14 | REJECT |
| T10 new zero-shot models | `it08/adamex-svk/t10` (9c51907) | best chronos-2-small 4th -0.10 | REJECT |

## Conclusion
**D40 is unchanged** (local 4-cut mean -13.67) as of **2026-11-21**; the decision is recorded as
**D41** in `decisions-to-make.md`. No task came within 0.4 points of the bar, so there was nothing to
stack. The merged branch `it08/adamex-svk/improvements` keeps every task's results and new forecast
files for later reuse.

Integration re-verified on the merged branch:
- `bash scripts/check.sh` passes.
- `python src/meta/ensemble_recipe.py evaluate --models moirai-2.0-small,tirex,chronos-t5-small`
  prints **-13.67** (-3.72 / -25.91 / -11.01 / -14.02).
- `python src/meta/ensemble_recipe.py build --models moirai-2.0-small,tirex,chronos-t5-small
  --lgbm-weight 0.3 --out /tmp/it08_check.csv` reproduces `submissions/ensemble_top3_blend_v1.csv`
  byte-identically (md5 81154895d89604d463313e3f74759894).

## What now exists (kept from the tasks)
- **T1 classics:** `sf-ses`, `sf-tsb`, `sf-croston`, `sf-croston-sba`, `sf-croston-opt`, `sf-adida`,
  `sf-imapa` (padded context, cuts 0-3 + final).
- **T6 variants:** `moirai-2.0-small-q40`, `moirai-2.0-small-q45`, `tirex-q40`,
  `chronos-t5-small-q40`, `moirai-1.1-base-s200`; additive SPECS, defaults unchanged.
- **T10 models:** `chronos-2-small`, `toto-open-base`, `sundial-base-128m`; new SPECS + adapters
  (Toto/Sundial need their own remote venvs).
- **T9 fine-tune:** `moirai-1.1-small-ft` (5 leakage-safe per-cut checkpoints).
- **T7 contexts:** `first_sale` and `last52` forecast files for the three D40 members;
  `run_model.py --context last52` and `ensemble_recipe.py --context` (both default `padded`).

## Still blocked
**T5 / D36 TabPFN-TS:** all five GPU jobs were CPU-bound and had produced no full forecasts after
about 3 h; they were left running per the task rules. Re-run T5 as soon as complete padded parquets
exist. TabPFN is therefore still untested in D40.
