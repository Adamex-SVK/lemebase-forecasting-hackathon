# Data overview and "train only after the jump" (2026-11-21, adamex)

## Plot
`weekly_overview.png`: weekly totals for Sets A-D and the test shop, plus the test shop's total `on_hand` and the share of missing `on_hand`.
- The jump is in the test shop's **stock**, not its sales: total `on_hand` is about 1.0-1.07M through 2020, drops to about 390k in late Jan 2021 and settles at about 100-200k from Apr 2021. The missing-`on_hand` share falls steadily from 62% to about 1%.
- **Units sold have no break**: a slow decline (about 33k per week in 2020 to about 27k in 2022-23), plus the Nov 2020 peak (53k).
- Sets A-D show no comparable break.

## Experiment
`src/experiments_trim.py` (no core-file change). LightGBM own arm, seed 0, same protocol as `experiments_02.run_lgbm_df`. Only the training origins change: origins whose 13 target weeks start before the trim date are dropped. Earlier weeks are still used as feature inputs. Note that with `MIN_HIST = 13` the reference already starts its targets in late Oct / early Nov 2020, so trimming removes only 7 (Feb) or 13 (Apr) of the 33-53 origins. Log: `logs/exp_trim.txt`; results: `results_adamex_trim.csv`.

The base reproduces the HANDOVER yardstick exactly (raw -15.52, rounded + zero rule -15.12).

| Variant | Post-processing | cut0 | cut1 | cut2 | cut3 | mean | vs base | cuts won |
|---|---|---|---|---|---|---|---|---|
| base | raw | -4.83 | -28.45 | -13.35 | -15.44 | -15.52 | | |
| trim from 2021-02-01 | raw | -4.64 | -28.39 | -12.92 | -15.62 | -15.39 | +0.13 | 3/4 |
| trim from 2021-04-26 | raw | -4.66 | -27.79 | -13.23 | -15.49 | -15.29 | +0.23 | 3/4 |
| base | zero rule + rounded | -4.04 | -28.04 | -13.08 | -15.31 | -15.12 | | |
| trim from 2021-02-01 | zero rule + rounded | -4.13 | -27.84 | -12.48 | -15.49 | -14.98 | +0.13 | 2/4 |
| trim from 2021-04-26 | zero rule + rounded | -4.02 | -27.22 | -13.10 | -15.36 | -14.93 | +0.19 | 2/4 |

Paired 95% intervals over articles include 0 on every cut except cut 2 for the Feb trim (+0.44 raw, +0.60 rounded). Most of the Apr trim's gain is cut 1 (+0.66 / +0.81, interval about [-0.6, +2.5]). Seed noise alone is about 0.8 on one cut.

## Verdict
**It does not clear the adoption bar** (>= 0.5 points on the mean and >= 3 of 4 cuts; `TEAM-RULES.md` section 2): the gains are +0.1 to +0.2, within noise. This is consistent with D33 (+0.0): the stock jump does not skew the LightGBM in a way trimming can fix. The direction is slightly positive, and the trimmed models train faster (fewer trees). Decision for the user: leave as is (recommended), or keep the trim as an option to retest in an ensemble or with 3 seeds.

## Live-probe files (2026-11-21, requested by Adamex-SVK)
`src/make_submission_trim.py`: final LightGBM on all weeks, origins from 2021-04-26 (46 origins, first target week 2021-05-03), 144 trees (1.1 x 130 + 1), seed 0.
- `submissions/lgbm_trim0426_v1.csv`: LightGBM alone. Mean |diff| vs `lgbm_own_v1.csv` 0.375 units; total forecast 285,638 vs 282,358 (+1.2%).
- `submissions/ensemble_top3_blend_trim_v1.csv`: D40 recipe with the trimmed LightGBM as the 30% share. 6.9% of cells (529 of 2,904 articles) differ from `ensemble_top3_blend_v1.csv`; total 293,957 vs 292,986.
Both pass `make_submission.validate_submission`. Not uploaded.

## Follow-up: soft trim and 3 seeds (`src/experiments_trim2.py`, `results_adamex_trim2.csv`, `logs/exp_trim2.txt`)
Soft trim keeps every origin but down-weights old ones by 0.5 ** (age / half-life). Mean over the 4 cuts:

| Variant | Raw | Zero rule + rounded |
|---|---|---|
| base, seed 0 / 1 / 2 | -15.52 / -15.46 / -15.38 | -15.12 / -15.20 / -14.98 |
| base, 3-seed average | -15.40 | -15.05 |
| hard trim 2021-04, seed 0 / 1 / 2 | -15.29 / -15.50 / -15.40 | -14.93 / -15.09 / -15.02 |
| hard trim 2021-04, 3-seed average | -15.35 (+0.05 vs base avg) | -14.88 (+0.17) |
| decay, half-life 26 weeks | -15.53 | -15.08 |
| decay, half-life 52 weeks | -15.41 | -14.94 |
| decay, half-life 104 weeks | -15.26 (+0.14 vs base avg) | -14.80 (+0.25) |

Seed 0 was the worst base seed, so part of the earlier +0.23 was seed luck: with seeds matched the hard trim is
+0.05 raw. The best soft trim (half-life 104) is +0.14 / +0.25 against the 3-seed base. **Nothing reaches +0.5;
the trim/recency line is closed.**
