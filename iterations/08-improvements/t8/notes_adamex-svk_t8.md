# T8 - other tree-model members in the 0.3 slot (iteration 08)

Worker: T8, branch `it08/adamex-svk/t8`. Code: `src/improve/t8_trees.py`.
Reference recipe D40 (4-cut mean **-13.67**, cuts -3.72 / -25.91 / -11.01 / -14.02):

    mean(moirai-2.0-small, tirex, chronos-t5-small raw forecasts)
    -> 0 for articles with no sale in the last 8 weeks
    -> 0.3 * LightGBM + 0.7 * ensemble -> round -> clip at 0

Test: keep the recipe fixed and replace only the 0.3 tree-member forecast. Weights are fixed
(no tuning). Candidate tree forecasts are read READ-ONLY from
`/Users/adomi/Documents/hackathlon/iterations/02-robust-eval/forecasts/`; their rows are in the same
`ShopMatrix` key order as `base_cut{K}_seed0.npy`, and each was produced per cut with only weeks before
that cut (no leakage). Scoring uses D26 (`src/validate.py`) on the 4 standard cuts.

## Availability (verified on disk)

| set | files | complete over 4 cuts x 3 seeds? | how it is used here |
|---|---|---|---|
| `loss_F_cut{k}_seed{s}` | 12 | yes | LightGBM l1 = the D40 slot; seed 0 is identical to `base_cut{k}_seed0.npy` (verified) |
| `loss_P_cut{k}_seed{s}` | 12 | yes | LightGBM P(units>0), NOT a units forecast; used only as the P<0.5 zero gate on `loss_F` (item 2.3) |
| `cb_F_cut{k}_seed{s}` | 12 | yes | CatBoost MAE |
| `d33_cut{k}_seed{s}` | 12 | yes | extra: LightGBM with the D33 feature set |
| `loss2_F_huber/fair/l2/tweedie1.2` | 3 seeds on cut 0 + seed 0 on cut 1 | **no** | diagnostic, cut 0-1 only |
| `loss2_F_tweedie1.5/poisson` | 3 seeds on cut 0 | **no** | diagnostic, cut 0 only |

`loss2_*` were not run on cuts 2-3, so they cannot be judged on the adoption bar; their rows below are
flagged incomplete and are shown for information only.

## Results (seed-average of the tree member; D40 is the seed-0 reference)

| variant (forecast in the 0.3 slot) | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 | cuts won | verdict |
|---|---|---|---|---|---|---|---|---|
| D40 reference (`loss_F` seed 0) | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | +0.00 | 0/4 | reference |
| D40 with LightGBM seed-average (control) | -3.66 | -25.94 | -10.98 | -14.02 | -13.65 | +0.01 | 2/4 | keep |
| LightGBM l1 + zero gate (`loss_P` < 0.5) | -3.41 | -25.98 | -10.94 | -14.04 | -13.59 | +0.07 | 2/4 | keep |
| CatBoost MAE (`cb_F`) | -3.71 | -26.13 | -10.94 | -14.11 | -13.72 | -0.05 | 2/4 | keep |
| LightGBM + CatBoost, 1/2 each | -3.69 | -26.01 | -10.99 | -14.07 | -13.69 | -0.03 | 2/4 | keep |
| LightGBM + CatBoost, 1/2 each, seed 0 | -3.64 | -25.99 | -11.01 | -14.03 | -13.67 | -0.00 | 2/4 | keep |
| 0.3 slot split 0.15 LGBM + 0.15 CatBoost | -3.69 | -26.01 | -10.99 | -14.07 | -13.69 | -0.03 | 2/4 | keep (identical to the 1/2 average) |
| d33 LightGBM (extra) | -3.68 | -25.93 | -10.95 | -14.11 | -13.66 | +0.00 | 2/4 | keep |
| `loss2_F_huber` (cut 0-1 only) | -3.82 | -26.11 | - | - | -14.96 | -0.15 | 0/2 | incomplete |
| `loss2_F_fair` (cut 0-1 only) | -3.94 | -25.69 | - | - | -14.81 | +0.00 | 1/2 | incomplete |
| `loss2_F_l2` (cut 0-1 only) | -6.49 | -28.03 | - | - | -17.26 | -2.45 | 0/2 | incomplete |
| `loss2_F_tweedie1.2` (cut 0-1 only) | -4.54 | -26.46 | - | - | -15.50 | -0.68 | 0/2 | incomplete |
| `loss2_F_tweedie1.5` (cut 0 only) | -4.48 | - | - | - | -4.48 | -0.75 | 0/1 | incomplete |
| `loss2_F_poisson` (cut 0 only) | -5.19 | - | - | - | -5.19 | -1.47 | 0/1 | incomplete |

For incomplete rows the mean/gain use only the cuts listed (the CSV also stores the D40 mean on the same
cuts), so they are not comparable with the 4-cut rows.

## Interpretation

- **No variant comes close to the bar** (+0.5 mean and 3 of 4 cuts). The best is the zero-gate LightGBM at
  +0.07, and it wins only 2/4 cuts (cut 0 +0.31, cut 1 -0.07, cut 2 +0.07, cut 3 -0.02).
  The gate gain seen on the full model in iteration 02 mostly disappears once it acts only through the
  0.3 slot.
- CatBoost is **slightly worse** than LightGBM in the slot (-0.05 mean). The plain 50/50 LightGBM+CatBoost
  average is also slightly worse (-0.03), so CatBoost adds no useful diversity here. This matches iteration
  02, where CatBoost was 0.6 worse alone and gave nothing in a 50/50 blend.
- The "average of LightGBM and CatBoost" and "0.3 slot split 0.15 + 0.15" are the **same forecast**
  (0.5*(L+C) on the slot); they are listed once and marked identical.
- The seed-averaging control moves the mean by only +0.01, so none of the small differences above is a
  seed artifact.
- `loss2_*` variants are clearly worse on the cuts that exist (huber/fair roughly at the reference, l2 and
  the count objectives much worse); they were never completed on cuts 2-3 and are not adoption candidates.
- Conclusion: **keep D40. No recipe change.** The 0.3 slot stays the LightGBM l1 forecast; no final-week
  forecast or submission CSV was produced (the task only asks for one after an adopted variant).

## Reproduce

    bash scripts/check.sh                       # passes
    .venv/bin/python src/improve/t8_trees.py evaluate

Runtime: about 11 s (loads the cut matrices and foundation parquets, scores 14 variants; no training).
Files: `src/improve/t8_trees.py`, `iterations/08-improvements/t8/results_adamex-svk_t8.csv`,
`iterations/08-improvements/t8/notes_adamex-svk_t8.md`.
