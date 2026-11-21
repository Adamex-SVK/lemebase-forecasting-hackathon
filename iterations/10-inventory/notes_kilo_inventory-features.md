# Inventory / intermittency feature engineering on the LightGBM `own` arm (iteration 10)

Date: 2026-11-21. Handle: `kilo`. Status: **tested, NOT adopted** (below the +0.5 / 3-of-4 bar).
Plan: `.kilo/plans/1795284706475-inventory-feature-engineering.md`.

## Question
The score is a per-article 13-week **level** problem (week-to-week shape is worth <1 point; oracle
selection among simple levels is +8.7 over D40). The one origin-time signal the LightGBM under-uses is
`on_hand` (inventory), plus the intermittency structure of `units`. Can a richer, internally derived
feature set improve the existing `own`-arm LightGBM — no new model, no new blend?

## What was added (`src/features.py`, opt-in `inv`, default path byte-identical)
30 features in 7 families, all computed at origin `o` from weeks `< o` only (leakage test extended):
- **availability**: `oos_share13/26`, `oos_run_len`, `weeks_since_stockout`, `zero_oos_share13`
  (separates "dead" zeros from "stocked-out" zeros), `oh_min/max4/13`
- **arrival**: `weeks_since_oh_arrival`, `arrival_magnitude`, `new_stock_flag` (the D5 launch signal),
  `replenish_count4/13`, `inter_replenish_interval`
- **dynamics**: `oh_diff1/4`, `depletion_rate4`, `oh_volatility13`
- **intermittency**: `adi`, `cv2`, `zshare52`, `skew13`
- **level**: `mean13_over_mean52`, `mean13_minus_median13`, `cv_units13`, `n_direction_changes13`
- **cross**: `shop_total13`, `article_share13`
- **demand**: `uncensored_demand_mean13` (demand proxy not censored by an empty shelf; feature only,
  the target is unchanged — D12)

`ShopMatrix`, the metric and the LightGBM are unchanged. `src/train_gbm.forecast_matrix` gained an
`inv=False` passthrough. Runner: `src/improve/inv_features.py` (own arm, 4 cuts, early stopping,
optional family ablation, optional D40 swap).

## Results (D26, test shop, 4 standard cuts; higher is better)
Seeds 0 and 1 averaged for `base`, `full`, `lofo:arrival`; the other `lofo:*` variants are seed 0 only.

| variant | cut 0 | cut 1 | cut 2 | cut 3 | mean | vs base | cuts won |
|---|---|---|---|---|---|---|---|
| `full` (all 30) | -4.08 | -28.88 | -12.71 | -15.41 | **-15.27** | **+0.22** | 3/4 |
| `lofo:arrival` | -4.15 | -28.89 | -12.80 | -15.36 | -15.30 | +0.19 | 3/4 |
| `lofo:dynamics` (seed0) | -4.40 | -28.59 | -12.82 | -15.25 | -15.27 | +0.22 | 4/4 |
| `lofo:availability` (seed0) | -4.12 | -29.01 | -12.90 | -15.33 | -15.34 | +0.15 | 3/4 |
| `lofo:level` (seed0) | -4.90 | -28.35 | -13.00 | -15.34 | -15.40 | +0.09 | 3/4 |
| `lofo:intermittency` (seed0) | -4.28 | -28.96 | -13.01 | -15.34 | -15.40 | +0.09 | 3/4 |
| `base` (no inv) | -4.57 | -28.80 | -13.09 | -15.50 | -15.49 | 0 | — |
| `lofo:cross` (seed0) | -4.79 | -29.48 | -13.00 | -15.41 | -15.67 | -0.18 | 2/4 |
| `lofo:demand` (seed0) | -4.52 | -29.80 | -12.85 | -15.67 | -15.71 | -0.22 | 2/4 |

Paired per-article 95% bootstrap (seed-averaged), gain over `base`:
- `full`: +0.50 / -0.08 / +0.38 / +0.09; intervals [-0.33,+1.42] [-0.46,+0.32] **[+0.13,+0.65]** [-0.12,+0.30]
- `lofo:arrival`: +0.42 / -0.10 / +0.29 / +0.14; intervals [-0.40,+1.30] [-0.46,+0.32] [+0.07,+0.49] [-0.16,+0.43]

`base` reproduces the documented own arm (-4.83 cut 0, -15.52/-15.46 seed means).

## D40 integration (same 0.3 x LightGBM + 0.7 x foundation ensemble, rounded; LightGBM slot swapped)
| | cut 0 | cut 1 | cut 2 | cut 3 | mean |
|---|---|---|---|---|---|
| D40 reference | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 |
| D40 + `full` | -3.41 | -25.93 | -10.93 | -14.02 | **-13.57** |
| gain | +0.31 | -0.02 | +0.08 | +0.00 | **+0.09** (3/4) |

The standalone +0.22 is diluted by the 0.3 LightGBM weight to +0.09.

## Reading and decision
- The enrichment gives a **small, fairly consistent gain** (+0.22 mean, 3/4 cuts over two seeds); only
  cut 2's interval excludes 0. It is **below the adoption bar** (+0.5 mean AND >=3/4 cuts).
- The seed-0 "remove `arrival`" variant won 4/4, but it did **not** survive seed 1 (`full` -15.27 vs
  `lofo:arrival` -15.30). So the D5 on_hand-arrival leading indicator did not transfer into a reliable
  gain; treat it as neutral-to-noisy, not a fix.
- `lofo:demand` and `lofo:cross` are *worse* than base, i.e. the `demand` (uncensored-demand) and
  `cross` (shop aggregate) features do carry a little signal; intermittency/level/availability are close
  to neutral.
- The +0.22 is in the range the repo already found does not transfer live (local gains <0.3 did not
  move the live board; cf. D33 ~0.0, D42/D44). **Not adopted; no submission file built (below bar).**

## Reproduction
```bash
.venv/bin/python tests/test_features.py                                     # default byte-identical + inv leakage
.venv/bin/python src/improve/inv_features.py --variants base,full --cuts 0 1 2 3 --seeds 0 1
.venv/bin/python src/improve/inv_features.py --variants lofo --cuts 0 1 2 3 --seeds 0
.venv/bin/python src/improve/inv_features.py --summary --d40 --d40-variant full
```
Outputs: `iterations/10-inventory/results_kilo_inventory.csv`, `forecasts/*.npy`, `detail/*.npy`.
