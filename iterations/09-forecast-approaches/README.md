# Iteration 09 — forecasting approaches with plausible headroom

Branch/worktree: `it09/adamex-svk/forecast-approaches` from `it08/adamex-svk/improvements` (`dfc50f1`).

Reference: D40, local 4-cut mean **-13.67** (-3.72 / -25.91 / -11.01 / -14.02), live -4.8.
Adoption bar: mean gain >= +0.5 and better on >= 3 of 4 cuts.

## Verdicts

| Task | Output | Verdict |
|---|---|---|
| 1. Error/headroom atlas | `results_adamex-svk_error-atlas.csv`, `notes_adamex-svk_error-atlas.md` | large oracle headroom (+8.3), but the purged pre-origin selector loses (1/8 wins, -1.59 mean); **gate closed** |
| 2. TabPFN-TS | `notes_adamex-svk_tabpfn-status.md` | **UNRESOLVED** — no full forecasts; no GPU access |
| 3. Distribution pool | `notes_adamex-svk_distribution-pool.md`, `results_adamex-svk_distribution-pool.csv` | **BLOCKED** — no quantile exports; additive code and tests ready |
| 4. Temporal reconciliation | `results_adamex-svk_temporal-reconcile.csv`, `notes_adamex-svk_temporal-reconcile.md` | **REJECT** — honest LOCO gain -0.32, 1/4 cuts |
| 5. Analog conditional median | `results_adamex-svk_analog.csv`, `notes_adamex-svk_analog.md` | **REJECT/STOP** — -17.07 alone vs median13 -16.54; best blend -0.13 |
| 6. Purged recency ensemble | `notes_adamex-svk_gated-followups.md` | **NOT RUN** — task 1 gate closed |
| 7. Moirai 2.0 fine-tune | `notes_adamex-svk_gated-followups.md` | **NOT ATTEMPTED** — optional, no GPU access |

**D40 is unchanged.** No submission file was built from a rejected candidate. The useful output is the
error atlas: scaled error is concentrated in low-scale, sparse, recently-active articles; the per-article
oracle is large but not predictable enough from pre-origin features to beat a fixed robust blend.

## Reproduction

```bash
.venv/bin/python src/improve/i09_error_atlas.py
.venv/bin/python src/improve/i09_temporal_reconcile.py
.venv/bin/python src/improve/i09_analog.py
.venv/bin/python src/improve/i09_distribution_pool.py   # reports BLOCKED without GPU quantiles
bash scripts/check.sh
.venv/bin/python src/meta/ensemble_recipe.py evaluate --models moirai-2.0-small,tirex,chronos-t5-small
```

Integration checks: D40 `evaluate` still prints -13.67; D40 `build` still reproduces
`submissions/ensemble_top3_blend_v1.csv` byte-identically (md5 `81154895d89604d463313e3f74759894`).
