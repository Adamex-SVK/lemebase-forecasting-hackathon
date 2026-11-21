# Iteration 07 log (reusable ensemble toolkit)

Append-only.

## 2026-11-21: packaged the ensemble toolkit
- Created this folder as a self-contained, reusable package of iteration 05 (meta-ensemble) plus the
  final recipe (D40), so a teammate can reproduce the model and plug in a better base model.
- Added `src/meta/ensemble_recipe.py`:
  - `evaluate --models a,b,c [--lgbm-weight W]` scores the fixed-weight blend on the 4 standard cuts
    and compares it with D40 on the same cuts, with the adoption bar (+0.5 mean AND >= 3/4 cuts).
  - `build --models ... --out submissions/x.csv` writes and validates a submission from the real-week
    forecasts.
- Verified: `evaluate` reproduces D40 = **-13.67** on the 4 standard cuts; `build` reproduces
  `submissions/ensemble_top3_blend_v1.csv` **byte-identically**. Adding `chronos-t5-base` to the set
  gives -13.68 (gain -0.01, 2/4) -> keep the current recipe.
- FFORMA (D37) status: built (`src/meta/fforma.py`) and evaluated under the purged protocol on the 13
  overlapping cuts; best variant +0.23 mean (3/4), all below the bar -> **not adopted**. Results are in
  `results/` here and the canonical copy is `iterations/05-meta-ensemble/`.
- Collected forecasts used by the tools: practice cuts and final weeks are already in
  `iterations/03-foundation-model/{forecasts,forecasts_final}/`; the 5-model x 13-cut overlapping set
  (tirex, moirai-2.0-small, chronos-t5-small, chronos-t5-base, moirai-1.1-base) is in `forecasts_cuts/`.
- TabPFN-TS (D36) was still running on the GPU machine at packaging time and is not scored yet.
