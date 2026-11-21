# X1: three cheap tests of unused information (2026-11-21, handle adamex-svk)

Question: can information the D40 models do not use (stock columns, covariates, real sibling-shop data) give a better score? All scored on the 4 rolling cuts against D40 (local -13.67: -3.72 / -25.91 / -11.01 / -14.02); adoption bar +0.5 mean AND 3/4 cuts. All GPU runs took minutes (RTX 3090). Forecast files are regenerable and not committed (`forecasts_oos-*` folders).

The test shop has only `units`, `on_hand` and an out-of-stock flag (no price or promotion), so "covariates" here means stock.

## A. Mask out-of-stock weeks in the context (`--oos interp|short` in `src/foundation/run_model.py`, scoring `src/improve/oos_eval.py`)
A zero-sales week while out of stock is censored demand, not zero demand. 31% of zero-sales weeks are out-of-stock weeks. Replace them by interpolation (`interp`) or only runs of at most 3 weeks (`short`) before running the three D40 foundation models.

| Variant | Mean | vs D40 | Cuts won |
|---|---|---|---|
| interp, all three masked, D40 blend | -32.84 | -19.17 | 0/4 |
| short, all three masked, D40 blend | -19.49 | -5.82 | 0/4 |
| short, only one member masked | -13.95 to -14.08 | -0.29 to -0.41 | 0/4 |
| short, masked + unmasked together | -14.14 | -0.47 | 0/4 |

**Reject.** The scored target is sales, which stay capped by stock in the forecast weeks too: imputing demand inflates the forecast. Full table: `results_adamex-svk_oos_mask.csv`.

**A2. Stock-status zero rule** (`src/improve/oos_zero_rule.py`, no fitted parameter): forecast 0 where the article is out of stock at the origin and has sold little recently. Gains +0.06 / +0.05 / +0.02, 4/4, 3/4, 3/4 cuts won, but far below +0.5. **Reject** (consistent but tiny; the other session's LightGBM stock features gave about +0.26).

## B. Chronos-2 with stock covariates (`src/foundation/x1_chronos2_cov.py`, `src/improve/chronos2_cov_eval.py`)
Past covariates `log1p(on_hand)` and the out-of-stock flag; no future covariates exist.

| Variant | Mean | vs D40 |
|---|---|---|
| Chronos-2 alone, units only (reproduces the saved forecasts exactly) | -22.96 | -9.29 |
| Chronos-2 alone, with stock covariates | -23.42 | -9.76 |
| D40 + Chronos-2 (units) as 4th member | -13.75 | -0.08 |
| D40 + Chronos-2 (stock) as 4th member | -13.95 | -0.29 |

**Reject.** Covariates make it slightly worse.

## C. LoRA fine-tune Chronos-2 on real sibling shops (`src/foundation/x1_chronos2_ft.py`, `src/improve/chronos2_ft_eval.py`)
Training data: weekly series of Sets A, C and D only (never the test shop, never synthetic Set B), zero-padded from each shop's first week, sampled at random (seed 0), context 104. Then zero-shot on the test shop.

| Config (steps, learning rate, series per shop) | Alone | D40 + it as 4th member |
|---|---|---|
| zero-shot (no fine-tune) | -22.96 | |
| 300, 1e-5, 2000 | **-13.73** (cuts -2.73 / -27.04 / -11.26 / -13.88) | -13.65 (+0.02, 3/4) |
| 300, 5e-5, 2000 | -14.83 | -13.80 |
| 1000, 1e-5, 2000 | -14.29 | -13.67 |
| 1000, 5e-5, 2000 | -14.32 | -13.70 |
| 3000, 5e-5, 5000 | -14.57 | -13.80 |

**What this shows:** a 30-second fine-tune on real retail series lifts Chronos-2 by about 9 points, from far below the others to the level of the **whole D40 blend as a single model** (-13.73 vs -13.67; cut 0 -2.73 vs -3.72). **But it adds nothing on top of D40**: the best 4th-member gain is +0.02. More steps, higher rate and more series did not help (worse alone). Part of the lift is probably the model learning the zero-heavy retail style that the other members already know, so it adds no new information.

**Not adopted** (bar not met). Caveats: the first row is the single best of five configs on the same 4 cuts, so -13.73 is optimistic; the configs were a fixed small grid and all are reported.

## What it means
- Stock information as used here is not a lever (A, B).
- Sibling-shop fine-tuning works as a way to fix a weak model, but **gives no gain over the blend that already exists**. The idea is not dead for stronger members (fine-tune Moirai 2.0 or TiRex on the siblings), which was not tried: Moirai/TiRex need their own training code.
- Next, if continued: fine-tune Moirai 2.0-small or TiRex on A, C, D (about 1-2 h each with the tooling), and add the Chronos-2 fine-tune to a blend with 3 or more fine-tuned members.

## D. Follow-up after the live score of the D40 + fine-tuned Chronos-2 file (-4.5 vs -4.8) (`chronos2_ft_multi_eval.py`, `results_adamex-svk_chronos2_ft_multi.csv`)
- Averaging siblings-only runs does not help: 3-run avg -14.07, 6-run avg -14.15 alone (the first run's -13.73 was partly luck); in the blend +0.02 to +0.03.
- Fine-tune on siblings plus the test shop's own pre-cut history (fine-tuned per cut, leakage-safe), alone: seeds 0-4 give -13.37, -14.03, -13.57, -13.17, -13.87 (spread about 0.9 from the seed alone); 5-seed average -13.27 (+0.39, 2/4 cuts); x3 repeat of own history -13.58; 1000 steps -13.87. As a 4th member of D40: -13.51 to -13.73.
- Nothing clears +0.5 on 3/4 cuts. Seed noise (about +-0.4) is as large as every difference seen. The local plateau is about -13.5.

## E. Where do more ensemble members stop adding value? (`src/improve/ens_curve.py`, `results_adamex-svk_ens_curve.csv`)
Pool of 30 members from every family: 15 foundation models (incl. TabPFN-TS), 2 fine-tuned Chronos-2, 7 level/intermittent classics (SES, ADIDA, Croston variants, IMAPA, TSB), 1 seasonal (52-week seasonal naive), LightGBM + median13, 3 deep-learning (N-HiTS, N-BEATS, pooled N-HiTS). Zero rule on every non-tree member; equal-weight average of the chosen set; leave-one-cut-out forward selection.
- Best single members by family (4-cut mean): fine-tuned Chronos-2 -13.4 / -13.7, Moirai 2.0-small -14.7, LightGBM -15.1, N-BEATS -15.1, N-HiTS -15.3, best classic (ADIDA) -19.2, seasonal naive -28.9.
- Curve (honest, mean of 4 cuts): size 1 -13.75, **size 2 -13.43 (best)**, size 3 -13.96, then flat to worse: -13.7 to -14.1 up to 12 members. In-sample (optimistic) it is also flat at -13.4 to -13.5. **Saturation after 2-3 members.**
- D40 + one member per family: classics and seasonal hurt (-0.17 to -0.87), deep learning and TabPFN about -0.07 to -0.15, median13 -0.09 to -0.15, fine-tuned Chronos-2 (own history) +0.04 / +0.13. Nothing near +0.5.
- Reading: all members see the same weekly history, so their errors are highly correlated; what remains is irreducible noise of the weekly counts. Weaker, differently-built models (classics, seasonal) pull the blend toward their own bias.

## F. 13 overlapping cuts for the own-history fine-tuned Chronos-2 (`ft_overlap_eval.py`, `results_adamex-svk_ft_overlap13.csv`; seed 0 only, seeds 1-4 still running at the time of writing)
Offsets 0, 4, ..., 48 weeks; paired against D40 (mean over the 13 cuts -12.74).
| Variant | Mean | vs D40 | Cuts won | Bootstrap 95% (optimistic, cuts overlap) |
|---|---|---|---|---|
| fine-tuned alone | -12.69 | +0.05 | 4/13 | [-0.44, +0.70] |
| D40 + it as 4th member | -12.60 | +0.14 | 9/13 | [+0.04, +0.26] |
| D40, half of the foundation part is it | -12.49 | +0.24 | 8/13 | [+0.03, +0.50] |
A small but consistent gain, in line with the live +0.3 (-4.5 vs -4.8); about +0.15 to +0.25 is the realistic size. Far from the +0.5 bar, so not an adoption by the rules, but the direction is real.
