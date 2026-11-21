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
