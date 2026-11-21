# T6 — Moirai / TiRex: more samples and a lower quantile (iteration 08)

**Result: REJECT all replacements. The D40 recipe is unchanged (-13.67 local 4-cut mean).**

## What was run
Five new `SPECS` in `src/foundation/models.py` (the default specs are untouched):

| name | change |
|---|---|
| `moirai-2.0-small-q45` | `forecast.quantile(0.45)` instead of 0.50 |
| `moirai-2.0-small-q40` | `forecast.quantile(0.40)` |
| `tirex-q40` | quantile index 3 of the trained 0.1..0.9 grid (= 0.4) |
| `chronos-t5-small-q40` | `predict_quantiles(..., quantile_levels=[0.4])` |
| `moirai-1.1-base-s200` | Moirai 1.1-base with `num_samples=200` (was 50) |

All 5 models were run for cuts 0-3 and `--final` in `~/hack-t6` on the GPU (own folder, symlinked
`.venv-{moirai,tirex,chronos}` and `data`, `OMP_NUM_THREADS=MKL_NUM_THREADS=4`; the 5 stalled TabPFN
processes were left alone). 25/25 runs completed; every output is 2,904 articles x 13 weeks, no NaN,
no negative values.

Quantile sanity check (mean of all articles, cut 0): Moirai q40 6.52 < median 7.81, q45 7.16;
TiRex q40 6.85 < margin 8.28; Chronos q40 6.92 < median 8.05. The adapters read the requested quantile.

Default behaviour was verified byte-identical after the edit:
`ensemble_recipe.py evaluate --models moirai-2.0-small,tirex,chronos-t5-small` prints
`-3.72 -25.91 -11.01 -14.02 mean -13.67`, and `build` reproduces `submissions/ensemble_top3_blend_v1.csv`
(md5 `81154895d89604d463313e3f74759894`).

## Solo scores (score_all.py, post = round+zero8; 4-cut mean)
| model | cut0 | cut1 | cut2 | cut3 | mean |
|---|---|---|---|---|---|
| moirai-2.0-small-q45 | -4.25 | -27.96 | -11.98 | -15.07 | **-14.82** |
| tirex-q40 | -3.83 | -29.02 | -13.28 | -15.51 | -15.41 |
| moirai-2.0-small-q40 | -4.61 | -29.09 | -13.05 | -15.91 | -15.66 |
| chronos-t5-small-q40 | -6.66 | -28.33 | -15.98 | -16.24 | -16.80 |
| moirai-1.1-base-s200 | -6.50 | -30.26 | -18.26 | -15.45 | -17.62 |

For reference the median versions score moirai-2.0-small -14.72, tirex -15.96, chronos-t5-small -17.94,
moirai-1.1-base -18.28. So a lower quantile helps the weak models *alone* (chronos +1.14, tirex +0.55,
Moirai 1.1-base with 200 samples +0.66) and q45 is neutral-to-worse for Moirai, but that gain does not
survive the averaging with the other members and the LightGBM blend.

## Recipe tests (fixed 0.3 LightGBM blend, zero rule, round; vs D40)
| variant | cut0 | cut1 | cut2 | cut3 | mean | gain | cuts won | verdict |
|---|---|---|---|---|---|---|---|---|
| D40 reference | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0 | - | - |
| q40 all three | -3.08 | -26.62 | -12.51 | -15.06 | -14.32 | **-0.65** | 1/4 | reject |
| q45 all (Moirai only, see note) | -3.71 | -25.88 | -11.21 | -14.16 | -13.74 | -0.07 | 2/4 | reject |
| Moirai q45 only | -3.71 | -25.88 | -11.21 | -14.16 | -13.74 | -0.07 | 2/4 | reject |
| Moirai q40 only | -3.71 | -25.97 | -11.39 | -14.19 | -13.82 | -0.15 | 1/4 | reject |
| TiRex q40 only | -3.59 | -26.28 | -11.59 | -14.34 | -13.95 | -0.28 | 1/4 | reject |
| Chronos q40 only | -3.41 | -25.74 | -11.15 | -14.26 | -13.64 | +0.03 | 2/4 | reject |
| Moirai 1.1-base-s200 replaces Moirai 2.0 | -3.65 | -25.76 | -12.55 | -13.82 | -13.95 | -0.28 | 3/4 | reject |
| Moirai 1.1-base-s200 as 4th member | -3.55 | -25.75 | -11.82 | -14.11 | -13.81 | -0.14 | 2/4 | reject |

No variant comes close to the +0.5 mean bar. The only positive mean is Chronos q40 alone (+0.03, noise).
The q40-all row is *worse* than D40: all three members pulled down together over-correct the right-skewed
high-volume articles. The q40 variants improve cut 0 (-3.08 vs -3.72) but lose cut 1/2/3, which matches
T2/T3's finding that per-cut differences of this size are noise.

**Note on "all three at q45":** only Moirai 2.0-small got a q45 spec in this task. TiRex publishes
quantiles on the 0.1..0.9 grid only (no 0.45 level), so a literal three-model q45 blend cannot be built;
the q45-all row therefore equals the Moirai-q45-only row. Chronos q45 was not run because the q40
one-at-a-time result (+0.03) already shows that side of the quantile has no headroom.

## Leakage
Each cut K used `run_model.py --cut-index K`, i.e. history ends at `max_week - 13*K` — only weeks before the
first holdout week. `--final` used the full history and wrote to `forecasts_final/`.

## Files
- `src/foundation/models.py` (5 new SPEC names + `quantile`/`q_index`/`num_samples` pass-through)
- `src/improve/t6_quantiles.py` (recipe table generator)
- `iterations/08-improvements/t6/results_adamex-svk_t6.csv` (recipe + solo table)
- `iterations/08-improvements/t6/results_zero_shot_t6.csv` (raw score_all.py output from the GPU)
- 25 new forecast parquets + 25 json: `iterations/03-foundation-model/{forecasts,forecasts_final}/{moirai-2.0-small-q40,moirai-2.0-small-q45,tirex-q40,chronos-t5-small-q40,moirai-1.1-base-s200}__test__*__padded.*`

## Runtime
GPU: 5 driver loops in parallel (18 min wall), ~41 min of model time in total
(chronos 7 min, Moirai 2.0 7 min each, TiRex 3 min, Moirai 1.1-base-200 18 min incl. model loads).
Local scoring/evaluation a few minutes. `bash scripts/check.sh` passes after the change.
