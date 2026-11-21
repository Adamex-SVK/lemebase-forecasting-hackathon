# Top models of each series in the winning blend (2026-11-21)

Request (user): test-run the current winning blend (D40) with the top model of each series instead of
the small ones, e.g. Chronos 2.0 and Moirai 2.0, and prepare a CSV for upload. Work by the `kilo`
session; no decision is taken here.

## What "top of each series" can mean with our forecasts

| Series | Top checkpoint | Available on 4 cuts | Available for the real weeks |
|---|---|---|---|
| Chronos 2.0 | `amazon/chronos-2` (120M, the full model; `chronos-2-small` 28M was never run) | yes | yes |
| Moirai 2.0 | `Salesforce/moirai-2.0-R-small` (11.4M) — **the only Moirai 2.0 checkpoint that exists**, so it is already the top of its series | yes | yes |
| Chronos 1.0 (T5) | `amazon/chronos-t5-large` (710M) | yes | yes |
| Moirai 1.1 | `Salesforce/moirai-1.1-R-large` (0.3B) | yes | **no** (no `forecasts_final/` file) |
| TimesFM | `google/timesfm-2.5-200m-pytorch` | yes | yes |
| TiRex | `NX-AI/TiRex` (single model) | yes | yes |

Consequence: a faithful upload file can use `chronos-2 + moirai-2.0-small + tirex` (the 3-family
structure of D40 with Chronos-2 swapped in); the larger Moirai 1.1 cannot go into a submission because
its real-week forecast was never produced.

## 4-cut result (D26; fixed LightGBM weight 0.3, as in the uploaded D40 file)

Reference D40 = moirai-2.0-small + tirex + chronos-t5-small, 0.3 x LightGBM: mean **-13.67**
(cuts -3.72 / -25.91 / -11.01 / -14.02).

| Candidate model set (w = 0.3) | cut0 | cut1 | cut2 | cut3 | mean | vs D40 | cuts won |
|---|---|---|---|---|---|---|---|
| **D40 reference** | -3.72 | -25.91 | -11.01 | -14.02 | **-13.67** | 0.00 | — |
| top generation 3: chronos-2 + moirai-2.0-small + tirex | -4.06 | -26.17 | -11.57 | -14.17 | -13.99 | -0.33 | 0/4 |
| top generation 4: + timesfm-2.5 | -4.08 | -25.96 | -12.02 | -13.94 | -14.00 | -0.33 | 1/4 |
| chronos-2 + chronos-t5-large + moirai-2.0-small + tirex | -4.69 | -25.96 | -11.46 | -14.14 | -14.06 | -0.40 | 0/4 |
| top of large series: chronos-t5-large + moirai-1.1-large + tirex | -4.27 | -25.88 | -11.99 | -14.42 | -14.14 | -0.47 | 1/4 |
| all top/large 6 models | -4.04 | -25.37 | -11.87 | -14.32 | -13.90 | -0.23 | 1/4 |
| chronos-2 + moirai-1.1-large + tirex | -3.76 | -25.84 | -12.43 | -14.57 | -14.15 | -0.49 | 1/4 |

Paired per-article 95% bootstrap intervals for the prepared file versus D40: cut0 -0.34 [-1.33, +0.70],
cut1 -0.26 [-1.67, +0.91], cut2 -0.56 [-1.94, +0.30], cut3 -0.15 [-4.86, +3.95]; every interval
includes 0. The difference is inside noise, but the point estimate is negative on every cut.

A weight scan over the LightGBM share (cuts 1-3 used for the choice, as everywhere) does not rescue the
top-series sets: best mean -14.00 at w = 0.4 for top generation 3, still 0.33 behind D40 at w = 0.3.

## Reading

- The bigger/newer checkpoints are not better here, which repeats the known finding: Moirai 2.0-small
  beats Moirai 1.1-base/large/MoE, and Chronos-2 scores -22.96 alone versus -17.94 for T5-small. In the
  blend the gain comes from **diversification**, not from raw model quality, so adding a weaker but
  correlated member (Chronos-2) removes part of the benefit that Chronos-T5-small provides.
- None of the top-series sets clears the adoption bar (+0.5 mean, 3/4 cuts). D40 stays the best recipe.

## File prepared for upload

`submissions/ensemble_top_series_v1.csv` — mean of chronos-2 + moirai-2.0-small + tirex raw forecasts,
8-week zero rule, 0.3 x `lgbm_own_v1` + 0.7 x ensemble, rounded, clipped at 0. 2,904 rows; passes
`src/meta/ensemble_recipe.py build` validation. Differences from the current D40 file
(`ensemble_top3_blend_v1.csv`, live -4.8): 0.41 units per article-week on average, 16.0% of article-weeks
changed, 65.3% of articles forecast 0 every week.

Expected local mean -13.99 versus -13.67 for D40, i.e. no reason to expect a live improvement. It is a
clean single-variable test of the "top models of each series" idea; uploading is done by a person
(`TEAM-RULES.md` section 6). Recording it in `submissions/UPLOADS.md` is for the uploader.

Raw numbers: `results_kilo_top_series.csv`. Evaluation and build tool: `src/meta/ensemble_recipe.py`.
