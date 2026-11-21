# Iteration 09 / task 5 — analog nearest-neighbour conditional median (adamex-svk)

- commit: `dfc50f1` (script is additive and untracked at the time of the run)
- command: `.venv/bin/python src/improve/i09_analog.py`
- outputs: `results_adamex-svk_analog.csv`, `results_adamex-svk_analog-donors.csv`.

## What was run

For each standard cut the target origin is the first holdout week (`M.W` = 130, 117, 104, 91).
A series-origin is represented by the trailing 26 weekly units divided by the pre-origin D26 scale
(`mean |week-to-week diff|` over all history `< o`), plus `log1p(mean13)`, the 26-week zero share and
`weeks_since_sale / 13` (cap 104). Donors are all `(series, o')` with `26 <= o' <= o - 13` and a positive
scale, so a donor's whole 13-week future ends before the target origin. Donor origins use a fixed stride
of 2 (the same stride as the training windows).

Each target retrieves its 25 nearest donors by Euclidean distance on that 29-dimensional representation.
Donor futures are divided by the donor's own scale, combined horizon-by-horizon with an inverse-distance
weighted median, and multiplied back by the target scale. The deterministic fallback is the target's
median of the last 13 weeks; it is used when the target scale is zero or fewer than 25 donors are finite.
The point forecast is rounded and clipped like D40.

Recipes: analog alone (with the zero rule), analog as a fourth raw member
`0.3*L + 0.7*zero8(mean(3 foundation + analog))`, and the fixed trained-slot split
`0.15*L + 0.15*analog + 0.7*zero8(mean(3 foundation))`.

## Donor coverage

| cut | eligible donors | target articles with 25 donors | fallback used |
|---|---:|---:|---:|
| 0 | 112,620 | 2,839 | 0 |
| 1 | 96,305 | 2,762 | 0 |
| 2 | 77,821 | 2,680 | 0 |
| 3 | 62,500 | 2,599 | 0 |

Coverage is uniform; the fallback never binds.

## Results (4 standard cuts, round + clip)

| variant | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 | cuts won |
|---|---:|---:|---:|---:|---:|---:|---:|
| D40 (reference) | -3.72 | -25.91 | -11.01 | -14.02 | **-13.67** | 0.00 | — |
| median13 (reference) | -6.68 | -30.53 | -12.59 | -16.34 | -16.54 | -2.87 | 0/4 |
| analog alone | -5.59 | -32.82 | -13.44 | -16.43 | **-17.07** | -3.40 | 0/4 |
| analog as 4th D40 member | -3.93 | -26.60 | -11.42 | -14.17 | -14.03 | -0.36 | 0/4 |
| 0.15/0.15/0.7 split | -3.57 | -26.53 | -11.09 | -13.98 | -13.79 | -0.13 | 2/4 |

Paired per-article bootstrap (95%, 1,000 resamples, seed 0; gain = candidate - D40, positive = better):

| variant | mean gain | interval |
|---|---:|---:|
| analog alone | -3.40 | [-5.43, -1.95] |
| 4th member | -0.36 | [-0.81, +0.02] |
| split | -0.13 | [-0.45, +0.09] |

## Stop gates (both met)

1. Analog alone (-17.07) is **worse than median13 (-16.54)** on the 4-cut mean, so the basic analog model
   is not competitive; no DTW or second representation was added (as required by the plan).
2. The best blend gain is **-0.13** (split), below the +0.25 stop threshold, and no blend wins the
   adoption bar. No k sweep was run because the stop gate was already met; k=25 was the pre-registered
   primary choice.

## Verdict

**STOP — negative result.** The analog conditional median does not beat median13 alone and adds no
adoptable value to D40 in either fixed recipe. The nearest-neighbour conditional median is less accurate
than a simple recent median for these intermittent series, and a 15% analog slot is too small to move
D40. No pseudo-model forecast file was saved (the plan saves one only if the full comparison passes the
stop gate).
