# T3 - robust combination of the members (iteration 08)

**Worker:** T3 (branch `it08/adamex-svk/t3`)
**Code:** `src/improve/t3_combine.py`
**Results:** `iterations/08-improvements/t3/results_adamex-svk_t3.csv`
**Runtime:** ~32 s for all 16 recipes (4 cuts each, forecasts cached after first load).

## What was tested

Per standard cut K (the last `13*K` weeks removed), the raw member forecasts were combined with
fixed, parameter-free rules. The rest of D40 was kept byte-for-byte: the zero rule (0 for articles
with no sale in the last 8 weeks), the fixed `0.3*LightGBM + 0.7*combination` blend, and the
`round`-then-`clip` step. No parameter was fitted on the scored cuts.

Rules:
- **mean** - plain average (D40 baseline for the 3 reference members)
- **median** - per-cell median
- **trimmed** - per-cell mean after dropping the min and max member when `n >= 4` (falls back to mean for `n < 4`)
- **geometric** - `expm1(mean(log1p(member)))`, i.e. geometric-style mean of `log1p`

Member pools:
- `REF3` = moirai-2.0-small, tirex, chronos-t5-small (D40 members)
- `FIVE` = REF3 + chronos-t5-base, moirai-1.1-base
- `*+LightGBM` = the pool with the LightGBM added as a member; the 0.3 blend is then **removed**
  and the combination output is used directly (zero rule still applied).

Reference D40 reproduced exactly: cuts `-3.72 / -25.91 / -11.01 / -14.02`, mean `-13.67`
(asserted in the script).

## Results

Fixed 0.3 LightGBM blend:

| combination | members | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 | cuts won | verdict |
|---|---|---|---|---|---|---|---|---|---|
| trimmed | FIVE (5) | -3.45 | -25.86 | -10.99 | -14.06 | **-13.59** | +0.08 | 3/4 | reject |
| mean (D40) | REF3 (3) | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 | 0/4 | baseline |
| mean | FIVE (5) | -3.64 | -25.80 | -11.08 | -14.19 | -13.68 | -0.01 | 2/4 | reject |
| geometric | FIVE (5) | -3.42 | -26.28 | -11.35 | -14.32 | -13.84 | -0.18 | 1/4 | reject |
| geometric | REF3 (3) | -3.53 | -26.39 | -11.17 | -14.33 | -13.86 | -0.19 | 1/4 | reject |
| median (of 5) | FIVE (5) | -3.61 | -26.16 | -11.75 | -14.30 | -13.96 | -0.29 | 1/4 | reject |
| median | REF3 (3) | -3.66 | -26.40 | -11.68 | -14.35 | -14.02 | -0.36 | 1/4 | reject |

LightGBM as a member (no fixed blend):

| combination | members | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 | cuts won | verdict |
|---|---|---|---|---|---|---|---|---|---|
| trimmed | FIVE+LightGBM (6) | -3.26 | -26.09 | -11.46 | -14.16 | -13.74 | -0.08 | 1/4 | reject |
| mean | REF3+LightGBM (4) | -3.83 | -26.10 | -11.17 | -13.97 | -13.77 | -0.10 | 1/4 | reject |
| median | REF3+LightGBM (4) | -3.40 | -26.26 | -11.57 | -14.12 | -13.83 | -0.17 | 1/4 | reject |
| trimmed | REF3+LightGBM (4) | -3.40 | -26.26 | -11.57 | -14.12 | -13.83 | -0.17 | 1/4 | reject |
| geometric | FIVE+LightGBM (6) | -3.31 | -26.45 | -11.22 | -14.67 | -13.91 | -0.24 | 1/4 | reject |
| geometric | REF3+LightGBM (4) | -3.50 | -26.56 | -11.16 | -14.56 | -13.95 | -0.28 | 1/4 | reject |
| median | FIVE+LightGBM (6) | -3.25 | -26.45 | -12.26 | -14.00 | -13.99 | -0.32 | 2/4 | reject |
| mean | FIVE+LightGBM (6) | -3.68 | -26.12 | -12.22 | -14.06 | -14.02 | -0.35 | 1/4 | reject |

## Conclusion

**No recipe change.** Nothing reaches the adoption bar (mean gain >= +0.5 AND better on >= 3/4 cuts).

- Best variant is the trimmed mean of the 5-model pool under the fixed 0.3 blend: **+0.08** mean
  and 3/4 cuts, but the mean gain is ~6x below the +0.5 bar and the per-cut gains are within seed
  noise (+0.28 / +0.05 / +0.02 / -0.04).
- Plain `median` and the `log1p` geometric mean are consistently **worse** than the mean,
  especially on cuts 1-2, so a single bad member is not what is hurting D40.
- Adding LightGBM as a member and dropping the fixed 0.3 blend does not help at any combination
  (best -0.08).
- D40 remains the reference: mean `-13.67` (`-3.72 / -25.91 / -11.01 / -14.02`).

## Notes / failures

- No files outside `src/improve/t3_combine.py` and `iterations/08-improvements/t3/` were touched;
  `src/meta/ensemble_recipe.py` was not modified.
- All five members are present on cuts 0-3 and on `forecasts_final/`, so a final build would be
  possible, but none was produced because no variant passed.
- All raw member forecasts are non-negative, so `log1p` needed no extra guard (a `clip(.,0)` is
  still applied defensively).
- `bash scripts/check.sh` passes.
