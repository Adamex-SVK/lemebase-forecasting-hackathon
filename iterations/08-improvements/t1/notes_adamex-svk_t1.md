# T1 — intermittent-demand classics as blend members

Worker: `adamex-svk` (T1), branch `it08/adamex-svk/t1`.
Code: `src/improve/t1_classics.py`. Outputs: this folder.

## What was done

`statsforecast` 2.1.1 was fit on the zero-padded (D26) history of every cut
(`cut_frame(full, K)` -> `validate.split`) and on the full padded history for the
real-future files (`validate.pad_to_shop_start`): `CrostonClassic`, `CrostonSBA`,
`CrostonOptimized`, `TSB(alpha_d=0.1, alpha_p=0.1)`, `ADIDA`, `IMAPA` and
`SimpleExponentialSmoothingOptimized` (non-intermittent reference).  Frequency
`W-MON`, horizon 13, `n_jobs=-1`.  Only active series (non-zero history) are sent
to statsforecast; the all-zero-history series get 0.  Saving was clipped at 0 and
non-finite values were set to 0.

Pseudo-models written (2,904 articles x 13 weeks, standard layout + a `.json`
metadata file):

| name | file stem |
|---|---|
| `sf-croston` | `sf-croston__test__cut{0..3}/final__padded` |
| `sf-croston-sba` | `sf-croston-sba__test__cut{0..3}/final__padded` |
| `sf-croston-opt` | `sf-croston-opt__test__cut{0..3}/final__padded` |
| `sf-tsb` | `sf-tsb__test__cut{0..3}/final__padded` |
| `sf-adida` | `sf-adida__test__cut{0..3}/final__padded` |
| `sf-imapa` | `sf-imapa__test__cut{0..3}/final__padded` |
| `sf-ses` | `sf-ses__test__cut{0..3}/final__padded` |

Generation runtime: 38 s for 4 cuts + final (all 7 models per context,
2.6k-2.9k active series).  One `ensemble_recipe.py evaluate` run: ~7 s.

## Solo scores (same cuts, D26; full table in `solo_scores.csv`)

`score_all.py` output, best post-processing per model (`round+zero8`; D40 reference
on the same cuts is **-13.67**):

| model | cut0 | cut1 | cut2 | cut3 | mean |
|---|---:|---:|---:|---:|---:|
| sf-adida | -8.70 | -33.67 | -15.07 | -19.26 | **-19.18** |
| sf-croston-sba | -9.33 | -33.80 | -15.58 | -18.33 | -19.26 |
| sf-imapa | -9.99 | -32.88 | -15.91 | -19.53 | -19.58 |
| sf-croston-opt | -9.47 | -33.58 | -15.75 | -19.81 | -19.65 |
| sf-croston | -9.83 | -34.29 | -15.99 | -18.99 | -19.77 |
| sf-tsb | -14.94 | -33.35 | -18.57 | -21.09 | -21.99 |
| sf-ses | -22.18 | -37.78 | -23.62 | -23.21 | -26.70 |

All classics are far worse than D40 alone.  Raw Croston variants are catastrophic
(-127 to -136) because a single large sale in a mostly-zero history becomes a
large constant forecast; rounding barely helps and the zero rule (which removes
the long-dead series) is what rescues them.  ADIDA/Croston-SBA are the least bad.

## Blend tests (honest table; `results_adamex-svk_t1.csv`)

Blend = mean of members -> zero rule (D39) -> `0.3*LightGBM + 0.7*ensemble` ->
round -> clip 0.  Higher is better; gain is vs D40 (`-13.67`).

| variant | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 | cuts won | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D40 reference | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 | - | - |
| D40 + sf-adida (4th member) | -4.04 | -26.57 | -11.50 | -14.27 | -14.10 | -0.43 | 0/4 | reject |
| D40 + sf-croston-sba (4th member) | -3.96 | -26.64 | -11.53 | -14.32 | -14.11 | -0.44 | 0/4 | reject |
| D40 + sf-croston-opt (4th member) | -4.02 | -26.49 | -11.52 | -14.35 | -14.10 | -0.42 | 0/4 | reject |
| D40 + sf-croston (4th member) | -4.01 | -26.66 | -11.55 | -14.37 | -14.15 | -0.48 | 0/4 | reject |
| D40 + sf-imapa (4th member) | -4.48 | -26.34 | -11.36 | -14.28 | -14.12 | -0.45 | 0/4 | reject |
| D40 + sf-tsb (4th member) | -4.80 | -26.12 | -10.86 | -14.69 | -14.12 | -0.45 | 1/4 | reject |
| D40 + sf-ses (4th member) | -5.21 | -25.90 | -11.50 | -14.24 | -14.21 | -0.54 | 1/4 | reject |
| D40 + best two classics (5 members) | -4.27 | -27.25 | -11.96 | -14.74 | -14.56 | -0.88 | 0/4 | reject |
| replace chronos-t5-small by sf-adida | -4.04 | -27.04 | -11.42 | -15.01 | -14.38 | -0.71 | 0/4 | reject |
| replace chronos-t5-small by sf-croston-sba | -4.09 | -27.14 | -11.76 | -14.83 | -14.46 | -0.79 | 0/4 | reject |
| drop chronos, two classics (4 members) [extra] | -4.16 | -28.00 | -11.97 | -14.82 | -14.74 | -1.07 | 0/4 | reject |

"Best two" = `sf-adida + sf-croston-sba` (the two best solo scores).  "Best
classic" for the replacement test = `sf-adida` (also the best 4th-member result).

### Small-weight sweep (extra, honest; `blend_sweep.csv`)

An equal-weight 4th/5th member already implies 25%/20% weight for the classic, so
a smaller weight was tested: `w` on the classic inside the foundation ensemble,
`w in {0, .05, .10, .15, .20, .25}`.  The LOCO row is the honest one (for each cut,
`w` chosen on the other three cuts).

| variant | cut0 | cut1 | cut2 | cut3 | mean | gain |
|---|---:|---:|---:|---:|---:|---:|
| D40 + LOCO weight on sf-adida | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 |
| D40 + LOCO weight on sf-croston-sba | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 |
| D40 + LOCO weight on sf-adida+sf-croston-sba | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 |
| D40 + in-sample-best weight (all three variants) | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 |

Every positive weight is monotonically worse (e.g. w=0.05 on sf-adida: -13.74;
w=0.25: -14.10), so the honest and in-sample optimum is `w = 0`.  No leakage-tuned
recipe exists here.

## Verdict

**No adoption.**  No variant comes within 0.9 of the +0.5 adoption bar and none
wins 3 of 4 cuts; the best tested variant is `sf-croston-opt` as a 4th member at
-0.42 mean gain with 0/4 cuts won.  The classics are individually much worse than
D40 and their errors are not diverse enough to help; a small weight cannot fix
that (minimum over the whole weight grid is 0 weight).  D40 stays unchanged.

## Files added

* `src/improve/t1_classics.py` — generation + `--blend-sweep` reporting.
* `iterations/08-improvements/t1/results_adamex-svk_t1.csv` — blend tests above.
* `iterations/08-improvements/t1/solo_scores.csv` — all 4 post-processing variants.
* `iterations/08-improvements/t1/blend_sweep.csv` — small-weight sweep.
* `iterations/08-improvements/t1/notes_adamex-svk_t1.md` — this file.
* `iterations/03-foundation-model/forecasts/sf-*__test__cut{0..3}__padded.{parquet,json}` (56 files).
* `iterations/03-foundation-model/forecasts_final/sf-*__test__final__padded.{parquet,json}` (14 files).
* Each new file is ~60 KB; no file over 5 MB, no credentials, no raw data.

Nothing failed.  The only suspicious behaviour is the Croston raw scale blow-up
noted above, which is a known property of the estimator on zero-heavy series and
disappears after rounding + the zero rule.
