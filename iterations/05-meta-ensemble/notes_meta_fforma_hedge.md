# FFORMA hedge vs pure: purged evaluation and two candidate submissions (D42)

2026-11-21. Code: `src/meta/fforma.py` (new options `--hedge-weight`, `--fit-all`, `--final`, `--out-pure`, `--out-hedge`; defaults unchanged and the default run reproduces the D37 numbers exactly).

## Purged evaluation (top-3 foundation models, FFORMA bases = LightGBM only, 1500 epochs)
Gain over the simple blend (weight chosen on the training cuts) on the 4 overlapping test cuts:

| Variant | W=0 | W=4 | W=8 | W=12 | Mean gain | Cuts won |
|---|---|---|---|---|---|---|
| Pure FFORMA (tested protocol, newest training cut held out) | +0.66 | +0.15 | +0.26 | -0.17 | +0.23 | 3/4 |
| Hedge 0.5 FFORMA + 0.5 simple (tested protocol) | +1.00 | +0.06 | +0.15 | -0.16 | **+0.26** | 3/4 |
| Pure FFORMA, `--fit-all` | +1.20 | +0.11 | +0.17 | -0.86 | +0.16 | 3/4 |
| Hedge, `--fit-all` | +1.25 | +0.06 | +0.11 | -0.40 | +0.26 | 3/4 |

All variants are below the adoption bar (+0.5 mean AND >= 3/4). The test cuts overlap, so they are not
independent evidence. Results: `results_meta_fforma_lgbm_only_hedge.csv`, `results_meta_fforma_lgbm_only_hedge_fitall.csv`.

## Submissions built (tested protocol: trained on W = 4..48, W = 0 held out)
```bash
.venv/bin/python src/meta/fforma.py --final --ff-bases lgbm \
    --out-pure submissions/ensemble_fforma_pure_v1.csv --out-hedge submissions/ensemble_fforma_hedge_v1.csv
```
- Mean FFORMA weights at the real origin: lgbm 0.37, moirai-2.0-small 0.23, tirex 0.25, chronos-t5-small 0.15.
- Hedge = 0.5 x FFORMA + 0.5 x D40 blend (LightGBM weight 0.3, as uploaded), then zero rule, round, clip.
- Against the uploaded `ensemble_top3_blend_v1.csv` (D40): pure differs in 25.4% of articles (mean |diff| 0.37
  units, +0.9% total units); hedge differs in 21.9% (mean |diff| 0.18, +0.4% total units).
- Both pass `make_submission.validate_submission`. Not uploaded (a person uploads; log in `submissions/UPLOADS.md`).

## How to read a live score
The live board scores weeks 1-7 with noise of about 2 points or more; the expected local gain is about +0.25.
A live difference against D40 (-4.8) smaller than about 2 points says nothing either way.

## Live result (2026-11-21, weeks 1-7)
| File | Live | vs D40 (-4.8) |
|---|---|---|
| `ensemble_fforma_pure_v1.csv` | -7.9 | -3.1 |
| `ensemble_fforma_hedge_v1.csv` | -6.2 | -1.4 |

Pure FFORMA loses by more than the live noise band (about 2 points) and scores below LightGBM alone (-7.6).
The hedge lands about halfway between D40 and pure, which fits mixing in a worse forecast rather than luck.
The local evidence already pointed this way: FFORMA lost the hardest cut (W=12) in every variant.
Conclusion: FFORMA not adopted; D40 stays the final submission. Do not tune FFORMA on the live board.
