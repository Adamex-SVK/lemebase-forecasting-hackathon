# Iteration 05 log (meta-ensemble on overlapping cuts)

Append-only. (This folder was first created as `04-meta-ensemble`; renamed to 05 because another session uses `04-rich-columns`, as in the roadmap of `pipeline-architecture.md`.)

## 2026-11-21: base forecasts at 13 overlapping cuts
- **Cuts:** offsets N = 0, 4, ..., 48 weeks removed from the end of the test-shop history; holdout = the 13 weeks after the cut. Runner option `--cut-weeks N` in `src/foundation/run_model.py` (output `iterations/03-foundation-model/forecasts_cuts/{model}__test__w{N}__padded.parquet`).
- **LightGBM (own arm, plain features, seed 0) at all 13 cuts: finished** (`src/meta/base_cuts_lgbm.py`; `base_forecasts/lgbm_w{N}.npy` and `median13_w{N}.npy`, rows in `ShopMatrix` key order of the training part). Rounded scores per cut (holdout start in brackets): N=0 -4.03 (2023-01-23), 4 -2.81, 8 -12.72, 12 -22.43, 16 -37.89 (2022-10-03), 20 -29.07, 24 -2.27, 28 -24.36, 32 -19.74, 36 -3.11, 40 -30.87, 44 +7.30 (2022-03-21), 48 +5.27 (2022-02-21). **Window difficulty varies by 45 points between neighbouring months**: compare models within a cut only; the Oct-Dec windows are the hardest.
- **Foundation models at the cuts** (GPU machine, `run_cuts.sh`, `OMP_NUM_THREADS=4`): tirex finished (13 cuts, about 50 s each); moirai-2.0-small, moirai-1.1-base and chronos-t5-small/base in progress at the time of writing (check `forecasts_cuts/`, target 5 models x 13 cuts = 65 files). **TimesFM was dropped**: its first overlapping cut sat at about 25% CPU for 11 minutes (it had run fine on the standard cuts); a restart behaved the same; not a top model (-18.7 alone), so it was left out.
- **Not done yet:** the meta-learner itself (`src/meta/fforma.py`), see `plan.md` for design and the leakage-free protocol (train on cuts with offset >= test offset + 13; test offsets 0, 4, 8, 12).

## 2026-11-21 (later): meta-learner built and evaluated (D37, NOT adopted)
- **Collected forecasts:** the full 5-model x 13-cut overlapping-cut set is now in `forecasts_cuts/` (tirex, moirai-2.0-small, chronos-t5-small were done; chronos-t5-base `STREAM_DONE 13:50`, moirai-1.1-base `STREAM_DONE 13:57`, copied). Also copied `forecasts_final/moirai-1.1-base`. **TabPFN full runs were still in progress** (5 processes, about 6h45 CPU each), so D36 (TabPFN) is not scored yet.
- **Code:** `src/meta/fforma.py`. Per overlapping cut it assembles the base forecasts (LightGBM `lgbm_w{N}.npy`, `median13_w{N}.npy`, constant zero, foundation models from `forecasts_cuts/`), the actuals, and the per-article MASE scale (`validate.history_scale`, D26). It compares (1) the simple blend with a global LightGBM weight chosen on the training cuts and (2) FFORMA-linear: per-article `softmax(X @ theta)` weights over all base methods trained on the exact scaled-absolute-error objective (mean_i mean_w |blend-actual| / scale), Adam, L2. Purged protocol: test offsets 0/4/8/12, training only on cuts with W >= W_t + 13. Post-processing: 8-week zero rule, round, clip (D38/D39).
- **Results** (mean over test cuts 0/4/8/12; `results_meta_fforma_*.csv`):

  | FFORMA bases | foundation models | simple blend | FFORMA | gain | cuts |
  |---|---|---|---|---|---|
  | LightGBM only | top-3 | -9.74 | -9.51 | +0.23 | 3/4 |
  | LightGBM + median13 + zero | top-3 | -9.74 | -9.79 | -0.05 | 1/4 |
  | LightGBM only | all 5 | -9.91 | -9.84 | +0.08 | 3/4 |
  | LightGBM + median13 + zero | all 5 | -9.91 | -9.67 | +0.24 | 3/4 |

  The learned weights are almost global (per-article variation tiny); for the best variant they are about LightGBM 0.34, moirai-2.0-small 0.24, tirex 0.25, chronos-t5-small 0.17.
- **Decision:** every variant is below the adoption bar (+0.5 mean AND >= 3/4 cuts), so **keep the simple blend (D40)**. Adding the 5th/4th foundation models does not help on these cuts (the 5-model simple blend -9.91 is worse than the top-3 -9.74).
- **Final-recipe check on the standard (non-overlapping) 4 cuts** with `src/meta/eval_global_weights_4cut.py`: D40 (0.30 LightGBM + equal top-3) gives **-13.67** (ensemble alone -14.05), and every alternative global weighting is worse (-13.70 to -14.05). The submission was rebuilt byte-identical; the recipe is unchanged.
