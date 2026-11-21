# Iteration 04: meta-ensemble (FFORMA style) on overlapping cuts

**Status (2026-11-21):** started. Base forecasts at 13 overlapping cuts are being produced; the meta-learner is next. Decision D37.

## Why
The simple blend (average of three foundation models, 8-week zero rule, 30% LightGBM, rounded) scored **-4.8 live** (LightGBM alone -7.6, Moirai 2.0-small alone -7.3), so combining models is where the gain is. FFORMA learns per-article weights from article features (how often it sells, level, scale, stock) instead of one global weight.

## Data
- **Cuts:** offsets W = 0, 4, 8, ..., 48 weeks removed from the end of the test-shop history (13 cuts, holdout = the next 13 weeks, so neighbouring cuts overlap in their holdout weeks).
- **Base forecasts per cut:** LightGBM (own, retrained per cut; `src/meta/base_cuts_lgbm.py`), median of the last 13 weeks, zero, and the zero-shot models moirai-2.0-small, moirai-1.1-base, tirex, timesfm-2.5, chronos-t5-small, chronos-t5-base (GPU machine, `run_model.py --cut-weeks`, `run_cuts.sh`; TabPFN joins if its runs finish).
- Actuals and the MASE scale of every article at every cut (D26 metric).

## Evaluation protocol (no leakage)
A test cut with offset W_t may only train the meta-learner on cuts whose whole holdout lies before the test cut starts, i.e. W >= W_t + 13. Test cuts: W_t in {0, 4, 8, 12} (training on 9, 8, 7 and 6 cuts). Overlapping cuts give more training examples, not independent evidence; the comparison on the 4 non-overlapping cuts of iterations 01-03 stays the honest yardstick. Reference to beat on the same test cuts: the simple blend with one global weight chosen on the same training cuts, rounded.

## Meta-learners
1. **Simple blend** (reference): average of the top-3 zero-shot models, zero rule, one weight on the LightGBM.
2. **FFORMA-linear:** per-article weights softmax(X theta) over the base models, trained on the exact objective (absolute error of the blended forecast / article scale), L2 penalty, early stopping on the latest training cuts.
3. **FFORMA-lite:** LightGBM regressors that predict each model's error per article, then softmax(-lambda * predicted error) weights.
Post-processing for all: 8-week zero rule, round to whole units, clip at 0.

## Adoption bar (TEAM-RULES.md section 2)
Mean gain over the reference >= 0.5 points on the test cuts and better on >= 3 of 4; otherwise keep the simple blend.
