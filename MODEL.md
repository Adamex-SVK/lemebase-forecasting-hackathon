# The current best model (D40)

One page that says what the best model is, how to rebuild it, what it is worth, and where it is weak. Everything here is backed by the logs; nothing is untested.

## Result
| | Score (higher is better) |
|---|---|
| Live board, weeks 1-7 | **-4.8** (LightGBM alone -7.6, median of last 13 weeks -9.8, AutoARIMA -33.7, Naive -64.5) |
| Local, mean of 4 rolling cuts | **-13.67** (cuts -3.72 / -25.91 / -11.01 / -14.02) |

Local cuts differ by up to 45 points between months (Oct-Dec are the hardest), so compare models inside a cut, never across cuts. The final score uses weeks 8-13, not the 7 weeks seen live.

## The recipe
1. **Foundation ensemble.** Average the raw zero-shot forecasts of Moirai 2.0-small, TiRex and Chronos-T5-small. They are used as published (no training on our data), with the shop's weekly history zero-padded from its first sale week.
2. **Zero rule (D39).** Forecast 0 for any article with no sale in the last 8 weeks (1,073 of 2,904 articles).
3. **LightGBM blend.** `0.3 x LightGBM + 0.7 x ensemble`. LightGBM (`lgbm_own_v1`): L1 loss on units divided by each article's MASE scale (D29), 41 lag and calendar features, training windows every 2 weeks, 276 trees, trained on the test shop's own history only.
4. **Round and clip (D38).** Round to whole units, clip at 0. The metric favours integer medians for slow sellers.

## Rebuild
```bash
.venv/bin/python src/prepare.py                    # weekly tables from the raw data (once)
.venv/bin/python src/build_best_model.py           # LightGBM + ensemble, validates, checks the hash
.venv/bin/python src/build_best_model.py --skip-lgbm   # reuse submissions/lgbm_own_v1.csv
```
Rebuilding reproduces `submissions/ensemble_top3_blend_v1.csv` **byte for byte** (sha256 checked in the script; verified 2026-11-21). The final foundation-model forecasts are committed in `iterations/03-foundation-model/forecasts_final/`, so no GPU is needed to rebuild. Regenerating them needs the GPU machine (`GPU-ACCESS.md`, `iterations/03-foundation-model/gpu-machine.md`).

## Why this recipe (evidence)
- **Combining different models is what worked**: LightGBM -7.6 and Moirai alone -7.3 live, the blend -4.8 (`iterations/03-foundation-model/`).
- **It sits at a robust optimum.** About thirty alternatives were scored on the same 4 cuts with leave-one-cut-out and none cleared the bar (+0.5 mean and better on 3 of 4 cuts): greedy selection, SLSQP weights, segment weights, zero voting, performance-based weights, FFORMA (live -7.9), TabPFN swap (live -5.2), ten improvement tasks T1-T10 (`decisions-to-make.md` D41), neural N-HiTS / N-BEATS, rich columns (D43). Every model set tried gave -13.7 to -13.9.

## Weaknesses
- Slow sellers dominate the error; the integer median is the best we found for them.
- One seed of LightGBM and no fine-tuning of any foundation model on retail data.
- Nothing in the model knows about holidays, price or stock; the shop's country is not identified.
- Local and live scores agree on direction, not on size (local -13.67 vs live -4.8).

## Next (not in this model yet)
`iterations/10-external-info/plan.md`: original public sources (X2), holiday features (X3), bias correction learned on sibling shops (X4); then fine-tuning on real sibling retail data. A change enters this model only through a decision entry and the 4-cut bar.
