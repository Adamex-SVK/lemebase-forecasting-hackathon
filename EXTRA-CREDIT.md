# Extra credit (draft): does what we learn on one shop help another?

The brief asks: run each shop on its own data only and with the other shops included, show both scores, explain why borrowing helped or hurt. The page says this is weighed when picking the winner. **Status: draft with partial evidence; step 1.3 (all shops x arms) was never finished.**

## Evidence so far (D26 score, 13-week local cut, higher is better)
**Test shop, LightGBM trained on different data** (`iterations/01-global-gbm/log.md`, first attempt, cut 0; the same own-arm number -4.83 is reproduced by the other session's reference file):

| Training data | Score | Note |
|---|---|---|
| own history only | **-4.83** | 127,143 windows |
| own + all four other shops, shops weighted equally | -5.84 | 7.7M rows |
| own + all four other shops, unweighted | -4.95 | 7.7M rows (Set D is 53% of them) |
| own + other shops (other session, `04-rich-columns`, no rich columns) | -5.69 | same protocol, seed 0 |
| the other four shops only (zero-shot transfer) | **-23.07** | about the same as forecasting zero (-23.4) |

Shop A, own data only: 14.49 (first attempt). Shops B, C, D and the pooled arms for A, B, C, D were **not** run to completion (`python src/run_arms.py`, 45-60 minutes; partial results in `iterations/01-global-gbm/results_1_3_partial_origin_step2.csv`).

**Pretrained models applied cold to the test shop** (`iterations/03-foundation-model`): models trained on other people's data, never on this shop. Zero-shot, rounded, mean over 4 cuts: Moirai 2.0-small -14.72, TiRex -15.96 versus the trained LightGBM -15.12 (rounded). A blend of three of them with the LightGBM reaches -13.67, and scored -4.8 live against -7.6 for the LightGBM alone.

## What we can already say (draft explanation)
1. **Borrowing the other shops' rows did not help our own-data LightGBM.** The shops share no articles and have different calendars, scales and sparsity (Set D is 94% zero weeks, the test shop 65%), so the extra rows mostly teach behaviours that do not match; equalising the shops made it slightly worse, letting the zero-heavy Set D dominate left it about equal. Learning *only* from the other shops fails (-23.07): the model needs the shop's own level and rhythm.
2. **Borrowing helped when it was done differently:** pretrained time-series models carry general knowledge about short, noisy series from very large corpora and add information the shop's own history cannot give. Used as an ensemble alongside the own-data model they improved the score by about 1.5 points locally and 2.8 live. Borrowing helps when it adds a *different view* (diversity) and hurts when it replaces the shop's own signal or imposes another shop's calendar and scale (as the brief's own experiment showed for Set B's seasonal shape).
3. **Sparsity matters:** on shops with very long zero runs (C, D) a plain zero forecast already beats fitted models (C 58.8, D 69.1 vs AutoARIMA 40.2 and 58.6), so what is borrowed must be robust to zeros.

## To finish
- Run `src/run_arms.py` for all five shops (own / pooled equal / pooled unweighted / others only) and fill the table; add the zero-shot foundation models and the rolling cuts for the test shop.
- Check the rich-column experiments (`iterations/04-rich-columns`, another session) as a third kind of borrowing.
- State both scores per shop in one table, with paired intervals, and keep the explanation above to what the numbers support.
