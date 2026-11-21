# T7 — Context variants (padded / first_sale / last52)

Worker T7, iteration 08, branch `it08/adamex-svk/t7`.
Question: does the long zero padding in the history hurt the zero-shot models?
Answer: **no. All three tested variants fail the adoption bar; keep D40 (-13.67).**

## TL;DR

| variant | cut0 | cut1 | cut2 | cut3 | mean | gain vs D40 | cuts won | verdict |
|---|---|---|---|---|---|---|---|---|
| D40 padded (reference) | -3.72 | -25.91 | -11.01 | -14.02 | **-13.67** | 0.00 | – | reference |
| first_sale | -14.98 | -41.42 | -29.88 | -17.36 | -25.91 | -12.25 | 0/4 | **REJECT** |
| last52 | -4.20 | -25.57 | -13.02 | -14.07 | -14.21 | -0.55 | 1/4 | **REJECT** |
| mixed_LOCO | -3.72 | -25.91 | -11.01 | -14.02 | -13.67 | 0.00 | 0/4 | **REJECT** (picks padded everywhere) |

Adoption bar: mean gain >= +0.5 AND better on >= 3 of 4 cuts. Nothing is close.

## Code changes (defaults byte-identical)

* `src/foundation/run_model.py`
  * new `--context last52` (keep the last 52 weekly columns of each series' context);
    default remains `padded`.
  * the "active" flag is now computed from the effective context row. For `padded`
    and `first_sale` this is provably identical to the old `A[k] > 0` test (trimming
    `first_sale` leading zeros cannot change whether a positive exists), and it makes
    `last52` skip series whose last 52 weeks are all zero.
* `src/meta/ensemble_recipe.py`
  * new `--context {padded,first_sale,last52}` on both `evaluate` and `build`
    (default `padded`). `evaluate` always scores the D40 reference on `padded`;
    only the candidate reads the requested context. Cache keys include the context.
  * `build` reads `forecasts_final/{model}__test__final__{context}.parquet`.

Default-preservation proof (run locally after the edit):

```
$ .venv/bin/python src/meta/ensemble_recipe.py evaluate --models moirai-2.0-small,tirex,chronos-t5-small
reference D40 ... by cut:   -3.72  -25.91  -11.01  -14.02   mean  -13.67
candidate     ... by cut:   -3.72  -25.91  -11.01  -14.02   mean  -13.67

$ .venv/bin/python src/meta/ensemble_recipe.py build --models moirai-2.0-small,tirex,chronos-t5-small \
      --lgbm-weight 0.3 --out /tmp/t7_build_check.csv
MD5 (/tmp/t7_build_check.csv)                 = 81154895d89604d463313e3f74759894
MD5 (submissions/ensemble_top3_blend_v1.csv)  = 81154895d89604d463313e3f74759894
```

I also compared the new `contexts_for` with a re-implementation of the old logic on
`padded` and `first_sale` for cuts 0-3 and `--final`: identical keys, contexts, active
flags and cut timestamps.

## GPU work

* Own remote folder `~/hack-t7`: copied `~/hack/src`, symlinked `~/hack/data`,
  `.venv-moirai`, `.venv-tirex`, `.venv-chronos`; pushed the edited `run_model.py`.
  All runs launched from `~/hack-t7` with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`.
* 3 models x 2 contexts (`first_sale`, `last52`) x 5 runs (cuts 0-3 + final) = **30 runs, all rc=0**.
* The 5 stalled TabPFN processes (PIDs 38996/38998/39001/39002/39007) were checked and left untouched.
* Wall time **09:50** (15:09:57 -> 15:19:47 UTC). Process time: load 174.4 s + forecast 377.2 s
  across the 30 runs (~9m12s). Slowest single forecast: moirai cut2 last52, 24 s.
* All 30 files verified: 2,904 series x 13 weeks, no NaN, minimum forecast 0.0.
* No leakage: every context for cut K uses only weeks before the first holdout week;
  the `last52` window is the last 52 pre-cut weeks.

## How the contexts behave

All series share the shop grid; `padded` keeps the whole (up to 130-week) history with
leading zeros, `first_sale` trims the leading zeros, `last52` keeps the last 52 pre-cut
weeks. Because history length differs per cut (130/117/104/91 weeks for cuts 0/1/2/3),
`last52` truncates 78/65/52/39 weeks respectively. Series sent to the model:
`first_sale` 2839/2762/2680/2599, `last52` 2356/2324/2306/2270 (the rest forecast 0 by the
all-zero-context rule and are then subject to the D39 zero rule in the blend).

## Solo scores (round + zero-8, no LightGBM)

| model | context | cut0 | cut1 | cut2 | cut3 | mean |
|---|---|---|---|---|---|---|
| moirai-2.0-small | padded | -4.60 | -27.68 | -11.76 | -14.83 | **-14.72** |
| moirai-2.0-small | first_sale | -20.80 | -46.07 | -34.98 | -19.95 | -30.45 |
| moirai-2.0-small | last52 | -6.92 | -27.00 | -13.72 | -15.57 | -15.80 |
| tirex | padded | -6.19 | -27.33 | -15.67 | -14.67 | **-15.97** |
| tirex | first_sale | -24.79 | -47.83 | -38.48 | -20.62 | -32.93 |
| tirex | last52 | -7.57 | -27.85 | -17.08 | -15.33 | -16.96 |
| chronos-t5-small | padded | -7.51 | -30.02 | -17.30 | -16.93 | **-17.94** |
| chronos-t5-small | first_sale | -23.39 | -47.46 | -37.44 | -22.84 | -32.78 |
| chronos-t5-small | last52 | -11.89 | -32.37 | -17.61 | -18.01 | -19.97 |

For **every** model, `padded` is the best context on the mean. `last52` is uniformly a
little worse; `first_sale` is catastrophic. Hence the leave-one-cut-out selection picks
`padded` for every (model, target cut):

```
cut 0/1/2/3: moirai-2.0-small=padded, tirex=padded, chronos-t5-small=padded
```

so `mixed_LOCO` is bit-identical to D40.

## Interpretation (why padding helps)

On cut 0 the plain (unscaled) MAE of the 0.3 LightGBM + 0.7 ensemble blend is almost
unchanged across contexts — padded 4.5865, first_sale 4.5898, last52 4.5518 — yet the MASE
score differs by up to 11 points. MASE averages per-series MAE divided by a per-series
scale (effectively equal weight per article), so the ~60-96%-zero, low-volume series
dominate it. Removing the leading zeros makes those histories look denser and biases their
forecasts up: small absolute changes on low-volume series move MASE a lot. The long zero
prefix is therefore **useful prior information**, not noise, and truncating it
(`first_sale`) or shortening it (`last52`) hurts exactly the series that drive the metric.

`last52` cut2 is the worst cut (-2.01 vs D40); its slightly *better* high-volume MAE does
not compensate the worse scaled errors on sparse series.

## Files added

* `src/foundation/run_model.py`, `src/meta/ensemble_recipe.py` (options only; defaults proven identical).
* `src/improve/t7_context.py` — evaluates each context and the mixed leave-one-cut-out
  selection, writes the results CSV.
* `iterations/08-improvements/t7/results_adamex-svk_t7.csv` — the table above.
* `iterations/08-improvements/t7/notes_adamex-svk_t7.md` — this file.
* 30 new forecast parquet + 30 json (names `*__first_sale.*`, `*__last52.*`), no existing file overwritten.

## Failed / suspicious

* Nothing failed: 30/30 rc=0, all files complete. `bash scripts/check.sh` passes before and after.
* Not suspicious after checking: the large `first_sale` drop is a MASE-scaling effect as
  explained above (plain MAE nearly unchanged), not a data or plumbing error; solo scores
  agree across all three model families and all four cuts.
* No submission built (nothing passed). No leaderboard interaction.

## Recommendation

Keep D40 unchanged. The recipe change proposed by this task (context switching) does not
help; with the current metric, the full zero-padded history is the right input.
