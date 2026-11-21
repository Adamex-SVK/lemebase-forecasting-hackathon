# Foundation models for iteration 03: facts and pitfalls

Compiled 2026-11-21 from the models' public pages by three read-only research agents (GitHub READMEs, Hugging Face cards, `pyproject.toml`). Not tested on our machine. Items marked **VERIFY** were not confirmed on the pages. User-chosen list (D35): Chronos 1.0, Chronos-Bolt, Chronos-2, TimesFM, Moirai 1.0/1.1, Moirai-MoE, Moirai 2.0, TiRex.

| Family | Model ids | Package | Python / torch pins | Licence |
|---|---|---|---|---|
| **Chronos 1.0** | `amazon/chronos-t5-{tiny 8M, mini 20M, small 46M, base 200M, large 710M}` | `chronos-forecasting` | Python >= 3.10, torch >= 2.2 < 3, transformers >= 4.41 < 6 | Apache-2.0 |
| **Chronos-Bolt** | `amazon/chronos-bolt-{tiny 9M, mini 21M, small 48M, base 205M}` | same | same | Apache-2.0 |
| **Chronos-2** | `amazon/chronos-2` (120M), `autogluon/chronos-2-small` (28M) | `chronos-forecasting>=2.0` | same | Apache-2.0 |
| **TimesFM** | `google/timesfm-2.5-200m-pytorch` (200M), `google/timesfm-2.0-500m-pytorch` (500M, needs `timesfm==1.3.0`), 3.0 (330M, `timesfm3`) | `timesfm[torch]` (also `[mlx]` for Apple silicon, `[flax]`) | Python >= 3.10, torch >= 2.0; pick ONE backend (jax vs torch) | 2.0/2.5 Apache-2.0; **3.0 weights non-commercial** |
| **Moirai 1.0 / 1.1** | `Salesforce/moirai-1.0-R-large`, `moirai-1.1-R-{small, base, large 0.3B}` | `uni2ts` | **torch >= 2.1 < 2.5, numpy ~= 1.26, gluonts ~= 0.14.3**; needs its own env (not Python 3.13) | **cc-by-nc-4.0, research only** |
| **Moirai-MoE** | `Salesforce/moirai-moe-1.0-R-{small, base 0.9B}` | `uni2ts` | same | cc-by-nc-4.0 |
| **Moirai 2.0** | `Salesforce/moirai-2.0-R-small` (11.4M, decoder-only) | `uni2ts` | same | cc-by-nc-4.0 |
| **TiRex** | `NX-AI/TiRex` (35M, xLSTM) | `tirex-ts` (`[cuda]` extra) | CUDA kernels compile on first load, GPU compute capability >= 8.0; tested on Linux and macOS | NXAI community licence (not read) |

## Inference API notes (what the adapters in `src/foundation/models.py` use)
- **Chronos 1.0 / Bolt:** `BaseChronosPipeline.from_pretrained(id, device_map=..., torch_dtype=torch.bfloat16)`; the docs show `predict(context, prediction_length)`. 1.0 returns sample paths `[series, samples, horizon]`, Bolt returns quantiles `[series, quantiles, horizon]`. Our adapter calls `predict_quantiles(..., quantile_levels=[0.5])` (**VERIFY** it exists on both).
- **Chronos-2:** `Chronos2Pipeline.from_pretrained("amazon/chronos-2")` and `predict_df(context_df, prediction_length, quantile_levels, id_column, timestamp_column, target)` with long-format input; output has a `predictions` column and one column per quantile named by its value (**VERIFY** that `"0.5"` is the median column). Native past/future covariates (not used in the zero-shot run). The page documents no fine-tuning procedure (`peft` is an optional extra).
- **TimesFM 2.5:** `timesfm.TimesFM_2p5_200M_torch.from_pretrained(id)`, `model.compile(ForecastConfig(...))`, `model.forecast(horizon, inputs=list of arrays)` returns point forecasts and quantiles `[n, H, 10]` = mean then the 10th to 90th percentile (**VERIFY** index 5 = median). Variable-length inputs are allowed. The card's `infer_is_positive=True` is the flag most relevant to counts. Context up to 16k. 2.0 expects a frequency code (1 = weekly); 2.5 and 3.0 do not.
- **Moirai:** GluonTS `PandasDataset` of series with a weekly `DatetimeIndex`; `MoiraiForecast(module=MoiraiModule.from_pretrained(id), prediction_length=13, context_length=..., patch_size="auto", num_samples=..., target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)`, `create_predictor(batch_size)`, `forecast.median`. MoE uses `MoiraiMoEForecast` with `patch_size=16`. **Moirai 2.0: the README snippet is broken** (missing imports, hard-coded `prediction_length=100`); the example notebook imports `Moirai2Forecast, Moirai2Module` from `uni2ts.model.moirai2`. Covariates for 1.x via `feat_dynamic_real_dim` / `past_feat_dynamic_real_dim`.
- **TiRex:** `from tirex import load_model; model = load_model("NX-AI/TiRex"); quantiles, mean = model.forecast(context=tensor[n, T], prediction_length=H)` (order: quantiles first). Quantile levels, NaN handling and context limit are not on the README (**VERIFY**).

## Fine-tuning (for the second half of iteration 03)
- **Chronos 1.0:** `scripts/training/train.py` with a YAML config, data as GluonTS arrow files; the README says it "may be outdated".
- **Chronos-Bolt:** through AutoGluon `TimeSeriesPredictor(...).fit(..., hyperparameters={"Chronos": {...}})`.
- **Chronos-2:** not documented on the pages read (**VERIFY**, `peft` extra suggests adapters).
- **Moirai:** uni2ts CLI (`python -m cli.train` with Hydra configs; data builder `uni2ts.data.builder.simple`); only 1.0 configs documented.
- **TimesFM:** a LoRA fine-tuning example (HF Transformers + PEFT) is listed in the repo (**VERIFY**).
- **TiRex:** no public recipe.

## Cautions for our data (60-96% zeros)
None of the pages discuss intermittent or count data. We clip forecasts at 0, take the median (our metric is absolute-error based), do not send all-zero contexts to a model, and compare every model with the median-of-last-13-weeks baseline and the LightGBM on the same cuts. TimesFM 2.0 interpolates NaNs linearly, so zeros and gaps must be explicit.

## Hardware (from the pages; no measurements)
Weights are small (Chronos-2 about 0.5 GB, TimesFM 2.5 about 0.8 GB, TiRex about 0.14 GB, Moirai-MoE-base about 3.6 GB, Chronos-T5-large about 2.8 GB in fp32). Chronos-2 is stated at over 300 series per second on an A10G (so 50,000 series take about 3 minutes). TimesFM has an Apple-silicon MLX backend. Apple MPS support is only mentioned for Chronos-Bolt.

## Verified from the package source (2026-11-21; downloaded the PyPI wheels and read the code, nothing was run)
- **chronos-forecasting 2.3.2:** `predict_quantiles` exists on `ChronosPipeline` (1.0), `ChronosBoltPipeline` and `Chronos2Pipeline`. `Chronos2Pipeline.predict_df` returns a long DataFrame with the id column, the timestamp, `target_name`, **`predictions`** (point forecast) and one column per requested quantile level (named by the level); the adapter asks for `0.5` and falls back to `predictions`.
- **timesfm 3.0.2 (PyPI latest) and 2.0.2** both ship `timesfm.TimesFM_2p5_200M_torch` (module `timesfm_2p5`); 3.0.2 additionally ships `timesfm3` (the 3.0 model, non-commercial weights). The unpinned `timesfm[torch]` therefore works for the 2.5 adapter. `forecast(horizon, inputs)` returns `(point_forecast, quantile_forecast)`.
- **tirex-ts 1.4.2:** `load_model(path, backend="torch" | "cuda")`; the default is `cuda` only when the `xlstm` package is installed, otherwise the pure-PyTorch `torch` backend, so **no nvcc is needed** for plain `tirex-ts`. The trained quantiles are 0.1 to 0.9; `forecast(context, prediction_length)` returns `(quantiles, mean)` where `mean` is the median (the middle quantile, 0.5). NaN values in the context are handled as missing through an input mask.

## TabPFN-TS (added 2026-11-21, from the package source and pages)
- Packages: `tabpfn-time-series` 1.4.0 (Apache-2.0; needs `tabpfn>=9.2.0`, `gluonts>=0.16`, `datasets>=4`, `fev`, `statsmodels`, `tabpfn-client`) and `tabpfn` 10.2.0 (needs `torch>=2.5`, which the GPU machine has: 2.5.1).
- API (read from the wheel): `TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL, tabpfn_output_selection="median", tabpfn_model_config={"model_path": ...})`; `predict_df(context_df, prediction_length, quantiles=[...])` with a long DataFrame (`item_id`, `timestamp`, `target`); the result is indexed by (item_id, timestamp) with `target` = the median and one column per quantile. **The code defaults to LOCAL mode** (the README says cloud: out of date); the cloud mode would send the data to Prior Labs. It builds calendar / seasonal features from the timestamps.
- Weights: the package pins `v3.5` by default (gated: Prior Labs account, licence acceptance, token; non-commercial). The **v2 regression checkpoint** `Prior-Labs/TabPFN-v2-reg/tabpfn-v2-regressor.ckpt` is ungated (HTTP 200 without a token; licence "Prior Labs License", Apache 2.0 with an attribution requirement); the file name must keep its version. Our adapter (`tabpfn-ts-v2`) downloads it with `hf_hub_download` and passes the path as `model_path`. Telemetry off: `TABPFN_DISABLE_TELEMETRY=1`.
- Not verified: speed for about 2,900 series (each series is its own table), behaviour on zero-heavy counts, whether `model_path` accepts the v2 file in this package version. Marked `VERIFY` in the adapter.
