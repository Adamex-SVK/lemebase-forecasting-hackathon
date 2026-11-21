"""Zero-shot adapters for the time-series foundation models (iteration 03).

PREPARED ON 2026-11-21 FROM THE MODELS' DOCUMENTATION; THE MODEL ADAPTERS HAVE NOT BEEN RUN
(no GPU and no packages installed yet). The two `dummy-*` models are real and test the plumbing.
Every call below that comes from memory or from a doc snippet that was not fully verified is
marked  # VERIFY.  Run `src/foundation/smoke.py` first on the GPU machine: it tells which adapter
needs a fix.

Interface (same for every model):
    handle = load(name, device)
    F = forecast(name, handle, contexts, horizon, end_week)
        contexts : list of 1-D float arrays (history, oldest first; lengths may differ)
        end_week : pandas Timestamp of the last week of every context
        returns  : float array [n_series, horizon], the MEDIAN forecast (the metric is absolute-error based)
Model packages are imported lazily inside each function, so this file imports without them.
Licences (check before any use beyond the hackathon): Chronos Apache-2.0; TimesFM 2.0/2.5 Apache-2.0,
TimesFM 3.0 weights non-commercial; Moirai models cc-by-nc-4.0 (non-commercial, research); TiRex NXAI community licence.
"""
import os

import numpy as np
import pandas as pd

SPECS = {
    # Chronos 1.0 (T5 tokenised), Chronos-Bolt, Chronos-2          package: chronos-forecasting
    "chronos-t5-small": dict(family="chronos", kind="t5", hf="amazon/chronos-t5-small", chunk=48),
    "chronos-t5-small-q40": dict(family="chronos", kind="t5", hf="amazon/chronos-t5-small", chunk=48, quantile=0.40),
    "chronos-t5-base": dict(family="chronos", kind="t5", hf="amazon/chronos-t5-base", chunk=48),
    "chronos-t5-large": dict(family="chronos", kind="t5", hf="amazon/chronos-t5-large", chunk=48),
    "chronos-bolt-small": dict(family="chronos", kind="bolt", hf="amazon/chronos-bolt-small"),
    "chronos-bolt-base": dict(family="chronos", kind="bolt", hf="amazon/chronos-bolt-base"),
    "chronos-2": dict(family="chronos", kind="c2", hf="amazon/chronos-2"),
    "chronos-2-small": dict(family="chronos", kind="c2", hf="autogluon/chronos-2-small"),
    # TimesFM                                                      package: timesfm[torch]
    "timesfm-2.5": dict(family="timesfm", kind="2.5", hf="google/timesfm-2.5-200m-pytorch"),
    # Moirai (uni2ts; needs its own environment: torch<2.5, numpy 1.26)
    "moirai-1.0-large": dict(family="moirai", kind="m1", hf="Salesforce/moirai-1.0-R-large"),
    "moirai-1.1-large": dict(family="moirai", kind="m1", hf="Salesforce/moirai-1.1-R-large"),
    "moirai-1.1-base": dict(family="moirai", kind="m1", hf="Salesforce/moirai-1.1-R-base"),
    "moirai-moe-small": dict(family="moirai", kind="moe", hf="Salesforce/moirai-moe-1.0-R-small"),
    "moirai-moe-base": dict(family="moirai", kind="moe", hf="Salesforce/moirai-moe-1.0-R-base"),
    "moirai-2.0-small": dict(family="moirai", kind="m2", hf="Salesforce/moirai-2.0-R-small"),
    # quantile / sample-count variants (iteration 08, T6); the default specs above are unchanged
    "moirai-2.0-small-q45": dict(family="moirai", kind="m2", hf="Salesforce/moirai-2.0-R-small", quantile=0.45),
    "moirai-2.0-small-q40": dict(family="moirai", kind="m2", hf="Salesforce/moirai-2.0-R-small", quantile=0.40),
    "moirai-1.1-base-s200": dict(family="moirai", kind="m1", hf="Salesforce/moirai-1.1-R-base", num_samples=200),
    # TiRex                                                        package: tirex-ts
    "tirex": dict(family="tirex", kind="v1", hf="NX-AI/TiRex"),
    "tirex-q40": dict(family="tirex", kind="v1", hf="NX-AI/TiRex", q_index=3),   # trained levels 0.1..0.9; index 3 = 0.4
    # Toto (Datadog, decoder-only, observability + GIFT-Eval pretraining)   package: toto-ts (own env)
    "toto-open-base": dict(family="toto", kind="v1", hf="Datadog/Toto-Open-Base-1.0", chunk=64, samples=256),
    # Sundial (THUML / Timer 3.0, generative flow-matching decoder)         package: transformers (trust_remote_code; own env, transformers==4.40.1)
    "sundial-base-128m": dict(family="sundial", kind="base", hf="thuml/sundial-base-128m", chunk=64, samples=32),
    # TabPFN-TS (tabular foundation model for time series), LOCAL mode only (the hosted API would send the shop's data to a third party),
    # TabPFN-v2 regression weights (ungated; the newer 2.5 / 3.5 weights need a Prior Labs account + licence acceptance + token)
    "tabpfn-ts-v2": dict(family="tabpfn", kind="v2", hf="Prior-Labs/TabPFN-v2-reg", ckpt="tabpfn-v2-regressor.ckpt", chunk=200),
    # plumbing tests, no dependencies
    "dummy-median13": dict(family="dummy", kind="median13"),
    "dummy-naive": dict(family="dummy", kind="naive"),
}
CHUNK = {"chronos": 1024, "timesfm": 1024, "moirai": 256, "tirex": 1024, "tabpfn": 200, "toto": 64, "sundial": 64,
         "dummy": 100000}


def load(name, device="cuda"):
    spec = SPECS[name]
    return _LOADERS[spec["family"]](spec, device)


def forecast(name, handle, contexts, horizon, end_week):
    spec = SPECS[name]
    out = np.zeros((len(contexts), horizon))
    # series per batch; T5 Chronos samples many paths per series and ran out of GPU memory at 1024 (24 GB card)
    step = int(os.environ.get("FM_CHUNK", spec.get("chunk", CHUNK[spec["family"]])))
    for i in range(0, len(contexts), step):
        out[i:i + step] = _FORECASTERS[spec["family"]](spec, handle, contexts[i:i + step], horizon, end_week)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, None)


# ---------------------------------------------------------------- dummy
def _load_dummy(spec, device):
    return None


def _fc_dummy(spec, handle, ctx, H, end_week):
    if spec["kind"] == "naive":
        return np.array([[c[-1]] * H for c in ctx])
    return np.array([[np.median(c[-13:])] * H for c in ctx])


# ---------------------------------------------------------------- Chronos
def _load_chronos(spec, device):
    import torch
    if spec["kind"] == "c2":
        from chronos import Chronos2Pipeline
        return Chronos2Pipeline.from_pretrained(spec["hf"], device_map=device)
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(spec["hf"], device_map=device, torch_dtype=torch.bfloat16)


def _fc_chronos(spec, pipe, ctx, H, end_week):
    if spec["kind"] == "c2":
        # long DataFrame API (documented): id, timestamp, target;  output has a column per quantile, e.g. "0.5"
        rows = []
        for k, c in enumerate(ctx):
            ts = pd.date_range(end=end_week, periods=len(c), freq="W-MON")
            rows.append(pd.DataFrame({"id": k, "timestamp": ts, "target": c.astype("float32")}))
        pred = pipe.predict_df(pd.concat(rows, ignore_index=True), prediction_length=H,
                               quantile_levels=[spec.get("quantile", 0.5)],
                               id_column="id", timestamp_column="timestamp", target="target")
        pred = pred.sort_values(["id", "timestamp"])
        ql = str(spec.get("quantile", 0.5))
        col = ql if ql in pred.columns else "predictions"   # VERIFY the column name
        return pred[col].to_numpy().reshape(len(ctx), H)
    import torch
    # predict_quantiles exists on the 1.0 / Bolt pipelines in the chronos package   # VERIFY
    q, _mean = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in ctx],
                                      prediction_length=H, quantile_levels=[spec.get("quantile", 0.5)])
    return q[..., 0].float().cpu().numpy()                         # [n, H, 1] -> [n, H]


# ---------------------------------------------------------------- TimesFM 2.5
def _load_timesfm(spec, device):
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(spec["hf"])
    m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=64, normalize_inputs=True,
                                     use_continuous_quantile_head=True, force_flip_invariance=False,
                                     infer_is_positive=True, fix_quantile_crossing=True))   # VERIFY
    return m


def _fc_timesfm(spec, m, ctx, H, end_week):
    point, quant = m.forecast(horizon=H, inputs=[c.astype("float32") for c in ctx])
    # quantile_forecast: [n, H, 10] = mean, then the 10th .. 90th percentile (card)   # VERIFY index 5 = median
    return quant[:, :, 5] if quant is not None and quant.shape[-1] >= 10 else point


# ---------------------------------------------------------------- Moirai (uni2ts, GluonTS)
def _load_moirai(spec, device):
    return {"device": device}


def _gluonts_dataset(ctx, end_week):
    from gluonts.dataset.pandas import PandasDataset
    series = {f"s{k}": pd.Series(c.astype("float32"), index=pd.date_range(end=end_week, periods=len(c), freq="W-MON"))
              for k, c in enumerate(ctx)}
    return PandasDataset(series, freq="W-MON")


def _fc_moirai(spec, handle, ctx, H, end_week):
    ds = _gluonts_dataset(ctx, end_week)
    L = int(max(len(c) for c in ctx))
    if spec["kind"] == "m1":
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        model = MoiraiForecast(module=MoiraiModule.from_pretrained(spec["hf"]), prediction_length=H, context_length=L,
                               patch_size="auto", num_samples=spec.get("num_samples", 50), target_dim=1,
                               feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    elif spec["kind"] == "moe":
        from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
        model = MoiraiMoEForecast(module=MoiraiMoEModule.from_pretrained(spec["hf"]), prediction_length=H,
                                  context_length=L, patch_size=16, num_samples=50, target_dim=1,
                                  feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    else:  # Moirai 2.0: the README snippet is broken, the example notebook imports from uni2ts.model.moirai2
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        extra = {"num_samples": spec["num_samples"]} if "num_samples" in spec else {}
        model = Moirai2Forecast(module=Moirai2Module.from_pretrained(spec["hf"]), prediction_length=H,
                                context_length=L, target_dim=1, feat_dynamic_real_dim=0,
                                past_feat_dynamic_real_dim=0, **extra)                          # VERIFY
    predictor = model.create_predictor(batch_size=64)
    out = []
    for f in predictor.predict(ds):
        try:
            out.append(np.asarray(f.quantile(spec.get("quantile", 0.5))))
        except Exception:                                                                       # VERIFY
            out.append(np.asarray(f.median))   # no quantile method: fall back to the median
    return np.stack(out)


# ---------------------------------------------------------------- TiRex
def _load_tirex(spec, device):
    from tirex import load_model
    return load_model(spec["hf"])


def _fc_tirex(spec, m, ctx, H, end_week):
    import torch
    L = max(len(c) for c in ctx)
    X = torch.full((len(ctx), L), float("nan"))              # left-pad with NaN (missing)   # VERIFY NaN handling
    for k, c in enumerate(ctx):
        X[k, L - len(c):] = torch.tensor(c, dtype=torch.float32)
    quantiles, mean = m.forecast(context=X, prediction_length=H)   # order: quantiles, then mean (README)
    q = quantiles.float().cpu().numpy()
    return q[..., spec.get("q_index", q.shape[-1] // 2)]      # middle quantile = median   # VERIFY quantile levels


# ---------------------------------------------------------------- Toto (Datadog)
def _load_toto(spec, device):
    import torch
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto
    m = Toto.from_pretrained(spec["hf"]).to(device)       # VERIFY ModelHubMixin needs huggingface_hub < 0.30
    m.eval()
    return TotoForecaster(m.model)


def _fc_toto(spec, forecaster, ctx, H, end_week):
    import torch
    from toto.data.util.dataset import MaskedTimeseries
    B, L = len(ctx), max(len(c) for c in ctx)
    series = torch.zeros((B, 1, L))
    mask = torch.zeros((B, 1, L), dtype=torch.bool)      # True = real value, False = padding
    stamps = torch.zeros((B, 1, L), dtype=torch.long)
    end_s = int(pd.Timestamp(end_week).timestamp())
    for k, c in enumerate(ctx):
        series[k, 0, L - len(c):] = torch.tensor(c, dtype=torch.float32)
        mask[k, 0, L - len(c):] = True
        # real POSIX seconds, one week apart, oldest first; the model is time-aware   # VERIFY it only needs spacing
        stamps[k, 0, L - len(c):] = torch.tensor(end_s - 604800 * np.arange(len(c) - 1, -1, -1), dtype=torch.long)
    inputs = MaskedTimeseries(series=series.to(forecaster.model.device), padding_mask=mask.to(forecaster.model.device),
                              id_mask=torch.zeros((B, 1, L), dtype=torch.long).to(forecaster.model.device),
                              timestamp_seconds=stamps.to(forecaster.model.device),
                              time_interval_seconds=torch.full((B, 1), 604800, dtype=torch.long).to(forecaster.model.device))
    n = int(os.environ.get("TOTO_SAMPLES", spec["samples"]))
    spb = 32 if n % 32 == 0 else n                     # forecaster requires samples divisible by this batch size
    f = forecaster.forecast(inputs, prediction_length=H, num_samples=n, samples_per_batch=spb)
    return f.median.float().cpu().numpy()[:, 0, :]       # median of the sample paths (their published point forecast)


# ---------------------------------------------------------------- Sundial (THUML)
def _load_sundial(spec, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(spec["hf"], trust_remote_code=True).to(device)
    m.eval()
    return m


def _fc_sundial(spec, m, ctx, H, end_week):
    # contexts may differ in length (first_sale): group by length, Sundial pads each batch on the left internally
    import torch
    by_len = {}
    for k, c in enumerate(ctx):
        by_len.setdefault(len(c), []).append(k)
    out = np.zeros((len(ctx), H))
    N = int(os.environ.get("SUNDIAL_SAMPLES", spec["samples"]))
    for L, ks in by_len.items():
        X = torch.tensor(np.stack([ctx[k] for k in ks]), dtype=torch.float32, device=m.device)
        gen = m.generate(X, max_new_tokens=H, num_samples=N)   # [batch, num_samples, horizon]   # VERIFY sample axis
        out[ks] = np.median(gen.float().cpu().numpy(), axis=1)
    return out


# ---------------------------------------------------------------- TabPFN-TS (local mode, v2 weights)
def _load_tabpfn(spec, device):
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download(spec["hf"], spec["ckpt"])      # ungated; the file name carries the model version (v2)
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
    return TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL, tabpfn_output_selection="median",
                            tabpfn_model_config={"model_path": ckpt})        # VERIFY: device handling is automatic


def _fc_tabpfn(spec, pipe, ctx, H, end_week):
    # ~1 s per series: series with no sale in the last 8 weeks are forecast 0 without calling the model
    # (the 8-week zero rule of the baselines; our best post-processing sets them to 0 anyway)
    alive = np.array([c[-8:].sum() > 0 for c in ctx])
    out = np.zeros((len(ctx), H))
    if alive.any():
        out[alive] = _fc_tabpfn_alive(pipe, [c for c, a in zip(ctx, alive) if a], H, end_week)
    return out


def _fc_tabpfn_alive(pipe, ctx, H, end_week):
    rows = []
    for k, c in enumerate(ctx):
        ts = pd.date_range(end=end_week, periods=len(c), freq="W-MON")
        rows.append(pd.DataFrame({"item_id": k, "timestamp": ts, "target": c.astype("float32")}))
    pred = pipe.predict_df(pd.concat(rows, ignore_index=True), prediction_length=H, quantiles=[0.5])
    pred = pred.reset_index().sort_values(["item_id", "timestamp"])
    return pred["target"].to_numpy(dtype=float).reshape(len(ctx), H)


_LOADERS = {"dummy": _load_dummy, "chronos": _load_chronos, "timesfm": _load_timesfm, "moirai": _load_moirai, "tirex": _load_tirex, "tabpfn": _load_tabpfn,
            "toto": _load_toto, "sundial": _load_sundial}
_FORECASTERS = {"dummy": _fc_dummy, "chronos": _fc_chronos, "timesfm": _fc_timesfm, "moirai": _fc_moirai, "tirex": _fc_tirex, "tabpfn": _fc_tabpfn,
                "toto": _fc_toto, "sundial": _fc_sundial}
