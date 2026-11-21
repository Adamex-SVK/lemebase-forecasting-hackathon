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
    "chronos-t5-base": dict(family="chronos", kind="t5", hf="amazon/chronos-t5-base", chunk=48),
    "chronos-t5-large": dict(family="chronos", kind="t5", hf="amazon/chronos-t5-large", chunk=48),
    "chronos-bolt-small": dict(family="chronos", kind="bolt", hf="amazon/chronos-bolt-small"),
    "chronos-bolt-base": dict(family="chronos", kind="bolt", hf="amazon/chronos-bolt-base"),
    "chronos-2": dict(family="chronos", kind="c2", hf="amazon/chronos-2"),
    # TimesFM                                                      package: timesfm[torch]
    "timesfm-2.5": dict(family="timesfm", kind="2.5", hf="google/timesfm-2.5-200m-pytorch"),
    # Moirai (uni2ts; needs its own environment: torch<2.5, numpy 1.26)
    "moirai-1.0-large": dict(family="moirai", kind="m1", hf="Salesforce/moirai-1.0-R-large"),
    "moirai-1.1-large": dict(family="moirai", kind="m1", hf="Salesforce/moirai-1.1-R-large"),
    "moirai-1.1-base": dict(family="moirai", kind="m1", hf="Salesforce/moirai-1.1-R-base"),
    "moirai-moe-small": dict(family="moirai", kind="moe", hf="Salesforce/moirai-moe-1.0-R-small"),
    "moirai-moe-base": dict(family="moirai", kind="moe", hf="Salesforce/moirai-moe-1.0-R-base"),
    "moirai-2.0-small": dict(family="moirai", kind="m2", hf="Salesforce/moirai-2.0-R-small"),
    # TiRex                                                        package: tirex-ts
    "tirex": dict(family="tirex", kind="v1", hf="NX-AI/TiRex"),
    # plumbing tests, no dependencies
    "dummy-median13": dict(family="dummy", kind="median13"),
    "dummy-naive": dict(family="dummy", kind="naive"),
}
CHUNK = {"chronos": 1024, "timesfm": 1024, "moirai": 256, "tirex": 1024, "dummy": 100000}


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
        pred = pipe.predict_df(pd.concat(rows, ignore_index=True), prediction_length=H, quantile_levels=[0.5],
                               id_column="id", timestamp_column="timestamp", target="target")
        pred = pred.sort_values(["id", "timestamp"])
        col = "0.5" if "0.5" in pred.columns else "predictions"   # VERIFY the column name
        return pred[col].to_numpy().reshape(len(ctx), H)
    import torch
    # predict_quantiles exists on the 1.0 / Bolt pipelines in the chronos package   # VERIFY
    q, _mean = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in ctx],
                                      prediction_length=H, quantile_levels=[0.5])
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
                               patch_size="auto", num_samples=50, target_dim=1, feat_dynamic_real_dim=0,
                               past_feat_dynamic_real_dim=0)
    elif spec["kind"] == "moe":
        from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
        model = MoiraiMoEForecast(module=MoiraiMoEModule.from_pretrained(spec["hf"]), prediction_length=H,
                                  context_length=L, patch_size=16, num_samples=50, target_dim=1,
                                  feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    else:  # Moirai 2.0: the README snippet is broken, the example notebook imports from uni2ts.model.moirai2
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        model = Moirai2Forecast(module=Moirai2Module.from_pretrained(spec["hf"]), prediction_length=H,
                                context_length=L, target_dim=1, feat_dynamic_real_dim=0,
                                past_feat_dynamic_real_dim=0)                                  # VERIFY
    predictor = model.create_predictor(batch_size=64)
    out = []
    for f in predictor.predict(ds):
        try:
            out.append(np.asarray(f.quantile(0.5)))
        except Exception:                                                                       # VERIFY
            out.append(np.asarray(f.median))
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
    return q[..., q.shape[-1] // 2]                           # middle quantile = median   # VERIFY quantile levels


_LOADERS = {"dummy": _load_dummy, "chronos": _load_chronos, "timesfm": _load_timesfm, "moirai": _load_moirai, "tirex": _load_tirex}
_FORECASTERS = {"dummy": _fc_dummy, "chronos": _fc_chronos, "timesfm": _fc_timesfm, "moirai": _fc_moirai, "tirex": _fc_tirex}
