"""Iteration 06: neural forecasters trained from scratch on our weekly tables (NHITS / NBEATS).

Part 3 of iterations/02-robust-eval/notes_adamex_loss-and-nn-plan.md, gamma of
iterations/06-neural/notes_kilo_neural.md.

Design that keeps the neural model comparable with the LightGBM (D29, D26):
  * target = units / the series' D26 MASE scale. The scale is the mean absolute week-to-week change of the
    zero-padded history (leading zeros from the shop's first week), exactly `features_at` / `validate.history_scale`;
    it is computed per shop from that shop's own first week, so pooling stays leak-free and comparable.
  * loss = MAE on the scaled target; `scaler_type="identity"` (no second normalisation).
  * one model per cut, trained only on weeks before the cut (own arm); `--arm pooled` adds all weeks of
    shops A-D (they share no stores or products with the test shop, D27).
  * constant-history series (scale 0) are dropped from training and get forecast 0, like the LightGBM;
    they keep their row so the matrices line up.
  * forecast = model output * scale, clipped at 0; seeds are averaged (3 seeds by default).

Outputs (kept out of the foundation-model folders so nothing collides):
  iterations/06-neural/forecasts/{model}__test__cut{k}__padded.parquet            seed average, store_id, article_id, week, forecast
  iterations/06-neural/forecasts/{model}__test__cut{k}__padded__seed{s}.parquet   single seeds
  iterations/06-neural/forecasts/{model}__cut{k}__mean.npy, ...__seed{s}.npy      same rows in ShopMatrix key order
  a small .json next to the parquet with the run settings (versions, input_size, steps, timing).
With --cut-weeks N the stem is w{N} and the files go to iterations/06-neural/forecasts_cuts/.

Run on the GPU machine (one env per family, see env/setup_gpu.sh):
  .venv-nf/bin/python src/neural/run_nf.py --arm own --cuts 0 1 2 3 --models NHITS NBEATS --seeds 0 1 2
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-nf/bin/python src/neural/run_nf.py --limit 200 --max-steps 100   # smoke
An existing output file is reused instead of retraining, so every grid config needs its own `--tag`
(e.g. `--tag _s3000_lr3e-4_in104`); the baseline keeps the empty tag. `--dropout`, `--pool-shops A C`
and `--start-padding` are the other knobs. Needs pandas / numpy / pyarrow plus neuralforecast; scoring is
done on the laptop by src/neural/score_nf.py.
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import features as f  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
H = f.H
OUT = ROOT / "iterations" / "06-neural"
FC = OUT / "forecasts"
FC_CUTS = OUT / "forecasts_cuts"
FC_FINAL = OUT / "forecasts_final"
SHOPS = ["A", "B", "C", "D"]
TARGET = "test"


def cut_frame(full, cut_index=0, cut_weeks=None):
    drop = cut_weeks if cut_weeks is not None else 13 * cut_index
    return full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=drop)]


def shop_matrix(df, cut=None, use_all=False):
    """ShopMatrix of the history before `cut` (default: the D26 split of df) plus its metric scale.

    use_all=True keeps every week of df (the pooled shops A-D, D27: all their weeks). The scale
    mirrors `features_at(M, M.W)`: mean |diff| over the full zero-padded history. It is duplicated
    here (3 lines) instead of building the 70-column feature matrix; tests/test_neural.py asserts it
    equals `features_at`'s scale."""
    if use_all:
        train = hold = None
        cut = None
    elif cut is None:
        train, hold, cut = v.split(df)
    else:
        train = v.pad_to_shop_start(df[df["week"] < cut], df[v.KEYS].drop_duplicates())
        hold = None
    M = f.ShopMatrix(df if use_all else df[df["week"] < cut])
    assert pd.Timestamp(M.weeks[0]).day_name() == "Monday", "weekly tables must be Monday-aligned (neuralforecast freq W-MON)"
    scale = np.abs(np.diff(M.U.astype(np.float64), axis=1)).mean(axis=1)
    return M, scale, scale > 0, cut, train, hold


def tagged_stem(name, tag):
    """File-stem prefix of a model config: the model plus an optional `--tag` (grid runs need distinct
    stems, otherwise the existing-file cache silently re-scores the baseline)."""
    return f"{name}{tag}"


def stem_for(name, tag):
    return tagged_stem(name, tag)


def output_files(stem, ktag, out_dir, lim=0):
    out_dir = Path(out_dir)
    lims = f"__limit{lim}" if lim else ""
    return {"parquet": out_dir / f"{stem}__test__{ktag}__padded{lims}.parquet",
            "npy_mean": out_dir / f"{stem}__{ktag}__mean{lims}.npy",
            "json": out_dir / f"{stem}__test__{ktag}__padded{lims}.json"}


def seed_files(stem, ktag, out_dir, seed, lim=0):
    out_dir = Path(out_dir)
    lims = f"__limit{lim}" if lim else ""
    return {"parquet": out_dir / f"{stem}__test__{ktag}__padded__seed{seed}{lims}.parquet",
            "npy": out_dir / f"{stem}__{ktag}__seed{seed}{lims}.npy"}


def unique_ids(keys):
    return (keys["store_id"].astype(str) + "|" + keys["article_id"].astype(str)).to_numpy()


def panel_for(M, scale, valid, limit=0):
    """Long panel [unique_id, ds, y] of one shop, y = units / scale, only scorable series."""
    idx = np.flatnonzero(valid)
    if limit:
        idx = idx[:limit]
    ids = unique_ids(M.keys)[idx]
    y = (M.U[idx].astype(np.float64) / scale[idx, None]).reshape(-1)
    df = pd.DataFrame({"unique_id": np.repeat(ids, M.W), "ds": np.tile(pd.DatetimeIndex(M.weeks), len(idx)),
                       "y": y})
    return df, idx


def build_panels(cut_index, cut_weeks, arm, limit, pool_shops=SHOPS):
    """Target-shop matrices plus (pooled) the other shops. Returns the training panel and the target rows."""
    full = v.load_weekly(TARGET)
    df = cut_frame(full, cut_index, cut_weeks)
    M, scale, valid, cut, train, hold = shop_matrix(df)
    tgt, idx = panel_for(M, scale, valid, limit)
    panels, meta = [tgt], {"shops": [TARGET], "n_series": {TARGET: int(valid.sum())}}
    if arm == "pooled":
        for shop in pool_shops:
            full_s = v.load_weekly(shop)
            Ms, ss, vs, _, _, _ = shop_matrix(full_s, use_all=True)
            ps, _ = panel_for(Ms, ss, vs, limit)
            panels.append(ps)
            meta["shops"].append(shop)
            meta["n_series"][shop] = int(vs.sum())
    panel = pd.concat(panels, ignore_index=True)
    if panel["unique_id"].nunique() != sum(p["unique_id"].nunique() for p in panels):
        raise ValueError("store_id|article_id is not unique across the pooled shops")
    return dict(panel=panel, M=M, scale=scale, valid=valid, idx=idx, cut=cut, hold=hold, train=train, meta=meta)


def final_panels(limit=0):
    """The real forecast setup: train on ALL weeks of the test shop, forecast the 13 weeks after the last."""
    full = v.load_weekly(TARGET)
    cut = full["week"].max() + pd.Timedelta(weeks=1)
    M, scale, valid, cut, train, hold = shop_matrix(full, cut=cut)
    panel, idx = panel_for(M, scale, valid, limit)
    return dict(panel=panel, M=M, scale=scale, valid=valid, idx=idx, cut=cut, hold=hold, train=train,
                meta={"shops": [TARGET], "n_series": {TARGET: int(valid.sum())}})


def make_model(name, input_size, args, seed):
    import inspect
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import NBEATS, NHITS
    cls = {"NHITS": NHITS, "NBEATS": NBEATS}[name]
    kw = dict(h=H, input_size=input_size, loss=MAE(), valid_loss=MAE(), max_steps=args.max_steps,
              learning_rate=args.lr, batch_size=args.series_batch, windows_batch_size=args.windows_batch,
              scaler_type="identity", random_seed=seed, early_stop_patience_steps=args.patience,
              val_check_steps=100, start_padding_enabled=args.start_padding,
              alias=name,
              logger=False, enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False)
    if "dropout_prob_theta" in inspect.signature(cls.__init__).parameters:
        kw["dropout_prob_theta"] = getattr(args, "dropout", 0.0)
    if args.accelerator != "auto":
        kw["accelerator"] = args.accelerator
        kw["devices"] = args.devices
    return cls(**kw)


def run_one(P, name, args, seed):
    """Fit one model+seed and return the [n x H] unit forecast of the target shop."""
    from neuralforecast import NeuralForecast
    t0 = time.time()
    longest = int(P["panel"].groupby("unique_id").size().max())
    if args.start_padding:   # the left pad makes short shops trainable, keep the requested window
        input_size = int(args.input_size)
    else:                    # neuralforecast needs max_len - val_size >= input_size for its window check
        input_size = int(min(args.input_size, longest - args.val_size))
        if input_size < 8:
            raise ValueError(f"series too short for input_size {args.input_size}; pass --start-padding")
        input_size = max(input_size, 8)
    nf = NeuralForecast(models=[make_model(name, input_size, args, seed)], freq="W-MON")
    nf.fit(df=P["panel"], val_size=args.val_size, verbose=False)
    pred = nf.predict()
    t1 = time.time()
    col = name
    weeks = pd.date_range(P["cut"], periods=H, freq="7D")
    Piv = pred.pivot_table(index="unique_id", columns="ds", values=col, aggfunc="mean")
    ids = unique_ids(P["M"].keys)
    F = np.nan_to_num(Piv.reindex(index=ids, columns=weeks).to_numpy(dtype=float), nan=0.0)
    F = np.clip(F * P["scale"][:, None], 0, None)
    F[~P["valid"]] = 0.0
    return F, {"input_size": input_size, "fit_seconds": round(t1 - t0, 1), "predict_seconds": round(time.time() - t1, 1)}


def to_long(M, F, weeks):
    return pd.DataFrame({"store_id": np.repeat(M.keys["store_id"].to_numpy(), H),
                         "article_id": np.repeat(M.keys["article_id"].to_numpy(), H),
                         "week": np.tile(np.asarray(weeks), M.n), "forecast": F.reshape(-1)})


def matrix_from_long(fc, M, weeks):
    """Long forecast frame -> [n x H] matrix, rows reindexed to ShopMatrix key order (the #1 pitfall:
    the .npy and the parquet must agree with M.keys or blend/scoring silently misaligns)."""
    P = fc.pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum")
    return P.reindex(index=pd.MultiIndex.from_frame(M.keys), columns=pd.DatetimeIndex(weeks)).fillna(0.0).to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="own", choices=["own", "pooled"])
    ap.add_argument("--cuts", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--cut-weeks", type=int, default=None, help="drop this many weeks before the cut (overlapping cuts, iteration 05); writes to forecasts_cuts/ as w{N}")
    ap.add_argument("--models", nargs="+", default=["NHITS", "NBEATS"], choices=["NHITS", "NBEATS"])
    ap.add_argument("--tag", default="", help="suffix appended to the model stem, e.g. _s3000_lr3e-4 (grid runs must not reuse the baseline files)")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout_prob_theta (NHITS/NBEATS)")
    ap.add_argument("--pool-shops", nargs="+", default=SHOPS, choices=SHOPS, help="shops added in the pooled arm (default A B C D)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--input-size", type=int, default=52)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--series-batch", type=int, default=32, help="series per training batch (neuralforecast batch_size)")
    ap.add_argument("--windows-batch", type=int, default=1024, help="windows per batch (neuralforecast windows_batch_size)")
    ap.add_argument("--patience", type=int, default=300, help="early-stopping patience in steps (-1 = off)")
    ap.add_argument("--val-size", type=int, default=H, help="last weeks of the history used for early stopping (never the holdout)")
    ap.add_argument("--start-padding", action="store_true", help="zero-pad the left of every series (needed when a shop is shorter than input_size)")
    ap.add_argument("--accelerator", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N scorable series per shop (smoke runs); output gets a __limitN suffix")
    ap.add_argument("--final", action="store_true", help="train on all weeks of the test shop and forecast the 13 weeks after the last; writes forecasts_final/")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if a.final:
        a.arm = "own"
        out_dir, tag, cuts = FC_FINAL, "final", [None]
    elif a.cut_weeks is not None:
        out_dir, tag, cuts = FC_CUTS, f"w{a.cut_weeks}", [None]
    else:
        out_dir, tag, cuts = FC, None, a.cuts
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in cuts:
        t0 = time.time()
        P = final_panels(a.limit) if a.final else build_panels(0 if k is None else k, a.cut_weeks, a.arm, a.limit, a.pool_shops)
        ktag = tag or f"cut{k}"
        print(f"{ktag} ({a.arm}): {P['meta']['n_series']} scorable series, panel {len(P['panel']):,} rows, "
              f"{P['panel']['unique_id'].nunique():,} series in training ({time.time()-t0:.0f}s)", flush=True)
        weeks = pd.date_range(P["cut"], periods=H, freq="7D")
        for name in a.models:
            stem = tagged_stem(name, a.tag)
            Fs, info = [], []
            lim = f"__limit{a.limit}" if a.limit else ""
            files = output_files(stem, ktag, out_dir, a.limit)
            for s in a.seeds:
                sf = seed_files(stem, ktag, out_dir, s, a.limit)
                fp = sf["parquet"]
                if fp.exists():
                    fc = pd.read_parquet(fp)
                    F = matrix_from_long(fc, P["M"], weeks)
                else:
                    F, inf = run_one(P, name, a, s)
                    to_long(P["M"], F, weeks).to_parquet(fp, index=False)
                    np.save(sf["npy"], F)
                    info.append({**inf, "seed": s})
                    print(f"  {stem} seed {s}: input_size {inf['input_size']}, fit {inf['fit_seconds']}s, "
                          f"predict {inf['predict_seconds']}s", flush=True)
                Fs.append(F)
            Fm = np.mean(Fs, axis=0)
            to_long(P["M"], Fm, weeks).to_parquet(files["parquet"], index=False)
            np.save(files["npy_mean"], Fm)
            import neuralforecast, torch
            jp = files["json"]
            if info or not jp.exists():
                jp.write_text(json.dumps({
                    "model": stem, "base_model": name, "tag": a.tag, "arm": a.arm, "cut_index": k,
                    "cut_weeks": a.cut_weeks, "cut": str(P["cut"].date()),
                    "seeds": a.seeds, "input_size": a.input_size, "max_steps": a.max_steps, "lr": a.lr,
                    "dropout": a.dropout, "pool_shops": a.pool_shops if a.arm == "pooled" else None,
                    "patience": a.patience, "start_padding": a.start_padding, "limit": a.limit,
                    "panels": P["meta"], "runs": info,
                    "neuralforecast": neuralforecast.__version__, "torch": torch.__version__,
                    "seconds": round(time.time() - t0, 1)}))
            print(f"  {stem} {ktag}: seed-average written ({len(a.seeds)} seeds){lim}", flush=True)


if __name__ == "__main__":
    main()
