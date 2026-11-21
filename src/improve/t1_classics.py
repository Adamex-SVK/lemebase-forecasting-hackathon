"""T1: intermittent-demand classics (statsforecast) as pseudo-models for the blend.

For every standard cut K = 0..3 the history is cut_frame(full, K) -> validate.split (D26, zero padded).
For --final the whole history is padded with validate.pad_to_shop_start. Each classic is fit with
statsforecast on the active series only (all-zero history -> forecast 0) and saved in the standard
pseudo-model layout:

    iterations/03-foundation-model/forecasts/<name>__test__cut{K}__padded.parquet
    iterations/03-foundation-model/forecasts_final/<name>__test__final__padded.parquet

Run: .venv/bin/python src/improve/t1_classics.py [--cuts 0,1,2,3] [--no-final]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    ADIDA, IMAPA, TSB, CrostonClassic, CrostonOptimized, CrostonSBA,
    SimpleExponentialSmoothingOptimized,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import experiments_02 as e  # noqa: E402
import validate as v  # noqa: E402

H = v.HORIZON
FM = ROOT / "iterations" / "03-foundation-model" / "forecasts"
FM_FINAL = ROOT / "iterations" / "03-foundation-model" / "forecasts_final"
FREQ = "W-MON"

MODELS = [
    CrostonClassic(alias="sf-croston"),
    CrostonSBA(alias="sf-croston-sba"),
    CrostonOptimized(alias="sf-croston-opt"),
    TSB(alpha_d=0.1, alpha_p=0.1, alias="sf-tsb"),
    ADIDA(alias="sf-adida"),
    IMAPA(alias="sf-imapa"),
    SimpleExponentialSmoothingOptimized(alias="sf-ses"),
]
NAMES = [m.alias for m in MODELS]


def long_history(train):
    """Padded wide history -> statsforecast long frame (unique_id, ds, y)."""
    W = train.pivot_table(index=v.KEYS, columns="week", values="units",
                          aggfunc="sum", fill_value=0.0).sort_index(axis=1)
    W = W.sort_index(axis=1)
    A = W.to_numpy(dtype="float64")
    active = (A > 0).any(axis=1)
    ids = W.index.get_level_values("article_id").to_numpy()
    long = pd.DataFrame({
        "unique_id": np.repeat(ids, A.shape[1]),
        "ds": np.tile(np.asarray(W.columns, dtype="datetime64[ns]"), len(ids)),
        "y": A.reshape(-1),
    })
    return W.index, A, active, long, np.asarray(W.columns, dtype="datetime64[ns]")


def fit_predict(long, active_ids, last_week):
    """Fit all classics on the active series, forecast H weeks after last_week."""
    t0 = time.time()
    sf = StatsForecast(models=MODELS, freq=FREQ, n_jobs=-1)
    out = sf.forecast(df=long, h=H)
    dt = time.time() - t0
    out = out.rename(columns={"unique_id": "article_id"})
    # statsforecast may return fewer rows for degenerate series; reindex defensively
    weeks = pd.date_range(last_week + pd.Timedelta(weeks=1), periods=H, freq="7D")
    return out, weeks, dt


def forecasts_for(train, cut_kind):
    index, A, active, long, weeks_hist = long_history(train)
    last_week = weeks_hist.max()
    active_ids = index.get_level_values("article_id").to_numpy()[active]
    if len(active_ids) == 0:
        raise SystemExit("no active series")
    long = long[long["unique_id"].isin(set(active_ids))]
    raw, weeks, dt = fit_predict(long, active_ids, last_week)
    n = len(index)
    ids = index.get_level_values("article_id").to_numpy()
    res = {}
    for name in NAMES:
        if name not in raw.columns:
            raise SystemExit(f"{name} missing from statsforecast output: {list(raw.columns)}")
        P = raw.pivot_table(index="article_id", columns="ds", values=name, aggfunc="mean")
        P = P.reindex(index=ids, columns=weeks)
        F = P.to_numpy(dtype="float64")
        F = np.where(np.isfinite(F), F, 0.0)
        F = np.clip(F, 0, None)
        F[~active] = 0.0
        fc = pd.DataFrame({
            "store_id": np.repeat(index.get_level_values(0).to_numpy(), H),
            "article_id": np.repeat(ids, H),
            "week": np.tile(np.asarray(weeks, dtype="datetime64[ns]"), n),
            "forecast": F.reshape(-1),
        })
        assert len(fc) == n * H, (name, len(fc), n * H)
        res[name] = fc
    return res, int(active.sum()), n, dt


def write(fc, name, shop, tag, extra):
    out_dir = FM_FINAL if tag == "final" else FM
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}__{shop}__{tag}__padded"
    fc.to_parquet(out_dir / f"{stem}.parquet", index=False)
    (out_dir / f"{stem}.json").write_text(json.dumps(extra))


def blend_sweep():
    """Extra: does a *small* weight for a classic help? Honest leave-one-cut-out weight.

    w is the weight of the classic inside the foundation ensemble (D40 uses equal weights,
    which is w = 1/4 for one extra member). For each cut K, w is chosen on the other 3 cuts.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("ensemble_recipe", ROOT / "src" / "meta" / "ensemble_recipe.py")
    er = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(er)
    ctx, cache = er.cut_contexts(), {}
    base = er.REFERENCE_MODELS
    WS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    variants = {
        "sf-adida": ["sf-adida"],
        "sf-croston-sba": ["sf-croston-sba"],
        "sf-adida+sf-croston-sba": ["sf-adida", "sf-croston-sba"],
    }
    rows = []
    for label, members in variants.items():
        sc = {w: [] for w in WS}
        for k in range(er.N_CUTS):
            M, _, _, _, dead, Fl = ctx[k]
            B = np.mean([er.raw_practice(m, k, ctx, cache) for m in base], axis=0)
            C = np.mean([er.raw_practice(m, k, ctx, cache) for m in members], axis=0)
            for w in WS:
                E = (1 - w) * B + w * C
                E[dead] = 0.0
                sc[w].append(er.score_cut(k, er.REFERENCE_W * Fl + (1 - er.REFERENCE_W) * E, ctx))
        for w in WS:
            rows.append({"variant": f"D40 + weight {w:.2f} on {label}", "scores": sc[w],
                         "mean": float(np.mean(sc[w])),
                         "gain_vs_D40": float(np.mean(sc[w]) - np.mean(sc[0.0]))})
        # leave-one-cut-out weight
        honest = []
        for k in range(er.N_CUTS):
            other = [j for j in range(er.N_CUTS) if j != k]
            w = max(WS, key=lambda x: np.mean([sc[x][j] for j in other]))
            honest.append(sc[w][k])
        rows.append({"variant": f"D40 + LOCO weight on {label}", "scores": honest,
                     "mean": float(np.mean(honest)),
                     "gain_vs_D40": float(np.mean(honest) - np.mean(sc[0.0]))})
        best = max(WS, key=lambda x: np.mean(sc[x]))
        rows.append({"variant": f"D40 + in-sample-best weight {best:.2f} on {label} (labelled)", "scores": sc[best],
                     "mean": float(np.mean(sc[best])),
                     "gain_vs_D40": float(np.mean(sc[best]) - np.mean(sc[0.0]))})
    out = pd.DataFrame(rows)
    out["cut0"] = out["scores"].apply(lambda x: x[0])
    out["cut1"] = out["scores"].apply(lambda x: x[1])
    out["cut2"] = out["scores"].apply(lambda x: x[2])
    out["cut3"] = out["scores"].apply(lambda x: x[3])
    out = out[["variant", "cut0", "cut1", "cut2", "cut3", "mean", "gain_vs_D40"]]
    dest = ROOT / "iterations" / "08-improvements" / "t1" / "blend_sweep.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    pd.set_option("display.width", 220)
    print(out.round(2).to_string(index=False))
    print(f"\nwrote {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shop", default="test")
    ap.add_argument("--cuts", default="0,1,2,3")
    ap.add_argument("--no-final", action="store_true", help="do not write the real-future forecasts")
    ap.add_argument("--limit", type=int, default=0, help="only the first N active series (smoke test)")
    ap.add_argument("--blend-sweep", action="store_true", help="run the small-weight blend sweep and exit")
    a = ap.parse_args()

    if a.blend_sweep:
        blend_sweep()
        return

    full = v.load_weekly(a.shop)
    cuts = [int(x) for x in a.cuts.split(",") if x.strip() != ""]
    for k in cuts:
        df = e.cut_frame(full, k)
        train, hold, cut = v.split(df)
        t0 = time.time()
        res, n_active, n_series, dt = forecasts_for(train, f"cut{k}")
        if a.limit:
            for name in res:
                res[name] = res[name][res[name]["article_id"].isin(res[name]["article_id"].unique()[:a.limit])]
        for name in NAMES:
            write(res[name], name, a.shop, f"cut{k}", {
                "model": name, "shop": a.shop, "cut_index": k, "context": "padded",
                "n_series": n_series, "n_sent_to_model": n_active,
                "load_seconds": 0.0, "forecast_seconds": round(dt, 1),
            })
        print(f"cut {k}: {n_active:,} of {n_series:,} active, {dt:.1f}s for all 7 models "
              f"({time.time()-t0:.1f}s total)", flush=True)

    if not a.no_final:
        train = v.pad_to_shop_start(full, full[v.KEYS].drop_duplicates())
        t0 = time.time()
        res, n_active, n_series, dt = forecasts_for(train, "final")
        if a.limit:
            for name in res:
                res[name] = res[name][res[name]["article_id"].isin(res[name]["article_id"].unique()[:a.limit])]
        for name in NAMES:
            write(res[name], name, a.shop, "final", {
                "model": name, "shop": a.shop, "cut_index": 0, "context": "padded",
                "n_series": n_series, "n_sent_to_model": n_active,
                "load_seconds": 0.0, "forecast_seconds": round(dt, 1),
            })
        print(f"final: {n_active:,} of {n_series:,} active, {dt:.1f}s for all 7 models "
              f"({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
