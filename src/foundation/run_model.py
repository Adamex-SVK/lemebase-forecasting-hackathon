"""Run one foundation model zero-shot on one shop at one cut and save the forecasts.

    python src/foundation/run_model.py MODEL SHOP [--cut-index K] [--context padded|first_sale|last52]
                                       [--device cuda] [--limit N]

Needs only pandas / numpy / pyarrow plus the model's own package (so it can run in the model's
own environment). Scoring is done by score_all.py in the main environment.

 * history = the shop's weekly table before the cut, every series zero-padded from the shop's first
   week (D26) when --context padded (default), starting at its first sale (first_sale), or truncated
   to the last 52 weeks (last52);
 * cut K drops the last 13*K weeks first (rolling origin, D32); K = 0 is the standard local cut (D14);
 * series whose context is all zeros are not sent to the model: forecast 0;
 * output: iterations/03-foundation-model/forecasts/{model}__{shop}__cut{K}__{context}.parquet
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
import validate as v  # noqa: E402
import models as fm  # noqa: E402

OUT = ROOT / "iterations" / "03-foundation-model" / "forecasts"
H = v.HORIZON


def contexts_for(shop, cut_index, context, final=False, cut_weeks=None):
    full = v.load_weekly(shop)
    if final:   # real forecast: ALL history is the context, the 13 weeks after the last week are forecast
        train = v.pad_to_shop_start(full, full[v.KEYS].drop_duplicates())
        hold, cut = None, full["week"].max() + pd.Timedelta(weeks=1)
    else:
        drop = cut_weeks if cut_weeks is not None else 13 * cut_index   # weeks removed from the end of the history
        df = full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=drop)]
        train, hold, cut = v.split(df)
    W = train.pivot_table(index=v.KEYS, columns="week", values="units", aggfunc="sum", fill_value=0.0).sort_index(axis=1)
    A = W.to_numpy(dtype="float64")
    ctx, active = [], []
    for k in range(len(A)):
        row = A[k]
        if context == "first_sale" and (row > 0).any():
            row = row[(row > 0).argmax():]
        elif context == "last52":
            row = row[-52:]
        ctx.append(row)
        active.append(bool((row > 0).any()))
    return W.index, ctx, np.array(active), hold, cut, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("shop")
    ap.add_argument("--cut-index", type=int, default=0)
    ap.add_argument("--context", default="padded", choices=["padded", "first_sale", "last52"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="only the first N active series (smoke runs)")
    ap.add_argument("--cut-weeks", type=int, default=None, help="drop this many weeks from the end before the cut (any offset, for overlapping cuts); output goes to forecasts_cuts/ as ..__w{N}__..")
    ap.add_argument("--final", action="store_true", help="forecast the real future weeks from the full history (no holdout); output goes to forecasts_final/")
    a = ap.parse_args()
    t0 = time.time()
    index, ctx, active, hold, cut, train = contexts_for(a.shop, a.cut_index, a.context, a.final, a.cut_weeks)
    end_week = cut - pd.Timedelta(weeks=1)
    idx = np.flatnonzero(active)
    if a.limit:
        idx = idx[:a.limit]
    handle = fm.load(a.model, a.device)
    t1 = time.time()
    F = np.zeros((len(index), H))
    if len(idx):
        F[idx] = fm.forecast(a.model, handle, [ctx[i] for i in idx], H, end_week)
    t2 = time.time()
    weeks = pd.date_range(cut, periods=H, freq="7D")
    fc = pd.DataFrame({
        "store_id": np.repeat(index.get_level_values(0).to_numpy(), H),
        "article_id": np.repeat(index.get_level_values(1).to_numpy(), H),
        "week": np.tile(weeks, len(index)), "forecast": F.reshape(-1)})
    out_dir = OUT.parent / "forecasts_final" if a.final else (OUT.parent / "forecasts_cuts" if a.cut_weeks is not None else OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.model}__{a.shop}__" + ("final" if a.final else (f"w{a.cut_weeks}" if a.cut_weeks is not None else f"cut{a.cut_index}")) + f"__{a.context}" + (f"__limit{a.limit}" if a.limit else "")
    fc.to_parquet(out_dir / f"{stem}.parquet", index=False)
    (out_dir / f"{stem}.json").write_text(json.dumps({"model": a.model, "shop": a.shop, "cut_index": a.cut_index, "context": a.context,
                                                   "n_series": len(index), "n_sent_to_model": int(len(idx)),
                                                   "load_seconds": round(t1 - t0, 1), "forecast_seconds": round(t2 - t1, 1)}))
    print(f"{stem}: {len(idx):,} of {len(index):,} series sent to the model; load {t1-t0:.0f}s, forecast {t2-t1:.0f}s")


if __name__ == "__main__":
    main()
