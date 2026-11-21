"""Score every saved foundation-model forecast with the D26 metric (main environment, pandas only).

    .venv/bin/python src/foundation/score_all.py

Reads iterations/03-foundation-model/forecasts/*.parquet (written by run_model.py), scores each on the
cut it was made for, and writes iterations/03-foundation-model/results_zero_shot.csv.
Also prints the median-of-last-13-weeks reference for the same cut.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import validate as v  # noqa: E402

DIR = ROOT / "iterations" / "03-foundation-model"
H = v.HORIZON


def median13_reference(train, hold):
    W = train.pivot_table(index=v.KEYS, columns="week", values="units", aggfunc="sum", fill_value=0.0).sort_index(axis=1)
    med = np.median(W.to_numpy()[:, -13:], axis=1)
    weeks = sorted(hold["week"].unique())
    fc = pd.DataFrame({"store_id": np.repeat(W.index.get_level_values(0), H), "article_id": np.repeat(W.index.get_level_values(1), H),
                       "week": np.tile(weeks, len(W)), "forecast": np.repeat(med, H)})
    return fc


def main():
    rows, cache = [], {}
    for p in sorted((DIR / "forecasts").glob("*.parquet")):
        model, shop, cut, ctxm, *rest = p.stem.split("__")
        k = int(cut.replace("cut", ""))
        key = (shop, k)
        if key not in cache:
            full = v.load_weekly(shop)
            df = full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=13 * k)]
            train, hold, _ = v.split(df)
            ref = median13_reference(train, hold)
            ref = hold[v.KEYS + ["week"]].merge(ref, on=v.KEYS + ["week"], how="left").fillna({"forecast": 0.0})
            cache[key] = (train, hold, v.score(train, hold, ref)["score"])
        train, hold, ref_score = cache[key]
        fc = pd.read_parquet(p)
        fc = hold[v.KEYS + ["week"]].merge(fc, on=v.KEYS + ["week"], how="left").fillna({"forecast": 0.0})
        r = v.score(train, hold, fc)
        rows.append({"model": model, "shop": shop, "cut_index": k, "context": ctxm, "variant": "_".join(rest) or "full",
                     "score": round(r["score"], 2), "median13_reference": round(ref_score, 2),
                     "n_scored": r["n_scored"], "n_series": r["n_series"]})
    res = pd.DataFrame(rows)
    if res.empty:
        print("no forecasts yet"); return
    res.to_csv(DIR / "results_zero_shot.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.sort_values(["shop", "cut_index", "score"], ascending=[True, True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
