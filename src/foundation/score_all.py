"""Score every saved foundation-model forecast with the D26 metric (pandas only, runs on the GPU machine too).

    python src/foundation/score_all.py

Reads iterations/03-foundation-model/forecasts/*.parquet (written by run_model.py) and scores each on the
cut it was made for. Besides the raw forecast it scores cheap post-processing variants, because the metric
punishes small non-zero forecasts for articles whose true level is 0 (D20):
    raw        the model's median forecast, clipped at 0
    round      rounded to whole units (counts have integer medians)
    zero8      0 for articles with no sale in the last 8 training weeks (the baselines' rule)
    round+zero8
Writes results_zero_shot.csv and prints per-cut scores and the mean over the cuts, next to the median-of-13-weeks reference.
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
KEYS = v.KEYS


def prepare_cut(shop, k):
    full = v.load_weekly(shop)
    df = full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=13 * k)]
    train, hold, cut = v.split(df)
    W = train.pivot_table(index=KEYS, columns="week", values="units", aggfunc="sum", fill_value=0.0).sort_index(axis=1)
    dead = W.iloc[:, -8:].sum(axis=1) == 0
    dead_keys = set(map(tuple, dead[dead].index.to_frame(index=False).to_numpy()))
    med = pd.DataFrame({"store_id": np.repeat(W.index.get_level_values(0), H), "article_id": np.repeat(W.index.get_level_values(1), H),
                        "week": np.tile(sorted(hold["week"].unique()), len(W)),
                        "forecast": np.repeat(np.median(W.to_numpy()[:, -13:], axis=1), H)})
    return train, hold, dead_keys, med


def variant(m, dead_keys, name):
    m = m.copy()
    if "round" in name:
        m["forecast"] = np.round(m["forecast"])
    if "zero8" in name:
        isdead = pd.Series(list(zip(m["store_id"], m["article_id"]))).isin(dead_keys).to_numpy()
        m.loc[isdead, "forecast"] = 0.0
    return m


def main():
    rows, cache = [], {}
    for p in sorted((DIR / "forecasts").glob("*.parquet")):
        model, shop, cut, ctxm, *rest = p.stem.split("__")
        k = int(cut.replace("cut", ""))
        if (shop, k) not in cache:
            train, hold, dead_keys, med = prepare_cut(shop, k)
            ref = hold[KEYS + ["week"]].merge(med, on=KEYS + ["week"], how="left").fillna({"forecast": 0.0})
            cache[(shop, k)] = (train, hold, dead_keys, v.score(train, hold, ref)["score"])
        train, hold, dead_keys, ref_score = cache[(shop, k)]
        fc = hold[KEYS + ["week"]].merge(pd.read_parquet(p), on=KEYS + ["week"], how="left").fillna({"forecast": 0.0})
        for name in ["raw", "round", "zero8", "round+zero8"]:
            r = v.score(train, hold, variant(fc, dead_keys, name))
            rows.append({"model": model, "shop": shop, "cut_index": k, "context": ctxm, "post": name, "limit": "_".join(rest) or "full",
                         "score": round(r["score"], 2), "median13_reference": round(ref_score, 2), "n_scored": r["n_scored"]})
    res = pd.DataFrame(rows)
    if res.empty:
        print("no forecasts yet"); return
    res.to_csv(DIR / "results_zero_shot.csv", index=False)
    pd.set_option("display.width", 220)
    full = res[res["limit"] == "full"]
    for shop, g in full.groupby("shop"):
        pt = g.pivot_table(index=["model", "post"], columns="cut_index", values="score")
        pt["mean"] = pt.mean(axis=1)
        pt["cuts"] = g.pivot_table(index=["model", "post"], columns="cut_index", values="score").notna().sum(axis=1)
        ref = g.groupby("cut_index")["median13_reference"].first()
        print(f"\n=== shop {shop}: score by cut (median13 reference: " + ", ".join(f"cut{c} {x}" for c, x in ref.items()) + f"; mean {ref.mean():.2f}) ===")
        print(pt.sort_values("mean", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
