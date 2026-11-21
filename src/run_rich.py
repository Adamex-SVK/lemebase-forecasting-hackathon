"""Iteration 04 (step 4.2/4.3): reference and rich-column arms on the 4 rolling test-shop cuts.

Arms: own, pooled_norich, pooled_rich_nodrop, pooled_rich_drop50, pooled_rich_drop80.
Cut k drops the last 13*k weeks of the test shop (same convention as src/rolling.py).
Pooled arms: test-shop windows (cut-truncated, origin step 2) + donor shops A, B, C, D (all weeks, origin
step 4, sample_inactive for D, shop weights 'equal'), fixed 300 trees. Rich arms use rich features; the rich
dropout share applies to the DONOR windows only (test-shop windows have the rich columns NaN by construction).
Seed s sets the LightGBM seed and the window rng (sampling of inactive D series, dropout masks).

Run: .venv/bin/python src/run_rich.py ARM[,ARM...] --seeds 0,1 [--cuts 0,1,2,3] [--topic NAME] [--smoke]
Results: iterations/04-rich-columns/results_adamex-svk_<topic>.csv (rows for the same arm/cut/seed are replaced).
"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as f  # noqa: E402
import rich as rich_mod  # noqa: E402
import run_arms as ra  # noqa: E402
import train_gbm as g  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
OUT = v.ROOT / "iterations" / "04-rich-columns"
SHOP = "test"
DONORS = ["A", "B", "C", "D"]
N_CUTS = 4
POOLED_TREES = ra.FALLBACK_ROUNDS["pooled"]   # 300
# arm -> (rich features, rich dropout share on donor windows)
ARM_SPEC = {"own": None, "pooled_norich": (False, 0.0), "pooled_rich_nodrop": (True, 0.0),
            "pooled_rich_drop50": (True, 0.5), "pooled_rich_drop80": (True, 0.8)}
COLS = ["arm", "cut_index", "seed", "score", "n_scored", "trees", "seconds"]


def parse_arms(s):
    arms = [a for a in s.split(",") if a]
    bad = [a for a in arms if a not in ARM_SPEC]
    if bad:
        raise ValueError(f"unknown arm(s) {bad}; choose from {list(ARM_SPEC)}")
    return arms


def truncate(full, k):
    """Cut k: drop the last 13*k weeks (identical to rolling.py)."""
    return full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=13 * k)]


def donor_windows(rich, rich_drop, seed, step):
    parts = []
    for s in DONORS:
        df = v.load_weekly(s)
        M = f.ShopMatrix(df, rich=rich_mod.load_rich(s) if rich else None)
        w = f.build_windows(M, f.make_origins(M.W, step=step), sample_inactive=(s == "D"),
                            rng=np.random.default_rng(seed), rich=rich, rich_drop=rich_drop)
        parts.append((s, w))
        del M
    return parts


def own_cut(df, seed):
    old = g.PARAMS["seed"]
    g.PARAMS["seed"] = seed          # fit() uses this dict as its default params
    try:
        res = g.own_arm(SHOP, df=df)
    finally:
        g.PARAMS["seed"] = old
    fc = g.complete(g.to_long(res["M"], res["F"], res["weeks"]), res["hold"])
    r = v.score(res["train"], res["hold"], fc)
    return r, int(res["best_iter"] * 1.1) + 1


def pooled_cut(df, seed, parts, rich, trees, step_test=2):
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=f.H, freq="7D")
    w_x = f.build_windows(M, f.make_origins(M.W, step=step_test), rng=np.random.default_rng(seed), rich=rich)
    full = ra.merge([(SHOP, w_x)] + parts, "equal")
    del w_x
    names = f.feature_names(rich)
    model, _ = g.fit(full, None, rounds=trees, params={**g.PARAMS, "seed": seed}, names=names)
    del full
    F = g.forecast_matrix(model, M, weeks, rich=rich)
    fc = g.complete(g.to_long(M, F, weeks), hold)
    r = v.score(train, hold, fc)
    del model
    return r, trees


def run(arms, seeds, cuts=range(N_CUTS), topic=None, smoke=False):
    topic = topic or "+".join(arms)
    path = OUT / f"results_adamex-svk_{topic}.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    full = v.load_weekly(SHOP)
    step = 26 if smoke else ra.OTHER_STEP
    trees = 20 if smoke else POOLED_TREES
    for arm in arms:
        for seed in seeds:
            parts, spec = None, ARM_SPEC[arm]
            if spec is not None:
                t0 = time.time()
                parts = donor_windows(spec[0], spec[1], seed, step)
                print(f"{arm} seed {seed}: donor windows " + ", ".join(f"{s} {len(w['y']):,}" for s, w in parts)
                      + f" rows ({time.time()-t0:.0f}s)", flush=True)
            for k in cuts:
                t0 = time.time()
                df = truncate(full, k)
                r, n_trees = own_cut(df, seed) if spec is None else pooled_cut(df, seed, parts, spec[0], trees)
                row = {"arm": arm, "cut_index": k, "seed": seed, "score": r["score"], "n_scored": r["n_scored"],
                       "trees": n_trees, "seconds": round(time.time() - t0)}
                rows = [x for x in rows if (x["arm"], x["cut_index"], x["seed"]) != (arm, k, seed)] + [row]
                OUT.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)
                print(f"{arm:20s} cut {k} seed {seed} score {r['score']:8.2f} scored {r['n_scored']} trees {n_trees} {row['seconds']}s", flush=True)
                gc.collect()
            del parts
            gc.collect()
    out = pd.DataFrame(rows, columns=COLS)
    return out[out["arm"].isin(arms) & out["seed"].isin(list(seeds)) & out["cut_index"].isin(list(cuts))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--cuts", default="0,1,2,3")
    ap.add_argument("--topic", default=None)
    ap.add_argument("--smoke", action="store_true", help="20 trees, donor origin step 26 (structure check only)")
    a = ap.parse_args()
    arms = parse_arms(a.arms)
    ints = lambda s: [int(x) for x in s.split(",") if x != ""]  # noqa: E731
    out = run(arms, ints(a.seeds), ints(a.cuts), a.topic, a.smoke)
    print(out.groupby("arm")["score"].agg(["mean", "count"]).round(2).to_string())


if __name__ == "__main__":
    main()
