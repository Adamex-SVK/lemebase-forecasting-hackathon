"""Step 1.3: all shops x training arms (D16, D27, D30).

Arms for each target shop X (always scored on X's local cut, D14, D26):
  own          X's training weeks only
  pooled       X's training weeks + ALL weeks of the other four shops (D27)
               weighting 'equal': every shop contributes the same total weight (D30 rec.);
               weighting 'unweighted': every row weight 1 (D dominates)
  others_only  the other four shops only (zero-shot transfer to X); 'equal' weights

Early stopping is used where X has enough windows for a validation block (A, C, test);
B and D have too little history (2 and 4 window origins): fixed number of trees
(own 150, pooled / others_only 300). With early stopping the final fit reuses 1.1 x best
iteration on all of X's windows.

Run: .venv/bin/python src/run_arms.py [shop ...]     (results saved after every arm)
"""
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as f  # noqa: E402
import train_gbm as g  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
SHOPS = ["test", "A", "C", "B", "D"]
OUT = v.ROOT / "iterations" / "01-global-gbm"
FC_DIR = OUT / "forecasts"
H = f.H
FALLBACK_ROUNDS = {"own": 150, "pooled": 300, "others_only": 300}
OTHER_STEP = 4   # D31 (b): window origins every 4 weeks for the other shops (the target shop's own windows keep step 2)
ARMS = [("own", "n/a"), ("pooled", "equal"), ("pooled", "unweighted"), ("others_only", "equal")]


def merge(parts, weighting):
    """Concatenate window dicts; 'equal' gives every shop the same total weight."""
    n = sum(len(p["y"]) for _, p in parts)
    k = len(parts)
    Xs, ys, ws = [], [], []
    for _, p in parts:
        mult = n / (k * len(p["y"])) if weighting == "equal" else 1.0
        Xs.append(p["X"]); ys.append(p["y"]); ws.append(p["w"] * np.float32(mult))
    return {"X": np.vstack(Xs), "y": np.concatenate(ys), "w": np.concatenate(ws)}


def main(targets):
    FC_DIR.mkdir(exist_ok=True)
    data, cache = {}, {}
    t0 = time.time()
    for s in SHOPS:
        df = v.load_weekly(s)
        train, hold, cut = v.split(df)
        data[s] = {"df": df, "train": train, "hold": hold, "cut": cut}
        M = f.ShopMatrix(df)
        # all weeks (D27, D28); D31: origins every OTHER_STEP weeks for shops used as "other" data
        cache[s] = f.build_windows(M, f.make_origins(M.W, step=OTHER_STEP), sample_inactive=(s == "D"))
        del M
    print(f"windows over all weeks built: " + ", ".join(f"{s} {len(cache[s]['y']):,}" for s in SHOPS) + f" rows ({time.time()-t0:.0f}s)", flush=True)
    results_path = OUT / "results_1_3.csv"
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() and len(targets) < len(SHOPS) else []

    for X in targets:
        d = data[X]
        MX = f.ShopMatrix(d["df"][d["df"]["week"] < d["cut"]])
        weeks = pd.date_range(d["cut"], periods=H, freq="7D")
        origins = f.make_origins(MX.W)
        val_o = origins[-7:]
        tr_o = origins[origins <= val_o[0] - H]
        es = len(tr_o) >= 10
        samp = X == "D"
        w_all_X = f.build_windows(MX, origins, sample_inactive=samp)
        w_tr_X = f.build_windows(MX, tr_o, sample_inactive=samp) if es else None
        w_va_X = f.build_windows(MX, val_o, sample_inactive=samp) if es else None
        others = [s for s in SHOPS if s != X]
        for arm, weighting in ARMS:
            ta = time.time()
            other_parts = [(s, cache[s]) for s in others] if arm in ("pooled", "others_only") else []
            if arm == "others_only":
                full = merge(other_parts, weighting)
                model, _ = g.fit(full, None, rounds=FALLBACK_ROUNDS[arm]); best, n_trees = None, FALLBACK_ROUNDS[arm]
            elif es:
                m1, best = g.fit(merge([("X", w_tr_X)] + other_parts, weighting), w_va_X)
                n_trees = int(best * 1.1) + 1
                model, _ = g.fit(merge([("X", w_all_X)] + other_parts, weighting), None, rounds=n_trees)
            else:
                best, n_trees = None, FALLBACK_ROUNDS[arm]
                model, _ = g.fit(merge([("X", w_all_X)] + other_parts, weighting), None, rounds=n_trees)
            F = g.forecast_matrix(model, MX, weeks)
            fc = g.complete(g.to_long(MX, F, weeks), d["hold"])
            r = v.score(d["train"], d["hold"], fc)
            n_rows = (len(w_all_X["y"]) if arm != "others_only" else 0) + sum(len(p["y"]) for _, p in other_parts)
            row = {"iteration": "01-global-gbm", "step": "1.3", "convention": "D26", "shop": X, "model": "lgbm_l1",
                   "training_data": arm, "weighting": weighting, **r, "best_iter": best, "trees": n_trees,
                   "train_rows": n_rows, "other_origin_step": OTHER_STEP, "seconds": round(time.time() - ta)}
            rows.append(row)
            fc.to_parquet(FC_DIR / f"{X}_{arm}_{weighting.replace('/', '')}.parquet", index=False)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            print(f"{X:4s} {arm:12s} {weighting:10s} score {r['score']:8.2f}  scored {r['n_scored']:6d}  trees {n_trees:4d}  rows {n_rows:>10,}  {time.time()-ta:5.0f}s", flush=True)
            del model; gc.collect()
        del w_all_X, w_tr_X, w_va_X; gc.collect()
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or SHOPS)
