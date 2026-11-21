"""Iteration 02 quick experiments on the test shop, scored on the 4 rolling cuts (D32, D26):
  base      LightGBM own arm as in step 1.2
  d33       + weeks-of-cover features and the old on_hand regime (before 2021-04-26) set to missing (D33)
  blends    w * LightGBM + (1 - w) * median of the last 13 weeks; w is chosen on cuts 1-3 and checked on cut 0
  seeds     (phase 2) the better variant with 3 seeds, averaged

Run: .venv/bin/python src/experiments_02.py 1      then      .venv/bin/python src/experiments_02.py 2
Forecasts are saved in iterations/02-robust-eval/forecasts/, results in results_experiments.csv.
"""
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
OUT = v.ROOT / "iterations" / "02-robust-eval"
FC = OUT / "forecasts"
H = f.H
MASK_DATE = "2021-04-26"
N_CUTS = 4
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)


def cut_frame(full, k):
    return full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=13 * k)]


def run_lgbm(full, k, d33, seed):
    """Own-arm LightGBM for cut k. Returns (keys, F[n x 13], train, hold, weeks, median13 F)."""
    df = cut_frame(full, k)
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut], mask_oh_before=MASK_DATE if d33 else None)
    weeks = pd.date_range(cut, periods=H, freq="7D")
    origins = f.make_origins(M.W)
    val_o = origins[-7:]
    tr_o = origins[origins <= val_o[0] - H]
    params = dict(g.PARAMS, seed=seed)
    w_tr = f.build_windows(M, tr_o, d33=d33)
    w_va = f.build_windows(M, val_o, d33=d33)
    names = f.FEATURES_D33 if d33 else f.FEATURES
    dtr_names = names
    import lightgbm as lgb
    dtr = lgb.Dataset(w_tr["X"], w_tr["y"], weight=w_tr["w"], feature_name=dtr_names)
    dva = lgb.Dataset(w_va["X"], w_va["y"], weight=w_va["w"], reference=dtr)
    m1 = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    best = m1.best_iteration
    w_all = f.build_windows(M, origins, d33=d33)
    m2 = lgb.train(params, lgb.Dataset(w_all["X"], w_all["y"], weight=w_all["w"], feature_name=dtr_names), num_boost_round=int(best * 1.1) + 1)
    X, scale, valid = f.inference_features(M, d33=d33)
    F = np.clip(m2.predict(X).reshape(M.n, H) * scale[:, None], 0, None)
    F[~valid] = 0.0
    Fmed = np.repeat(np.median(M.U[:, -13:], axis=1, keepdims=True), H, axis=1).astype(float)
    return M, F, Fmed, train, hold, weeks, best


def score_F(M, F, train, hold, weeks):
    fc = pd.DataFrame({"store_id": np.repeat(M.keys["store_id"].to_numpy(), H), "article_id": np.repeat(M.keys["article_id"].to_numpy(), H),
                       "week": np.tile(np.asarray(weeks), M.n), "forecast": F.reshape(-1)})
    fc = g.complete(fc, hold)
    return v.score(train, hold, fc, detail=True)


def phase1():
    FC.mkdir(exist_ok=True)
    full = v.load_weekly("test")
    store, rows = {}, []
    for k in range(N_CUTS):
        for name, d33 in [("base", False), ("d33", True)]:
            t0 = time.time()
            M, F, Fmed, train, hold, weeks, best = run_lgbm(full, k, d33, seed=0)
            store[(name, k)] = (M, F, Fmed, train, hold, weeks)
            np.save(FC / f"{name}_cut{k}_seed0.npy", F)
            r = score_F(M, F, train, hold, weeks)
            rows.append({"variant": name, "cut": k, "seed": 0, "score": r["score"], "best_iter": best})
            print(f"cut {k} {name:5s} score {r['score']:7.2f}  best_iter {best}  ({time.time()-t0:.0f}s)", flush=True)
        # median13 for the same cut
        M, F, Fmed, train, hold, weeks = store[("base", k)]
        r = score_F(M, Fmed, train, hold, weeks)
        rows.append({"variant": "median13", "cut": k, "seed": 0, "score": r["score"], "best_iter": None})
    res = pd.DataFrame(rows)
    # blends: weight picked on cuts 1-3, reported on cut 0 and on all four cuts
    out = []
    for name in ["base", "d33"]:
        sc = {}
        for w in WEIGHTS:
            sc[w] = [score_F(store[(name, k)][0], w * store[(name, k)][1] + (1 - w) * store[(name, k)][2], store[(name, k)][3], store[(name, k)][4], store[(name, k)][5])["score"] for k in range(N_CUTS)]
        wbest = max(WEIGHTS, key=lambda w: np.mean(sc[w][1:]))
        print(f"\n{name}: blend weight on LightGBM chosen on cuts 1-3 = {wbest}; scores by weight (mean of cuts 1-3 | cut 0):")
        print("  " + "  ".join(f"{w:.1f}: {np.mean(sc[w][1:]):6.2f}|{sc[w][0]:6.2f}" for w in WEIGHTS))
        for k in range(N_CUTS):
            rows.append({"variant": f"{name}+median13 blend (w={wbest})", "cut": k, "seed": 0, "score": sc[wbest][k], "best_iter": None})
        out.append((name, wbest))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results_experiments.csv", index=False)
    pt = res.pivot_table(index="variant", columns="cut", values="score")
    pt["mean"] = pt.mean(axis=1)
    print("\nScore by cut (test shop, higher is better):"); print(pt.sort_values("mean", ascending=False).round(2).to_string())
    base_pt = pt.loc["median13", list(range(N_CUTS))]
    print("\nGain over median13 per cut, and cuts won:")
    for variant in pt.index:
        if variant == "median13":
            continue
        d = pt.loc[variant, list(range(N_CUTS))] - base_pt
        print(f"  {variant:36s} " + " ".join(f"{x:+6.2f}" for x in d) + f"   mean {d.mean():+.2f}  won {(d > 0).sum()}/4")
    # paired per-article comparison base vs d33 at each cut
    print("\nd33 minus base, paired over articles (score points, 95% bootstrap interval):")
    rng = np.random.default_rng(0)
    for k in range(N_CUTS):
        a = score_F(*[store[("d33", k)][i] for i in (0, 1, 3, 4, 5)])["detail"]["mase"]
        b = score_F(*[store[("base", k)][i] for i in (0, 1, 3, 4, 5)])["detail"]["mase"]
        d = (a - b).dropna()
        bs = [-100 * rng.choice(d.to_numpy(), len(d)).mean() for _ in range(1000)]
        print(f"  cut {k}: {-100*d.mean():+.2f}  [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]")


def phase2():
    res = pd.read_csv(OUT / "results_experiments.csv")
    pt = res[res.variant.isin(["base", "d33"])].pivot_table(index="variant", columns="cut", values="score")
    win = pt.mean(axis=1).idxmax()
    d33 = win == "d33"
    print(f"phase 2: seeds 1 and 2 for the better single variant: {win}", flush=True)
    full = v.load_weekly("test")
    rows = []
    for k in range(N_CUTS):
        Fs = []
        for seed in (0, 1, 2):
            if seed == 0:
                F0 = np.load(FC / f"{win}_cut{k}_seed0.npy")
                Fs.append(F0)
                continue
            M, F, Fmed, train, hold, weeks, best = run_lgbm(full, k, d33, seed)
            np.save(FC / f"{win}_cut{k}_seed{seed}.npy", F)
            Fs.append(F)
            print(f"cut {k} seed {seed} score {score_F(M, F, train, hold, weeks)['score']:.2f}", flush=True)
        Fm = np.mean(Fs, axis=0)
        r1 = score_F(M, Fs[0], train, hold, weeks)["score"]
        r3 = score_F(M, Fm, train, hold, weeks)["score"]
        rows += [{"variant": f"{win} seed0", "cut": k, "score": r1}, {"variant": f"{win} 3-seed average", "cut": k, "score": r3}]
        # 3-seed average blended with the median at 0.7 / 0.5
        for w in (0.7, 0.5):
            rows.append({"variant": f"{win} 3-seed avg + median13 (w={w})", "cut": k, "score": score_F(M, w * Fm + (1 - w) * Fmed, train, hold, weeks)["score"]})
        print(f"cut {k}: seed0 {r1:.2f}  3-seed average {r3:.2f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "results_experiments_seeds.csv", index=False)
    pt = out.pivot_table(index="variant", columns="cut", values="score"); pt["mean"] = pt.mean(axis=1)
    print("\nSeeds (test shop):"); print(pt.sort_values("mean", ascending=False).round(2).to_string())


if __name__ == "__main__":
    {"1": phase1, "2": phase2}[sys.argv[1]]()
