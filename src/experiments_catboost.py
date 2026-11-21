"""Item 2.4 (+ blends 2.6) of iterations/02-robust-eval/notes_adamex_loss-and-nn-plan.md.

CatBoost with MAE loss on the same windows, features, weights and early-stopping protocol as the LightGBM
(target = units / scale, D29). Needs the LightGBM forecasts and gate probabilities saved by experiments_loss.py.
Blends use FIXED equal weights (not tuned on the cuts). Scored on the 4 rolling cuts, D26.

Run: .venv/bin/python src/experiments_catboost.py [--cuts 0 1 2 3] [--seeds 0 1 2]
Outputs: forecasts/cb_F_cut{k}_seed{s}.npy, results_adamex_catboost.csv
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_02 as e  # noqa: E402
import experiments_loss as el  # noqa: E402
import features as f  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
OUT, FC, H = el.OUT, el.FC, f.H
PARAMS = dict(loss_function="MAE", learning_rate=0.08, depth=8, l2_leaf_reg=3.0, thread_count=8, verbose=False, border_count=128)


def fit_cb(P, seed):
    M = P["M"]
    X, scale, valid = P["inf"]
    tr, va, al = P["w_tr"], P["w_va"], P["w_all"]
    m1 = CatBoostRegressor(iterations=3000, random_seed=seed, early_stopping_rounds=50, **PARAMS)
    m1.fit(Pool(tr["X"], tr["y"], weight=tr["w"]), eval_set=Pool(va["X"], va["y"], weight=va["w"]), use_best_model=True)
    best = m1.get_best_iteration()
    m2 = CatBoostRegressor(iterations=int(best * 1.1) + 1, random_seed=seed, **PARAMS)
    m2.fit(Pool(al["X"], al["y"], weight=al["w"]))
    F = np.clip(m2.predict(X).reshape(M.n, H) * scale[:, None], 0, None)
    F[~valid] = 0.0
    return F, best


def combos(P, Fl, Fc, Pr):
    M = P["M"]
    med = np.repeat(np.median(M.U[:, -13:], axis=1, keepdims=True), H, axis=1).astype(float)
    gate = lambda F: np.where(Pr < 0.5, 0.0, F)
    avg = 0.5 * (Fl + Fc)
    return {"lgbm": Fl, "catboost": Fc, "lgbm+cb": avg, "lgbm+gate": gate(Fl), "catboost+gate": gate(Fc),
            "lgbm+cb+gate": gate(avg), "lgbm+cb+median13 (1/3 each)": (Fl + Fc + med) / 3,
            "lgbm+cb+gate, then 1/3 median": gate((Fl + Fc) / 3 * 2 + med / 3) if False else gate((Fl + Fc + med) / 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    full = v.load_weekly("test")
    rows = []
    for k in a.cuts:
        P = el.prep(full, k)
        Fl = [np.load(FC / f"loss_F_cut{k}_seed{s}.npy") for s in a.seeds]
        Pl = [np.load(FC / f"loss_P_cut{k}_seed{s}.npy") for s in a.seeds]
        Fc = []
        for s in a.seeds:
            fp = FC / f"cb_F_cut{k}_seed{s}.npy"
            if fp.exists():
                Fc.append(np.load(fp)); continue
            t0 = time.time()
            F, best = fit_cb(P, s)
            np.save(fp, F); Fc.append(F)
            print(f"cut {k} seed {s}: catboost {best} iters ({time.time()-t0:.0f}s)", flush=True)
        for tag, sel in [("seed-mean", None), ("avg", "avg")]:
            if sel is None:   # score each seed, record the mean over seeds
                per = [combos(P, Fl[i], Fc[i], Pl[i]) for i in range(len(a.seeds))]
                sc = {n: np.mean([e.score_F(P["M"], p[n], P["train"], P["hold"], P["weeks"])["score"] for p in per]) for n in per[0]}
            else:
                c = combos(P, np.mean(Fl, 0), np.mean(Fc, 0), np.mean(Pl, 0))
                sc = {n: e.score_F(P["M"], Fv, P["train"], P["hold"], P["weeks"])["score"] for n, Fv in c.items()}
            for n, s_ in sc.items():
                rows.append({"cut": k, "mode": tag, "variant": n, "score": s_})
        print(f"cut {k} done", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "results_adamex_catboost.csv", index=False)
    df = pd.DataFrame(rows)
    for tag in ("seed-mean", "avg"):
        pt = df[df["mode"] == tag].pivot_table(index="variant", columns="cut", values="score")
        pt["mean"] = pt.mean(axis=1)
        print(f"\n{tag}:"); print(pt.sort_values("mean", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
