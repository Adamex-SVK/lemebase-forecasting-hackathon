"""Part 2 of iterations/02-robust-eval/notes_adamex_loss-and-nn-plan.md (items 2.0 and 2.3).

  2.0  reference: global LightGBM, L1 on units / scale (same as experiments_02 `base`)
  2.3  zero gate: a classifier predicts P(units > 0 in week o+h-1) from the same features and windows;
       forecast is set to 0 where P < threshold (0.5 by default). Compared with no gate and with
       the hand rule D20 (zero if no sale in the last 8 weeks).

Scored on the 4 rolling cuts (D32) with the D26 metric. Seeds average both the regression and the gate probability.
Run: .venv/bin/python src/experiments_loss.py [--cuts 0 1 2 3] [--seeds 0 1 2]
Outputs: iterations/02-robust-eval/forecasts/loss_{F,P}_cut{k}_seed{s}.npy, results_adamex_loss.csv
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_02 as e  # noqa: E402
import features as f  # noqa: E402
import train_gbm as g  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
OUT = v.ROOT / "iterations" / "02-robust-eval"
FC = OUT / "forecasts"
H = f.H
CLF = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)


def prep(full, k):
    df = e.cut_frame(full, k)
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=H, freq="7D")
    origins = f.make_origins(M.W)
    val_o = origins[-7:]
    tr_o = origins[origins <= val_o[0] - H]
    return dict(M=M, train=train, hold=hold, weeks=weeks, origins=origins,
                w_tr=f.build_windows(M, tr_o), w_va=f.build_windows(M, val_o), w_all=f.build_windows(M, origins),
                inf=f.inference_features(M))


def fit_pair(P, seed):
    """Returns F [n x 13] (units) and prob [n x 13] (P(sale > 0))."""
    M = P["M"]
    X, scale, valid = P["inf"]
    out = {}
    for name, params, target in [("reg", dict(g.PARAMS, seed=seed), lambda w: w["y"]),
                                 ("clf", dict(CLF, seed=seed), lambda w: (w["y"] > 0).astype(np.float32))]:
        dtr = lgb.Dataset(P["w_tr"]["X"], target(P["w_tr"]), weight=P["w_tr"]["w"], feature_name=f.FEATURES)
        dva = lgb.Dataset(P["w_va"]["X"], target(P["w_va"]), weight=P["w_va"]["w"], reference=dtr)
        m1 = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        best = m1.best_iteration
        m2 = lgb.train(params, lgb.Dataset(P["w_all"]["X"], target(P["w_all"]), weight=P["w_all"]["w"], feature_name=f.FEATURES),
                       num_boost_round=int(best * 1.1) + 1)
        out[name] = (m2.predict(X).reshape(M.n, H), best)
    F = np.clip(out["reg"][0] * scale[:, None], 0, None)
    F[~valid] = 0.0
    Pr = out["clf"][0]
    Pr[~valid] = 0.0
    return F, Pr, out["reg"][1], out["clf"][1]


def variants(P, F, Pr, thr=0.5):
    M = P["M"]
    d20 = F.copy(); d20[M.U[:, -8:].sum(axis=1) == 0] = 0.0
    gate = F.copy(); gate[Pr < thr] = 0.0
    both = d20.copy(); both[Pr < thr] = 0.0
    return {"l1": F, "l1+D20": d20, f"l1+gate{thr}": gate, f"l1+D20+gate{thr}": both}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    FC.mkdir(exist_ok=True)
    full = v.load_weekly("test")
    path = OUT / "results_adamex_loss.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["cut"], r["seed"], r["variant"]) for r in rows}
    for k in a.cuts:
        t0 = time.time()
        P = prep(full, k)
        print(f"cut {k}: windows train {P['w_tr']['n_windows']:,} all {P['w_all']['n_windows']:,} ({time.time()-t0:.0f}s)", flush=True)
        Fs, Ps = [], []
        for s in a.seeds:
            fF, fP = FC / f"loss_F_cut{k}_seed{s}.npy", FC / f"loss_P_cut{k}_seed{s}.npy"
            if fF.exists() and fP.exists():
                F, Pr = np.load(fF), np.load(fP)
            else:
                t1 = time.time()
                F, Pr, b1, b2 = fit_pair(P, s)
                np.save(fF, F); np.save(fP, Pr)
                print(f"  cut {k} seed {s}: trees reg {b1} clf {b2} ({time.time()-t1:.0f}s)", flush=True)
            Fs.append(F); Ps.append(Pr)
            for name, Fv in variants(P, F, Pr).items():
                if (k, s, name) not in done:
                    r = e.score_F(P["M"], Fv, P["train"], P["hold"], P["weeks"])
                    rows.append({"cut": k, "seed": s, "variant": name, "score": r["score"]})
                    print(f"    {name:18s} {r['score']:7.2f}", flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
        if len(Fs) > 1:
            tag = -1  # seed = -1 marks the average over the seeds run
            Fm, Pm = np.mean(Fs, axis=0), np.mean(Ps, axis=0)
            for name, Fv in variants(P, Fm, Pm).items():
                r = e.score_F(P["M"], Fv, P["train"], P["hold"], P["weeks"])
                rows = [x for x in rows if not (x["cut"] == k and x["seed"] == tag and x["variant"] == name)]
                rows.append({"cut": k, "seed": tag, "variant": name, "score": r["score"]})
                print(f"    avg {name:14s} {r['score']:7.2f}", flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    df = pd.DataFrame(rows)
    for label, sub in [("single seeds (mean over seeds)", df[df.seed >= 0]), ("seed average", df[df.seed == -1])]:
        if len(sub):
            pt = sub.pivot_table(index="variant", columns="cut", values="score", aggfunc="mean")
            pt["mean"] = pt.mean(axis=1)
            print(f"\n{label}:"); print(pt.round(2).to_string())


if __name__ == "__main__":
    main()
