"""Items 2.1 and 2.2 of iterations/02-robust-eval/notes_adamex_loss-and-nn-plan.md.

  2.1  LightGBM `huber` (alpha = 1 on the scaled target), `fair` and `l2` instead of `l1`.
  2.2  LightGBM `tweedie` (p = 1.2, 1.5) and `poisson`, each followed by a multiplicative
       **median correction** fitted on the training windows (minimises the weighted L1 error).

Same windows, features, weights and early-stopping protocol as src/experiments_loss.py (which runs
the l1 reference and the zero gate); the target stays units / MASE scale (D29) and every objective
early-stops on the validation L1, so the comparison is fair. Diagnostics only: the plan expects
"within noise", so nothing here changes src/train_gbm.py.

Run: .venv/bin/python src/experiments_loss2.py [--cuts 0 1 2 3] [--seeds 0 1 2]
Outputs: forecasts/loss2_F_{variant}_cut{k}_seed{s}.npy, rows appended to results_adamex_loss.csv
         (same columns as experiments_loss.py: cut, seed, variant, score).
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
import experiments_loss as el  # noqa: E402
import features as f  # noqa: E402
import train_gbm as g  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
OUT, FC, H = el.OUT, el.FC, f.H

# objective -> extra LightGBM parameters. `metric="l1"` is added for every config in fit_objective,
# so early stopping always watches the metric's own loss (not the objective's default).
CONFIGS = {
    "huber": dict(objective="huber", alpha=1.0),
    "fair": dict(objective="fair", fair_c=1.0),
    "l2": dict(objective="regression"),
    "tweedie1.2": dict(objective="tweedie", tweedie_variance_power=1.2),
    "tweedie1.5": dict(objective="tweedie", tweedie_variance_power=1.5),
    "poisson": dict(objective="poisson"),
}
NEEDS_MEDIAN = {"tweedie1.2", "tweedie1.5", "poisson"}


def weighted_median(x, w):
    """Minimiser of sum_i w_i |x_i - c| (the weighted median)."""
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    return float(x[np.searchsorted(cw, 0.5 * cw[-1])])


def median_factor(y, p, w):
    """Factor c > 0 minimising sum_i w_i |y_i - c p_i|: the weighted median of y/p with
    weights w*p (because |y - c p| = p * |y/p - c|)."""
    ok = p > 0
    if not ok.any():
        return 1.0
    c = weighted_median(y[ok] / p[ok], w[ok] * p[ok])
    return c if np.isfinite(c) and c > 0 else 1.0


def fit_objective(P, name, seed):
    """Train one objective with the LightGBM protocol of experiments_loss.fit_pair.

    Returns (F [n x 13] in units, median factor c, best_iteration)."""
    M = P["M"]
    params = dict(g.PARAMS, metric="l1", seed=seed, **CONFIGS[name])
    dtr = lgb.Dataset(P["w_tr"]["X"], P["w_tr"]["y"], weight=P["w_tr"]["w"], feature_name=f.FEATURES)
    dva = lgb.Dataset(P["w_va"]["X"], P["w_va"]["y"], weight=P["w_va"]["w"], reference=dtr)
    m1 = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    best = m1.best_iteration
    m2 = lgb.train(params, lgb.Dataset(P["w_all"]["X"], P["w_all"]["y"], weight=P["w_all"]["w"], feature_name=f.FEATURES),
                   num_boost_round=int(best * 1.1) + 1)
    c = 1.0
    if name in NEEDS_MEDIAN:   # fit the correction on the training windows only (no holdout weeks)
        c = median_factor(P["w_all"]["y"].astype(np.float64), m2.predict(P["w_all"]["X"]).astype(np.float64),
                          P["w_all"]["w"].astype(np.float64))
    X, scale, valid = P["inf"]
    F = np.clip(m2.predict(X).reshape(M.n, H) * c * scale[:, None], 0, None)
    F[~valid] = 0.0
    return F, c, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    a = ap.parse_args()
    FC.mkdir(exist_ok=True)
    full = v.load_weekly("test")
    path = OUT / "results_adamex_loss.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["cut"], r["seed"], r["variant"]) for r in rows}
    for k in a.cuts:
        t0 = time.time()
        P = el.prep(full, k)
        print(f"cut {k}: windows train {P['w_tr']['n_windows']:,} all {P['w_all']['n_windows']:,} ({time.time()-t0:.0f}s)", flush=True)
        Fs = {name: [] for name in a.variants}
        for s in a.seeds:
            for name in a.variants:
                fp = FC / f"loss2_F_{name}_cut{k}_seed{s}.npy"
                if fp.exists():
                    F = np.load(fp)
                else:
                    t1 = time.time()
                    F, c, best = fit_objective(P, name, s)
                    np.save(fp, F)
                    print(f"  cut {k} seed {s} {name:11s}: trees {best:4d}, median factor {c:.3f} ({time.time()-t1:.0f}s)", flush=True)
                Fs[name].append(F)
                if (k, s, name) in done:
                    continue
                r = e.score_F(P["M"], F, P["train"], P["hold"], P["weeks"])
                rows.append({"cut": k, "seed": s, "variant": name, "score": r["score"]})
                done.add((k, s, name))
                pd.DataFrame(rows).to_csv(path, index=False)   # append-safe: write after every variant
                print(f"    {name:16s} {r['score']:7.2f}", flush=True)
        if len(a.seeds) > 1:
            tag = -1   # seed = -1 marks the average over the seeds, as in experiments_loss.py
            for name, Fs_ in Fs.items():
                Fm = np.mean(Fs_, axis=0)
                r = e.score_F(P["M"], Fm, P["train"], P["hold"], P["weeks"])
                rows = [x for x in rows if not (x["cut"] == k and x["seed"] == tag and x["variant"] == name)]
                rows.append({"cut": k, "seed": tag, "variant": name, "score": r["score"]})
                print(f"    avg {name:12s} {r['score']:7.2f}", flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    df = pd.DataFrame(rows)
    for label, sub in [("single seeds (mean over seeds)", df[df.seed >= 0]), ("seed average", df[df.seed == -1])]:
        if len(sub):
            pt = sub.pivot_table(index="variant", columns="cut", values="score", aggfunc="mean")
            pt["mean"] = pt.mean(axis=1)
            print(f"\n{label}:"); print(pt.round(2).to_string())


if __name__ == "__main__":
    main()
