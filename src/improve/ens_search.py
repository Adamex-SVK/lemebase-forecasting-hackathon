"""Ensemble approaches not covered by iteration 08 tasks T1-T8, scored honestly on the 4 standard cuts.

    .venv/bin/python src/improve/ens_search.py

Pool (raw forecasts on all 4 standard cuts): 14 zero-shot models (zero rule applied to each), LightGBM `base` and
`d33`, CatBoost (seed average, other session's forecasts in iterations/02-robust-eval/forecasts, git-ignored),
median of the last 13 weeks, and zero. Every blend is rounded and clipped at 0 at the end (D38).

Approaches (all leave-one-cut-out, LOCO: choose on three cuts, score the fourth; the mean of the four honest
scores is compared with D40 = -13.67):
  1. greedy ensemble selection with replacement (Caruana et al.), from empty and warm-started at D40,
     maximising the ROUNDED D26 score on the training cuts;
  2. non-negative global weights summing to 1, minimising the unrounded scaled absolute error (SLSQP);
  3. zero voting: D40, but 0 where at least half of the foundation models forecast < 0.5 (no parameter);
  4. one LightGBM weight per sales segment (mean of the last 13 weeks < 1, 1-5, > 5), chosen LOCO.
Results: iterations/08-improvements/ens-search/results_ens_search.csv and printed tables.
"""
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import experiments_02 as e  # noqa: E402
import features as f  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
H = 13
FM_DIR = ROOT / "iterations" / "03-foundation-model" / "forecasts"
TREE_DIR = ROOT / "iterations" / "02-robust-eval" / "forecasts"
OUT = ROOT / "iterations" / "08-improvements" / "ens-search"
FMS = ["moirai-2.0-small", "tirex", "chronos-t5-small", "chronos-t5-base", "chronos-t5-large", "chronos-2", "timesfm-2.5",
       "moirai-1.1-base", "moirai-1.1-large", "moirai-moe-small", "moirai-moe-base", "moirai-1.0-large",
       "chronos-bolt-small", "chronos-bolt-base"]
D40_FM = ["moirai-2.0-small", "tirex", "chronos-t5-small"]


def load_cut(full, k):
    df = e.cut_frame(full, k)
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    idx = pd.MultiIndex.from_frame(M.keys)
    Y = hold.pivot_table(index=v.KEYS, columns="week", values="units", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float)
    s = v.history_scale(train, "full").reindex(idx).to_numpy(float)
    valid = np.isfinite(s) & (s > 0)
    dead = M.U[:, -8:].sum(axis=1) == 0
    mem = {}
    for m in FMS:
        p = FM_DIR / f"{m}__test__cut{k}__padded.parquet"
        if p.exists():
            F = pd.read_parquet(p).pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float)
            F[dead] = 0.0                                                     # zero rule on every foundation model (D39)
            mem[m] = F
    mem["lgbm"] = np.load(TREE_DIR / f"base_cut{k}_seed0.npy")
    if (TREE_DIR / f"d33_cut{k}_seed0.npy").exists():
        mem["lgbm-d33"] = np.load(TREE_DIR / f"d33_cut{k}_seed0.npy")
    cbs = [np.load(p) for p in sorted(TREE_DIR.glob(f"cb_F_cut{k}_seed*.npy"))]
    if cbs:
        mem["catboost"] = np.mean(cbs, axis=0)
    mem["median13"] = np.repeat(np.median(M.U[:, -13:], axis=1, keepdims=True), H, axis=1).astype(float)
    mem["zero"] = np.zeros_like(mem["lgbm"])
    return {"Y": Y[valid], "s": s[valid], "mem": {m: F[valid] for m, F in mem.items()}, "M": M, "train": train, "hold": hold,
            "weeks": pd.date_range(cut, periods=H, freq="7D"), "valid": valid, "seg": M.U[:, -13:].mean(axis=1)[valid]}


def score(c, F):
    """D26 score of a forecast restricted to scorable series (rounded, clipped)."""
    Fr = np.clip(np.round(F), 0, None)
    return 100 * (1 - np.mean(np.abs(Fr - c["Y"]).mean(axis=1) / c["s"]))


def blend(c, weights):
    return sum(w * c["mem"][m] for m, w in weights.items() if w > 0)


def d40(c):
    return 0.3 * c["mem"]["lgbm"] + 0.7 * np.mean([c["mem"][m] for m in D40_FM], axis=0)


def caruana(cuts, pool, steps=40, init=None):
    bag = Counter(init or {})
    best_bag, best_val = Counter(bag), (np.mean([score(c, d40(c)) for c in cuts]) if init else -1e9)
    for _ in range(steps):
        cand = []
        for m in pool:
            b = bag.copy(); b[m] += 1; n = sum(b.values())
            cand.append((np.mean([score(c, blend(c, {k: q / n for k, q in b.items()})) for c in cuts]), m))
        val, m = max(cand)
        bag[m] += 1
        if val > best_val + 1e-9:
            best_val, best_bag = val, Counter(bag)
    n = sum(best_bag.values())
    return {k: q / n for k, q in best_bag.items()}


def opt_weights(cuts, pool):
    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return np.mean([np.mean(np.abs(sum(wi * c["mem"][m] for wi, m in zip(w, pool)) - c["Y"]).mean(axis=1) / c["s"]) for c in cuts])
    x0 = np.full(len(pool), 1 / len(pool))
    r = minimize(loss, x0, method="SLSQP", bounds=[(0, 1)] * len(pool), constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                 options={"maxiter": 200})
    w = np.clip(r.x, 0, None); w = w / w.sum()
    return {m: wi for m, wi in zip(pool, w) if wi > 1e-3}


def zero_vote(c, members=FMS):
    F = d40(c).copy()
    raw = [c["mem"][m] for m in members if m in c["mem"]]
    votes = np.mean([r < 0.5 for r in raw], axis=0)
    F[votes >= 0.5] = 0.0
    return F


def seg_weights(cuts, grid=np.round(np.arange(0, 1.01, 0.1), 1)):
    out = {}
    for name, lo, hi in [("slow", -1, 1), ("mid", 1, 5), ("fast", 5, np.inf)]:
        best = max(grid, key=lambda w: np.mean([seg_score(c, {name: w}, (lo, hi)) for c in cuts]))
        out[name] = (best, lo, hi)
    return out


def seg_blend(c, segw):
    E = np.mean([c["mem"][m] for m in D40_FM], axis=0)
    F = 0.3 * c["mem"]["lgbm"] + 0.7 * E
    for name, (w, lo, hi) in segw.items():
        sel = (c["seg"] > lo) & (c["seg"] <= hi) if lo >= 0 else c["seg"] <= hi
        F[sel] = w * c["mem"]["lgbm"][sel] + (1 - w) * E[sel]
    return F


def seg_score(c, wmap, rng):
    lo, hi = rng
    E = np.mean([c["mem"][m] for m in D40_FM], axis=0)
    w = list(wmap.values())[0]
    F = 0.3 * c["mem"]["lgbm"] + 0.7 * E
    sel = (c["seg"] > lo) & (c["seg"] <= hi) if lo >= 0 else c["seg"] <= hi
    F[sel] = w * c["mem"]["lgbm"][sel] + (1 - w) * E[sel]
    return score(c, F)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full = v.load_weekly("test")
    cuts = [load_cut(full, k) for k in range(4)]
    pool = [m for m in cuts[0]["mem"] if all(m in c["mem"] for c in cuts)]
    ref = [score(c, d40(c)) for c in cuts]
    # sanity: our fast score equals the official one for D40 on cut 0
    c0 = cuts[0]; F0 = np.zeros((c0["M"].n, H)); F0[c0["valid"]] = d40(c0)
    official = e.score_F(c0["M"], np.round(F0), c0["train"], c0["hold"], c0["weeks"])["score"]
    print(f"pool ({len(pool)}): {pool}\nD40 by cut: {np.round(ref, 2)}  mean {np.mean(ref):.2f}  (official cut-0 check {official:.2f})\n")
    rows = []

    def report(name, honest, picks):
        g = np.array(honest) - np.array(ref)
        rows.append({"approach": name, **{f"cut{k}": round(x, 2) for k, x in enumerate(honest)}, "mean": round(np.mean(honest), 2),
                     "gain_vs_D40": round(g.mean(), 2), "cuts_won": int((g > 0).sum()), "picks": picks})
        verdict = "ADOPT" if g.mean() >= 0.5 and (g > 0).sum() >= 3 else "reject"
        print(f"{name:42s} " + " ".join(f"{x:7.2f}" for x in honest) + f"   mean {np.mean(honest):7.2f}  gain {g.mean():+.2f}  won {(g > 0).sum()}/4  {verdict}")
        if picks:
            print(f"   {picks}")

    for label, init in [("greedy selection (from empty)", None), ("greedy selection (warm start at D40)", {"lgbm": 9, **{m: 7 for m in D40_FM}})]:
        honest, picks = [], []
        for k in range(4):
            w = caruana([cuts[j] for j in range(4) if j != k], pool, init=init)
            honest.append(score(cuts[k], blend(cuts[k], w))); picks.append({m: round(x, 2) for m, x in sorted(w.items(), key=lambda t: -t[1])[:5]})
        report(label, honest, picks)
    honest, picks = [], []
    for k in range(4):
        w = opt_weights([cuts[j] for j in range(4) if j != k], pool)
        honest.append(score(cuts[k], blend(cuts[k], w))); picks.append({m: round(x, 2) for m, x in sorted(w.items(), key=lambda t: -t[1])[:5]})
    report("optimised global weights (SLSQP)", honest, picks)
    report("zero voting (>= half of 14 models < 0.5)", [score(c, zero_vote(c)) for c in cuts], "")
    report("zero voting (D40's 3 models)", [score(c, zero_vote(c, D40_FM)) for c in cuts], "")
    honest, picks = [], []
    for k in range(4):
        sw = seg_weights([cuts[j] for j in range(4) if j != k])
        honest.append(score(cuts[k], seg_blend(cuts[k], sw))); picks.append({n: w for n, (w, _, _) in sw.items()})
    report("LightGBM weight per sales segment", honest, picks)
    pd.DataFrame(rows).to_csv(OUT / "results_ens_search.csv", index=False)


if __name__ == "__main__":
    main()
