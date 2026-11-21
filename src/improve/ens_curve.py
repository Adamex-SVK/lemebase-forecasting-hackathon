"""When do more ensemble members stop adding value? Members from every family on the 4 standard cuts.

    .venv/bin/python src/improve/ens_curve.py

Families: level/intermittent classics (sf-ses, adida, croston*, imapa, tsb), seasonal (52-week seasonal naive, smoothed),
ML (LightGBM base/d33, CatBoost, median13), deep learning (N-HiTS, N-BEATS, pooled N-HiTS; seed means), foundation models (14 zero-shot,
TabPFN-TS, fine-tuned Chronos-2 on siblings and on siblings + own history). Zero rule (8 weeks) on every non-tree member, rounding at the end.
1. Forward selection curve, leave-one-cut-out: pick members one at a time on 3 cuts (equal-weight average of the chosen set), score the 4th.
2. Same curve in-sample (optimistic ceiling).
3. D40 plus the best member of each family added at weights 0.1 / 0.2 (weights fixed in advance).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "improve"))
import ens_search as es, validate as v  # noqa: E402
warnings.filterwarnings("ignore")
H = 13
IT3 = ROOT / "iterations" / "03-foundation-model"
MAIN = Path("/Users/adomi/Documents/hackathlon")                      # shared checkout holding tabpfn and neural forecasts
NEURAL = MAIN / "iterations" / "06-neural" / "forecasts"
SF = ["sf-ses", "sf-adida", "sf-croston", "sf-croston-sba", "sf-croston-opt", "sf-imapa", "sf-tsb"]
FAMILY = {}


def pivot(c, p):
    idx = pd.MultiIndex.from_frame(c["M"].keys)
    return pd.read_parquet(p).pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float)


def build(full, k):
    c = es.load_cut(full, k); mem = dict(c["mem"]); dead = (c["M"].U[:, -8:].sum(axis=1) == 0)
    def add(name, F, fam, zero=True):
        F = np.asarray(F, float).copy()
        if zero:
            F[dead] = 0.0
        mem[name] = F[c["valid"]]; FAMILY[name] = fam
    for m in es.FMS:
        FAMILY[m] = "foundation"
    for m in ["lgbm", "lgbm-d33", "catboost", "median13"]:
        FAMILY[m] = "ML/robust"
    FAMILY["zero"] = "other"
    for m in SF:
        add(m, pivot(c, IT3 / "forecasts" / f"{m}__test__cut{k}__padded.parquet"), "level (classics)")
    p = MAIN / "iterations" / "03-foundation-model" / "forecasts" / f"tabpfn-ts-v2__test__cut{k}__padded.parquet"
    if p.exists():
        add("tabpfn-ts-v2", pivot(c, p), "foundation")
    for tag, name in [("ft", "chronos-2-ft-siblings"), ("ftT1", "chronos-2-ft-own")]:
        add(name, pivot(c, IT3 / f"forecasts_oos-{tag}" / f"chronos-2-{tag}__test__cut{k}__padded.parquet"), "foundation (fine-tuned)")
    for m, fam in [("NHITS", "deep learning"), ("NBEATS_base", "deep learning"), ("NHITS_pooled", "deep learning")]:
        p = NEURAL / f"{m}__cut{k}__mean.npy"
        if p.exists():
            add(m, np.load(p), fam)
    U = c["M"].U                                                     # seasonal naive: same weeks one year ago, 3-week smoothed
    S = np.zeros((U.shape[0], H))
    for h in range(H):
        j = U.shape[1] - 52 + h
        S[:, h] = U[:, max(j - 1, 0):j + 2].mean(axis=1) if j >= 1 else U[:, -13:].mean(axis=1)
    add("seasonal-naive-52", S, "seasonal")
    c["mem"] = mem
    return c


def avg(c, names):
    return np.mean([c["mem"][m] for m in names], axis=0)


def curve(cuts, pool, steps, train_idx_fn):
    """greedy forward selection (without replacement) on the training cuts; returns list of chosen sets."""
    chosen = []
    for _ in range(steps):
        best = max((m for m in pool if m not in chosen), key=lambda m: np.mean([es.score(c, avg(c, chosen + [m])) for c in train_idx_fn]))
        chosen.append(best)
    return chosen


def main():
    full = v.load_weekly("test")
    cuts = [build(full, k) for k in range(4)]
    pool = [m for m in cuts[0]["mem"] if m != "zero" and all(m in c["mem"] for c in cuts)]
    ref = np.array([es.score(c, es.d40(c)) for c in cuts])
    print(f"pool {len(pool)} members; families: " + ", ".join(f"{f} {sum(1 for m in pool if FAMILY.get(m) == f)}" for f in sorted(set(FAMILY.values()))))
    print(f"D40 by cut {np.round(ref, 2)} mean {ref.mean():.2f}\n")
    print("single members (mean of 4 cuts, rounded), by family:")
    solo = {m: np.mean([es.score(c, c["mem"][m]) for c in cuts]) for m in pool}
    for f in sorted(set(FAMILY[m] for m in pool)):
        ms_ = sorted([m for m in pool if FAMILY[m] == f], key=lambda m: -solo[m])
        print(f"  {f:24s} " + ", ".join(f"{m} {solo[m]:.1f}" for m in ms_[:5]))
    steps = 12
    honest = np.zeros((4, steps)); sets = []
    for k in range(4):
        tr = [cuts[j] for j in range(4) if j != k]
        ch = curve(cuts, pool, steps, tr); sets.append(ch)
        for s in range(steps):
            honest[k, s] = es.score(cuts[k], avg(cuts[k], ch[:s + 1]))
    insample = []; ch = curve(cuts, pool, steps, cuts)
    for s in range(steps):
        insample.append(np.mean([es.score(c, avg(c, ch[:s + 1])) for c in cuts]))
    rows = []
    print("\nforward selection curve (equal-weight average of the chosen set), leave-one-cut-out vs in-sample:")
    print(f"{'size':>4} {'LOCO mean':>10} {'gain vs D40':>12} {'cuts won':>9} {'in-sample':>10}   added (last fold's choice order)")
    for s in range(steps):
        g = honest[:, s] - ref
        print(f"{s + 1:4d} {honest[:, s].mean():10.2f} {g.mean():+12.2f} {(g > 0).sum():7d}/4 {insample[s]:10.2f}   {ch[s]} [{FAMILY[ch[s]]}]")
        rows.append({"size": s + 1, "loco_mean": round(honest[:, s].mean(), 2), "gain_vs_D40": round(g.mean(), 2), "cuts_won": int((g > 0).sum()), "in_sample": round(insample[s], 2), "added_in_sample": ch[s], "family": FAMILY[ch[s]]})
    print("\nD40 + one member of each family (best by mean solo score, chosen without the held-out cut only in spirit: fixed list), at weight w of the final blend:")
    d40 = lambda c: es.d40(c)
    for f in sorted(set(FAMILY[m] for m in pool)):
        ms_ = sorted([m for m in pool if FAMILY[m] == f and m not in es.D40_FM + ["lgbm"]], key=lambda m: -solo[m])
        if not ms_:
            continue
        m = ms_[0]
        for w in [0.1, 0.2]:
            s = np.array([es.score(c, (1 - w) * d40(c) + w * c["mem"][m]) for c in cuts]); g = s - ref
            print(f"  + {m:24s} [{f}] w={w}: mean {s.mean():7.2f}  gain {g.mean():+.2f}  won {(g > 0).sum()}/4")
            rows.append({"size": f"D40 + {m} w={w}", "loco_mean": round(s.mean(), 2), "gain_vs_D40": round(g.mean(), 2), "cuts_won": int((g > 0).sum()), "in_sample": "", "added_in_sample": m, "family": f})
    out = ROOT / "iterations" / "10-external-info" / "x1"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "results_adamex-svk_ens_curve.csv", index=False)


if __name__ == "__main__":
    main()
