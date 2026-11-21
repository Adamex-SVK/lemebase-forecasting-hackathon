"""Performance-based blend weights, scored honestly against D40.

    .venv/bin/python src/improve/perf_weights.py

A. Global performance weights (leave-one-cut-out on the 4 standard cuts): each member's stand-alone MASE on the three
   training cuts (rounded, zero rule on foundation models) gives its weight:
     inverse: w_m ~ 1 / MASE_m ;  softmax: w_m ~ exp(-MASE_m / tau), tau chosen LOCO from {0.02, 0.05, 0.1, 0.2}.
   Pools: the 4 D40 members, the 6 best members (ranked on the training cuts), all 19 members.
B. Per-article performance weights (the article's own track record): for test cut K (offset 13K weeks), each member's
   mean scaled absolute error for THAT article is measured on the overlapping cuts whose whole holdout ends before the test
   cut starts (offset >= 13K + 13; forecasts in iterations/03-foundation-model/forecasts_cuts and
   iterations/05-meta-ensemble/base_forecasts). w_im ~ 1 / (e_im + eps), shrunk toward D40's global weights:
   w = alpha * w_article + (1 - alpha) * w_D40, alpha in {0.25, 0.5, 1.0} (all reported, none selected).
   Possible only for cuts 0, 1, 2 (cut 3 has no earlier overlapping cut); compared with D40 on the same cuts.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "improve"))
import ens_search as es  # noqa: E402
import features as f  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
H = 13
CUTS_DIR = ROOT / "iterations" / "03-foundation-model" / "forecasts_cuts"
BASE_DIR = ROOT / "iterations" / "05-meta-ensemble" / "base_forecasts"
OUT = ROOT / "iterations" / "08-improvements" / "ens-search"
D40_W = {"lgbm": 0.3, "moirai-2.0-small": 0.7 / 3, "tirex": 0.7 / 3, "chronos-t5-small": 0.7 / 3}
TRACK_MEMBERS = ["lgbm", "moirai-2.0-small", "tirex", "chronos-t5-small"]


def standalone_mase(c, m):
    return 1 - es.score(c, c["mem"][m]) / 100


def article_errors(full, W):
    """Per-article mean scaled absolute error of each track member at overlapping cut W (Series by article_id)."""
    df = full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=W)]
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    idx = pd.MultiIndex.from_frame(M.keys)
    Y = hold.pivot_table(index=v.KEYS, columns="week", values="units", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float)
    s = v.history_scale(train, "full").reindex(idx).to_numpy(float)
    ok = np.isfinite(s) & (s > 0)
    dead = M.U[:, -8:].sum(axis=1) == 0
    out = {}
    for m in TRACK_MEMBERS:
        if m == "lgbm":
            F = np.load(BASE_DIR / f"lgbm_w{W}.npy")
        else:
            F = pd.read_parquet(CUTS_DIR / f"{m}__test__w{W}__padded.parquet").pivot_table(
                index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float)
            F[dead] = 0.0
        e = np.abs(F - Y).mean(axis=1) / np.where(ok, s, np.nan)
        out[m] = pd.Series(e, index=M.keys["article_id"].to_numpy())[ok]
    return out


def main():
    full = v.load_weekly("test")
    cuts = [es.load_cut(full, k) for k in range(4)]
    pool_all = [m for m in cuts[0]["mem"] if all(m in c["mem"] for c in cuts)]
    ref = [es.score(c, es.d40(c)) for c in cuts]
    print(f"D40 by cut: {np.round(ref, 2)} mean {np.mean(ref):.2f}\n")
    rows = []

    def report(name, honest, ref_cuts, picks=""):
        g = np.array(honest) - np.array(ref_cuts)
        verdict = "ADOPT" if g.mean() >= 0.5 and (g > 0).sum() >= max(3, len(g) - 1) else "reject"
        rows.append({"approach": name, "scores": [round(x, 2) for x in honest], "mean": round(np.mean(honest), 2),
                     "D40_same_cuts": round(np.mean(ref_cuts), 2), "gain": round(g.mean(), 2), "won": f"{(g > 0).sum()}/{len(g)}", "picks": str(picks)})
        print(f"{name:52s} " + " ".join(f"{x:7.2f}" for x in honest) + f"  mean {np.mean(honest):7.2f} (D40 {np.mean(ref_cuts):.2f})  gain {g.mean():+.2f}  won {(g > 0).sum()}/{len(g)}  {verdict}")
        if picks:
            print(f"   {picks}")

    # A. global performance weights, LOCO on the 4 standard cuts
    for pool_name in ["D40 members", "top 6", "all 19"]:
        for scheme in ["inverse", "softmax"]:
            honest, picks = [], []
            for k in range(4):
                tr = [cuts[j] for j in range(4) if j != k]
                mase = {m: np.mean([standalone_mase(c, m) for c in tr]) for m in pool_all}
                if pool_name == "D40 members":
                    pool = list(D40_W)
                elif pool_name == "top 6":
                    pool = sorted(mase, key=mase.get)[:6]
                else:
                    pool = [m for m in pool_all if m != "zero"]
                if scheme == "inverse":
                    w = {m: 1 / mase[m] for m in pool}
                else:
                    def wts(tau):
                        return {m: np.exp(-(mase[m] - min(mase[p] for p in pool)) / tau) for m in pool}
                    tau = max([0.02, 0.05, 0.1, 0.2], key=lambda t: np.mean([es.score(c, es.blend(c, norm(wts(t)))) for c in tr]))
                    w = wts(tau)
                w = norm(w)
                honest.append(es.score(cuts[k], es.blend(cuts[k], w)))
                picks.append({m: round(x, 2) for m, x in sorted(w.items(), key=lambda t: -t[1])[:4]})
            report(f"global {scheme} weights, pool {pool_name}", honest, ref, picks[0])

    # B. per-article track-record weights on cuts 0, 1, 2
    track = {W: article_errors(full, W) for W in range(0, 49, 4)}
    for alpha in [0.25, 0.5, 1.0]:
        honest, refk = [], []
        for k in range(3):
            hist = [W for W in track if W >= 13 * k + 13]
            c = cuts[k]
            arts = c["M"].keys["article_id"].to_numpy()[c["valid"]]
            err = {m: pd.concat([track[W][m] for W in hist], axis=1).mean(axis=1).reindex(arts).to_numpy() for m in TRACK_MEMBERS}
            inv = np.column_stack([1 / (err[m] + 0.05) for m in TRACK_MEMBERS])
            have = np.isfinite(inv).all(axis=1)
            wa = np.where(have[:, None], inv / np.where(have, np.nansum(inv, axis=1), 1)[:, None], 0)
            wg = np.array([D40_W[m] for m in TRACK_MEMBERS])
            W_ = np.where(have[:, None], alpha * wa + (1 - alpha) * wg, wg)
            F = sum(W_[:, i:i + 1] * c["mem"][m] for i, m in enumerate(TRACK_MEMBERS))
            honest.append(es.score(c, F)); refk.append(ref[k])
        report(f"per-article track-record weights, alpha {alpha} (cuts 0-2)", honest, refk, f"history cuts per test cut: {[len([W for W in track if W >= 13 * k + 13]) for k in range(3)]}")
    pd.DataFrame(rows).to_csv(OUT / "results_perf_weights.csv", index=False)


def norm(w):
    s = sum(w.values())
    return {m: x / s for m, x in w.items()}


if __name__ == "__main__":
    main()
