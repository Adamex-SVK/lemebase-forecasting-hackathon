"""Score the iteration-06 neural forecasts with the D26 metric and test them as blend members.

    .venv/bin/python src/neural/score_nf.py [--variant base|d33] [--post round+zero8]

Reads the seed-average forecasts written by src/neural/run_nf.py
(iterations/06-neural/forecasts/{model}__test__cut{k}__padded.parquet) and scores each beside:
  * the LightGBM own arm (iterations/02-robust-eval/forecasts/{variant}_cut{k}_seed0.npy),
  * the median of the last 13 weeks,
  * the D40 foundation ensemble (mean of moirai-2.0-small, tirex, chronos-t5-small zero-shot forecasts).
Blend weights are chosen on cuts 1-3 and reported on cut 0 and on the mean of all 4 cuts (same protocol as
blend_eval.py, no tuning on the reported cuts). Rounding and the 8-week zero rule are applied as in
score_all.py. Writes results_kilo_neural.csv and prints the tables plus a paired per-article interval.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
import experiments_02 as e  # noqa: E402
import features as f  # noqa: E402
import score_all as sa  # noqa: E402
import validate as v  # noqa: E402

NF_DIR = ROOT / "iterations" / "06-neural"
FC = NF_DIR / "forecasts"
FM_DIR = ROOT / "iterations" / "03-foundation-model" / "forecasts"
LGBM_DIR = ROOT / "iterations" / "02-robust-eval" / "forecasts"
OUT = NF_DIR / "results_kilo_neural.csv"
H = f.H
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
FM_SET = ["moirai-2.0-small", "tirex", "chronos-t5-small"]


def neural_models():
    if not FC.exists():
        return []
    names = sorted({p.stem.split("__")[0] for p in FC.glob("*__test__cut0__padded.parquet")})
    return [m for m in names if all((FC / f"{m}__test__cut{k}__padded.parquet").exists() for k in range(4))]


def foundation_models():
    return [m for m in FM_SET if all((FM_DIR / f"{m}__test__cut{k}__padded.parquet").exists() for k in range(4))]


def post_process(F, dead, name):
    F = np.array(F, dtype=float, copy=True)
    if "round" in name:
        F = np.round(F)
    if "zero8" in name:
        F[dead] = 0.0
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base", choices=["base", "d33"])
    ap.add_argument("--post", default="round+zero8", choices=["raw", "round", "zero8", "round+zero8"])
    a = ap.parse_args()
    full = v.load_weekly("test")
    models, fms = neural_models(), foundation_models()
    if not models:
        print(f"no neural forecasts in {FC} yet (run src/neural/run_nf.py first)"); return
    if not fms:
        print(f"warning: no D40 foundation model has all 4 cuts in {FM_DIR}; skipping that blend")

    ctx = {}
    for k in range(4):
        df = e.cut_frame(full, k)
        train, hold, cut = v.split(df)
        M = f.ShopMatrix(df[df["week"] < cut])
        _, _, dead_keys, _ = sa.prepare_cut("test", k)
        dead = np.array([tuple(x) in dead_keys for x in M.keys[v.KEYS].to_numpy()])
        ctx[k] = dict(M=M, train=train, hold=hold, weeks=pd.date_range(cut, periods=H, freq="7D"), dead=dead,
                      Fl=np.load(LGBM_DIR / f"{a.variant}_cut{k}_seed0.npy"),
                      Fmed=np.repeat(np.median(M.U[:, -13:], axis=1, keepdims=True), H, axis=1).astype(float))

    def score(k, F):
        c = ctx[k]
        return e.score_F(c["M"], post_process(F, c["dead"], a.post), c["train"], c["hold"], c["weeks"])["score"]

    def matrix(fc, k):
        P = fc.pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum")
        return P.reindex(pd.MultiIndex.from_frame(ctx[k]["M"].keys)).fillna(0.0).to_numpy(float)

    def fm_matrix(k):
        if not fms:
            return None
        return np.mean([matrix(sa.variant(pd.read_parquet(FM_DIR / f"{m}__test__cut{k}__padded.parquet"),
                                          set(), "round"), k) for m in fms], axis=0)

    rows = []
    add = lambda k, name, s: rows.append({"model": name, "cut": k, "variant": a.post, "score": s})
    ref = [score(k, ctx[k]["Fl"]) for k in range(4)]
    med = [score(k, ctx[k]["Fmed"]) for k in range(4)]
    FM = {k: fm_matrix(k) for k in range(4)}
    fmref = [score(k, FM[k]) for k in range(4)] if fms else None
    for k in range(4):
        add(k, f"lgbm_{a.variant}", ref[k]); add(k, "median13", med[k])
        if fms:
            add(k, f"fm_ensemble({'+'.join(fms)})", fmref[k])
            add(k, "d40 0.3*lgbm+0.7*fm", score(k, 0.3 * ctx[k]["Fl"] + 0.7 * FM[k]))
    print(f"LightGBM variant {a.variant}; neural post-processing {a.post}; neural models {models}; foundation set {fms}\n")
    print(f"{'lgbm_' + a.variant:18s}" + " ".join(f"{x:7.2f}" for x in ref) + f"  mean {np.mean(ref):.2f}")
    print(f"{'median13':18s}" + " ".join(f"{x:7.2f}" for x in med) + f"  mean {np.mean(med):.2f}")
    if fms:
        print(f"{'fm ensemble':18s}" + " ".join(f"{x:7.2f}" for x in fmref) + f"  mean {np.mean(fmref):.2f}")

    blends = {}
    for m in models:
        N = {k: matrix(pd.read_parquet(FC / f"{m}__test__cut{k}__padded.parquet"), k) for k in range(4)}
        alone = [score(k, N[k]) for k in range(4)]
        for k in range(4):
            add(k, m, alone[k])
        print(f"\n{m}: alone        " + " ".join(f"{x:7.2f}" for x in alone) + f"   mean {np.mean(alone):.2f}")
        others = [("lgbm", lambda k: ctx[k]["Fl"])] + ([("fm", lambda k: FM[k])] if fms else [])
        for label, other in others:
            sc = {w: [score(k, w * N[k] + (1 - w) * other(k)) for k in range(4)] for w in WEIGHTS}
            wbest = max(WEIGHTS, key=lambda w: np.mean(sc[w][1:]))
            for k in range(4):
                add(k, f"{m}+{label} w={wbest}", sc[wbest][k])
            blends[(m, label)] = sc[wbest]
            d = np.array(sc[wbest]) - np.array(ref)
            print(f"{m}+{label}: w={wbest} (chosen on cuts 1-3) " + " ".join(f"{x:7.2f}" for x in sc[wbest]) +
                  f"   mean {np.mean(sc[wbest]):.2f}   vs lgbm {np.mean(d):+.2f}, cuts won {int((d > 0).sum())}/4")

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    pt = res.pivot_table(index="model", columns="cut", values="score")
    pt["mean"] = pt.mean(axis=1)
    print("\nmean over cuts (sorted):"); print(pt.sort_values("mean", ascending=False).round(2).to_string())

    if blends:   # paired per-article interval of the best neural blend against the LightGBM, per cut
        (m, label), _ = max(blends.items(), key=lambda kv: np.mean(kv[1]))
        N = {k: matrix(pd.read_parquet(FC / f"{m}__test__cut{k}__padded.parquet"), k) for k in range(4)}
        other = (lambda k: ctx[k]["Fl"]) if label == "lgbm" else (lambda k: FM[k])
        sc = {w: [score(k, w * N[k] + (1 - w) * other(k)) for k in range(4)] for w in WEIGHTS}
        wbest = max(WEIGHTS, key=lambda w: np.mean(sc[w][1:]))
        rng = np.random.default_rng(0)
        print(f"\nbest neural blend {m}+{label} w={wbest} vs lgbm_{a.variant}, paired over articles "
              f"(score points, 95% bootstrap interval):")
        for k in range(4):
            c = ctx[k]
            db = e.score_F(c["M"], post_process(wbest * N[k] + (1 - wbest) * other(k), c["dead"], a.post),
                           c["train"], c["hold"], c["weeks"])["detail"]["mase"]
            dr = e.score_F(c["M"], post_process(c["Fl"], c["dead"], a.post), c["train"], c["hold"], c["weeks"])["detail"]["mase"]
            d = (db - dr).dropna()
            if not len(d):
                continue
            bs = [-100 * rng.choice(d.to_numpy(), len(d)).mean() for _ in range(1000)]
            print(f"  cut {k}: {-100*d.mean():+.2f}  [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")


if __name__ == "__main__":
    main()
