"""Does a foundation model add anything to our LightGBM?  (run on the laptop, where the LightGBM forecasts are)

    .venv/bin/python src/foundation/blend_eval.py

For every foundation model with forecasts on all 4 test-shop cuts, blend w * LightGBM + (1 - w) * model
and score the result with the D26 metric. The weight is chosen on cuts 1-3 and reported on cut 0 and on the
mean of all 4 cuts (same protocol as iteration 02). LightGBM forecasts: the `d33` variant from
src/experiments_02.py (iterations/02-robust-eval/forecasts/d33_cut{k}_seed0.npy, rows in ShopMatrix key order);
use --variant base for the plain one.
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

FM_DIR = ROOT / "iterations" / "03-foundation-model" / "forecasts"
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
H = f.H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base", choices=["base", "d33"])
    ap.add_argument("--post", default="raw", choices=["raw", "round", "zero8", "round+zero8"])
    a = ap.parse_args()
    full = v.load_weekly("test")
    models = sorted({p.stem.split("__")[0] for p in FM_DIR.glob("*__test__cut0__padded.parquet")})
    models = [m for m in models if all((FM_DIR / f"{m}__test__cut{k}__padded.parquet").exists() for k in range(4))]
    ctx = {}
    for k in range(4):
        df = e.cut_frame(full, k)
        train, hold, cut = v.split(df)
        M = f.ShopMatrix(df[df["week"] < cut])
        _, _, dead_keys, _ = sa.prepare_cut("test", k)
        weeks = pd.date_range(cut, periods=H, freq="7D")
        Fl = np.load(ROOT / "iterations" / "02-robust-eval" / "forecasts" / f"{a.variant}_cut{k}_seed0.npy")
        Fmed = np.repeat(np.median(M.U[:, -13:], axis=1, keepdims=True), H, axis=1).astype(float)
        ctx[k] = (M, train, hold, weeks, dead_keys, Fl, Fmed)

    def matrix(model, k):
        M, train, hold, weeks, dead_keys, *_ = ctx[k]
        fc = pd.read_parquet(FM_DIR / f"{model}__test__cut{k}__padded.parquet")
        fc = sa.variant(fc, dead_keys, a.post) if a.post != "raw" else fc
        P = fc.pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(pd.MultiIndex.from_frame(M.keys)).fillna(0.0)
        return P.to_numpy(dtype=float)

    def score(k, F):
        M, train, hold, weeks, *_ = ctx[k]
        return e.score_F(M, F, train, hold, weeks)["score"]

    print(f"LightGBM variant: {a.variant}; foundation post-processing: {a.post}; models with all 4 cuts: {models}\n")
    ref = [score(k, ctx[k][5]) for k in range(4)]
    print("LightGBM alone:", " ".join(f"{x:7.2f}" for x in ref), f"  mean {np.mean(ref):.2f}")
    for m in models:
        FM = {k: matrix(m, k) for k in range(4)}
        alone = [score(k, FM[k]) for k in range(4)]
        sc = {w: [score(k, w * ctx[k][5] + (1 - w) * FM[k]) for k in range(4)] for w in WEIGHTS}
        wbest = max(WEIGHTS, key=lambda w: np.mean(sc[w][1:]))
        bl = sc[wbest]
        print(f"\n{m}: alone       " + " ".join(f"{x:7.2f}" for x in alone) + f"   mean {np.mean(alone):.2f}")
        print(f"{m}: blend w={wbest} (chosen on cuts 1-3) " + " ".join(f"{x:7.2f}" for x in bl) + f"   mean {np.mean(bl):.2f}   vs LightGBM alone {np.mean(bl) - np.mean(ref):+.2f}, cuts won {sum(b > r for b, r in zip(bl, ref))}/4")
        print("   mean over 4 cuts by weight on LightGBM: " + "  ".join(f"{w:.1f}:{np.mean(sc[w]):.2f}" for w in WEIGHTS))


if __name__ == "__main__":
    main()
