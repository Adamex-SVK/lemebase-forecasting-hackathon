"""Refit the best rich arm once (pooled_rich_nodrop, cut 0, seed 0) and print gain importance of the
rich features. Results CSVs do not store the model, so this is a one-off refit for the notes.

Run: .venv/bin/python iterations/04-rich-columns/feature_importance_rich.py [--seed 0] [--cut 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import features as f  # noqa: E402
import run_arms as ra  # noqa: E402
import run_rich as rr  # noqa: E402
import train_gbm as g  # noqa: E402
import validate as v  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cut", type=int, default=0)
    a = ap.parse_args()
    seed, k = a.seed, a.cut
    rich = True
    full = v.load_weekly(rr.SHOP)
    df = rr.truncate(full, k)
    parts = rr.donor_windows(rich, 0.0, seed, ra.OTHER_STEP)  # pooled_rich_nodrop: rich, no dropout
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=f.H, freq="7D")
    w_x = f.build_windows(M, f.make_origins(M.W, step=2), rng=np.random.default_rng(seed), rich=rich)
    fullwin = ra.merge([(rr.SHOP, w_x)] + parts, "equal")
    del w_x
    names = f.feature_names(rich)
    model, _ = g.fit(fullwin, None, rounds=rr.POOLED_TREES, params={**g.PARAMS, "seed": seed}, names=names)
    imp = pd.Series(model.feature_importance("gain"), index=names)
    share = 100 * imp / imp.sum()
    print(f"pooled_rich_nodrop cut {k} seed {seed}: {len(fullwin['y']):,} rows, {rr.POOLED_TREES} trees")
    print("\nAll features by gain share (%):")
    print(share.sort_values(ascending=False).round(2).to_string())
    print("\nRich features by gain share (%):")
    print(share[f.RICH_FEATURES].sort_values(ascending=False).round(2).to_string())
    print(f"\nRich features combined: {share[f.RICH_FEATURES].sum():.1f}% of total gain")


if __name__ == "__main__":
    main()
