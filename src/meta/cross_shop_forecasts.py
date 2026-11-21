"""Cross-shop LightGBM forecasts at the 4 rolling test-shop cuts (for the own-only vs cross-shop check).

    .venv/bin/python src/meta/cross_shop_forecasts.py [--cuts 0,1,2,3] [--seed 0]

The model is the pooled arm of `src/run_rich.py` (`pooled_norich`): test-shop windows before the cut
(origin step 2) plus all weeks of donor shops A-D (origin step 4, equal shop weights, sample_inactive
for D), fixed 300 trees, `src/train_gbm.py` parameters. It saves the forecast matrix as long parquet in
iterations/03-foundation-model/forecasts_lgbm_cross/ so the blend checks can use it. No decision is
implied; this only materialises the cross-shop (own + other shops) LightGBM at the standard cuts.
"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import features as f  # noqa: E402
import run_arms as ra  # noqa: E402
import run_rich as rr  # noqa: E402
import train_gbm as g  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "iterations" / "03-foundation-model" / "forecasts_lgbm_cross"
SHOP = "test"


def pooled_cut_F(df, seed, parts, trees=rr.POOLED_TREES, step_test=2):
    """Same model as run_rich.pooled_cut (rich=False), but returns the forecast long frame."""
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=f.H, freq="7D")
    w_x = f.build_windows(M, f.make_origins(M.W, step=step_test), rng=np.random.default_rng(seed), rich=False)
    full = ra.merge([(SHOP, w_x)] + parts, "equal")
    del w_x
    model, _ = g.fit(full, None, rounds=trees, params={**g.PARAMS, "seed": seed}, names=f.feature_names(False))
    del full
    F = g.forecast_matrix(model, M, weeks, rich=False)
    fc = g.to_long(M, F, weeks)
    del model
    return fc, M, F, weeks


def final_F(full, seed, parts, trees=rr.POOLED_TREES, step_test=2):
    """Cross-shop model trained on ALL test-shop history plus the donors; forecasts the real 13 weeks.

    Same protocol as make_lgbm_final.py for the own model: M on all weeks, origins inside the history,
    forecast weeks 2023-04-24 .. 2023-07-17."""
    M = f.ShopMatrix(full)
    weeks = pd.date_range(full["week"].max() + pd.Timedelta(weeks=1), periods=f.H, freq="7D")
    w_x = f.build_windows(M, f.make_origins(M.W, step=step_test), rng=np.random.default_rng(seed), rich=False)
    panel = ra.merge([(SHOP, w_x)] + parts, "equal")
    del w_x
    model, _ = g.fit(panel, None, rounds=trees, params={**g.PARAMS, "seed": seed}, names=f.feature_names(False))
    del panel
    F = g.forecast_matrix(model, M, weeks, rich=False)
    fc = g.to_long(M, F, weeks)
    del model
    return fc, M, F, weeks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", default="0,1,2,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="forecast the real 13 weeks from all history")
    a = ap.parse_args()
    cuts = [int(x) for x in a.cuts.split(",") if x != ""]
    OUT.mkdir(parents=True, exist_ok=True)
    full = v.load_weekly(SHOP)
    t0 = time.time()
    parts = rr.donor_windows(rich=False, rich_drop=0.0, seed=a.seed, step=ra.OTHER_STEP)
    print("donor windows: " + ", ".join(f"{s} {len(w['y']):,}" for s, w in parts)
          + f" rows ({time.time()-t0:.0f}s)", flush=True)
    if a.final:
        t1 = time.time()
        fc, M, F, weeks = final_F(full, a.seed, parts)
        path = OUT / f"pooled_equal__test__final__seed{a.seed}.parquet"
        fc.to_parquet(path, index=False)
        print(f"final: wrote {path.name} ({len(fc)} rows, mean {F.mean():.3f}, "
              f"{time.time()-t1:.0f}s)", flush=True)
        del fc, M, F, weeks
        gc.collect()
    for k in cuts:
        t1 = time.time()
        df = rr.truncate(full, k)
        fc, M, F, weeks = pooled_cut_F(df, a.seed, parts)
        path = OUT / f"pooled_equal__test__cut{k}__seed{a.seed}.parquet"
        fc.to_parquet(path, index=False)
        print(f"cut {k}: wrote {path.name} ({len(fc)} rows, mean {F.mean():.3f}, "
              f"{time.time()-t1:.0f}s)", flush=True)
        del fc, M, F, weeks
        gc.collect()
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
