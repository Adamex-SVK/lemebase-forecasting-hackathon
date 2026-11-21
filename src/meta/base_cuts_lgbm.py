"""LightGBM (own arm, plain features, seed 0, same recipe as iterations 01/02) at overlapping cuts for the meta-ensemble.

    .venv/bin/python src/meta/base_cuts_lgbm.py

Cut offset W = number of weeks removed from the end of the test-shop history (0, 4, ..., 48); the last 13 weeks of the
remaining history are the holdout. Saves iterations/04-meta-ensemble/base_forecasts/lgbm_w{W}.npy
(rows in ShopMatrix key order = sorted by store_id, article_id of the training part) and median13_w{W}.npy.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import experiments_02 as e  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "iterations" / "04-meta-ensemble" / "base_forecasts"
OFFSETS = list(range(0, 49, 4))

if __name__ == "__main__":
    full = v.load_weekly("test")
    OUT.mkdir(parents=True, exist_ok=True)
    for W in OFFSETS:
        t0 = time.time()
        df = full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=W)]
        M, F, Fmed, train, hold, weeks, best = e.run_lgbm_df(df, False, 0)
        np.save(OUT / f"lgbm_w{W}.npy", F)
        np.save(OUT / f"median13_w{W}.npy", Fmed)
        r = e.score_F(M, np.round(F), train, hold, weeks)
        print(f"W={W:2d}: holdout starts {weeks[0].date()}  train weeks {M.W}  LightGBM (rounded) {r['score']:7.2f}  best_iter {best}  ({time.time()-t0:.0f}s)", flush=True)
    print("DONE")
