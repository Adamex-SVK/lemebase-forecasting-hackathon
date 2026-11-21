"""Reproduce the LightGBM forecast of the real future weeks (the base of our submissions).

    .venv/bin/python src/make_lgbm_final.py            # writes submissions/lgbm_own_repro.csv and compares with lgbm_own_v1.csv

Recipe (iteration 01, step 1.2): LightGBM, L1 loss on units / MASE scale (D29), features from src/features.py (default set),
trained on ALL test-shop weeks up to 2023-04-17 with windows every 2 weeks, 276 trees (1.1 x the best iteration 250 found by early
stopping on the cut-0 training part), seed 0. `submissions/lgbm_own_v1.csv` (live -7.6) was produced this way; the foundation-model
ensemble file blends 30% of it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as f  # noqa: E402
import make_submission as ms  # noqa: E402
import train_gbm as g  # noqa: E402
import validate as v  # noqa: E402

if __name__ == "__main__":
    df = v.load_weekly("test")
    M = f.ShopMatrix(df)                                       # all weeks, up to 2023-04-17: no holdout
    windows = f.build_windows(M, f.make_origins(M.W))
    model, _ = g.fit(windows, None, rounds=276)
    F = g.forecast_matrix(model, M, pd.date_range("2023-04-24", periods=13, freq="7D"))
    ids = list(M.keys["article_id"])
    path = ms.SUB_DIR / "lgbm_own_repro.csv"
    ms.write_submission(F, ids, path)
    ms.validate_submission(path)
    ref = ms.SUB_DIR / "lgbm_own_v1.csv"
    if ref.exists():
        a = pd.read_csv(path).set_index("article_id"); b = pd.read_csv(ref).set_index("article_id").loc[a.index]
        print(f"max |difference| against lgbm_own_v1.csv: {np.abs(a.to_numpy() - b.to_numpy()).max():.6f}")
