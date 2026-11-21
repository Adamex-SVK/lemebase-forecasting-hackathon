"""Checks for src/features.py. Run: .venv/bin/python tests/test_features.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as f  # noqa: E402
import validate as v  # noqa: E402


def synthetic(n_series=40, n_weeks=70, seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2024-01-01", periods=n_weeks, freq="7D")
    rows = []
    for i in range(n_series):
        start = int(rng.integers(0, 15))
        lam = rng.uniform(0.2, 6)
        for j in range(start, n_weeks):
            oh = float(rng.integers(0, 30))
            rows.append({"store_id": "s", "article_id": f"a{i:03d}", "week": weeks[j], "units": float(rng.poisson(lam)),
                         "on_hand": oh, "on_hand_missing": bool(rng.random() < 0.2), "on_hand_was_negative": False,
                         "out_of_stock_flag": oh == 0})
    return pd.DataFrame(rows)


df = synthetic()
M = f.ShopMatrix(df)
assert M.n == 40 and M.W == 70

# 1) LEAKAGE: features at origin o must not change when everything at or after o changes
for o in (13, 30, 57):
    X1, s1, v1 = f.features_at(M, o)
    M2 = f.ShopMatrix(df)
    rng = np.random.default_rng(1)
    M2.U[:, o:] = rng.integers(0, 500, size=M2.U[:, o:].shape)
    M2.OH[:, o:] = rng.integers(0, 500, size=M2.OH[:, o:].shape)
    M2.OOS[:, o:] = rng.random(M2.OOS[:, o:].shape) < 0.5
    X2, s2, v2 = f.features_at(M2, o)
    assert np.array_equal(X1, X2, equal_nan=True) and np.array_equal(s1, s2), f"future leaks into features at origin {o}"

# 2) targets are the future units divided by the history scale, aligned with h
origins = f.make_origins(M.W, step=2)
assert origins[-1] == M.W - f.H and origins[0] >= f.MIN_HIST and (np.diff(origins) == 2).all()
win = f.build_windows(M, origins)
hcol = f.FEATURES.index("h")
for r in np.random.default_rng(2).integers(0, len(win["y"]), 200):
    i, o, h = win["series"][r], win["origin"][r], int(win["X"][r, hcol])
    scale = np.abs(np.diff(M.U[i, :o].astype(float))).mean()
    assert abs(win["y"][r] - M.U[i, o + h - 1] / scale) < 1e-4
assert win["origin"].max() + f.H <= M.W   # all target weeks inside the data

# 3) windows with scale 0 are skipped
dd = df.copy()
const = dd["article_id"] == "a000"
dd.loc[const, "units"] = 0.0   # all-zero history -> scale 0
M3 = f.ShopMatrix(dd)
w3 = f.build_windows(M3, f.make_origins(M3.W))
assert 0 not in set(w3["series"]) and len(set(w3["series"])) == 39, "all-zero series (scale 0) must be skipped"

# 4) weighted sampling: all active windows kept, inactive ones ~10% with weight 10
dz = df.copy()
dead = dz["article_id"].isin([f"a{i:03d}" for i in range(20, 40)])
dz.loc[dead & (dz["week"] < dz["week"].min() + pd.Timedelta(weeks=60)), "units"] = 0.0
dz.loc[dead & (dz["week"] == dz["week"].min() + pd.Timedelta(weeks=3)), "units"] = 1.0   # keeps scale > 0
Mz = f.ShopMatrix(dz)
ws = f.build_windows(Mz, f.make_origins(Mz.W), sample_inactive=True, rng=np.random.default_rng(3))
assert set(np.unique(ws["w"])) <= {1.0, 10.0} and (ws["w"] == 10.0).any() and (ws["w"] == 1.0).any()

# 5) the scale equals the metric's scale (src/validate.py, D26 padding) on real data (test shop)
real = v.load_weekly("test")
train, hold, cut = v.split(real)
Mr = f.ShopMatrix(real[real["week"] < cut])
X, scale, valid = f.features_at(Mr, Mr.W)
ref = v.history_scale(train, "full").reindex(pd.MultiIndex.from_frame(Mr.keys))
assert np.allclose(scale, ref.to_numpy(), atol=1e-6), "feature scale differs from the metric's scale"
assert Mr.W == 130 and valid.sum() == 2839   # 65 zero-scale articles, as in the metric

# 6) inference rows: n * 13, series-major, h = 1..13 within each series
Xi, si, vi = f.inference_features(Mr)
assert Xi.shape == (Mr.n * 13, len(f.FEATURES)) and (Xi[:13, hcol] == np.arange(1, 14)).all()

# 7) D33 option: no leakage, the regime mask works, the default feature set is unchanged
for o in (13, 40):
    M1 = f.ShopMatrix(df, mask_oh_before="2024-03-04")
    X1, s1, _ = f.features_at(M1, o, d33=True)
    M2 = f.ShopMatrix(df, mask_oh_before="2024-03-04")
    M2.U[:, o:] = 999; M2.OH[:, o:] = 999
    X2, s2, _ = f.features_at(M2, o, d33=True)
    assert np.array_equal(X1, X2, equal_nan=True), "future leaks into the D33 features"
    assert X1.shape[1] == len(f.FEATURES_D33) - 1
Mm = f.ShopMatrix(df, mask_oh_before="2024-04-08")          # week index 14 is 2024-04-08: weeks 0..13 masked
assert np.isnan(Mm.OH[:, :14]).all() and not np.isnan(f.ShopMatrix(df).OH[:, :14]).all()
Xm, _, _ = f.features_at(Mm, 13, d33=True)                   # origin 13: all history masked
names = [n for n in f.FEATURES_D33 if n != "h"]
assert np.isnan(Xm[:, names.index("cover_last")]).all() and np.isnan(Xm[:, names.index("oh_last_log")] ).all()
assert (Xm[:, names.index("oh_missing_last")] == 1).all()
assert f.features_at(M, 30)[0].shape[1] == len(f.FEATURES) - 1   # default unchanged

print("all feature checks passed")
