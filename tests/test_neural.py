"""Checks for src/neural/run_nf.py that need no GPU and no neuralforecast.

The failure this guards against is silent misalignment: the .npy rows and the parquet rows must be in
ShopMatrix key order, and the target scale must be the D26 metric scale. Run: .venv/bin/python tests/test_neural.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as f  # noqa: E402
import validate as v  # noqa: E402
from neural import run_nf as rn  # noqa: E402

full = v.load_weekly("test")

# 1) the cut frame is the last 13*k weeks dropped, weeks stay Monday and contiguous
for k in (0, 1, 3):
    df = rn.cut_frame(full, k)
    assert df["week"].max() == full["week"].max() - pd.Timedelta(weeks=13 * k)
df = rn.cut_frame(full, 2)
step = np.diff(np.sort(df["week"].unique())).astype("timedelta64[D]").astype(int)
assert (step == 7).all() and pd.Timestamp(df["week"].min()).day_name() == "Monday"

# 2) the target scale is exactly features_at's D26 scale (the metric's scale), and key order matches
for k in (0, 2, 3):
    M, scale, valid, cut, train, hold = rn.shop_matrix(rn.cut_frame(full, k))
    _, s_ref, valid_ref = f.features_at(M, M.W)
    assert np.allclose(scale, s_ref), f"cut {k}: scale differs from features_at"
    assert np.array_equal(valid, valid_ref)
    assert len(scale) == len(M.keys) == M.n

# 3) the long panel keeps ShopMatrix row order and the scaled target (units / scale)
M, scale, valid, *_ = rn.shop_matrix(rn.cut_frame(full, 0))
panel, idx = rn.panel_for(M, scale, valid)
assert len(panel) == int(valid.sum()) * M.W
assert np.array_equal(idx, np.flatnonzero(valid))
Y = panel["y"].to_numpy().reshape(len(idx), M.W)
assert np.allclose(Y, M.U[idx] / scale[idx, None], atol=1e-6)
assert np.array_equal(panel["unique_id"].to_numpy()[::M.W],
                      (M.keys["store_id"].astype(str) + "|" + M.keys["article_id"].astype(str)).to_numpy()[idx])
assert panel.groupby("unique_id")["ds"].nunique().eq(M.W).all()
assert panel["y"].min() >= 0

# 4) parquet round trip: rows -> matrix is exactly the ShopMatrix key order (the blend/scoring alignment)
weeks = pd.date_range(M.weeks[-1] + pd.Timedelta(weeks=1), periods=rn.H, freq="7D")
rng = np.random.default_rng(0)
F = rng.uniform(0, 5, size=(M.n, rn.H)).astype(np.float32)
F[~valid] = 0.0
back = rn.matrix_from_long(rn.to_long(M, F, weeks), M, weeks)
assert back.shape == F.shape and np.array_equal(back, F.astype(float))
assert list(rn.to_long(M, F, weeks).columns) == ["store_id", "article_id", "week", "forecast"]

# 5) the limit used by smoke runs keeps the first scorable series only
lim, idx_lim = rn.panel_for(M, scale, valid, limit=200)
assert len(idx_lim) == 200 and np.array_equal(idx_lim, np.flatnonzero(valid)[:200])
assert lim["unique_id"].nunique() == 200

# 6) --tag only changes the file stem (grid runs must not overwrite the baseline outputs)
assert rn.stem_for("NHITS", "") == "NHITS"
assert rn.stem_for("NHITS", "_s3000_lr3e-4") == "NHITS_s3000_lr3e-4"
assert rn.stem_for("NBEATS", "_smoke").startswith("NBEATS_smoke")
assert rn.tagged_stem("NHITS", "") == "NHITS"
assert rn.tagged_stem("NHITS", "_s3000_lr3e-4_in104") == "NHITS_s3000_lr3e-4_in104"
pf = rn.output_files(rn.tagged_stem("NHITS", "_x"), "cut0", rn.FC)["parquet"]
pf0 = rn.output_files(rn.tagged_stem("NHITS", ""), "cut0", rn.FC)["parquet"]
assert pf == rn.FC / "NHITS_x__test__cut0__padded.parquet", pf
assert pf0 == rn.FC / "NHITS__test__cut0__padded.parquet", pf0
assert str(pf.parent).endswith("iterations/06-neural/forecasts")

# 7) the pooled arm never sees a week at or after the test cut, and pools exactly test + A B C D
P = rn.build_panels(0, None, "pooled", 0)
test_ids = set(rn.unique_ids(P["M"].keys))
test_rows = P["panel"][P["panel"]["unique_id"].isin(test_ids)]
assert len(test_rows) and (test_rows["ds"] < P["cut"]).all()
assert P["meta"]["shops"] == ["test", "A", "B", "C", "D"]

# 8) the --final setup trains on all 143 weeks and forecasts the 13 weeks after the last
full_final = v.load_weekly("test")
cut_final = full_final["week"].max() + pd.Timedelta(weeks=1)
Mf, *_ = rn.shop_matrix(full_final, cut=cut_final)
assert Mf.W == full_final["week"].nunique() == 143
assert cut_final == full_final["week"].max() + pd.Timedelta(weeks=1)

print("all neural checks passed")
