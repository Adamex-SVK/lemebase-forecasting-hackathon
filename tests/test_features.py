"""Checks for src/features.py. Run: .venv/bin/python tests/test_features.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as f  # noqa: E402
import validate as v  # noqa: E402
from rich import RICH_COLS  # noqa: E402


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

# 8) rich features (iteration 04): defaults bit-identical, no leakage, missing table, hand-computed values
import hashlib  # noqa: E402
# SHA-256 of the default-path output, computed on this synthetic data BEFORE the rich-feature edit of src/features.py
HASH_FEATURES_AT_30 = "b26736279e2bee50aa1a9831c97b9264f4244ac265380a0a185648dddfa17055"   # features_at(M, 30)[0].tobytes()
HASH_WINDOWS_X = "46284b1735a53e2d30de909a86f4ac61b99bfb706fed01306d112441047ae85a"        # build_windows(M, make_origins(M.W, 2))["X"].tobytes()
assert hashlib.sha256(f.features_at(M, 30)[0].tobytes()).hexdigest() == HASH_FEATURES_AT_30, "default features_at changed"
assert hashlib.sha256(f.build_windows(M, origins)["X"].tobytes()).hexdigest() == HASH_WINDOWS_X, "default build_windows changed"
assert f.feature_names() == f.FEATURES and f.feature_names(rich=True) == f.FEATURES[:-1] + f.RICH_FEATURES + ["h"]


def synthetic_rich(df, seed=5):
    rng = np.random.default_rng(seed)
    r = df[["store_id", "article_id", "week"]].copy()
    n = len(r)
    r["price"] = rng.uniform(1, 20, n)
    r["disc_share"] = rng.uniform(0, 0.5, n)
    r["promo_flag"] = (rng.random(n) < 0.2).astype(float)
    r["markdown_level"] = rng.integers(0, 4, n).astype(float)
    for c in ("price", "disc_share"):
        r.loc[rng.random(n) < 0.1, c] = np.nan
    extra = r.head(3).copy(); extra["article_id"] = "zzz_not_in_df"   # keys absent from df are ignored
    return pd.concat([r, extra], ignore_index=True)


rich_df = synthetic_rich(df)
Mr1 = f.ShopMatrix(df, rich=rich_df)
assert Mr1.has_rich and not M.has_rich and set(Mr1.R) == set(RICH_COLS)
assert all(a.shape == (40, 70) and a.dtype == np.float32 for a in Mr1.R.values())

# (a) leakage: overwrite everything at or after o in the rich matrices
for o in (13, 30, 57):
    X1, s1, _ = f.features_at(Mr1, o, rich=True)
    assert X1.shape[1] == len(f.feature_names(True)) - 1
    M2 = f.ShopMatrix(df, rich=rich_df)
    rng = np.random.default_rng(7)
    for c in RICH_COLS:
        M2.R[c][:, o:] = rng.uniform(-50, 500, size=M2.R[c][:, o:].shape)
    X2, s2, _ = f.features_at(M2, o, rich=True)
    assert np.array_equal(X1, X2, equal_nan=True) and np.array_equal(s1, s2), f"future rich values leak at origin {o}"
    Xd = f.features_at(Mr1, o)[0]   # base columns are unchanged by rich=True
    assert np.array_equal(X1[:, :Xd.shape[1]], Xd, equal_nan=True)

# (c) no rich table: rich features all NaN, rich_missing == 1
Xn, _, _ = f.features_at(M, 30, rich=True)
rn = f.feature_names(True)[:-1]
for name in f.RICH_FEATURES:
    col = Xn[:, rn.index(name)]
    if name == "rich_missing":
        assert (col == 1).all()
    else:
        assert np.isnan(col).all(), name

# (d) hand-computed promo4 / weeks_since_promo for one series
full = df.groupby("article_id")["week"].min()
aid = full[full == df["week"].min()].index[0]               # a series present in every week
hand = df[["store_id", "article_id", "week"]].copy()
hand["price"] = 10.0; hand["disc_share"] = 0.0; hand["markdown_level"] = 0.0; hand["promo_flag"] = 0.0
wk = lambda j: M.weeks[j]
for j in (20, 27):
    hand.loc[(hand["article_id"] == aid) & (hand["week"] == wk(j)), "promo_flag"] = 1.0
Mh = f.ShopMatrix(df, rich=hand)
Xh, _, _ = f.features_at(Mh, 30, rich=True)
i = int(Mh.keys.index[Mh.keys["article_id"] == aid][0])
assert Xh[i, rn.index("promo4")] == 0.25                      # weeks 26..29: one promo week (27)
assert abs(Xh[i, rn.index("promo13")] - 2 / 13) < 1e-6        # weeks 17..29: weeks 20 and 27
assert Xh[i, rn.index("weeks_since_promo")] == 2              # last promo at 27, last history week 29
assert Xh[i, rn.index("rich_missing")] == 0 and Xh[i, rn.index("price_ratio4")] == 1.0
Xe, _, _ = f.features_at(Mh, 20, rich=True)                    # origin 20: promos at >= 20 are the future
assert Xe[i, rn.index("promo4")] == 0 and Xe[i, rn.index("weeks_since_promo")] == f.DEAD_CAP
Xe2, _, _ = f.features_at(Mh, 21, rich=True)
assert Xe2[i, rn.index("weeks_since_promo")] == 0 and Xe2[i, rn.index("promo4")] == 0.25
# a series that is all-NaN over o-13..o-1 is flagged, others are not
nanh = hand.copy()
nanh.loc[(nanh["article_id"] == aid) & (nanh["week"] >= wk(17)), RICH_COLS] = np.nan
Xq, _, _ = f.features_at(f.ShopMatrix(df, rich=nanh), 30, rich=True)
assert Xq[i, rn.index("rich_missing")] == 1 and Xq[:, rn.index("rich_missing")].sum() >= 1
assert (np.delete(Xq[:, rn.index("rich_missing")], i) == 0).sum() > 0

# zero-price series: price_ratio4 is NaN (never inf), and no rich column is ever inf
zp = hand.copy()
zp.loc[zp["article_id"] == aid, "price"] = 0.0
Mz0 = f.ShopMatrix(df, rich=zp)
for o in (13, 30, 57):
    Xz, _, _ = f.features_at(Mz0, o, rich=True)
    assert np.isnan(Xz[i, rn.index("price_ratio4")]) and not np.isinf(Xz).any()
    assert not np.isinf(f.features_at(Mr1, o, rich=True)[0]).any()

# (e) window-level rich dropout and model plumbing
rorig = f.make_origins(Mr1.W)[:6]
rnames = f.feature_names(True)
rich_cols_idx = [rnames.index(c) for c in f.RICH_FEATURES]
miss_idx = rnames.index("rich_missing")
oth_idx = [i for i in range(len(rnames)) if i not in rich_cols_idx]
W0 = f.build_windows(Mr1, rorig, rich=True)
assert W0["X"].shape[1] == len(rnames) and set(np.unique(W0["X"][:, -1])) == set(range(1, 14))
Wz = f.build_windows(Mr1, rorig, rich=True, rich_drop=0.0)
assert np.array_equal(W0["X"], Wz["X"], equal_nan=True), "rich_drop=0 must be the identity"
W1 = f.build_windows(Mr1, rorig, rich=True, rich_drop=1.0)
assert np.array_equal(W1["X"][:, oth_idx], W0["X"][:, oth_idx], equal_nan=True)
assert (W1["X"][:, miss_idx] == 1).all()
assert all(np.isnan(W1["X"][:, rnames.index(c)]).all() for c in f.RICH_FEATURES if c != "rich_missing")
Wa = f.build_windows(Mr1, rorig, rich=True, rich_drop=0.5, rng=np.random.default_rng(3))
Wb = f.build_windows(Mr1, rorig, rich=True, rich_drop=0.5, rng=np.random.default_rng(3))
assert np.array_equal(Wa["X"], Wb["X"], equal_nan=True), "same seed must give the same mask"
mk = np.isnan(Wa["X"][:, rnames.index("price_ratio4")]) | (Wa["X"][:, miss_idx] == 1)
grp = pd.Series(mk).groupby([Wa["series"], Wa["origin"]]).agg(["min", "max", "size"])
assert (grp["min"] == grp["max"]).all() and (grp["size"] == 13).all(), "all 13 h rows of a window are masked alike"
Wn = f.build_windows(Mr1, f.make_origins(Mr1.W), rich=True, rich_drop=0.5, rng=np.random.default_rng(5))
share = (Wn["X"][:, miss_idx] == 1).mean()
assert abs(share - 0.5) < 0.05, share
# no leakage into base columns: rich=False path unchanged by the new arguments
assert np.array_equal(f.build_windows(Mr1, rorig)["X"], f.build_windows(Mr1, rorig, rich=False, rich_drop=0.7)["X"], equal_nan=True)
Xi_r, _, _ = f.inference_features(Mr1, rich=True)
assert Xi_r.shape == (Mr1.n * 13, len(rnames))

import train_gbm as tg  # noqa: E402
wk = f.make_origins(M.W)[:4]
wtr = f.build_windows(M, wk, rich=True, rich_drop=0.5, rng=np.random.default_rng(1))
tiny = dict(tg.PARAMS, num_threads=2, verbose=-1, min_data_in_leaf=5)
mdl, _ = tg.fit(wtr, None, rounds=5, params=tiny, names=rnames)
Fc = tg.forecast_matrix(mdl, M, pd.date_range("2025-01-01", periods=13, freq="7D"), rich=True)
assert Fc.shape == (M.n, 13) and np.isfinite(Fc).all() and (Fc >= 0).all()

# (f) rich + d33 together: names, masking and widths follow feature_names(True, True)
dn = f.feature_names(True, True)
assert dn[-1] == "h" and f.feature_names() is f.FEATURES and f.feature_names(True) == f.FEATURES[:-1] + f.RICH_FEATURES + ["h"]
assert f.feature_names(False, True) == f.FEATURES_D33
Wd0 = f.build_windows(Mr1, rorig, d33=True, rich=True)
Wd1 = f.build_windows(Mr1, rorig, d33=True, rich=True, rich_drop=1.0)
assert Wd0["X"].shape[1] == len(dn) and Wd1["X"].shape[1] == len(dn)
rset = {dn.index(c) for c in f.RICH_FEATURES if c != "rich_missing"}
for j in range(len(dn)):
    if j in rset:
        assert np.isnan(Wd1["X"][:, j]).all(), dn[j]
    elif dn[j] == "rich_missing":
        assert (Wd1["X"][:, j] == 1).all()
    else:
        assert np.array_equal(Wd1["X"][:, j], Wd0["X"][:, j], equal_nan=True), dn[j]
assert not np.isnan(Wd1["X"][:, dn.index("cover_last")]).all()
assert f.build_windows(Mr1, [], d33=True, rich=True)["X"].shape == (0, len(dn))
assert f.build_windows(Mr1, [])["X"].shape == (0, len(f.FEATURES))
assert f.inference_features(Mr1, d33=True, rich=True)[0].shape[1] == len(dn)

print("all feature checks passed")
