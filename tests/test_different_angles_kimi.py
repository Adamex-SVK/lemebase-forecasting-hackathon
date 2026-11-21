"""Fast action/leakage checks for iteration 09 (Kimi). Run:
  .venv/bin/python tests/test_different_angles_kimi.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "meta"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "foundation"))
import experiments_different_angles_kimi as da  # noqa: E402
import features as f  # noqa: E402
import validate as v  # noqa: E402

H = v.HORIZON
KEYS = v.KEYS


def toy_frame(n_articles=6, n_weeks=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    weeks = pd.date_range("2022-01-03", periods=n_weeks, freq="7D")
    for a in range(n_articles):
        base = rng.integers(0, 8)
        units = np.maximum(0, base + rng.integers(0, 4, size=n_weeks) - 1)
        units[::7] = 0                                    # keep zeros, avoid constant history
        for i, w in enumerate(weeks):
            rows.append({"store_id": "T", "article_id": f"A{a}", "week": w,
                         "units": float(units[i]), "on_hand": float(10 + a),
                         "on_hand_missing": False, "out_of_stock_flag": False})
    return pd.DataFrame(rows)


def test_post_action():
    c = {"dead": np.array([False, True, True])}
    F = np.array([[1.4, 1.5, 1.6], [5.0, 6.0, 7.0], [-1.0, 0.2, 0.3]])
    G = da.post(F, c, dead_zero=True)
    assert G.shape == F.shape
    assert np.all(G == np.round(G)) and np.all(G >= 0)
    assert np.all(G[1] == 0)
    G2 = da.post(F, c, dead_zero=False)
    assert np.all(G2[1] > 0)


def test_score_matches_validate():
    df = toy_frame()
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=H, freq="7D")
    c = da.context_from_frames(df)
    F = np.repeat(np.arange(M.n, dtype=float)[:, None], H, axis=1)
    G = da.post(F, c, dead_zero=False)
    fc = pd.DataFrame({
        "store_id": np.repeat(M.keys["store_id"].to_numpy(), H),
        "article_id": np.repeat(M.keys["article_id"].to_numpy(), H),
        "week": np.tile(np.asarray(weeks), M.n),
        "forecast": G.reshape(-1)})
    ref = v.score(train, hold, fc)["score"]
    assert abs(da.score_G(c, G) - ref) < 1e-9


def test_pooled_windows_drop_h_and_weight():
    df = toy_frame()
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    origins = f.make_origins(M.W)[-3:]
    w = da.pooled_windows(M, origins)
    assert w["X"].shape[1] == len(f.FEATURES) - 1
    assert np.allclose(w["w"], 1.0 / H)
    assert w["n_windows"] == len(origins) * M.n


def test_stump_learns_a_real_split():
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(size=(n, 3))
    loss = np.ones((n, 2))
    high = X[:, 0] > 0.0
    loss[high, 0] = 1.5                                # D40 bad above zero
    loss[high, 1] = 0.5                                # mean26 good above zero
    loss[~high, 0] = 0.5
    loss[~high, 1] = 1.5
    stump = da.fit_stump([{"X": X, "loss": loss, "valid": np.ones(n, bool)}])
    assert stump["feature"] == 0
    assert stump["threshold"] > -0.3 and stump["threshold"] < 0.3
    pick = da.apply_stump(np.array([[1.0, 0, 0], [-1.0, 0, 0]]), stump)
    assert pick.tolist() == [1, 0]


def test_oracle_selects_perfect_candidate():
    A = np.zeros((3, H))
    A[0] = 10
    A[1] = 7
    c = {"A": A, "s": np.ones(3), "valid": np.ones(3, bool),
         "dead": np.zeros(3, bool)}
    levels = {"good": np.array([10.0, 7.0, 0.0]), "bad": np.zeros(3)}
    G = da.oracle_actions(levels, c, ["bad", "good"], dead_zero=True)
    loss = np.abs(G - A[:, None, :]).mean(axis=2)
    assert np.all(loss[:, 1] <= loss[:, 0])
    out = da.oracle_select(c, levels, ["bad", "good"], True)
    assert out["score"] == 100.0 and out["pick"].tolist() == [1, 1, 0]


def test_features_at_ignores_future_weeks():
    df = toy_frame()
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    o = M.W - 5
    X1, s1, ok1 = f.features_at(M, o)
    M2 = f.ShopMatrix(df[df["week"] < cut])
    M2.U[:, o:] = 999.0                                # only weeks at/after the origin
    X2, s2, ok2 = f.features_at(M2, o)
    assert np.array_equal(X1, X2, equal_nan=True)
    assert np.array_equal(s1, s2) and np.array_equal(ok1, ok2)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
