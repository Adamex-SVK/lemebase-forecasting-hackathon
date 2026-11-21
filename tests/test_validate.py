"""Hand-computed checks of the metric. Run: .venv/bin/python tests/test_validate.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import validate as v  # noqa: E402

H = 2  # short horizon for hand-computable cases


def frame(rows):
    """rows: (store, article, [units per week]) -> weekly frame starting 2024-01-01"""
    out = []
    for s, a, units in rows:
        for i, u in enumerate(units):
            out.append({"store_id": s, "article_id": a, "week": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=i), "units": float(u)})
    return pd.DataFrame(out)


def fc_from(hold, values):
    f = hold[v.KEYS + ["week"]].copy()
    f["forecast"] = values
    return f


# 1) split: last 2 weeks held out, scale from the history [1,3,1,3,1,3] = mean |diff| = 2
df = frame([("s", "x", [1, 3, 1, 3, 1, 3, 1, 3])])
train, hold, cut = v.split(df, horizon=H)
assert cut == pd.Timestamp("2024-02-12") and len(train) == 6 and len(hold) == 2
assert v.history_scale(train).iloc[0] == 2.0

# 2) perfect forecast -> MASE 0, score 100
r = v.score(train, hold, fc_from(hold, hold["units"].to_numpy()), horizon=H)
assert r["mase"] == 0 and r["score"] == 100 and r["n_scored"] == 1

# 3) errors [2, 0] -> mean 1; scale 2 -> MASE 0.5 -> score 50
r = v.score(train, hold, fc_from(hold, [hold["units"].iloc[0] + 2, hold["units"].iloc[1]]), horizon=H)
assert abs(r["mase"] - 0.5) < 1e-12 and abs(r["score"] - 50) < 1e-9

# 4) MASE of exactly 1 -> score 0; error worse than 1 -> negative score
r = v.score(train, hold, fc_from(hold, hold["units"].to_numpy() + 2), horizon=H)  # err 2 / scale 2
assert abs(r["score"]) < 1e-9
r = v.score(train, hold, fc_from(hold, hold["units"].to_numpy() + 4), horizon=H)
assert r["score"] < 0

# 5) average over articles (equal weight): MASE 0.5 and 1.5 -> 1.0 -> score 0
df = frame([("s", "x", [1, 3, 1, 3, 1, 3, 1, 3]), ("s", "y", [2, 4, 2, 4, 2, 4, 2, 4])])
train, hold, _ = v.split(df, horizon=H)
f = hold[v.KEYS + ["week"]].copy()
f["forecast"] = hold["units"] + np.where(hold["article_id"] == "x", 1.0, 3.0)  # err 1/2 and 3/2
r = v.score(train, hold, f, horizon=H)
assert abs(r["mase"] - 1.0) < 1e-12 and r["n_scored"] == 2

# 6) constant history (zero scale) and a series without history are excluded and counted
df = frame([("s", "x", [1, 3, 1, 3, 1, 3, 1, 3]), ("s", "flat", [5] * 8), ("s", "new", [0] * 6 + [4, 4])])
train, hold, _ = v.split(df, horizon=H)
f = hold[v.KEYS + ["week"]].copy(); f["forecast"] = hold["units"]
r = v.score(train, hold, f, horizon=H)
# 'flat' (constant 5) and 'new' (six zeros) both have scale 0 -> zero_scale, not no_history
assert r["n_series"] == 3 and r["n_zero_scale"] == 2 and r["n_scored"] == 1 and r["n_no_history"] == 0

# 7) series that only starts inside the holdout has no training rows -> no history
df = frame([("s", "x", [1, 3, 1, 3, 1, 3, 1, 3])])
late = frame([("s", "late", [4, 4])]); late["week"] += pd.Timedelta(weeks=6)
df = pd.concat([df, late], ignore_index=True)
train, hold, _ = v.split(df, horizon=H, pad=False)
assert len(hold) == 4 and "late" not in set(train["article_id"])
f = hold[v.KEYS + ["week"]].copy(); f["forecast"] = hold["units"]
r = v.score(train, hold, f, horizon=H)
assert r["n_no_history"] == 1 and r["n_scored"] == 1
# D26 (pad=True, default): the late series gets a padded all-zero history -> zero scale, still excluded
train, hold, _ = v.split(df, horizon=H)
assert "late" in set(train["article_id"]) and train["units"].groupby(train["article_id"]).sum()["late"] == 0
r = v.score(train, hold, f, horizon=H)
assert r["n_no_history"] == 0 and r["n_zero_scale"] == 1 and r["n_scored"] == 1

# 7b) padding: a series that starts 3 weeks late is zero-filled from the shop's first week,
#     which lowers its scale: history [0,0,0,2,4,2,4] (not [2,4,2,4])
df = pd.concat([frame([("s", "full", [1, 3, 1, 3, 1, 3, 1, 3, 1])]),
                frame([("s", "late3", [2, 4, 2, 4, 2, 4])]).assign(week=lambda d: d["week"] + pd.Timedelta(weeks=3))], ignore_index=True)
train, hold, _ = v.split(df, horizon=H)
assert abs(v.history_scale(train, "full").loc[("s", "late3")] - 8 / 6) < 1e-12
train_np, _, _ = v.split(df, horizon=H, pad=False)
assert v.history_scale(train_np, "full").loc[("s", "late3")] == 2.0

# 8) from_first_sale ignores leading zeros. Train = [0,0,0,2,4,2,4]:
#    full: diffs 0,0,2,2,2,2 -> 8/6; from first sale [2,4,2,4]: diffs 2,2,2 -> 2
df = frame([("s", "x", [0, 0, 0, 2, 4, 2, 4, 2, 4])])
train, hold, _ = v.split(df, horizon=H)
assert abs(v.history_scale(train, "full").iloc[0] - 8 / 6) < 1e-12
assert v.history_scale(train, "from_first_sale").iloc[0] == 2.0

# 9) level='article' sums over stores first: two stores that offset each other
#    give a constant article total (scale 0 -> excluded), but have scale 2 as separate series
df = frame([("a", "x", [1, 3, 1, 3, 1, 3, 1, 3]), ("b", "x", [3, 1, 3, 1, 3, 1, 3, 1])])
train, hold, _ = v.split(df, horizon=H)
f = hold[v.KEYS + ["week"]].copy(); f["forecast"] = hold["units"]
assert v.score(train, hold, f, level="series", horizon=H)["n_scored"] == 2
r = v.score(train, hold, f, level="article", horizon=H)
assert r["n_series"] == 1 and r["n_zero_scale"] == 1 and r["n_scored"] == 0

print("all metric checks passed")
