"""Checks for the pure helpers of src/run_rich.py. Run: .venv/bin/python tests/test_run_rich.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_rich as rr  # noqa: E402


def test_truncate_matches_rolling():
    wk = pd.date_range("2022-01-03", periods=60, freq="7D")
    df = pd.DataFrame({"week": wk, "units": 1})
    for k in range(4):
        d = rr.truncate(df, k)
        assert len(d) == 60 - 13 * k
        assert d["week"].max() == wk.max() - pd.Timedelta(weeks=13 * k)


def test_parse_arms():
    assert rr.parse_arms("own,pooled_norich") == ["own", "pooled_norich"]
    try:
        rr.parse_arms("own,bogus")
    except ValueError:
        return
    raise AssertionError("unknown arm accepted")


if __name__ == "__main__":
    test_truncate_matches_rolling(); test_parse_arms(); print("ok")
