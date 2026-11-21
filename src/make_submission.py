"""Write and validate the submission CSV for the real test shop (step 7 of the pipeline).

Format (from https://lemebase.com/hackathon/submit): CSV with header
    article_id, w1, w2, ..., w13
one row per article (all 2,904), w1 = the week starting Mon 2023-04-24 (the week after the
history ends), w13 = the week starting Mon 2023-07-17. Fractional values are allowed (D17);
we clip at 0. The page does not state whether negative values are allowed.

Dry run (no model, nothing is uploaded): the median of the last 13 weeks of history.
    .venv/bin/python src/make_submission.py
Uploading is done by the user (D25).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import validate as v  # noqa: E402

SUB_DIR = ROOT / "submissions"
H = 13
FIRST_FORECAST_WEEK = pd.Timestamp("2023-04-24")


def history_ids():
    t = pd.read_parquet(ROOT / "hackathon-retail-sets/test-shop/history.parquet", columns=["article_id", "week_start"])
    assert pd.to_datetime(t["week_start"]).max() == FIRST_FORECAST_WEEK - pd.Timedelta(weeks=1)
    return sorted(t["article_id"].unique())


def write_submission(F, article_ids, path):
    """F: array [n_articles x 13] in the order of article_ids."""
    F = np.clip(np.asarray(F, dtype=float), 0, None)
    assert F.shape == (len(article_ids), H) and np.isfinite(F).all()
    df = pd.DataFrame(F, columns=[f"w{i}" for i in range(1, H + 1)])
    df.insert(0, "article_id", list(article_ids))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def validate_submission(path):
    """Raises AssertionError if the file does not match the published format."""
    df = pd.read_csv(path)
    assert list(df.columns) == ["article_id"] + [f"w{i}" for i in range(1, H + 1)], "wrong header"
    ids = history_ids()
    assert len(df) == len(ids) == 2904, f"expected 2904 rows, got {len(df)}"
    assert df["article_id"].is_unique and set(df["article_id"]) == set(ids), "article ids differ from the history"
    vals = df.drop(columns="article_id")
    assert vals.notna().all().all() and np.isfinite(vals.to_numpy()).all(), "missing or non-finite values"
    assert (vals.to_numpy() >= 0).all(), "negative values"
    return True


def median13_forecast():
    """Dry-run forecast: median of the last 13 weeks of history for every article."""
    t = pd.read_parquet(ROOT / "hackathon-retail-sets/test-shop/history.parquet")
    t["w"] = pd.to_datetime(t["week_start"])
    P = t.pivot(index="article_id", columns="w", values="units").sort_index(axis=1)
    ids = list(P.index)
    med = np.median(P.iloc[:, -13:].to_numpy(), axis=1, keepdims=True)
    return ids, np.repeat(med, H, axis=1)


if __name__ == "__main__":
    ids, F = median13_forecast()
    path = SUB_DIR / "dry_run_median13.csv"
    df = write_submission(F, ids, path)
    validate_submission(path)
    print(f"wrote {path} ({len(df)} rows) and it passes the format checks")
    print(df.head(3).to_string(index=False))
