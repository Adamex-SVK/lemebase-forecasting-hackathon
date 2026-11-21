"""Step 0.4: build one weekly table per shop in the common schema.

Follows decisions D1-D7, D21, D24 (see decisions-to-make.md and
iterations/00-baselines/plan.md). Raw files are only read.

Output: data/interim/weekly_{A,B,C,D,test}.parquet with columns
  shop, store_id, article_id, week, units, on_hand, on_hand_missing,
  on_hand_was_negative, out_of_stock_flag
and data/interim/prepare_stats.json.

Run: .venv/bin/python src/prepare.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "hackathon-retail-sets"
OUT = ROOT / "data" / "interim"

# D2: first and last COMPLETE weeks per shop (Mondays)
RANGES = {
    "A": ("2021-04-12", "2024-06-03"),
    "B": ("2025-01-06", "2025-10-20"),
    "C": ("2024-01-01", "2026-04-13"),
    "D": ("2025-06-02", "2026-04-13"),
    "test": ("2020-07-27", "2023-04-17"),
}
COLS = ["shop", "store_id", "article_id", "week", "units", "on_hand",
        "on_hand_missing", "on_hand_was_negative", "out_of_stock_flag"]


def monday(s):
    s = pd.to_datetime(s)
    return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.normalize()


def build_grid(starts, first, last):
    """One row per series per week from the series' start week to `last` (D5).

    `starts` has store_id, article_id, start. Series whose start is after `last`
    are dropped and counted.
    """
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    weeks = pd.date_range(first, last, freq="7D")
    s = starts.copy()
    s["start"] = s["start"].clip(lower=first)
    si = ((s["start"] - first).dt.days // 7).to_numpy()
    n = len(weeks) - si
    keep = n > 0
    dropped = int((~keep).sum())
    s, si, n = s[keep], si[keep], n[keep]
    rep = np.repeat(np.arange(len(s)), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + np.repeat(si, n)
    grid = pd.DataFrame({
        "store_id": s["store_id"].to_numpy()[rep],
        "article_id": s["article_id"].to_numpy()[rep],
        "week": weeks[offs],
    })
    return grid, dropped


def finish(shop, df):
    df["shop"] = shop
    df["on_hand_missing"] = df["on_hand"].isna()
    df["on_hand_was_negative"] = df["on_hand"] < 0
    df["on_hand"] = df["on_hand"].clip(lower=0)
    if "out_of_stock_flag" not in df:
        df["out_of_stock_flag"] = df["on_hand"] == 0  # NaN == 0 is False
    df["out_of_stock_flag"] = df["out_of_stock_flag"].fillna(False).astype(bool)
    df["units"] = df["units"].astype("float64")
    return df[COLS].sort_values(["store_id", "article_id", "week"]).reset_index(drop=True)


def clip_units(df, stats):
    neg = int((df["units"] < 0).sum())
    stats["series_weeks_clipped_at_0"] = neg
    df["units"] = df["units"].clip(lower=0)
    return df


def prep_a():
    first, last = RANGES["A"]
    a = pd.read_parquet(RAW / "set-a/weekly.parquet")
    a["week"] = pd.to_datetime(a["week_start"])
    stats = {"raw_units_total": float(a["units"].sum()), "raw_rows": len(a)}
    a = a[(a["week"] >= first) & (a["week"] <= last)].copy()
    df = pd.DataFrame({
        "store_id": a["store_id"].astype(str), "article_id": a["article_id"].astype(str),
        "week": a["week"], "units": a["units"], "on_hand": np.nan,
        "out_of_stock_flag": ~a["in_stock"],
    })
    df = clip_units(df, stats)
    out = finish("A", df)
    out["on_hand_missing"] = True  # A has no on_hand at all (D21)
    return out, stats


def prep_test():
    first, last = RANGES["test"]
    t = pd.read_parquet(RAW / "test-shop/history.parquet")
    t["week"] = pd.to_datetime(t["week_start"])
    stats = {"raw_units_total": float(t["units"].sum()), "raw_rows": len(t)}
    t = t[(t["week"] >= first) & (t["week"] <= last)].copy()
    df = pd.DataFrame({
        "store_id": "T", "article_id": t["article_id"], "week": t["week"],
        "units": t["units"], "on_hand": t["on_hand"],
    })
    df = clip_units(df, stats)
    return finish("test", df), stats


def prep_b():
    first, last = RANGES["B"]
    sl = pd.read_parquet(RAW / "set-b/sales_lines.parquet")
    bk = pd.read_parquet(RAW / "set-b/baskets.parquet")[["transaction_id", "store_id", "timestamp"]]
    assert bk["transaction_id"].is_unique  # D7: many lines -> one basket
    m = sl.merge(bk, on="transaction_id", how="left", validate="many_to_one")
    assert m["store_id"].notna().all()
    m["store_id"] = m["store_id"].astype(str)
    m["week"] = monday(m["timestamp"])
    stats = {"raw_units_total": float(m["units"].sum()), "raw_rows": len(m)}
    first_sale = m.groupby(["store_id", "article_id"])["week"].min().rename("first_sale")
    ev = pd.read_parquet(RAW / "set-b/stock_events.parquet")
    ev["store_id"] = ev["store_id"].astype(str)
    ev["week"] = monday(ev["timestamp"])
    first_ev = ev.groupby(["store_id", "article_id"])["week"].min().rename("first_event")
    st = pd.concat([first_sale, first_ev], axis=1).loc[first_sale.index]  # series with sales (D5)
    stats["series_without_stock_event"] = int(st["first_event"].isna().sum())
    stats["series_sale_before_first_event"] = int((st["first_sale"] < st["first_event"]).sum())
    st["start"] = st[["first_sale", "first_event"]].min(axis=1)
    starts = st["start"].reset_index()
    grid, dropped = build_grid(starts, first, last)
    stats["series_dropped_start_after_last_week"] = dropped
    wk = m[(m["week"] >= first) & (m["week"] <= last)].groupby(["store_id", "article_id", "week"], as_index=False)["units"].sum()
    df = grid.merge(wk, on=["store_id", "article_id", "week"], how="left")
    df["units"] = df["units"].fillna(0)
    so = ev[(ev["reason"] == "STOCKOUT")][["store_id", "article_id", "week"]].drop_duplicates()
    so["out_of_stock_flag"] = True
    df = df.merge(so, on=["store_id", "article_id", "week"], how="left")
    df["on_hand"] = np.nan  # D21: B has no weekly stock
    df = clip_units(df, stats)
    out = finish("B", df)
    out["on_hand_missing"] = True
    return out, stats


def prep_c():
    first, last = RANGES["C"]
    c = pd.read_parquet(RAW / "set-c/daily.parquet")
    c["d"] = pd.to_datetime(c["date"])
    c["units"] = pd.to_numeric(c["units"])
    c["on_hand"] = pd.to_numeric(c["on_hand"])
    c["week"] = monday(c["d"])
    stats = {"raw_units_total": float(c["units"].sum()), "raw_rows": len(c)}
    # D7 grain check: one row per store x article x day
    assert not c.duplicated(["store_id", "article_id", "d"]).any()
    fr = c.groupby(["store_id", "article_id"])["week"].min().rename("start").reset_index()
    sales_weeks = c.loc[c["units"] != 0].groupby(["store_id", "article_id"])["week"].min().rename("fs").reset_index()
    fr = fr.merge(sales_weeks, on=["store_id", "article_id"], how="left")
    stats["series_first_sale_before_first_row"] = int((fr["fs"] < fr["start"]).sum())
    fr["start"] = fr[["start", "fs"]].min(axis=1)
    grid, dropped = build_grid(fr[["store_id", "article_id", "start"]], first, last)
    stats["series_dropped_start_after_last_week"] = dropped
    cc = c[(c["week"] >= first) & (c["week"] <= last)]
    wk = cc.groupby(["store_id", "article_id", "week"], as_index=False)["units"].sum()
    oh = (cc.sort_values("d").groupby(["store_id", "article_id", "week"], as_index=False)["on_hand"].last())
    df = grid.merge(wk, on=["store_id", "article_id", "week"], how="left").merge(
        oh, on=["store_id", "article_id", "week"], how="left")
    df["units"] = df["units"].fillna(0)
    df = clip_units(df, stats)
    return finish("C", df), stats


def prep_d():
    first, last = RANGES["D"]
    d = pd.read_parquet(RAW / "set-d/sales.parquet")
    d["week"] = monday(d["date"])
    stats = {"raw_units_total": float(d["units"].sum()), "raw_rows": len(d)}
    first_sale = d.groupby(["store_id", "article_id"])["week"].min().rename("first_sale")
    s = pd.read_parquet(RAW / "set-d/stock.parquet")
    s["s"] = pd.to_datetime(s["start_date"])
    s["e"] = pd.to_datetime(s["end_date"].astype(str).str.replace("9999-12-31", "2262-01-01"), errors="coerce")
    universe = first_sale.index  # series with at least one sale (D5)
    stock_series = s.groupby(["store_id", "article_id"]).size().index
    stats["stock_only_series_excluded"] = int((~stock_series.isin(universe)).sum())
    first_stock = (s.assign(w=monday(s["s"])).groupby(["store_id", "article_id"])["w"].min().rename("first_stock"))
    st = pd.concat([first_sale, first_stock], axis=1).loc[universe]
    stats["series_without_stock_record"] = int(st["first_stock"].isna().sum())
    stats["series_sale_before_first_stock"] = int((st["first_sale"] < st["first_stock"]).sum())
    st["start"] = st[["first_sale", "first_stock"]].min(axis=1)
    grid, dropped = build_grid(st["start"].reset_index(), first, last)
    stats["series_dropped_start_after_last_week"] = dropped
    dd = d[(d["week"] >= first) & (d["week"] <= last)]
    wk = dd.groupby(["store_id", "article_id", "week"], as_index=False)["units"].sum()
    df = grid.merge(wk, on=["store_id", "article_id", "week"], how="left")
    df["units"] = df["units"].fillna(0)
    # on_hand at week end (Sunday): the stock range covering that day (ranges do not overlap)
    ss = s.set_index(["store_id", "article_id"]).loc[lambda x: x.index.isin(universe)].reset_index()
    ss = ss[["store_id", "article_id", "s", "e", "on_hand"]]
    g = df[["store_id", "article_id", "week"]].copy()
    g["we"] = g["week"] + pd.Timedelta(days=6)
    mg = g.merge(ss, on=["store_id", "article_id"], how="inner")
    mg = mg[(mg["s"] <= mg["we"]) & (mg["e"] >= mg["we"])]
    assert not mg.duplicated(["store_id", "article_id", "week"]).any()
    df = df.merge(mg[["store_id", "article_id", "week", "on_hand"]], on=["store_id", "article_id", "week"], how="left")
    df = clip_units(df, stats)
    return finish("D", df), stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allstats = {}
    for name, fn in [("A", prep_a), ("B", prep_b), ("C", prep_c), ("D", prep_d), ("test", prep_test)]:
        df, stats = fn()
        stats.update({
            "rows": len(df), "series": int(df.groupby(["store_id", "article_id"]).ngroups),
            "weeks": int(df["week"].nunique()), "first_week": str(df["week"].min().date()),
            "last_week": str(df["week"].max().date()), "units_total_kept": float(df["units"].sum()),
            "zero_rate": float((df["units"] == 0).mean()),
            "on_hand_missing_rate": float(df["on_hand_missing"].mean()),
            "on_hand_negative_clipped": int(df["on_hand_was_negative"].sum()),
            "out_of_stock_flag_rate": float(df["out_of_stock_flag"].mean()),
        })
        allstats[name] = stats
        df.to_parquet(OUT / f"weekly_{name}.parquet", index=False)
        print(name, {k: stats[k] for k in ("rows", "series", "weeks", "first_week", "last_week", "zero_rate")}, flush=True)
    (OUT / "prepare_stats.json").write_text(json.dumps(allstats, indent=2))


if __name__ == "__main__":
    main()
