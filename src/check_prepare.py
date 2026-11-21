"""Checks for step 0.4 output. Run: .venv/bin/python src/check_prepare.py"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IM = ROOT / "data" / "interim"
stats = json.loads((IM / "prepare_stats.json").read_text())
shops = ["A", "B", "C", "D", "test"]
dfs = {s: pd.read_parquet(IM / f"weekly_{s}.parquet") for s in shops}
rows = []
for s, df in dfs.items():
    k = ["store_id", "article_id", "week"]
    assert not df.duplicated(k).any(), f"duplicate keys in {s}"
    assert (df["units"] >= 0).all() and (df["on_hand"].dropna() >= 0).all()
    assert (df["week"].dt.weekday == 0).all()
    # complete grid: each series is contiguous weekly from its start to the last week
    g = df.groupby(["store_id", "article_id"])["week"].agg(["min", "max", "size"])
    span = ((g["max"] - g["min"]).dt.days // 7 + 1)
    assert (span == g["size"]).all() and (g["max"] == df["week"].max()).all(), f"gaps in {s}"
    st = stats[s]
    rows.append({
        "shop": s, "rows": len(df), "series": st["series"], "weeks": st["weeks"],
        "first_week": st["first_week"], "last_week": st["last_week"],
        "zero_rate": round(st["zero_rate"], 3), "on_hand_missing": round(st["on_hand_missing_rate"], 3),
        "oos_flag": round(st["out_of_stock_flag_rate"], 3),
        "raw_units": int(st["raw_units_total"]), "kept_units": int(st["units_total_kept"]),
        "kept_pct": round(100 * st["units_total_kept"] / st["raw_units_total"], 1),
        "clipped_weeks": st["series_weeks_clipped_at_0"], "neg_on_hand": st["on_hand_negative_clipped"],
    })
print(pd.DataFrame(rows).to_string(index=False))
print()
for s in shops:
    extra = {k: v for k, v in stats[s].items() if k.startswith("series_") and k != "series" or k.startswith("stock_only")}
    print(s, extra)

# D cold start with the brief's cut (13 weeks ending 2026-04-13)
d = dfs["D"]; cut = pd.Timestamp("2026-01-19")
pre = d[d["week"] < cut].groupby(["store_id", "article_id"])["units"].sum()
post = d[d["week"] >= cut].groupby(["store_id", "article_id"])["units"].sum()
allser = d.groupby(["store_id", "article_id"]).size().index
in_hold = d[d["week"] >= cut].groupby(["store_id", "article_id"]).size().index
exist_pre = d[d["week"] < cut].groupby(["store_id", "article_id"]).size().index
cold = post.index[(post > 0) & (post.index.map(lambda i: pre.get(i, 0)) == 0)]
print(f"\nD cut {cut.date()}: series {len(allser)}; with weeks before the cut {len(exist_pre)}; "
      f"zero units before cut but sold after: {len(cold)}; "
      f"of which present in grid before cut (stock record, history of zeros): {len(cold.intersection(exist_pre))}; "
      f"their share of holdout units: {post[cold].sum() / post.sum():.3f}")
# test shop sanity: identical to raw (no weeks dropped, no clipping other than on_hand)
t = pd.read_parquet(ROOT / "hackathon-retail-sets/test-shop/history.parquet")
tt = dfs["test"].merge(t.rename(columns={"units": "u_raw"}), left_on=["article_id", dfs["test"]["week"].dt.strftime("%Y-%m-%d")], right_on=["article_id", "week_start"])
print("test units identical to raw:", (tt["units"] == tt["u_raw"]).all(), "| rows", len(tt))

fig, ax = plt.subplots(5, 1, figsize=(10, 12))
for a, s in zip(ax, shops):
    tot = dfs[s].groupby("week")["units"].sum()
    a.plot(tot.index, tot.values, lw=1.2); a.set_title(f"Set {s}: weekly total units (complete weeks only)", fontsize=10)
    a.grid(alpha=.3)
plt.tight_layout(); fig.savefig(ROOT / "iterations/00-baselines/weekly_totals.png", dpi=110)
print("plot saved")
