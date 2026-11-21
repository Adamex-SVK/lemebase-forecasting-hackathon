"""Additional source checks and plots. Run after audit.py; no training or input writes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lemebase-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
raw = ROOT / "hackathon-retail-sets"
t = pd.read_parquet(raw / "test-shop/history.parquet")
e = pd.read_parquet(raw / "set-b/stock_events.parquet")
z = e[e.reason == "STOCKOUT"]
c = pd.read_parquet(raw / "set-c/daily.parquet")
d = pd.read_parquet(raw / "set-d/sales.parquet")
b = pd.read_parquet(raw / "set-b/baskets.parquet")
ds = pd.read_parquet(raw / "set-d/stock.parquet")
sales_keys = pd.MultiIndex.from_frame(d[["store_id", "article_id"]].drop_duplicates())
stock_keys = pd.MultiIndex.from_frame(ds[["store_id", "article_id"]].drop_duplicates())
errors = pd.read_csv(OUT / "saved_lgbm_per_series.csv")
concentration = {}
for cut, group in errors.groupby("cut"):
    n = max(1, int(len(group)*.1))
    concentration[int(cut)] = dict(top_n=n, n_scored=len(group),
        top10pct_share_scaled_error=float(group.nlargest(n, "lgbm_mase").lgbm_mase.sum()/group.lgbm_mase.sum()))
    legacy = pd.read_csv(ROOT / f"logs/chronos_four/LightGBM_{cut}.csv")
    expected = legacy.loc[legacy.period == "all", "score"].iloc[0]
    assert np.isclose(100*(1-group.lgbm_mase.mean()), expected)
baselines = pd.read_csv(ROOT / "iterations/00-baselines/results.csv")
feasible = pd.read_csv(OUT / "validation_feasibility.csv")
for shop in ("A", "B", "C", "D", "test"):
    for model, col in (("zero", "zero_score"), ("median13", "median13_score")):
        expected = baselines[(baselines.shop == shop) & (baselines.forecast_level == "series") &
                            (baselines.scale_mode == "full") & (baselines.model == model)]
        if len(expected):
            actual = feasible[(feasible.shop == shop) & (feasible.cut == 0)][col].iloc[0]
            assert np.isclose(actual, expected.score.iloc[0]), (shop, model)
info = dict(
    test_fractional_units_rows=int(((t.units.astype(float)%1).abs() > 1e-8).sum()),
    b_stockout_events=len(z), b_stockout_first=str(z.timestamp.min()),
    b_stockout_last=str(z.timestamp.max()),
    c_series_raw_records_stop_before_dataset_end=int((pd.to_datetime(c.date).groupby(
        [c.store_id, c.article_id]).max() < pd.to_datetime(c.date).max()).sum()),
    d_stock_only_series_excluded_by_existing_universe=int((~stock_keys.isin(sales_keys)).sum()),
    raw_date_ranges={"B": [str(b.timestamp.min()), str(b.timestamp.max())],
                     "C": [str(c.date.min()), str(c.date.max())],
                     "D": [str(d.date.min()), str(d.date.max())]},
    error_concentration=concentration,
    saved_forecast_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (ROOT / "logs/chronos_four").glob("LightGBM_*.parquet")},
    checks="Saved LGBM scores reproduced on all four cuts; stored baseline scores reproduced where available.")
(OUT / "supplemental_checks.json").write_text(json.dumps(info, indent=2))

s = pd.read_csv(OUT / "shop_summary.csv")
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
labels = ["Target" if x == "test" else x for x in s.shop]
axes[0].bar(labels, 100*s.zero_share_prepared, color="#4169a1")
axes[0].set(title="Zero-sales weeks", ylabel="% of prepared series-weeks", ylim=(0, 105))
axes[1].bar(labels, s.weeks, color="#258a7a")
axes[1].axhline(52, color="gray", linestyle="--", linewidth=1)
axes[1].set(title="Available complete history", ylabel="Weeks")
axes[2].bar(labels, 100*s.stock_missing_share, color="#a06a42")
axes[2].set(title="Missing numeric stock", ylabel="% of prepared series-weeks", ylim=(0, 105))
fig.savefig(OUT / "dataset_comparison.png", dpi=150)
plt.close(fig)
print(json.dumps(info, indent=2))
