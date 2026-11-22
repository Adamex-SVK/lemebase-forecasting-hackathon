"""Score the two full blends against identical actuals and summarize paired effects."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import validate as v

BASE = ROOT / "iterations/28-pr20-four-shops/results"
OUT = HERE / "results"
KEY = v.KEYS + ["week"]


def read_forecast(path, index):
    frame = pd.read_parquet(path).set_index(KEY).sort_index()
    assert index.is_unique and frame.index.equals(index)
    values = frame.forecast.to_numpy()
    assert np.isfinite(values).all()
    return values


def evaluate(shop):
    train, hold, cut = v.split(v.load_weekly(shop))
    grid = hold.set_index(KEY).sort_index().index
    tree = read_forecast(BASE / f"lgbm_{shop}.parquet", grid)
    fixed = np.mean([read_forecast(BASE / f"{name}_{shop}.parquet", grid)
                     for name in ("moirai-2.0-small", "tirex", "chronos-t5-small")], axis=0)
    recent = train.loc[train.week >= cut - pd.Timedelta(weeks=8)].groupby(v.KEYS).units.sum()
    inactive = recent.reindex(grid.droplevel("week")).to_numpy() == 0

    def blend(adapter):
        foundation = (3 * fixed + adapter) / 4
        foundation[inactive] = 0
        return (grid.to_frame(index=False)
                .assign(forecast=np.maximum(0, np.round(.3 * tree + .7 * foundation))))

    # Reproduce the previous whole-model forecasts exactly before substituting the adapter.
    previous = blend(read_forecast(BASE / f"chronos-2-ft_{shop}.parquet", grid))
    assert np.array_equal(previous.forecast, read_forecast(BASE / f"blend_{shop}.parquet", grid))
    runs = []
    under_errors = []
    actual = hold.set_index(KEY).sort_index().units
    for donors in [shop, "ABCD"]:
        adapter = read_forecast(OUT / donors / f"chronos-2-ft_{shop}.parquet", grid)
        forecast = blend(adapter)
        metrics = v.score(train, hold, forecast, detail=True)
        runs.append(metrics)
        shortfall = (actual - forecast.set_index(KEY).forecast).clip(lower=0)
        scales = metrics["detail"].scale
        mean_shortfall = shortfall.groupby(level=v.KEYS).mean().reindex(scales.index)
        under_errors.append((mean_shortfall / scales).mean())
    own, pooled = runs
    assert own["detail"].index.equals(pooled["detail"].index)
    gain = own["detail"].mase - pooled["detail"].mase
    assert np.isclose(gain.mean() * 100, pooled["score"] - own["score"])
    under_gain = 100 * (under_errors[0] - under_errors[1])
    return dict(underforecast_gain_points=under_gain, overforecast_gain_points=gain.mean()*100 - under_gain,
                shop=shop, cut=cut.date(), own=own["score"], pooled=pooled["score"],
                change=pooled["score"]-own["score"], improved=int((gain>0).sum()),
                worsened=int((gain<0).sum()), tied=int((gain==0).sum()), scored=len(gain),
                no_history=own["n_no_history"], zero_scale=own["n_zero_scale"],
                unit_coverage=own["holdout_units_share_scored"])


rows = pd.DataFrame([evaluate(shop) for shop in "ABCD"])
rows.to_csv(OUT / "paired_scores.csv", index=False)
display = rows.set_index("shop")[["own","pooled","change"]].copy()
display.loc["Average"] = display.mean()
display.to_csv(OUT / "comparison.csv")
print(display.round(4))
fig, ax = plt.subplots(figsize=(11, 5.9))
fig.subplots_adjust(left=.07, right=.95, top=.70, bottom=.15)
ax.axis("off")
fig.text(.07,.89,"Did learning from other shops help?",fontsize=25,weight="bold",color="#173246")
fig.text(.07,.82,"Shared feature: weekly units sold per product at each store",fontsize=16,weight="bold",color="#173246")
fig.text(.07,.754,"Full five-model blend  ·  Final 13 weeks of each shop  ·  Higher score is better",fontsize=13,color="#506676")
table=ax.table(cellText=[[f"Set {name}" if len(name)==1 else "Equal-weight average",
                         f"{r.own:.3f}",f"{r.pooled:.3f}",f"{r.change:+.3f}"]
                        for name,r in display.iterrows()],colLabels=["","Own shop","All four shops","Change (points)"],
               colWidths=[.31,.21,.23,.25],bbox=[0,0,1,1],cellLoc="right",colLoc="right")
table.auto_set_font_size(False)
table.set_fontsize(14)
for (i,j),cell in table.get_celld().items():
    cell.set_linewidth(0)
    cell.set_facecolor("#edf3f5" if i in (0,5) else "white")
    if i in (0,5): cell.get_text().set_weight("bold")
    if j==0: cell.get_text().set_ha("left")
fig.text(.07,.074,"Only the Chronos-2 training data changed. Scores use unrounded values. Set D covers 53.2% of held-out units.",fontsize=10,color="#506676")
fig.savefig(OUT/"borrowing-comparison.png",dpi=180)
