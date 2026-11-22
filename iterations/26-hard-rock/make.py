"""Make the fixed-origin forecast and its presentation PDF."""
from pathlib import Path
import sys

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE.parents[1] / "src"))
from train_gbm import PARAMS

data = pd.read_json(OUT / "filings.json").set_index("obligation_end_date_yyyymmdd")
data.index = pd.to_datetime(data.index).to_period("M")
data = data.sort_index()
assert data.index.equals(pd.period_range("2022-01", "2026-03", freq="M"))
categories = ["beer_receipts", "wine_receipts", "liquor_receipts"]
assert data[categories + ["cover_charge_receipts"]].sum(axis=1).equals(data.total_receipts)
train = data.loc[:"2025-12"]
assert len(train) == 48 and train.cover_charge_receipts.eq(0).all()

def inputs(history, horizon):
    scale = np.abs(np.diff(history)).mean()
    assert scale > 0
    previous_year = history[-12 + horizon - 1]
    values = [previous_year, history[-1], history[-3:].mean(), history[-12:].mean()]
    recent_growth = history[-3:].mean() / history[-15:-12].mean()
    return [*(np.array(values) / scale), recent_growth, horizon], scale

features, targets = [], []
for name in categories:
    series = train[name].to_numpy()
    for origin in range(15, len(series) - 2):
        for horizon in (1, 2, 3):
            row, scale = inputs(series[:origin], horizon)
            features.append(row)
            targets.append(series[origin + horizon - 1] / scale)

model = lgb.train(dict(PARAMS, num_leaves=7, min_data_in_leaf=15, num_threads=1),
                  lgb.Dataset(np.array(features), label=targets), num_boost_round=276)
forecast = np.zeros(3)
for name in categories:
    for h in (1, 2, 3):
        row, scale = inputs(train[name].to_numpy(), h)
        forecast[h - 1] += max(0,model.predict([row])[0] * scale)

months = pd.period_range("2026-01", periods=3, freq="M")
report = pd.DataFrame({"our_forecast": forecast.round().astype(int),
                       "same_month_last_year": train.loc["2025-01":"2025-03","total_receipts"].to_numpy()},
                      index=months)
report["actual"] = data.loc[months,"total_receipts"]
report.to_csv(OUT / "comparison.csv",index_label="month")
error = report[["our_forecast","same_month_last_year"]].sub(report.actual,axis=0).abs().sum() / report.actual.sum()
print(report, error, sep="\n")

fig = plt.figure(figsize=(12,7), facecolor="white")
ink, teal, muted = "#183344", "#007970", "#526675"
fig.text(.07,.91,"Hard Rock Cafe, San Antonio",fontsize=28,weight="bold",color=ink)
fig.text(.07,.845,"How closely could we forecast beverage receipts for January–March 2026?",fontsize=18,color=ink)
fig.text(.07,.785,"We used four years of sales history through December 2025 to forecast all three months at once.",fontsize=12,color=muted)
ax = fig.add_axes([.12,.32,.83,.36])
x = np.arange(3)
for offset,col,label,color in [(-.18,"our_forecast","Our forecast",teal),(.18,"actual","Actual receipts",ink)]:
    bars=ax.bar(x+offset,report[col],.33,label=label,color=color)
    ax.bar_label(bars,labels=[f"${v:,}" for v in report[col]],padding=5,fontsize=12,fontweight="bold",color=color)
ax.set_ylim(0, report[["our_forecast","actual"]].max().max()*1.20)
ax.set_xticks(x,["January","February","March"],fontsize=13)
ax.set_ylabel("Receipts in US dollars",fontsize=11,color=muted)
ax.yaxis.set_major_formatter(lambda amount,pos:f"${amount/1000:.0f}k")
ax.tick_params(length=0,pad=8)
ax.spines[:].set_visible(False)
ax.grid(axis="y",alpha=.16)
ax.set_axisbelow(True)
ax.legend(loc="lower left",bbox_to_anchor=(-.01,1.02),ncols=2,frameon=False)
fig.text(.07,.202,f"Across the quarter, our monthly forecasts missed by ${report.our_forecast.sub(report.actual).abs().mean():,.0f} on average.",fontsize=17,weight="bold",color=ink)
fig.text(.07,.148,f"The three misses totaled {error.our_forecast:.1%} of actual receipts. Using the same months in 2025 gave {error.same_month_last_year:.1%} error.",fontsize=13,color=ink)
fig.text(.07,.086,"One location on the San Antonio River Walk. Receipts cover alcohol and cover charges. Food sales are excluded.",fontsize=10,color=muted)
fig.text(.07,.048,"Monthly adaptation of our LightGBM component  •  Source: Texas Comptroller, Mixed Beverage Gross Receipts",fontsize=10,color=muted,url="https://data.texas.gov/d/naix-2893")
fig.savefig(OUT / "hard-rock-cafe-forecast.pdf")
