"""Run from repo root with the three publisher CSVs under /tmp/brammibal_*.csv."""
from pathlib import Path
import io
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
sys.path.insert(0, str(ROOT / "src"))
import features as f
import train_gbm as gb
import validate as v

def load(kind):
    return pd.read_csv(f"/tmp/brammibal_{kind}.csv")

def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=200, facecolor="white")
    plt.close(fig)

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
source = subprocess.check_output(["git", "show", "ca3a0ed:iterations/10-external-info/x1/results_adamex-svk_chronos2_ft_ft.csv"])
final = pd.read_csv(io.BytesIO(source)).set_index("variant").loc["[ft] D40 with it as 4th member"]
base = pd.read_csv(ROOT / "iterations/02-robust-eval/results_rolling_test.csv")
cuts = base.pivot(index=["cut_index","first_holdout_week"], columns="model", values="score")
scores = cuts[["autoarima","topdown_mstl"]].copy()
scores["ensemble"] = [final[f"cut{i}"] for i, date in scores.index]
scores.to_csv(OUT / "hackathon_scores.csv")
mase = 1 - scores.mean()/100
fig, ax = plt.subplots(figsize=(10,5.3))
fig.subplots_adjust(left=.2, right=.93, bottom=.26, top=.78)
ax.barh(["AutoARIMA","Top-down MSTL","Our model"], mase, color=["#8596a6","#8596a6","#007b70"], height=.57)
for i,val in enumerate(mase):
    ax.text(val+.017,i,f"{val:.3f}",va="center",weight="bold")
ax.set_xlim(0, max(mase)*1.13)
ax.set_xlabel("Mean absolute scaled error   ·   lower is better")
fig.text(.07,.9,"Our model beat both baselines on the target shop",fontsize=19,weight="bold")
fig.text(.07,.82,"Four historical 13-week forecasts, weighted equally",color="#485566")
fig.text(.07,.08,"Source: project backtests and PR 20.",fontsize=10)
save(fig,"hackathon")

test = load("test").merge(load("truth"), on="row_id", validate="one_to_one")
assert test.sales.notna().all() and len(test)==len(load("test"))==len(load("truth"))
d = pd.concat([load("train"),test],ignore_index=True)
d.date = pd.to_datetime(d.date)
raw = d.pivot(index="date",columns="store",values="sales")
end = raw.index.max() - pd.Timedelta(days=(raw.index.max().weekday()+1)%7)
cut = end - pd.Timedelta(days=90)
days = pd.date_range(end=cut-pd.Timedelta(days=1), periods=728)
hist = raw.reindex(days)
eligible = [s for s in hist if pd.notna(hist[s].iloc[0]) and hist[s].count() >= .98*len(hist)]
hist = hist[eligible].copy()
missing = hist.isna().sum()
for i in range(7,len(hist)):
    hist.iloc[i] = hist.iloc[i].fillna(hist.iloc[i-7])
assert not hist.isna().any().any()
weekly = hist.resample("W-MON",label="left",closed="left").sum()
# Affine translation to the nonnegative domain expected by project features.
shift = weekly.min().clip(upper=0).abs()
def long(frame, value):
    return frame.rename_axis(index="week",columns="article_id").stack().rename(value).reset_index().assign(store_id="Brammibals")
train = long(weekly+shift,"units").assign(on_hand_missing=True,on_hand=np.nan,out_of_stock_flag=False)
M = f.ShopMatrix(train)
model,_ = gb.fit(f.build_windows(M,f.make_origins(M.W)),rounds=276)
weeks = pd.date_range(cut,periods=13,freq="7D")
pred = pd.DataFrame(gb.forecast_matrix(model,M,weeks).T,index=weeks,columns=M.keys.article_id) - shift
baseline = weekly.reindex(weeks-pd.Timedelta(weeks=52)).set_axis(weeks)
assert baseline.notna().all().all()
actual_days = raw.loc[cut:end,eligible]
assert actual_days.shape==(91,len(eligible)) and actual_days.notna().all().all()
actual = actual_days.resample("W-MON",label="left",closed="left").sum()
records = long(actual,"actual").merge(long(pred,"model")).merge(long(baseline,"last_year"))
records.to_csv(OUT/"retailer_forecasts.csv",index=False)
metrics = {}
for label, forecasts in [("LightGBM",pred),("Same week last year",baseline)]:
    result = v.score(long(weekly,"units"),long(actual,"units"),long(forecasts,"forecast"),detail=True)
    metrics[label] = result["detail"].reset_index().set_index("article_id").mase
report = pd.DataFrame(metrics)
report.loc["Equal-store mean"] = report.mean()
report["error_reduction_pct"] = 100*(1-report.LightGBM/report["Same week last year"])
report.to_csv(OUT/"retailer_scores.csv")
print(report,missing,days[0],cut,end,mase)
# Compare all methods on the same five stores. Each store gets equal weight.
median_sales = weekly.tail(13).median()
median_forecast = pd.DataFrame([median_sales] * len(weeks), index=weeks)
median_result = v.score(long(weekly, "units"), long(actual, "units"),
                        long(median_forecast, "forecast"), detail=True)
errors = pd.Series({
    "Our model": report.loc["Equal-store mean", "LightGBM"],
    "Recent 13-week median": median_result["mase"],
    "Same week last year": report.loc["Equal-store mean", "Same week last year"],
})
relative = errors / errors["Recent 13-week median"] * 100
pd.DataFrame({"mean_scaled_error": errors, "relative_error": relative}).to_csv(
    OUT / "retailer_baselines.csv", index_label="method")
fig, ax = plt.subplots(figsize=(11, 5.8))
fig.subplots_adjust(left=.27, right=.92, top=.70, bottom=.24)
fig.text(.07, .89, "Brammibal’s Donuts", fontsize=26, weight="bold", color="#19354A")
fig.text(.07, .79, "Berlin-based doughnut retailer   ·   Five stores   ·   13-week forecast", fontsize=16, color="#354A56")
bars = ax.barh(relative.index, relative, height=.52, color=["#007970", "#8295A4", "#8295A4"])
ax.invert_yaxis()
ax.bar_label(bars, labels=[f"{n:.1f}".removesuffix(".0") for n in relative],
             padding=7, fontsize=14, weight="bold")
ax.set_xlim(0, relative.max()*1.13)
ax.set_xticks([0, 50, 100, 150, 200])
ax.tick_params(length=0, pad=7)
ax.spines[:].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="x", alpha=.14)
ax.set_xlabel("Relative forecast error (recent median = 100; lower is better)", labelpad=10)
fig.text(.07, .105, f"Our model had {100-relative['Our model']:.1f}% less error than the recent median.",
         fontsize=16, weight="bold", color="#19354A")
fig.text(.07, .043, "Model tested: LightGBM component   ·   Data: Green-AI Hub Mittelstand", fontsize=10, color="#475966")
save(fig, "brammibals-forecast-error")
