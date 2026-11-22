"""Regenerate the single-quarter Germany backtest from the transcribed disclosures."""
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

filings = pd.read_csv(HERE / "disclosures.csv")
sales = filings.groupby("year").ytd_eur_m.diff().fillna(filings.ytd_eur_m).to_numpy()
assert len(sales) == 15 and filings.iloc[-1].period == "Q3"
def predictors(target):
    # Last public quarter ends two quarter indices before the target.
    return [sales[target-lag]/sales[target-lag-4]-1 for lag in (2,3)]

indices = range(7,len(sales))
x = np.array([predictors(t) for t in indices])
y = [sales[t]/sales[t-4]-1 for t in indices]
params = dict(PARAMS,num_leaves=2,min_data_in_leaf=3,min_data_in_bin=1,
              feature_fraction=1,bagging_fraction=1,bagging_freq=0,num_threads=1)
model = lgb.train(params,lgb.Dataset(x,label=y),num_boost_round=20)
baseline = sales[12]
forecast = baseline * (1+model.predict([predictors(16)])[0])
actual = 372.6  # Reported 2026 Q1, used only after the single prediction.
error = abs(forecast-actual)
base_error = abs(baseline-actual)
pd.DataFrame([dict(actual_eur_m=actual, forecast_eur_m=forecast,
                   seasonal_naive_eur_m=baseline, absolute_error_pct=100*error/actual)]
             ).to_csv(OUT/"comparison.csv",index=False)
print(forecast,actual,baseline,error,base_error)
fig=plt.figure(figsize=(12,7),facecolor="white")
ink,teal,muted="#183344","#007970","#526675"
fig.text(.07,.90,"Fielmann’s German sales",size=29,weight="bold",color=ink)
fig.text(.07,.827,"What would we have forecast for January–March 2026?",size=20,color=ink)
fig.text(.07,.76,"One quarter of external sales in Germany, using the history available at the end of 2025.",size=13,color=muted)
ax=fig.add_axes([.21,.415,.67,.26])
for i,(v,c) in enumerate([(forecast,teal),(actual,ink)]):
    ax.barh(1-i,v,height=.51,color=c)
    ax.text(v+5,1-i,f"€{v:,.1f}m",va="center",size=16,weight="bold",color=c)
ax.set_yticks([1,0],["Our forecast","Actual sales"],fontsize=14)
ax.set_xlim(0,435)
ax.set_xticks([0,100,200,300,400],["€0","€100m","€200m","€300m","€400m"])
ax.set_axisbelow(True)
ax.grid(axis="x",alpha=.15)
ax.spines[:].set_visible(False)
ax.tick_params(length=0,pad=8)
direction="high" if forecast>actual else "low"
fig.text(.07,.285,f"Our forecast was €{error:.1f} million too {direction} ({error/actual:.1%} of actual sales).",size=20,weight="bold",color=ink)
fig.text(.07,.208,f"Repeating the published Q1 2025 figure of €{baseline:.1f}m would have missed by €{base_error:.1f}m ({base_error/actual:.1%}).",size=13,color=ink)
fig.text(.07,.12,"A retrospective test of our LightGBM component adapted to quarterly sales. Latest available sales period: Q3 2025.",size=10,color=muted)
fig.text(.07,.077,"Fielmann later revised Q1 2025 to €367.1m. The test uses the figure available before 2026.",size=10,color=muted)
fig.text(.07,.035,"Source: Fielmann Group financial reports, Germany segment, 2022–2026",size=10,color=muted,url="https://www.fielmann-group.com/investor-relations/praesentationen-und-publikationen/")
fig.savefig(OUT/"fielmann-germany-forecast.pdf")
