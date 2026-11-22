"""Score all five complete keyed component forecasts and render the A–D comparison."""
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
OUT = HERE / "results"
keys = v.KEYS + ["week"]
base = pd.read_csv(ROOT / "iterations/00-baselines/results.csv")
base = base.query("convention == 'D26 leading-zero grid' and forecast_level == 'series' and scale_mode == 'full'")
summary = []
for shop in "ABCD":
    train,hold,cut = v.split(v.load_weekly(shop))
    target = hold.set_index(keys).sort_index()
    expected = target.index
    parts = {}
    for name in ["lgbm","moirai-2.0-small","tirex","chronos-t5-small","chronos-2-ft"]:
        component = pd.read_parquet(OUT/f"{name}_{shop}.parquet").set_index(keys).sort_index()
        assert component.index.is_unique and component.index.equals(expected)
        parts[name] = component.forecast.to_numpy()
        assert np.isfinite(parts[name]).all()
    ave = np.mean([parts[name] for name in ("moirai-2.0-small","tirex","chronos-t5-small")],axis=0)
    combined = (3*ave+parts["chronos-2-ft"])/4
    recent = train[train.week>=cut-pd.Timedelta(weeks=8)].groupby(v.KEYS).units.sum()
    inactive = recent.reindex(expected.droplevel("week")).to_numpy()==0
    combined[inactive] = 0
    forecast = np.maximum(0,np.round(.3*parts["lgbm"]+.7*combined))
    fc = pd.DataFrame({"forecast":forecast},index=expected).reset_index()
    fc.to_parquet(OUT/f"blend_{shop}.parquet",index=False)
    res = v.score(train,hold,fc)
    refs = base[base.shop==shop].set_index("model")
    assert (refs.loc[["autoarima","topdown_mstl"],"n_scored"]==res["n_scored"]).all()
    summary.append(dict(shop=shop,cut=cut.date(),autoarima=refs.loc["autoarima","score"],
                        mstl=refs.loc["topdown_mstl","score"],blend=res["score"],
                        mase=res["mase"],scored=res["n_scored"],excluded=res["n_series"]-res["n_scored"],
                        unit_coverage=res["holdout_units_share_scored"]))
scores = pd.DataFrame(summary).set_index("shop")
scores.to_csv(OUT/"shop_scores.csv")
values = scores[["autoarima","mstl","blend"]].copy()
values.loc["Equal-weight average"]=values.mean()
values.to_csv(OUT/"chart_data.csv")
print(values.round(3), flush=True)

fig,ax=plt.subplots(figsize=(13,7.5))
fig.subplots_adjust(left=.36,right=.93,top=.84,bottom=.14)
fig.text(.055,.932,"Our model across four retail datasets",fontsize=24,weight="bold",color="#243646")
fig.text(.055,.874,"Final PR 20 blend and two baselines  ·  Last 13 weeks of each shop",fontsize=14,color="#526474")
bound=np.ceil(values.max().max()/10)*10+8
for g,(shop,row) in enumerate(values.iterrows()):
    ax.text(-.51,g*4+1,("Set "+shop if len(shop)==1 else "All four, equal weight"),
            transform=ax.get_yaxis_transform(),va="center",fontsize=14,weight="bold",color="#243646")
    for k,(val,label,color) in enumerate(zip(row,["AutoARIMA","Top-down MSTL","Our model"],["#477477","#ca8c30","#23589b"])):
        y=g*4+k
        ax.barh(y,val,height=.69,color=color)
        ax.text(-.018,y,label,transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=10.7)
        ax.text(val+.7,y,f"{val:.1f}",va="center",fontsize=11,weight="bold")
ax.set_ylim(19,-1);ax.set_xlim(0,bound);ax.set_yticks([]);ax.set_xticks(np.arange(0,bound,10))
ax.grid(axis="x",alpha=.16);ax.set_axisbelow(True);ax.spines[:].set_visible(False);ax.tick_params(length=0)
ax.set_xlabel("Competition score   ·   Higher is better   ·   0 corresponds to MASE = 1",fontsize=12)
fig.text(.055,.047,"Local backtests. Set D scored 53.2% of held-out units; series with zero historical variation are excluded.",fontsize=10,color="#526474")
fig.savefig(OUT/"pr20-four-shop-comparison.png",dpi=180)
