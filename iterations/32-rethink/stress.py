"""One older-regime assessment of the frozen inventory-plus-pack model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import evaluate as e

s=e.s
v=e.v
fold=65
df=v.load_weekly("test")
frame=df[df.week <= df.week.max()-pd.Timedelta(weeks=fold)]
train,hold,cut=v.split(frame)
grid=hold.set_index(s.KEY).sort_index().index
hist=train.pivot(index=v.KEYS,columns="week",values="units").sort_index()
assert hist.index.equals(grid.droplevel("week").unique())
sources=e.HERE/"research/stress"
parts=[]
for model in s.FM+["chronos-2"]:
    matches=list(sources.glob(f"{model}*.parquet"))
    if model=="chronos-2":
        matches=[p for p in matches if "ft" in p.name or "stress" in p.name or "dist" in p.name or p.stem=="chronos-2"]
    assert len(matches)==1,(model,matches)
    parts.append(s.matrix(matches[0],grid))
fm=(3*np.mean(parts[:3],axis=0)+parts[3])/4
fm[hist.to_numpy()[:,-8:].sum(axis=1)==0]=0
actual=hold.set_index(s.KEY).loc[grid].units.to_numpy().reshape(-1,13)
rows=[]
reference=None
for model,file in [("incumbent","incumbent"),("history_pack","history")]:
    tree=s.matrix(e.HERE/"history/stress"/f"{file}.parquet",grid)
    raw=np.maximum(0,.7*fm+.3*tree)
    forecast=np.rint(e.project(hist.to_numpy(),raw) if model=="history_pack" else raw)
    res=v.score(train,hold,grid.to_frame(index=False).assign(forecast=forecast.ravel()),detail=True)
    scale=res["detail"].scale.reindex(hist.index).to_numpy()
    ok=np.isfinite(scale)
    errors=np.abs(actual[ok]-forecast[ok])/scale[ok,None]
    metrics=np.array([res["mase"],errors[:,:7].mean(),errors[:,7:].mean()])
    np.testing.assert_allclose(metrics[0],errors.mean())
    if reference is None:
        reference=metrics.copy()
    gain=100*(reference-metrics)
    rows.append(dict(model=model,offset=fold,cut=cut.date(),history_weeks=hist.shape[1],mase=metrics[0],
                     early_mase=metrics[1],late_mase=metrics[2],gain=gain[0],early_gain=gain[1],late_gain=gain[2],
                     scored=ok.sum(),excluded=(~ok).sum(),coverage=res["holdout_units_share_scored"]))
df=pd.DataFrame(rows)
df.to_csv(e.OUT/"stress.csv",index=False)
(e.OUT/"stress_sources.json").write_text(json.dumps(s.HASHES,indent=2))
print(df.to_string(index=False))
