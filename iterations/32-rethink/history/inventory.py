"""Compare historical sale-frequency changes across levels of pre-cut ledger movement."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
records = pd.read_parquet(ROOT / "hackathon-retail-sets/test-shop/history.parquet")
def matrix(field):
    return records.pivot(index="article_id", columns="week_start", values=field).sort_index(axis=1)
units = matrix("units")
stock = matrix("on_hand")
assert units.index.equals(stock.index) and units.columns.equals(stock.columns)
rows, matched = [], []
for offset in [0,13,26,39]:
    end = units.shape[1] - 13 - offset
    u = units.iloc[:,end-13:end].to_numpy()
    s = stock.iloc[:,end-14:end].to_numpy()
    y = units.iloc[:,end:end+13].to_numpy()
    n = (u > 0).sum(axis=1)
    observed = (s[:,:-1] >= 0) & (s[:,1:] >= 0)
    flow = np.where(observed, np.maximum(np.diff(s)+u,0),np.nan)
    v = u.sum(axis=1)
    coverage = np.nansum(flow,axis=1)/(v+1)
    eligible = (s[:,-1]>0) & (observed.sum(axis=1)==13) & (n>=1) & (n<=8)
    grp = np.select([coverage < .01, coverage > 1.4], ['low','high'], 'middle')
    d = pd.DataFrame(dict(n=n, band=np.where(n<=3,'1-3','4-8'), grp=grp,
               recent_units=np.digitize(v,[4,16,64]), cover_bin=np.digitize(s[:,-1]/(v+1),[.25,1,4]),
               delta7=(y[:,:7]>0).mean(axis=1)-n/13, delta6=(y[:,7:]>0).mean(axis=1)-n/13,
               delta13=(y>0).mean(axis=1)-n/13, positive_median=(np.median(y,axis=1)>0).astype(float)))[eligible]
    for (band,group),sample in d.groupby(['band','grp']):
        rows.append(dict(cut=units.columns[end],band=band,group=group,N=len(sample),
                         **sample[['delta7','delta6','delta13','positive_median']].mean().to_dict()))
    # Match by exact number of sale weeks, pooled volume bin, stock cover bin.
    # Each shared stratum's weight is the smaller of its high/low counts.
    for band, sample in d.groupby('band'):
        diffs, weights = [], []
        cols=['delta7','delta6','delta13','positive_median']
        for _,g in sample.groupby(['n','recent_units','cover_bin']):
            hi,lo=g[g.grp=='high'],g[g.grp=='low']
            if len(hi) and len(lo):
                diffs.append(hi[cols].mean().to_numpy()-lo[cols].mean().to_numpy());weights.append(min(len(lo),len(hi)))
        matched.append(dict(cut=units.columns[end],band=band,strata=len(diffs),pairs=sum(weights),
                            **dict(zip(cols,np.average(diffs,axis=0,weights=weights)))))
pd.DataFrame(rows).to_csv(HERE/'stock_groups.csv',index=False)
pd.DataFrame(matched).to_csv(HERE/'stock_matched.csv',index=False)
print(pd.DataFrame(matched).round(3).to_string(index=False))
