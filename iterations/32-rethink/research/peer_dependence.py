"""Measure repeatability of cross-item demand co-movement. No model forecasts."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
df = pd.read_parquet(ROOT / "data/interim/weekly_test.parquet")
x = df.pivot(index="article_id", columns="week", values="units").fillna(0)
results = []
for offset in [0,13,26,39]:
    start = x.shape[1]-13-offset
    tr = np.log1p(x.iloc[:,start-52:start].to_numpy())
    active = ((tr>0).sum(axis=1)>=8) & (np.ptp(tr,axis=1)>0)
    u = tr[active]
    freq = (u>0).mean(axis=1)
    # First difference helps prevent slow shared starts/ends driving high correlation.
    train = np.diff(u,axis=1)
    z = train - train.mean(axis=1,keepdims=True)
    z /= np.maximum(1e-9, np.linalg.norm(z,axis=1,keepdims=True))
    sim = z @ z.T
    np.fill_diagonal(sim,-np.inf)
    peers = np.argpartition(-sim,7,axis=1)[:,:8]
    rng = np.random.default_rng(0)
    controls = np.stack([rng.choice(np.flatnonzero((np.arange(len(freq))!=i) & (np.abs(freq-freq[i])<=.1)),8,replace=False) for i in range(len(freq))])
    block = np.diff(np.log1p(x.iloc[active,start-1:start+13].to_numpy()),axis=1)
    valid = np.ptp(block,axis=1)>0
    for label,idx in [('nearest',peers),('frequency_matched_random',controls)]:
        peer = block[idx].mean(axis=1)
        for lag in [0,1]:
            a = block[:,lag:]
            b = peer[:,:13-lag]
            a = a-a.mean(axis=1,keepdims=True)
            b = b-b.mean(axis=1,keepdims=True)
            valid_pair = valid & (np.linalg.norm(b,axis=1)>0)
            z = np.divide(np.sum(a*b,axis=1),np.maximum(1e-9,np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)))
            results.append(dict(offset=offset,cut=x.columns[start].date(),method=label,lag=lag,eligible=len(freq),nonconstant_pairs=valid_pair.sum(),corr_mean=z[valid_pair].mean(),corr_median=np.median(z[valid_pair])))
pd.DataFrame(results).to_csv(OUT/'peer_dependence.csv',index=False)
print(pd.DataFrame(results).to_string(index=False))
