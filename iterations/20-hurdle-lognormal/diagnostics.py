"""Post-fit distribution diagnostics; never changes fitted models or selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr
import run as e

out=Path(__file__).resolve().parent/'results'
full=e.v.load_weekly('test')
rows=[]
for offset in e.OFFSETS:
    prefix=full[full.week<=full.week.max()-pd.Timedelta(weeks=offset)]
    history,hold,cut=e.v.split(prefix)
    M=e.f.ShopMatrix(prefix[prefix.week<cut])
    weeks=pd.date_range(cut,periods=13,freq='7D')
    y=e.review.matrix(hold.rename(columns={'units':'forecast'}),M.keys,weeks)
    scale=e.v.history_scale(history).reindex(pd.MultiIndex.from_frame(M.keys)).to_numpy()
    ok=np.isfinite(scale)&(scale>0)
    d=np.load(out/f'runs/w{offset}_hurdle/distribution.npz')
    p=d['probability'][ok];mu=d['mu'][ok];sigma=float(d['sigma']);a=y[ok];s=d['scale'][ok,None]
    positive=a>0
    z=np.log(a[positive]/np.broadcast_to(s,a.shape)[positive])
    conditional_cdf=ndtr((z-mu[positive])/sigma)
    # Deterministic midpoint PIT at zero; positive observations have continuous CDF.
    pit=np.array((1-p)/2)
    pit[positive]=1-p[positive]+p[positive]*conditional_cdf
    info=json.loads((out/f'runs/w{offset}_hurdle/training.json').read_text())
    rows.append(dict(offset=offset,primary=offset in [0,13,26,39],sigma=sigma,
                     brier=float(np.mean((p-positive)**2)),mean_sale_probability=float(p.mean()),
                     observed_sale_frequency=float(positive.mean()),
                     fraction_median_zero=float((p<=.5).mean()),pit_mean=float(pit.mean()),
                     pit_note='midpoint at zero; count-like positive data not randomized; diagnostic only'))
pd.DataFrame(rows).to_csv(out/'distribution_diagnostics.csv',index=False)
(out/'diagnostics_versions.json').write_text(json.dumps({'scipy':scipy.__version__},indent=2))
print(pd.DataFrame(rows).drop(columns='pit_note').round(5).to_string(index=False))
