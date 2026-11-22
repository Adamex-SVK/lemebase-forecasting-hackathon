"""Verify the selected CSV and quantify concentration of its historical gains."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

import evaluate as e


def check():
    deltas, category_rows = [], []
    for fold in e.PRIMARY:
        run,hist,train,hold,cut,grid,foundation=e.inputs("test",fold,"target")
        pred=dict(e.forecasts(run,hist,grid,foundation,["history"],[]))
        y=hold.set_index(e.s.KEY).loc[grid].units.to_numpy().reshape(-1,13)
        u=hist.to_numpy()
        scale=np.mean(np.abs(np.diff(u,axis=1)),axis=1)
        ok=scale>0
        gain=(np.abs(y-pred["incumbent"])-np.abs(y-pred["history_pack"])).mean(axis=1)
        gain=np.divide(100*gain,scale,out=np.full_like(gain,np.nan),where=ok)
        deltas.append(pd.Series(gain,index=hist.index.get_level_values("article_id")))
        age=u.shape[1]-np.argmax(u>0,axis=1)
        for name,mask in [("first sale in last 13 weeks",age<=13),("older",age>13)]:
            inside=ok & mask
            category_rows.append(dict(offset=fold,group=name,n=inside.sum(),
                                      contribution=np.sum(gain[inside])/ok.sum()))
    d=pd.concat(deltas,axis=1).to_numpy()
    rng=np.random.default_rng(0)
    values=[]
    for _ in range(100):
        indices=rng.integers(d.shape[0],size=(20,d.shape[0]))
        values.extend(np.nanmean(d[indices],axis=1).mean(axis=1))
    per_product=np.nansum(d / np.sum(np.isfinite(d),axis=0),axis=1)/len(e.PRIMARY)
    exclude=per_product.argmax()
    remaining=np.delete(d,exclude,axis=0)
    out=dict(mean_gain=np.nanmean(d,axis=0).mean(),p025=np.quantile(values,.025),p975=np.quantile(values,.975),
             mean_without_best_product=np.nanmean(remaining,axis=0).mean(),seed=0,draws=2000)
    pd.DataFrame([out]).to_csv(e.OUT/"selected_product_uncertainty.csv",index=False)
    pd.DataFrame(category_rows).to_csv(e.OUT/"selected_activation.csv",index=False)
    run,hist,_,_,_,grid,foundation=e.inputs("test","final","target")
    pred=dict(e.forecasts(run,hist,grid,foundation,["history"],[]))
    path=e.s.ROOT/"submissions/ensemble32_history_pack.csv"
    published=pd.read_csv(path).set_index("article_id").loc[hist.index.get_level_values("article_id")]
    np.testing.assert_array_equal(published.to_numpy(),pred["history_pack"])
    e.sub.validate_submission(path)
    print(out, "file_sha256",hashlib.sha256(path.read_bytes()).hexdigest())
    print("changed cells by stock and subsequent pack correction",
          (pred["history"]!=pred["incumbent"]).sum(),(pred["history_pack"]!=pred["history"]).sum())


if __name__=="__main__":
    check()
