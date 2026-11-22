"""Run in a PR ca3a0ed checkout, from the checkout root, with one model per matching GPU environment."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

root = Path.cwd()
sys.path[:0] = [str(root / "src"), str(root / "src/foundation")]
import models
import validate as v
from run_model import contexts_for

H = 13
def fit_adapter():
    from chronos import Chronos2Pipeline
    donors = []
    rng = np.random.default_rng(0)
    for shop in "ACD":
        df = v.load_weekly(shop)
        cut = v.cut_week(df)
        history = df.loc[df.week < cut]
        pivot = history.pivot_table(index=v.KEYS, columns="week", values="units", aggfunc="sum",fill_value=0)
        assert pivot.columns.max() < cut
        values = pivot.to_numpy(dtype=np.float32)
        eligible = np.flatnonzero((values > 0).sum(axis=1) >= 8)
        sample = rng.choice(eligible, min(2000,len(eligible)),replace=False)
        donors.extend({"target": values[i]} for i in sample)
        print("donor",shop,"last week",history.week.max(),"series",len(sample),flush=True)
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    return pipe.fit(inputs=donors, prediction_length=H,finetune_mode="lora",
                    learning_rate=1e-5,num_steps=300,batch_size=32,context_length=104)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("name",choices=["moirai-2.0-small","tirex","chronos-t5-small","chronos-2-ft"])
    name=ap.parse_args().name
    torch.manual_seed(0)
    np.random.seed(0)
    out=root/"predictions"
    out.mkdir(exist_ok=True)
    tuned=name=="chronos-2-ft"
    m=fit_adapter() if tuned else models.load(name,"cuda")
    for shop in "ABCD":
        dest=out/f"{name}_{shop}.parquet"
        if dest.exists(): continue
        index,contexts,_,hold,cut,history=contexts_for(shop,0,"padded")
        active=np.array([np.any(row[-8:]>0) for row in contexts])
        selected=[row for row,include in zip(contexts,active) if include]
        end=cut-pd.Timedelta(weeks=1)
        forecast=np.zeros((len(index),H))
        if tuned:
            # Same median prediction as PR; preserve signed raw FT forecasts until final clip.
            ids=np.flatnonzero(active)
            for start in range(0,len(ids),256):
                chunk=ids[start:start+256]
                forecast[chunk]=models._fc_chronos({"kind":"c2"},m,[contexts[i] for i in chunk],H,end)
        else:
            forecast[active]=models.forecast(name,m,selected,H,end)
        assert np.isfinite(forecast).all()
        keys=index.to_frame(index=False).loc[index.to_frame(index=False).index.repeat(H)].reset_index(drop=True)
        keys["week"]=np.tile(pd.date_range(cut,periods=H,freq="7D"),len(index))
        keys["forecast"]=forecast.ravel()
        temp=dest.with_suffix(".tmp")
        keys.to_parquet(temp,index=False)
        temp.replace(dest)
        print(name,shop,"completed",len(selected),"recently active series",flush=True)
