"""Run the prespecified paired Chronos-T5 experiment on the training GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import chronos
from chronos import ChronosPipeline
import numpy as np
import pandas as pd
import torch
import transformers

SHOPS = ("test","A","B","C","D")
MODEL_REV = "a971ba21945c4f1796b17a91fe69214b5f4ad472"
CHECKPOINT = Path.home()/".cache/huggingface/hub/models--amazon--chronos-t5-small/snapshots"/MODEL_REV
H = 13
BATCH = 32
CHUNK = 48
SEED = 0


def load_tables(folder):
    tables = {}
    for s in SHOPS:
        path = folder / f"weekly_{s}.parquet"
        df = pd.read_parquet(path,columns=["store_id","article_id","week","units"])
        panel = df.pivot(index=["store_id","article_id"],columns="week",values="units").sort_index().sort_index(axis=1)
        assert (np.diff(panel.columns.to_numpy()) == np.timedelta64(7,"D")).all()
        first = panel.notna().to_numpy().argmax(axis=1)
        series = panel.fillna(0).to_numpy(dtype=np.float32)
        tables[s] = (series,first,panel.index,panel.columns)
    return tables


def fresh():
    torch.manual_seed(SEED)
    pipe = ChronosPipeline.from_pretrained(str(CHECKPOINT),device_map="cuda",torch_dtype=torch.float32)
    pipe.tokenizer.config.prediction_length = H
    return pipe


def predict(pipe, context):
    pipe.model.model.eval()
    out = np.zeros((len(context),H),np.float32)
    active = np.flatnonzero((context>0).any(axis=1))
    with torch.random.fork_rng(devices=[0]),torch.inference_mode(),torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(SEED)
        for i in range(0,len(active),CHUNK):
            rows = active[i:i+CHUNK]
            q,_ = pipe.predict_quantiles(torch.from_numpy(context[rows]),prediction_length=H,quantile_levels=[.5])
            out[rows] = q[...,0].float().cpu().numpy()
    assert np.isfinite(out).all()
    return np.maximum(out,0)


class Windows:
    def __init__(self, tables, shop, target_weeks, pooled):
        self.groups = []
        donors = [s for s in SHOPS if s!=shop] if pooled else []
        for s in [shop]+donors:
            series, first, _, _ = tables[s]
            end = target_weeks if s==shop else series.shape[1]
            last_origin = end-H
            eligible = np.flatnonzero(first<last_origin)
            if not len(eligible):
                raise ValueError(f"No eligible history for {s} before week {end}")
            self.groups.append((series[:,:end],first,eligible,last_origin))
        self.rng = np.random.default_rng(SEED)

    def batch(self):
        xs, ys = [], []
        counts = [16,4,4,4,4] if len(self.groups)>1 else [32]
        for (values,first,eligible,last), count in zip(self.groups,counts,strict=True):
            ids = self.rng.choice(eligible,size=count)
            origins = self.rng.integers(first[ids]+1,last+1)
            for idx,o in zip(ids,origins):
                xs.append(values[idx,:o])
                ys.append(values[idx,o:o+H])
        X = np.full((BATCH,max(map(len,xs))),np.nan,np.float32)
        for i,x in enumerate(xs):
            X[i,-len(x):] = x
        return torch.from_numpy(X),torch.from_numpy(np.stack(ys))


def adapt(pipe,windows,steps,check=None):
    net = pipe.model.model
    opt = torch.optim.AdamW(net.parameters(),lr=1e-5)
    for step in range(1,steps+1):
        net.train()
        context,label = windows.batch()
        ids,mask,scale = pipe.tokenizer.context_input_transform(context)
        target,valid = pipe.tokenizer.label_input_transform(label,scale)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss = net(input_ids=ids.cuda(),attention_mask=mask.cuda(),
                       labels=target.masked_fill(~valid,-100).cuda()).loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if check is not None and step in (100,250,500):
            check(step)


def pick_steps(tables,shop,end,pooled):
    values = tables[shop][0]
    inner_end = end-H
    # Purge full 13-week labels: all training labels end before this validation block.
    history = values[:,:inner_end]
    scales = np.abs(np.diff(history,axis=1)).mean(axis=1)
    scored = np.flatnonzero(scales>0)
    ids = np.sort(np.random.default_rng(SEED).choice(scored,min(len(scored),256),replace=False))
    pipe = fresh()
    scores = {}
    def check(step):
        pred = predict(pipe,history[ids])
        scores[step] = float((np.abs(values[ids,inner_end:end]-pred).mean(axis=1)/scales[ids]).mean())
    adapt(pipe,Windows(tables,shop,inner_end,pooled),500,check)
    return min(scores,key=scores.get), scores


def run_job(tables,shop,offset,arm,out,final=False):
    name = f"{shop}-{'final' if final else offset}-{arm}"
    folder = out/name
    if (folder/"complete.json").exists():
        return
    folder.mkdir(exist_ok=True,parents=True)
    values,_,keys,dates = tables[shop]
    end = values.shape[1] if final else values.shape[1]-offset-H
    print(f"{name} starts, history through {dates[end-1].date()}",flush=True)
    start = time.monotonic()
    steps, scores = 0,{}
    if arm!="zero":
        steps,scores = pick_steps(tables,shop,end,arm=="pooled")
    pipe = fresh()
    if steps:
        adapt(pipe,Windows(tables,shop,end,arm=="pooled"),steps)
        pipe.model.model.save_pretrained(folder/"weights")
    fc = predict(pipe,values[:,:end])
    np.savez_compressed(folder/"forecast.npz",forecast=fc,
                        store=keys.get_level_values(0).to_numpy(dtype=str),
                        article=keys.get_level_values(1).to_numpy(dtype=str))
    meta = dict(shop=shop,offset=offset,arm=arm,training_end=str(dates[end-1].date()),
                inner_start=str(dates[end-H].date()),steps=steps,inner_mase=scores,
                checkpoint=MODEL_REV,seed=SEED,seconds=round(time.monotonic()-start,1))
    (folder/"complete.tmp").write_text(json.dumps(meta,indent=2)+"\n")
    (folder/"complete.tmp").replace(folder/"complete.json")
    print(f"{name} complete, steps={steps}, seconds={meta['seconds']}",flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--mode",choices=["smoke","historical","final"],required=True)
    ap.add_argument("--arm",choices=["own","pooled"])
    a = ap.parse_args()
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    tables = load_tables(a.data)
    a.out.mkdir(exist_ok=True,parents=True)
    if a.mode=="smoke":
        pipe = fresh()
        before = tables["test"][0][:16,:65]
        first = predict(pipe,before)
        np.testing.assert_array_equal(first,predict(pipe,before))
        adapt(pipe,Windows(tables,"test",65,True),1)
        assert predict(pipe,before).shape==(16,H)
        print(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,chronos=chronos.__version__,
                   transformers=transformers.__version__,python=platform.python_version(),
                   peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,3)),flush=True)
    elif a.mode=="historical":
        for shop in SHOPS:
            for offset in ([0,13,26,39,16,32] if shop=="test" else [0]):
                for arm in ("zero","own","pooled"):
                    run_job(tables,shop,offset,arm,a.out)
    else:
        if not a.arm:
            ap.error("--final requires --arm")
        run_job(tables,"test",0,a.arm,a.out,final=True)


if __name__=="__main__":
    main()
