"""Fixed local comparison: 60% FM + 30% LGBM + 10% newly trained XGBoost.

.venv-autogluon/bin/python iterations/08-xgboost-blend/run.py
Foundation predictions are reused; XGBoost is fitted locally with temporal early stopping.
"""
import argparse
import gc
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "iterations/07-ensemble-review"))
import features as f
import validate as v
import evaluate as review


def fit_predict(M, path, params, max_rounds):
    origins = f.make_origins(M.W)
    val_origins = origins[-7:]
    train_origins = origins[origins <= val_origins[0]-13]
    if len(train_origins) == 0:
        raise ValueError("Not enough origins for purged early stopping")
    assert train_origins.max()+13 <= val_origins.min()
    print(f"Building windows: {len(train_origins)} training, {len(val_origins)} internal validation origins", flush=True)
    tr = f.build_windows(M, train_origins)
    va = f.build_windows(M, val_origins)
    dt = xgb.QuantileDMatrix(tr["X"], tr["y"], weight=tr["w"], feature_names=f.FEATURES,
                            nthread=params["nthread"], max_bin=params["max_bin"])
    dv = xgb.QuantileDMatrix(va["X"], va["y"], weight=va["w"], feature_names=f.FEATURES,
                            nthread=params["nthread"], max_bin=params["max_bin"], ref=dt)
    ntr, nva = len(tr["y"]), len(va["y"])
    del tr, va
    gc.collect()
    model = xgb.train(params, dt, num_boost_round=max_rounds, evals=[(dv,"validation")],
                      early_stopping_rounds=30, verbose_eval=50)
    best = model.best_iteration+1
    best_mae = float(model.best_score)
    rounds = int(best*1.1)+1
    del model, dt, dv
    gc.collect()
    print(f"Refitting all pre-cut windows: {rounds} trees (inner best {best})", flush=True)
    allwin = f.build_windows(M, origins)
    dall = xgb.QuantileDMatrix(allwin["X"], allwin["y"], weight=allwin["w"], feature_names=f.FEATURES,
                              nthread=params["nthread"], max_bin=params["max_bin"])
    nall = len(allwin["y"])
    del allwin
    gc.collect()
    model = xgb.train(params, dall, num_boost_round=rounds)
    model.save_model(path / "model.ubj")
    X, scale, valid = f.inference_features(M)
    pred = model.inplace_predict(X).reshape(M.n,13)*scale[:,None]
    pred = np.clip(pred,0,None)
    pred[~valid] = 0
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite XGBoost predictions")
    return pred, dict(inner_best_rounds=best, final_rounds=rounds, inner_mae=best_mae,
                      train_rows=ntr, validation_rows=nva, final_train_rows=nall,
                      last_training_origin=int(train_origins.max()), first_validation_origin=int(val_origins.min()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cuts", type=int, nargs="+", choices=range(4), default=[0,1,2,3])
    ap.add_argument("--cpus", type=int, default=2)
    ap.add_argument("--max-rounds", type=int, default=400)
    ap.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"runs"/"fixed_60_30_10")
    args = ap.parse_args()
    if args.cpus < 1 or args.max_rounds < 1:
        ap.error("CPUs and rounds must be positive")
    args.output.mkdir(parents=True,exist_ok=False)
    params = dict(objective="reg:absoluteerror", eval_metric="mae", tree_method="hist", device="cpu",
                  max_depth=5, min_child_weight=50, learning_rate=.05, subsample=.8,
                  colsample_bytree=.8, reg_lambda=1., max_bin=128, seed=0, nthread=args.cpus)
    hashes, rows, status = {}, [], []
    cfg = dict(recipe="0.6*mean(FM after 8-week zero rule)+0.3*LGBM+0.1*XGB; round once then clip",
               reference="0.7*mean(FM after 8-week zero rule)+0.3*LGBM; round once then clip",
               params=params, max_rounds=args.max_rounds, cuts=args.cuts, xgboost=xgb.__version__,
               numpy=np.__version__, pandas=pd.__version__, python=sys.version,
               foundation_source=review.SOURCE, note="Fixed exploratory target-shop experiment; no private-score claim")
    cfg["source_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__),ROOT/"src/features.py",ROOT/"src/validate.py",
                                    ROOT/"iterations/07-ensemble-review/evaluate.py")}
    cfg["input_sha256"]=hashlib.sha256((v.IM/"weekly_test.parquet").read_bytes()).hexdigest()
    (args.output/"config.json").write_text(json.dumps(cfg,indent=2))
    (args.output/"environment.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    full=v.load_weekly("test")
    for k in args.cuts:
        start=time.time()
        path=args.output/f"cut{k}"
        path.mkdir()
        d=full[full.week<=full.week.max()-pd.Timedelta(weeks=13*k)]
        train,hold,cut=v.split(d)
        M=f.ShopMatrix(d[d.week<cut])
        weeks=pd.date_range(cut,periods=13,freq="7D")
        components=[]
        for name in review.MODELS:
            source=f"iterations/03-foundation-model/forecasts/{name}__test__cut{k}__padded.parquet"
            components.append(review.matrix(pd.read_parquet(io.BytesIO(review.load_blob(review.SOURCE,source,hashes))),M.keys,weeks))
        source=f"iterations/02-robust-eval/forecasts/base_cut{k}_seed0.npy"
        L=np.load(io.BytesIO(review.load_blob(review.SOURCE,source,hashes)),allow_pickle=False)
        local=review.matrix(pd.read_parquet(ROOT/f"logs/chronos_four/LightGBM_{k}.parquet"),M.keys,weeks)
        assert L.shape==local.shape and np.allclose(L,local,rtol=0,atol=1e-8)
        E=np.mean(components,axis=0)
        E[M.U[:,-8:].sum(axis=1)==0]=0
        current=np.clip(np.round(.7*E+.3*L),0,None)
        print(f"CUT {k}: train ends {train.week.max().date()}, holdout begins {cut.date()}",flush=True)
        X,info=fit_predict(M,path,params,args.max_rounds)
        np.save(path/"xgboost.npy",X)
        variants={"current_70_30":current,"candidate_60_30_10":np.clip(np.round(.6*E+.3*L+.1*X),0,None),
                  "xgboost_alone":np.clip(np.round(X),0,None)}
        baseline=None
        for name,F in variants.items():
            fc=pd.DataFrame({"store_id":np.repeat(M.keys.store_id.to_numpy(),13),
                             "article_id":np.repeat(M.keys.article_id.to_numpy(),13),
                             "week":np.tile(weeks,M.n),"forecast":F.reshape(-1)})
            fc.to_parquet(path/f"{name}.parquet",index=False)
            result=v.score(train,hold,fc,detail=True)
            detail=result.pop("detail")
            if baseline is None:
                baseline=detail.mase
            gain=100*(baseline-detail.mase)
            rng=np.random.default_rng(0)
            bs=[rng.choice(gain.to_numpy(),len(gain)).mean() for _ in range(500)]
            m=hold.merge(fc,on=v.KEYS+["week"],validate="one_to_one").merge(
                detail[["scale"]].reset_index(),on=v.KEYS,validate="many_to_one")
            m["h"]=(m.week-cut).dt.days//7+1
            m["scaled_error"]=abs(m.units-m.forecast)/m.scale
            early=100*(1-m.loc[m.h<=7,"scaled_error"].mean())
            late=100*(1-m.loc[m.h>=8,"scaled_error"].mean())
            rows.append(dict(cut=k,variant=name,gain_vs_current=float(gain.mean()),early_score=early,late_score=late,
                             gain_ci025=float(np.percentile(bs,2.5)),gain_ci975=float(np.percentile(bs,97.5)),**result))
            print(f"cut {k} {name}: {result['score']:.4f}, gain {gain.mean():+.4f}",flush=True)
        status.append(dict(cut=k,status="complete",seconds=time.time()-start,**info))
        pd.DataFrame(rows).to_csv(args.output/"results.csv",index=False)
        (args.output/"status.json").write_text(json.dumps(status,indent=2))
        del M,X,components,E,L,local
        gc.collect()
    cfg["forecast_source_sha256"]=hashes
    (args.output/"config.json").write_text(json.dumps(cfg,indent=2))
    summary=pd.DataFrame(rows).groupby("variant").agg(mean_score=("score","mean"),gain_vs_current=("gain_vs_current","mean"),
        early_score=("early_score","mean"),late_score=("late_score","mean"),cuts_won=("gain_vs_current",lambda s:int((s>1e-8).sum())))
    summary.to_csv(args.output/"summary.csv")
    print(summary.to_string(),flush=True)


if __name__=="__main__":
    main()
