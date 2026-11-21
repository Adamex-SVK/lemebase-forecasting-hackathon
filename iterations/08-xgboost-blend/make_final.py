"""Fit XGBoost on all history and write the user-requested 60/30/10 CSV. Never uploads.

.venv-autogluon/bin/python iterations/08-xgboost-blend/make_final.py
Tree count is fixed from the newest historical validation cut, not hidden outcomes.
"""
import argparse
import gc
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"iterations/07-ensemble-review"))
import features as f
import validate as v
import make_submission as ms
import evaluate as review


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts",type=Path,default=Path(__file__).resolve().parent/"runs/final_60_30_10")
    ap.add_argument("--output",type=Path,default=ROOT/"submissions/ensemble_top3_lgbm30_xgb10_v1.csv")
    args=ap.parse_args()
    if args.output.exists():
        ap.error("Submission path already exists; choose a new filename")
    cfg=json.loads((Path(__file__).resolve().parent/"evidence/config.json").read_text())
    statuses=json.loads((Path(__file__).resolve().parent/"evidence/status.json").read_text())
    rounds=next(s["final_rounds"] for s in statuses if s["cut"]==0)
    assert xgb.__version__==cfg["xgboost"], "Use the validated XGBoost environment"
    assert hashlib.sha256((ROOT/"src/features.py").read_bytes()).hexdigest()==cfg["source_sha256"]["src/features.py"]
    assert hashlib.sha256((v.IM/"weekly_test.parquet").read_bytes()).hexdigest()==cfg["input_sha256"]
    args.artifacts.mkdir(parents=True,exist_ok=False)
    full=v.load_weekly("test")
    M=f.ShopMatrix(full)
    ids=M.keys.article_id.to_list()
    assert ids==ms.history_ids() and M.keys.store_id.nunique()==1
    weeks=pd.date_range(full.week.max()+pd.Timedelta(weeks=1),periods=13,freq="7D")
    assert weeks[0]==ms.FIRST_FORECAST_WEEK and weeks[-1]==pd.Timestamp("2023-07-17")
    source=cfg["foundation_source"]
    hashes={}
    parts=[]
    for name in review.MODELS:
        p=f"iterations/03-foundation-model/forecasts_final/{name}__test__final__padded.parquet"
        parts.append(review.matrix(pd.read_parquet(io.BytesIO(review.load_blob(source,p,hashes))),M.keys,weeks))
    E=np.mean(parts,axis=0)
    E[M.U[:,-8:].sum(axis=1)==0]=0
    reference_path=ROOT/"submissions/lgbm_own_v1.csv"
    ref=pd.read_csv(reference_path)
    assert ref.article_id.is_unique and set(ref.article_id)==set(ids)
    L=ref.set_index("article_id").loc[ids,[f"w{i}" for i in range(1,14)]].to_numpy(float)
    incumbent_path=ROOT/"submissions/ensemble_top3_blend_v1.csv"
    incumbent_bytes=incumbent_path.read_bytes()
    incumbent=pd.read_csv(io.BytesIO(incumbent_bytes)).set_index("article_id").loc[ids].to_numpy(float)
    assert np.array_equal(np.clip(np.round(.7*E+.3*L),0,None),incumbent)
    params=cfg["params"]
    print(f"Building all-history windows ({M.W} weeks); fitting {rounds} trees fixed from cut 0",flush=True)
    windows=f.build_windows(M,f.make_origins(M.W))
    nrows=len(windows["y"])
    dm=xgb.QuantileDMatrix(windows["X"],windows["y"],weight=windows["w"],feature_names=f.FEATURES,
                          max_bin=params["max_bin"],nthread=params["nthread"])
    del windows
    gc.collect()
    model=xgb.train(params,dm,num_boost_round=rounds)
    model.save_model(args.artifacts/"model.ubj")
    X,scale,valid=f.inference_features(M)
    Xgb=np.clip(model.inplace_predict(X).reshape(M.n,13)*scale[:,None],0,None)
    Xgb[~valid]=0
    assert np.isfinite(Xgb).all()
    np.save(args.artifacts/"xgboost_final.npy",Xgb)
    F=np.clip(np.round(.6*E+.3*L+.1*Xgb),0,None)
    ms.write_submission(F,ids,args.output)
    ms.validate_submission(args.output)
    assert incumbent_path.read_bytes()==incumbent_bytes
    saved=pd.read_csv(args.output).set_index("article_id").loc[ids].to_numpy(float)
    assert np.array_equal(saved,F)
    metadata=dict(recipe=cfg["recipe"],params=params,rounds=rounds,rounds_source="cut 0 historical early-stopping/refit count",
                  train_end=str(full.week.max().date()),train_weeks=M.W,train_rows=nrows,
                  forecast_start=str(weeks[0].date()),forecast_end=str(weeks[-1].date()),
                  foundation_source=source,forecast_source_sha256=hashes,xgboost=xgb.__version__,
                  input_sha256=cfg["input_sha256"],lgbm_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                  incumbent_sha256=hashlib.sha256(incumbent_bytes).hexdigest(),
                  builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  output=str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else str(args.output),
                  output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  n_products=len(ids),changed_products=int(np.any(F!=incumbent,axis=1).sum()),
                  mean_absolute_change=float(np.abs(F-incumbent).mean()),
                  warning="User-requested experimental submission; historical mean -13.85 vs incumbent -13.67; not uploaded.")
    (args.artifacts/"metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)
    print(f"VALIDATED CSV: {args.output}",flush=True)


if __name__=="__main__":
    main()
