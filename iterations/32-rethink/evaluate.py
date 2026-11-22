"""Reconstruct the −4.4 file, compare keyed changes and export a selected forecast."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "31-demand-distribution"))
from diagnosis.pack import project
import score as s
import validate as v
import make_submission as sub

OUT = HERE / "results"
METHODS = ["history", "forest", "research"]
PRIMARY = [0,13,26,39]
EXTRA = [16,32]


def inputs(shop, fold, arm):
    df = v.load_weekly(shop)
    final = fold == "final"
    if final:
        cut = df.week.max() + pd.Timedelta(weeks=1)
        train = v.pad_to_shop_start(df,df[v.KEYS].drop_duplicates())
        hist = train.pivot(index=v.KEYS,columns="week",values="units").sort_index()
        weeks = pd.date_range(cut,periods=13,freq="7D")
        grid = pd.MultiIndex.from_frame(hist.index.to_frame(index=False).merge(pd.DataFrame({"week":weeks}),how="cross"))
        hold = None
    else:
        frame = df[df.week <= df.week.max() - pd.Timedelta(weeks=fold)]
        train,hold,cut = v.split(frame)
        grid = hold.set_index(s.KEY).sort_index().index
        hist = train.pivot(index=v.KEYS, columns="week",values="units").sort_index()
    assert hist.index.equals(grid.droplevel("week").unique()) and hist.notna().all().all()
    if shop == "test":
        run = f"test_{fold}"
        if final:
            base = "iterations/03-foundation-model/forecasts_final"
            parts = [s.matrix(s.blob(s.PR,f"{base}/{m}__test__final__padded.parquet"),grid) for m in s.FM]
            ft = s.blob(s.PR,"iterations/03-foundation-model/forecasts_oos-ft/chronos-2-ft__test__final__padded.parquet")
        else:
            primary = fold in PRIMARY
            commit, folder, suffix = ((s.FROZEN,"forecasts",f"cut{fold//13}") if primary else (s.BATCH,"forecasts_cuts",f"w{fold}"))
            parts = [s.matrix(s.blob(commit,f"iterations/03-foundation-model/{folder}/{m}__test__{suffix}__padded.parquet"),grid) for m in s.FM]
            model = "ft" if primary else "dist31"
            ft = s.SOURCE / f"chronos-2-{model}__test__{suffix}__padded.parquet"
    else:
        run = f"{shop}_{arm}"
        base = HERE.parent / "28-pr20-four-shops/results"
        parts = [s.matrix(base / f"{m}_{shop}.parquet",grid) for m in s.FM]
        donor = shop if arm=="own" else "ABCD"
        ft = HERE.parent / f"29-borrowing/results/{donor}/chronos-2-ft_{shop}.parquet"
    foundation = (3*np.mean(parts,axis=0) + s.matrix(ft,grid)) / 4
    foundation[hist.to_numpy()[:,-8:].sum(axis=1)==0]=0
    return run, hist, train, hold, cut, grid, foundation


def forecasts(run,hist,grid,foundation,methods,combination):
    trees = {"incumbent": s.matrix(s.OUT / f"{run}.parquet",grid)}
    for name in methods:
        trees[name] = s.matrix(HERE / name / "results" / f"{run}.parquet",grid)
    if len(combination)>1:
        trees["mix"] = np.mean([trees[name] for name in combination],axis=0)
    for name, tree in trees.items():
        raw = np.maximum(0,.7*foundation+.3*tree)
        yield name,np.rint(raw)
        yield name+"_pack",np.rint(project(hist.to_numpy(),raw))


def historical(stage, methods, combination):
    specs = ([("test",k,"target") for k in PRIMARY+EXTRA] if stage=="target" else
             [(shop,0,arm) for shop in "ABCD" for arm in ["own","pooled"]])
    refs = pd.read_csv(s.OUT / ("target_metrics.csv" if stage=="target" else "shops_metrics.csv")).query('model == "distribution"')
    rows,product_rows = [], []
    for shop,fold,arm in specs:
        run,hist,train,hold,cut,grid,foundation=inputs(shop,fold,arm)
        measures = None
        ref_row = refs[(refs.shop==shop)&(refs.offset==fold)]
        if stage=="shops":
            ref_row = ref_row[ref_row.arm==arm]
        actual = hold.set_index(s.KEY).loc[grid].units.to_numpy().reshape(-1,13)
        for name, pred in forecasts(run,hist,grid,foundation,methods,combination):
            long = grid.to_frame(index=False).assign(forecast=pred.ravel())
            assessed = v.score(train,hold,long,detail=True)
            sc = assessed["detail"].scale.reindex(hist.index).to_numpy()
            valid = np.isfinite(sc)
            err = np.abs(actual[valid]-pred[valid])/sc[valid,None]
            per_product = err.mean(axis=1)
            vals = np.array([assessed["mase"],err[:,:7].mean(),err[:,7:].mean()])
            np.testing.assert_allclose(vals[0],per_product.mean())
            if measures is None:
                measures = vals
                ref_product = per_product
                np.testing.assert_allclose(vals,ref_row[["mase","early_mase","late_mase"]].iloc[0],atol=1e-10)
            gain = 100*(measures-vals)
            rows.append(dict(shop=shop,offset=fold,cut=str(cut.date()),arm=arm,model=name,mase=vals[0],
                             early_mase=vals[1],late_mase=vals[2],gain=gain[0],early_gain=gain[1],late_gain=gain[2],
                             scored=valid.sum(),excluded=(~valid).sum(),all_zero_history=(hist.max(axis=1)==0).sum(),
                             coverage=assessed["holdout_units_share_scored"]))
            if stage=="target" and fold in PRIMARY:
                product_rows.append(pd.DataFrame(dict(article=hist.index[valid].get_level_values("article_id"),
                                     offset=fold,model=name,contribution=100*(ref_product-per_product)/valid.sum())))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/f"{stage}_periods.csv",index=False)
    group = np.where(df.offset.isin(PRIMARY),"primary","supplementary") if stage=="target" else df.arm
    summary = df.groupby([group,"model"])[["mase","early_mase","late_mase","gain","early_gain","late_gain"]].mean()
    summary["wins"] = df.gain.gt(1e-10).groupby([group,df.model]).sum()
    summary.index.names=["period_or_arm","model"]
    summary.to_csv(OUT/f"{stage}_average.csv")
    print(summary.round(4).to_string())
    if product_rows:
        result=pd.concat(product_rows).groupby(["model","article"]).contribution.sum().div(4)
        result.groupby(level="model").agg(total="sum",best_article="max",worst_article="min").to_csv(OUT/"influence.csv")


def export(methods,combination,selected):
    run,hist,_,_,_,grid,foundation=inputs("test","final","target")
    preds=dict(forecasts(run,hist,grid,foundation,methods,combination))
    old_path=s.ROOT/"submissions/ensemble_pr20_weekly_distribution_v1.csv"
    ids=hist.index.get_level_values("article_id")
    old=pd.read_csv(old_path).set_index("article_id").loc[ids].to_numpy()
    np.testing.assert_array_equal(preds["incumbent"],old)
    path=s.ROOT/f"submissions/ensemble32_{selected}.csv"
    if path.exists():
        np.testing.assert_array_equal(pd.read_csv(path).set_index("article_id").loc[ids],preds[selected])
    else:
        sub.write_submission(preds[selected],ids,path)
    sub.validate_submission(path)
    print(path, (preds[selected]!=old).sum(),"changed cells", (preds[selected]!=old).any(axis=1).sum(),"changed articles")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["target","shops","export"])
    p.add_argument("--methods",nargs="*",choices=METHODS,default=METHODS)
    p.add_argument("--mix",nargs="*",choices=METHODS,default=[])
    p.add_argument("--selected")
    args=p.parse_args()
    if args.stage=="export":
        export(args.methods,args.mix,args.selected)
    else:
        historical(args.stage,args.methods,args.mix)
