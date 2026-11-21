"""Phase 1 only: immutable benchmark diagnostics and forecast availability inventory.

.venv/bin/python iterations/10-benchmark-diagnosis/diagnose.py
No candidate training, weight tuning, or final submissions. All outputs are derived.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"lemebase-diagnostics-mpl"))
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"iterations/07-ensemble-review"))
import evaluate as review
import validate as v

ADDITIONAL_SOURCE="5592b3593c794dd21b441b0fdd564eea90f78248"
ORIGINAL_OFFSETS=[0,13,26,39]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_groups(U,OH,forecasts,scales):
    """History/prediction-only groups; never receives actual holdout values."""
    positive=(U[:,-13:]>0).sum(axis=1)
    zshare=(U[:,-13:]==0).mean(axis=1)
    ratio=(U[:,-4:].mean(axis=1)+1)/(U[:,-17:-4].mean(axis=1)+1)
    last=OH[:,-1]
    nz=(U[:,-26:]>0).sum(axis=1)
    sum_u=U[:,-26:].sum(axis=1)
    positive_mean=np.divide(sum_u,nz,out=np.zeros(len(U)),where=nz>0)
    var=np.divide((U[:,-26:]**2).sum(axis=1),nz,out=np.zeros(len(U)),where=nz>0)-positive_mean**2
    cv2=np.divide(np.maximum(var,0),positive_mean**2,out=np.zeros(len(U)),where=positive_mean>0)
    spread=np.ptp(np.stack(forecasts),axis=0).mean(axis=1)
    spread=np.divide(spread,scales,out=np.zeros(len(U)),where=scales>0)
    any_sale=(U>0).any(axis=1)
    age=np.where(any_sale,U.shape[1]-(U>0).argmax(axis=1),0)
    return {
        "activity13":np.where(positive==0,"inactive",np.where(positive<=3,"rare_1to3","active_4plus")),
        "sparsity13":np.where(zshare>=.75,"zeros_ge75pct",np.where(zshare>=.25,"zeros_25to75pct","zeros_lt25pct")),
        "recent_level":np.where(ratio<2/3,"falling",np.where(ratio>1.5,"rising","stable")),
        "positive_size_variability26":np.where(nz<2,"insufficient_sales",np.where(cv2>=.49,"variable","stable")),
        "last_stock":np.where(np.isnan(last),"missing",np.where(last==0,"zero","positive")),
        "stock_missing13":np.where(np.isnan(OH[:,-13:]).mean(axis=1)>.25,"missing_gt25pct","missing_le25pct"),
        "forecast_disagreement":np.where(spread>2,"high_gt2scale",np.where(spread>.5,"medium","low_le0.5scale")),
        "observed_activation_age":np.where(~any_sale,"never_sold",np.where(age<=13,"first_sale_le13weeks",np.where(age<=26,"first_sale_14to26weeks","older"))),
        "history_scale":np.where(scales<.1,"scale_lt0.1",np.where(scales<1,"scale_0.1to1","scale_ge1")),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"results")
    args=ap.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    (out/"details").mkdir()
    full=v.load_weekly("test")
    inputhash=sha(v.IM/"weekly_test.parquet")
    incumbenthash=sha(ROOT/"submissions/ensemble_top3_blend_v1.csv")
    tree=subprocess.check_output(["git","ls-tree","-r","--name-only",ADDITIONAL_SOURCE],cwd=ROOT,text=True).splitlines()
    tree=set(tree)
    inventory=[]
    for offset in list(range(0,49,4))+[52,65,78]:
        item=dict(offset=offset,source_commit=ADDITIONAL_SOURCE)
        for name in review.MODELS:
            item[name]=f"iterations/03-foundation-model/forecasts_cuts/{name}__test__w{offset}__padded.parquet" in tree
        item["lgbm"]=f"iterations/05-meta-ensemble/base_forecasts/lgbm_w{offset}.npy" in tree
        item["complete"]=all(item[name] for name in review.MODELS+["lgbm"])
        start=v.cut_week(full)-pd.Timedelta(weeks=offset)
        end=start+pd.Timedelta(weeks=12)
        item.update(holdout_start=str(start.date()),holdout_end=str(end.date()),train_weeks=full.week.nunique()-13-offset,
                    role="additional_development" if offset<=48 else "missing_future_stress_input")
        inventory.append(item)
    pd.DataFrame(inventory).to_csv(out/"forecast_inventory.csv",index=False)
    offsets=sorted(set(ORIGINAL_OFFSETS+[x["offset"] for x in inventory if x["complete"]]))
    scores,groups,horizons,concentration,products,source_checks=[],[],[],[],[],[]
    blobhash={}
    for offset in offsets:
        original=offset in ORIGINAL_OFFSETS
        source=review.SOURCE if original else ADDITIONAL_SOURCE
        family="original_four" if original else "additional_overlapping"
        d=full[full.week<=full.week.max()-pd.Timedelta(weeks=offset)]
        train,hold,cut=v.split(d)
        keys=train[v.KEYS].drop_duplicates().sort_values(v.KEYS).reset_index(drop=True)
        index=pd.MultiIndex.from_frame(keys)
        weeks=pd.date_range(cut,periods=13,freq="7D")
        U=train.pivot(index=v.KEYS,columns="week",values="units").reindex(index).sort_index(axis=1).to_numpy(float)
        OH=train.pivot(index=v.KEYS,columns="week",values="on_hand").reindex(index).sort_index(axis=1).to_numpy(float)
        scale=v.history_scale(train).reindex(index).to_numpy()
        ok=np.isfinite(scale)&(scale>0)
        actual=review.matrix(hold.rename(columns={"units":"forecast"}),keys,weeks)
        forecasts=[]
        for name in review.MODELS:
            fname=f"{name}__test__cut{offset//13}__padded.parquet" if original else f"{name}__test__w{offset}__padded.parquet"
            folder="forecasts" if original else "forecasts_cuts"
            path=f"iterations/03-foundation-model/{folder}/{fname}"
            temp={}
            forecasts.append(review.matrix(pd.read_parquet(io.BytesIO(review.load_blob(source,path,temp))),keys,weeks))
            blobhash.update({f"{source}:{p}":h for p,h in temp.items()})
        lp=f"iterations/02-robust-eval/forecasts/base_cut{offset//13}_seed0.npy" if original else f"iterations/05-meta-ensemble/base_forecasts/lgbm_w{offset}.npy"
        temp={}
        L=np.load(io.BytesIO(review.load_blob(source,lp,temp)),allow_pickle=False)
        blobhash.update({f"{source}:{p}":h for p,h in temp.items()})
        assert L.shape==actual.shape and np.isfinite(L).all() and (L>=0).all()
        # Original arrays have a keyed independent cross-check; new arrays retain documented sorted key order.
        if original:
            ref=review.matrix(pd.read_parquet(ROOT/f"logs/chronos_four/LightGBM_{offset//13}.parquet"),keys,weeks)
            assert np.allclose(L,ref,atol=1e-8,rtol=0)
        dead=U[:,-8:].sum(axis=1)==0
        F=review.compose(forecasts,L,dead)
        if offset==0 and inventory[0]["complete"]:
            duplicate=[]
            for name,old in zip(review.MODELS,forecasts):
                p=f"iterations/03-foundation-model/forecasts_cuts/{name}__test__w0__padded.parquet"
                temp={}
                new=review.matrix(pd.read_parquet(io.BytesIO(review.load_blob(ADDITIONAL_SOURCE,p,temp))),keys,weeks)
                blobhash.update({f"{ADDITIONAL_SOURCE}:{p}":h for p,h in temp.items()})
                duplicate.append(new)
                source_checks.append(dict(model=name,max_abs_difference=float(np.abs(old-new).max()),
                                          mean_abs_difference=float(np.abs(old-new).mean())))
            p="iterations/05-meta-ensemble/base_forecasts/lgbm_w0.npy"
            temp={}
            newL=np.load(io.BytesIO(review.load_blob(ADDITIONAL_SOURCE,p,temp)),allow_pickle=False)
            blobhash.update({f"{ADDITIONAL_SOURCE}:{p}":h for p,h in temp.items()})
            assert newL.shape==L.shape
            source_checks.append(dict(model="lgbm",max_abs_difference=float(np.abs(L-newL).max()),
                                      mean_abs_difference=float(np.abs(L-newL).mean())))
            newF=review.compose(duplicate,newL,dead)
            source_checks.append(dict(model="whole_recipe",max_abs_difference=float(np.abs(F-newF).max()),
                                      mean_abs_difference=float(np.abs(F-newF).mean())))
        signed=(F[ok]-actual[ok])/scale[ok,None]
        absolute=np.abs(signed)
        over=np.maximum(signed,0)
        under=np.maximum(-signed,0)
        err=absolute.mean(axis=1)
        nsc=int(ok.sum())
        result=dict(offset=offset,family=family,holdout_start=str(cut.date()),score=100*(1-err.mean()),
                    mase=err.mean(),n_series=len(keys),n_scored=nsc,excluded=int((~ok).sum()),
                    scored_units_share=float(actual[ok].sum()/actual.sum()),
                    over_mase=over.mean(),under_mase=under.mean(),scaled_bias=signed.mean(),
                    early_mase=absolute[:,:7].mean(),late_mase=absolute[:,7:].mean(),
                    excluded_unscaled_mae=float(np.abs(F[~ok]-actual[~ok]).mean()) if (~ok).any() else 0)
        scores.append(result)
        labels=history_groups(U,OH,forecasts+[L],scale)
        for dimension,lab in labels.items():
            labs=lab[ok]
            shares=0
            for label in sorted(set(labs)):
                mask=labs==label
                share=float(err[mask].sum()/err.sum())
                shares+=share
                groups.append(dict(offset=offset,family=family,dimension=dimension,group=label,n_series=int(mask.sum()),
                                   product_share=mask.mean(),mase=err[mask].mean(),scaled_error_share=share,
                                   over_mase=over[mask].mean(),under_mase=under[mask].mean(),scaled_bias=signed[mask].mean(),
                                   over_error_share=float(over[mask].sum()/absolute[mask].sum()) if absolute[mask].sum() else 0,
                                   late_mase=absolute[mask,7:].mean(),early_mase=absolute[mask,:7].mean()))
            assert np.isclose(shares,1), dimension
        for h in range(13):
            horizons.append(dict(offset=offset,family=family,horizon=h+1,mase=absolute[:,h].mean(),
                                 over_mase=over[:,h].mean(),under_mase=under[:,h].mean(),scaled_bias=signed[:,h].mean()))
        for fraction in [.01,.05,.1]:
            n=max(1,int(nsc*fraction))
            concentration.append(dict(offset=offset,family=family,fraction=fraction,n_products=n,
                                      error_share=float(np.sort(err)[-n:].sum()/err.sum())))
        detail=keys.loc[ok].copy()
        detail["offset"]=offset
        detail["family"]=family
        detail["mase"]=err
        detail["scaled_bias"]=signed.mean(axis=1)
        detail["over_mase"]=over.mean(axis=1)
        detail["under_mase"]=under.mean(axis=1)
        detail["scale"]=scale[ok]
        detail["recent13_units"]=U[ok,-13:].sum(axis=1)
        for dim,lab in labels.items():
            detail[dim]=lab[ok]
        detail.to_csv(out/"details"/f"products_w{offset}.csv",index=False)
        products.append(detail)
        # Verify vectorized decomposition against the canonical metric, including exclusions.
        fc=pd.DataFrame({"store_id":np.repeat(keys.store_id.to_numpy(),13),"article_id":np.repeat(keys.article_id.to_numpy(),13),
                         "week":np.tile(weeks,len(keys)),"forecast":F.reshape(-1)})
        authoritative=v.score(train,hold,fc)
        assert np.isclose(result["score"],authoritative["score"])
        assert nsc==authoritative["n_scored"]
        assert np.isclose(over.mean()+under.mean(),err.mean())
        print(f"offset {offset:2d} [{family}]: score {result['score']:.3f}; over-error share {over.sum()/absolute.sum():.1%}",flush=True)
    for name,data in [("scores.csv",scores),("groups.csv",groups),("horizons.csv",horizons),("error_concentration.csv",concentration)]:
        pd.DataFrame(data).to_csv(out/name,index=False)
    g=pd.DataFrame(groups)
    summary=g.groupby(["family","dimension","group"]).agg(
        mean_error_share=("scaled_error_share","mean"),min_error_share=("scaled_error_share","min"),max_error_share=("scaled_error_share","max"),
        mean_mase=("mase","mean"),mean_bias=("scaled_bias","mean"),over_bias_windows=("scaled_bias",lambda x:int((x>0).sum())),
        windows=("offset","nunique"),mean_product_share=("product_share","mean"),
        mean_over_mase=("over_mase","mean"),mean_under_mase=("under_mase","mean"))
    summary.to_csv(out/"group_summary.csv")
    # Product persistence is diagnostic only: no post-outcome product IDs become model rules.
    allproducts=pd.concat(products,ignore_index=True)
    primary=allproducts[allproducts.family=="original_four"]
    persist=primary.groupby(v.KEYS).agg(windows=("offset","nunique"),mean_mase=("mase","mean"),
        mean_scaled_bias=("scaled_bias","mean"),over_bias_windows=("scaled_bias",lambda x:int((x>0).sum())))
    persist.to_csv(out/"details/product_persistence.csv")
    pd.DataFrame(scores).query("family == 'original_four'").to_csv(out/"benchmark_lock.csv",index=False)
    pd.DataFrame(source_checks).to_csv(out/"duplicate_origin_checks.csv",index=False)
    assert sha(v.IM/"weekly_test.parquet")==inputhash
    assert sha(ROOT/"submissions/ensemble_top3_blend_v1.csv")==incumbenthash
    metadata=dict(primary_source=review.SOURCE,additional_source=ADDITIONAL_SOURCE,offsets=offsets,
        immutable_recipe="0.7*mean(three raw foundation forecasts with zero8)+0.3*raw LightGBM; round; clip",
        input_sha256=inputhash,incumbent_sha256=incumbenthash,forecast_sha256=blobhash,
        script_sha256=sha(Path(__file__)),numpy=np.__version__,pandas=pd.__version__,python=sys.version,
        limitations=["All evaluated windows are development evidence; additional windows overlap and some were used in donor meta-model experiments.",
                     "Additional LightGBM NPY row order follows donor runner contract; unlike original cuts it has no independent keyed reference.",
                     "Groups use history or forecasts only; error-based product rankings are post hoc and not deployable selection rules.",
                     "Bias is relative to an absolute-error median objective; underprediction of mean volume does not itself justify an upward correction.",
                     "Stock association is observational, not causal. No target-shop result establishes unseen-shop transfer."])
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    plot(out,pd.DataFrame(scores),g,pd.DataFrame(horizons))
    print(summary.loc["original_four"].round(3).to_string(),flush=True)


def plot(out,scores,groups,horizons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    activity=groups[(groups.family=="original_four")&(groups.dimension=="activity13")]
    p=activity.pivot(index="offset",columns="group",values="scaled_error_share")
    p.plot.bar(stacked=True,ax=axes[0,0],color=["#397e9b","#b7c7cf","#ba7149"])
    axes[0,0].set(title="Where incumbent error occurs",ylabel="Share of scaled absolute error",xlabel="Historical offset (weeks)")
    axes[0,0].legend(fontsize=8)
    h=horizons[horizons.family=="original_four"].groupby("horizon")[["over_mase","under_mase"]].mean()
    axes[0,1].bar(h.index,h.over_mase,label="Overprediction",color="#ba7149")
    axes[0,1].bar(h.index,h.under_mase,bottom=h.over_mase,label="Underprediction",color="#397e9b")
    axes[0,1].set(title="Error direction by forecast week",xlabel="Forecast horizon",ylabel="Mean scaled absolute error")
    axes[0,1].legend(fontsize=8)
    for family,color in [("original_four","#397e9b"),("additional_overlapping","#ba7149")]:
        sub=scores[scores.family==family]
        axes[1,0].scatter(sub.offset,sub.score,label=family.replace("_"," "),color=color)
    axes[1,0].set(title="Benchmark across historical periods",xlabel="Weeks removed from history",ylabel="Score (higher is better)")
    axes[1,0].legend(fontsize=8)
    for label,color in [("falling","#ba7149"),("stable","#397e9b"),("rising","#31806c")]:
        sub=groups[(groups.family=="original_four")&(groups.dimension=="recent_level")&(groups.group==label)]
        axes[1,1].plot(sub.offset,sub.scaled_bias,marker="o",label=label,color=color)
    axes[1,1].axhline(0,color="gray",linewidth=1)
    axes[1,1].set(title="History-defined trend and signed error",xlabel="Historical offset (weeks)",ylabel="Scaled bias (+ over / − under)")
    axes[1,1].legend(fontsize=8)
    fig.savefig(out/"diagnosis.png",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    main()
