"""Verify keyed GPU artifacts, score complete ensembles and apply the agreed gate."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/"iterations/15-sale-probability-gate"))
from run import diagnose, read_components, review, validate as v

PRIMARY = [0,13,26,39]
EXTRA = [16,32]
ARMS = ["zero","own","pooled"]


def component(folder, keys, training_end):
    meta = json.loads((folder/"complete.json").read_text())
    assert meta["training_end"] == str(pd.Timestamp(training_end).date())
    with np.load(folder/"forecast.npz") as saved:
        np.testing.assert_array_equal(saved["store"],keys.store_id.to_numpy(dtype=str))
        np.testing.assert_array_equal(saved["article"],keys.article_id.to_numpy(dtype=str))
        pred = saved["forecast"].copy()
    assert pred.shape == (len(keys),13) and np.isfinite(pred).all() and (pred>=0).all()
    return pred,meta


def metrics(F,history,hold,keys,weeks,actual,scale):
    ok = np.isfinite(scale) & (scale>0)
    long = pd.DataFrame({"store_id":np.repeat(keys.store_id,13),"article_id":np.repeat(keys.article_id,13),
                         "week":np.tile(weeks,len(keys)),"forecast":F.reshape(-1)})
    official = v.score(history,hold,long)
    err = np.abs(actual[ok]-F[ok])/scale[ok,None]
    np.testing.assert_allclose(official["mase"],err.mean(),rtol=0,atol=1e-10)
    return dict(**official,early=100*(1-err[:,:7].mean()),late=100*(1-err[:,7:].mean())),err


def summarize(frame):
    base = frame[frame.variant=="incumbent"].set_index("offset")
    rows = []
    for variant,group in frame.groupby("variant"):
        cmp = group.set_index("offset")
        dif = cmp[["score","early","late"]]-base[["score","early","late"]]
        gains = dif.loc[PRIMARY].mean()
        wins = int((dif.loc[PRIMARY,"score"]>0).sum())
        extra = dif.loc[EXTRA,"score"]
        passed = gains.score>=.5 and wins>=3 and gains.early>=0 and gains.late>=0 and (extra>=0).all()
        rows.append(dict(variant=variant,mean_score=cmp.loc[PRIMARY,"score"].mean(),
                         mean_mase=cmp.loc[PRIMARY,"mase"].mean(),gain=gains.score,wins=wins,
                         early_gain=gains.early,late_gain=gains.late,gain_w16=extra.loc[16],
                         gain_w32=extra.loc[32],passes=bool(passed)))
    return pd.DataFrame(rows)


def score(root):
    rows,groups = [],[]
    locked = pd.read_csv(ROOT/"iterations/10-benchmark-diagnosis/results/scores.csv").set_index("offset")
    daily = pd.read_csv(HERE/"results/lightgbm/paired.csv").set_index("offset")
    for shop in ["test","A","B","C","D"]:
        full = v.load_weekly(shop)
        for offset in (PRIMARY+EXTRA if shop=="test" else [0]):
            prefix = full[full.week<=full.week.max()-pd.Timedelta(days=7*offset)]
            history,hold,cut = v.split(prefix)
            keys = history[v.KEYS].drop_duplicates().sort_values(v.KEYS).reset_index(drop=True)
            weeks = pd.date_range(cut,periods=13,freq="7D")
            idx = pd.MultiIndex.from_frame(keys)
            hist = history.pivot(index=v.KEYS,columns="week",values="units").reindex(idx)
            scale = v.history_scale(history).reindex(idx).to_numpy()
            actual = review.matrix(hold.rename(columns={"units":"forecast"}),keys,weeks)
            cold = (hist.to_numpy()==0).all(axis=1)
            counts = dict(cold_excluded=int(cold.sum()),cold_sold=int((cold&(actual.sum(axis=1)>0)).sum()))
            preds = {}
            for arm in ARMS:
                pred,meta = component(root/f"{shop}-{offset}-{arm}",keys,history.week.max())
                preds[arm] = pred
                m,_ = metrics(pred,history,hold,keys,weeks,actual,scale)
                rows.append(dict(shop=shop,offset=offset,start=str(cut.date()),level="chronos",variant=arm,
                                 steps=meta["steps"],**m,**counts))
            if shop!="test":
                continue
            parts,tree = read_components(offset,keys,weeks,offset in PRIMARY,{})
            recent_zero = hist.iloc[:,-8:].sum(axis=1).to_numpy()==0
            # Preserved pinned Moirai/TiRex forecasts and original LightGBM for all arms.
            assert review.MODELS[-1]=="chronos-t5-small"
            variants = {"incumbent":review.compose(parts,tree,recent_zero)}
            daily_tree = np.load(HERE/f"results/lightgbm/tree_{offset}.npy")
            variants["daily"] = review.compose(parts,daily_tree,recent_zero)
            for arm,pred in preds.items():
                variants[arm] = review.compose(parts[:-1]+[pred],tree,recent_zero)
            stock = history.pivot(index=v.KEYS,columns="week",values="on_hand").reindex(idx).to_numpy()
            labels = diagnose.history_groups(hist.to_numpy(),stock,parts+[tree],scale)
            for name,F in variants.items():
                m,err = metrics(F,history,hold,keys,weeks,actual,scale)
                if name=="incumbent":
                    np.testing.assert_allclose(m["score"],locked.loc[offset,"score"],rtol=0,atol=1e-8)
                if name=="daily":
                    np.testing.assert_allclose(m["score"],daily.loc[offset,"daily"],rtol=0,atol=1e-8)
                rows.append(dict(shop=shop,offset=offset,start=str(cut.date()),level="ensemble",variant=name,
                                 **m,**counts))
                for dim in ("observed_activation_age","activity13","last_stock","stock_missing13"):
                    lab = labels[dim][np.isfinite(scale)&(scale>0)]
                    for group in np.unique(lab):
                        mask = lab==group
                        groups.append(dict(offset=offset,variant=name,dimension=dim,group=group,
                                           scored=int(mask.sum()),mase=err[mask].mean()))
    results = pd.DataFrame(rows)
    out = HERE/"results"
    results.to_csv(out/"scores.csv",index=False)
    pd.DataFrame(groups).to_csv(out/"groups.csv",index=False)
    summary = summarize(results[results.level=="ensemble"])
    summary.to_csv(out/"promotion.csv",index=False)
    cross = results[results.shop!="test"].pivot(index="shop",columns="variant",values="score")
    cross["pooled_minus_own"] = cross.pooled-cross.own
    cross.loc["equal_weight_A_D"] = cross.mean()
    cross.to_csv(out/"cross_shop.csv")
    # LightGBM failed its standalone gate, so the registered combined arm is ineligible.
    assert not summary.set_index("variant").loc["daily","passes"]
    passing = summary[summary.passes & summary.variant.isin(["own","pooled"])]
    winner = "incumbent" if passing.empty else passing.sort_values(["mean_score","variant"],ascending=[False,True]).iloc[0].variant
    (out/"selection.json").write_text(json.dumps(dict(winner=winner),indent=2)+"\n")
    print(summary.to_string(index=False))
    print(cross.to_string())
    print("Selected",winner)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-results",type=Path,default=HERE/"results/gpu")
    score(ap.parse_args().gpu_results)
