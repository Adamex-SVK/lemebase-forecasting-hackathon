"""Evaluate two source-derived timing features in the fixed competition ensemble."""
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from daily import HERE, ROOT, NAMES
sys.path.insert(0,str(ROOT / "iterations/15-sale-probability-gate"))
from run import features as f, read_components, review, train_gbm as gb, validate as v, diagnose

PRIMARY = [0,13,26,39]
EXTRA = [16,32]


def train(matrix, extra, output):
    def dataset(origins):
        w = f.build_windows(matrix, origins)
        timing = extra[w["origin"]-13, w["series"]]
        X = np.column_stack((w["X"][:,:-1],timing,w["X"][:,-1]))
        return lgb.Dataset(X,w["y"],weight=w["w"],feature_name=f.FEATURES[:-1]+NAMES+["h"])
    origins = f.make_origins(matrix.W)
    inner = origins[-7:]
    earlier = origins[origins<=inner[0]-13]
    assert earlier[-1]+13 <= inner[0] and origins[-1]+13 <= matrix.W
    tr, va = dataset(earlier), dataset(inner)
    va.set_reference(tr)
    trial = lgb.train(gb.PARAMS,tr,num_boost_round=3000,valid_sets=[va],
                      callbacks=[lgb.early_stopping(50,verbose=False)])
    rounds = int(trial.best_iteration*1.1)+1
    del tr, va, trial
    model = lgb.train(gb.PARAMS,dataset(origins),num_boost_round=rounds)
    model.save_model(str(output))
    X, scale, valid = f.inference_features(matrix)
    timing = np.repeat(extra[matrix.W-13],13,axis=0)
    X = np.column_stack((X[:,:-1],timing,X[:,-1]))
    F = np.maximum(model.predict(X).reshape(matrix.n,13)*scale[:,None],0)
    F[~valid] = 0
    assert np.isfinite(F).all()
    return F,rounds


def run():
    full = v.load_weekly("test")
    with np.load(HERE/"results/daily_features.npz") as saved:
        extra, ids = saved["features"],saved["ids"]
    rows,groups,hashes = [],[],{}
    out = HERE/"results/lightgbm"
    out.mkdir(exist_ok=False)
    benchmark = pd.read_csv(ROOT/"iterations/10-benchmark-diagnosis/results/scores.csv").set_index("offset")
    for offset in PRIMARY+EXTRA:
        prefix = full[full.week<=full.week.max()-pd.Timedelta(days=7*offset)]
        history,hold,cut = v.split(prefix)
        matrix = f.ShopMatrix(prefix[prefix.week<cut])
        np.testing.assert_array_equal(matrix.keys.article_id.to_numpy(),ids)
        weeks = pd.date_range(cut,periods=13,freq="7D")
        parts,own = read_components(offset,matrix.keys,weeks,offset in PRIMARY,hashes)
        tree,rounds = train(matrix,extra,out/f"model_{offset}.txt")
        np.save(out/f"tree_{offset}.npy",tree)
        scale = v.history_scale(history).reindex(pd.MultiIndex.from_frame(matrix.keys)).to_numpy()
        ok = np.isfinite(scale) & (scale>0)
        actual = review.matrix(hold.rename(columns={"units":"forecast"}),matrix.keys,weeks)
        categories = diagnose.history_groups(matrix.U,matrix.OH,parts+[own],scale)
        row = dict(offset=offset,start=str(cut.date()),trees=rounds,
                   count28_known=int(np.isfinite(extra[matrix.W-13,:,0]).sum()),
                   recency_known=int(np.isfinite(extra[matrix.W-13,:,1]).sum()))
        for label, component in [("original",own),("daily",tree)]:
            forecast = review.compose(parts,component,matrix.U[:,-8:].sum(axis=1)==0)
            score = v.score(history,hold,gb.to_long(matrix,forecast,weeks))
            err = np.abs(actual[ok]-forecast[ok])/scale[ok,None]
            row.update({label:score["score"],label+"_early":100*(1-err[:,:7].mean()),
                        label+"_late":100*(1-err[:,7:].mean()),"n_scored":score["n_scored"],
                        "excluded":matrix.n-score["n_scored"],"unit_coverage":score["holdout_units_share_scored"]})
            for dim in ("observed_activation_age","activity13","last_stock","stock_missing13"):
                labels = categories[dim][ok]
                for name in np.unique(labels):
                    mask = labels==name
                    groups.append(dict(offset=offset,model=label,dimension=dim,group=name,
                                       n_scored=mask.sum(),mase=err[mask].mean()))
        np.testing.assert_allclose(row["original"],benchmark.loc[offset,"score"],rtol=0,atol=1e-8)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out/"paired.csv",index=False)
        pd.DataFrame(groups).to_csv(out/"groups.csv",index=False)
        print(f"Offset {offset}: {row['daily']-row['original']:+.4f} points",flush=True)
    (out/"source_blobs.json").write_text(json.dumps(hashes,indent=2)+"\n")


if __name__ == "__main__":
    run()
