"""Freeze a single model forecast, validate its format, then report the outcome."""
import hashlib
import io
import json

import numpy as np
import pandas as pd

from score_trial import HERE, ROOT, component, review, v
from make_submission import validate_submission, write_submission


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_forecasts(keys, weeks):
    parts,hashes = [],{}
    for name in review.MODELS:
        p = f"iterations/03-foundation-model/forecasts_final/{name}__test__final__padded.parquet"
        fc = pd.read_parquet(io.BytesIO(review.load_blob(review.SOURCE,p,hashes)))
        parts.append(review.matrix(fc,keys,weeks))
    own = pd.read_csv(io.BytesIO(review.load_blob(review.SOURCE,"submissions/lgbm_own_v1.csv",hashes)))
    wcols = [f"w{h}" for h in range(1,14)]
    tree = own.set_index("article_id",verify_integrity=True).loc[keys.article_id,wcols].to_numpy()
    return parts, tree, hashes


def main():
    results = HERE/"results"
    arm = json.loads((results/"selection.json").read_text())["winner"]
    history = v.load_weekly("test")
    keys = history[v.KEYS].drop_duplicates().sort_values(v.KEYS).reset_index(drop=True)
    ids = keys.article_id.to_numpy()
    weeks = pd.date_range(history.week.max()+pd.Timedelta(days=7),periods=13,freq="7D")
    dead = history.pivot(index=v.KEYS,columns="week",values="units").reindex(
        pd.MultiIndex.from_frame(keys)).iloc[:,-8:].sum(axis=1).to_numpy()==0
    parts, tree, hashes = base_forecasts(keys, weeks)
    wcols = [f"w{h}" for h in range(1,14)]
    path = ROOT/"submissions/ensemble_top3_blend_v1.csv"
    champion = review.compose(parts,tree,dead)
    np.testing.assert_array_equal(champion,pd.read_csv(path).set_index("article_id").loc[ids,wcols])
    if arm!="incumbent":
        assert arm in ("own","pooled")
        pred,_ = component(results/"gpu"/f"test-final-{arm}",keys,history.week.max())
        selected = review.compose(parts[:-1]+[pred],tree,dead)
        path = ROOT/"submissions"/f"ensemble_top3_chronos_{arm}_finetuned_v1.csv"
        if path.exists():
            np.testing.assert_array_equal(selected,pd.read_csv(path).set_index("article_id").loc[ids,wcols])
        else:
            write_submission(selected,ids,path)
    assert validate_submission(path)
    freeze = dict(selected=arm,forecast=str(path.relative_to(ROOT)),sha256=digest(path),
                  training_through=str(history.week.max().date()),n_articles=len(ids),
                  week_starts=[str(x.date()) for x in weeks],base_blob_hashes=hashes)
    (results/"frozen_forecast.json").write_text(json.dumps(freeze,indent=2)+"\n")
    print(freeze["forecast"], freeze["sha256"])


if __name__=="__main__":
    main()
