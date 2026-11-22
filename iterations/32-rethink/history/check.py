"""Verify causal stock features, identical training samples, and keyed forecasts."""
import argparse
import importlib.util
import json
import platform
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import distribution as flow
import train

BASE = flow.HERE.parents[1] / '31-demand-distribution'
spec = importlib.util.spec_from_file_location('iteration31_distribution', BASE/'distribution.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def check_features():
    M, cut = train.history('test', 39)
    origin = 40
    x0, scale0, valid0 = flow.features_at(M, origin)
    saved = M.ledger[:, origin:].copy()
    M.ledger[:, origin:] = 1e9
    x1, scale1, valid1 = flow.features_at(M, origin)
    np.testing.assert_array_equal(x1,x0)
    np.testing.assert_array_equal(scale1,scale0)
    np.testing.assert_array_equal(valid0,valid1)
    M.ledger[:, origin:] = saved
    # Use a shorter prefix for an independently sampled-window comparison.
    M.W = origin
    x0, y0 = original.examples(M)
    x1, y1 = flow.examples(M)
    assert np.array_equal(y0,y1)
    indices = [flow.NAMES.index(name) for name in original.NAMES]
    np.testing.assert_array_equal(x1[:,indices], x0)
    print('Past-only features and matched sampling:', len(y1), 'binary examples at', cut.date())


def check_results():
    names = [f'test_{k}' for k in train.TARGET]
    names += [f'{s}_{a}' for s in 'ABCD' for a in ['own','pooled']]
    rows = []
    for name in names:
        df = pd.read_parquet(train.OUT/f'{name}.parquet')
        ref = pd.read_parquet(BASE/'results'/f'{name}.parquet')
        keys = ['store_id','article_id','week']
        assert not df.duplicated(keys).any() and df[keys].equals(ref[keys])
        assert np.isfinite(df.forecast).all() and (df.forecast>=0).all()
        rows.append(dict(run=name,forecasts=len(df),different_from_original=int((df.forecast!=ref.forecast).sum())))
    results = pd.DataFrame(rows)
    results.to_csv(train.OUT/'checks.csv',index=False)
    M, cut = train.history('test','final')
    model = lgb.Booster(model_file=str(train.OUT/'target_full_model.txt'))
    y = flow.medians(model, M, flow.support([M]))
    saved = pd.read_parquet(train.OUT/'test_final.parquet').forecast.to_numpy()
    np.testing.assert_array_equal(y.ravel(),saved)
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    lightgbm=lgb.__version__, seed=0, threads=3)
    (train.OUT/'versions.json').write_text(json.dumps(versions,indent=2))
    print(results.to_string(index=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['features','results'])
    {'features':check_features,'results':check_results}[p.parse_args().stage]()
