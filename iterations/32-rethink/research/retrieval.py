"""Forecast conditional weekly medians from observed historical continuations."""
import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'iterations/31-demand-distribution'))
from train import history, OUT as BASE, TARGET
import validate as v

OUT = HERE / 'results'
H = 13
N_NEIGHBORS = 96


def descriptors(series, origin):
    recent = series[:, max(0, origin-26):origin]
    ids = np.flatnonzero((recent > 0).any(axis=1))
    recent = recent[ids].astype(float)
    positive = recent > 0
    typical = np.nanmedian(np.where(positive, recent, np.nan), axis=1)
    recency = positive[:, ::-1].argmax(axis=1)
    age = origin - (series[ids, :origin] > 0).argmax(axis=1)
    last = recent[np.arange(len(ids)), recent.shape[1]-1-recency]
    features = []
    for length in (4, 8, 13, 26):
        values = recent[:, -length:]
        features += [2*(values > 0).mean(axis=1), np.log1p(values.mean(axis=1)/typical)]
    features += [.5*np.log1p(typical), .5*np.log1p(recency), .5*np.log1p(np.minimum(age,52)),
                 .5*np.log1p(last/typical)]
    return ids, typical, np.column_stack(features)


def examples(matrix):
    xx, yy = [], []
    for origin in range(matrix.W-H,12,-4):
        i, qty, x = descriptors(matrix.U, origin)
        if len(i):
            xx.append(x)
            yy.append(matrix.U[i,origin:origin+H] / qty[:,None])
    return np.concatenate(xx), np.concatenate(yy)


def database(samples):
    x = np.concatenate([row[0] for row in samples])
    y = np.concatenate([row[1] for row in samples])
    return cKDTree(x), y


def export(matrix, cut, name, db):
    base = pd.read_parquet(BASE / f'{name}.parquet').set_index(v.KEYS+['week']).sort_index()
    rows = pd.MultiIndex.from_frame(matrix.keys)
    assert base.index.droplevel('week').unique().equals(rows)
    assert base.index.get_level_values('week').unique().equals(pd.date_range(cut, periods=H, freq='7D'))
    reference = base.forecast.to_numpy().reshape(matrix.n,H)
    output = reference.copy()
    tree, normalized = db
    ids, qty, target = descriptors(matrix.U, matrix.W)
    if tree.n >= N_NEIGHBORS:
        for start in range(0,len(ids),500):
            chunk = slice(start,start+500)
            neighbor = tree.query(target[chunk], k=N_NEIGHBORS, workers=2)[1]
            analogous = np.median(normalized[neighbor], axis=1) * qty[chunk,None]
            output[ids[chunk]] = .5*reference[ids[chunk]] + .5*analogous
    assert np.all(np.isfinite(output)) and np.all(output>=0)
    base['forecast'] = output.reshape(-1)
    base.reset_index().to_parquet(OUT/f'{name}.parquet',index=False)
    print(name, cut.date(), tree.n, len(ids), int((output!=reference).any(axis=1).sum()), flush=True)


def run(stages):
    OUT.mkdir(exist_ok=True)
    if 'target' in stages:
        for offset in TARGET:
            m, cut = history('test',offset)
            export(m, cut, f'test_{offset}',database([examples(m)]))
    if 'shops' in stages:
        matrices = {shop:history(shop,0) for shop in 'ABCD'}
        samples = {shop:examples(m) for shop, (m,cut) in matrices.items()}
        pooled = database(list(samples.values()))
        for shop,(m,cut) in matrices.items():
            export(m,cut,f'{shop}_own',database([samples[shop]]))
            export(m,cut,f'{shop}_pooled',pooled)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stages',nargs='*',default=['target','shops'])
    run(p.parse_args().stages)
