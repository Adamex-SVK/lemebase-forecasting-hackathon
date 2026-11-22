"""A leaf-distribution forest for weekly demand, with origin-specific normalization."""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import ExtraTreeRegressor

HERE = Path(__file__).resolve().parent
OUT = HERE / "forest/results"
sys.path.insert(0, str(HERE.parent / "31-demand-distribution"))
from train import history, TARGET
import features as f
import train_gbm as gb

N_TREES = 40
MAX_ROWS = 48000
PROBS = (np.arange(31) + .5) / 31


def context(m, o):
    base, _, valid = f.features_at(m, o)
    recent = m.U[:, max(0, o-26):o]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        size = np.nanmedian(np.where(recent > 0, recent, np.nan), axis=1)
    size = np.nan_to_num(size, nan=1.)
    summaries = []
    for n in (4, 13, 26):
        a = recent[:, -n:] / size[:, None]
        summaries.extend([a.mean(axis=1), (a>0).mean(axis=1),
                          np.quantile(a, .75, axis=1), np.max(a, axis=1)])
    # Encode missing history/stock summaries explicitly without imputing future values.
    x = np.column_stack([base, np.log1p(size), *summaries]).astype('float32')
    return np.nan_to_num(x, nan=-1.), size, valid


def data(m):
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for o in f.make_origins(m.W):
        base, size, valid = context(m, o)
        i = np.repeat(np.flatnonzero(valid), 6)
        h = rng.integers(1, 14, len(i))
        xs.append(np.column_stack([base[i], h]).astype('float32'))
        ys.append(m.U[i, o+h-1] / size[i])
    return np.concatenate(xs), np.concatenate(ys)


def fit(x, y):
    rng = np.random.default_rng(0)
    members = []
    for seed in range(N_TREES):
        count = min(len(y)//2, MAX_ROWS)
        grow, estimate = np.split(rng.choice(len(y), 2*count, replace=False), 2)
        tree = ExtraTreeRegressor(max_features=.85, min_samples_leaf=38,
                                  max_depth=18, random_state=seed)
        tree.fit(x[grow], np.log1p(y[grow]))
        leaves = tree.apply(x[estimate])
        order = np.lexsort((y[estimate], leaves))
        labels, first, length = np.unique(leaves[order], return_index=True, return_counts=True)
        indices = first[:, None] + (length[:, None]*PROBS).astype(int)
        lookup = np.full((tree.tree_.node_count, len(PROBS)), np.nan, dtype='float32')
        lookup[labels] = y[estimate][order[indices]]
        # A terminal node can occasionally receive no independent estimation rows.
        absent = np.setdiff1d(np.flatnonzero(tree.tree_.children_left == -1), labels)
        if len(absent):
            grow_leaves = tree.apply(x[grow])
            for leaf in absent:
                lookup[leaf] = np.quantile(y[grow][grow_leaves == leaf], PROBS)
        members.append((tree,lookup))
    return members


def predict(members, m, cut, name):
    base, size, valid = context(m, m.W)
    forecast = np.zeros((m.n, 13))
    active = np.flatnonzero(valid)
    for pos in range(0,len(active),256):
        ids = active[pos:pos+256]
        x = np.column_stack([base[np.repeat(ids,13)], np.tile(np.arange(1,14), len(ids))]).astype('float32')
        draws = np.column_stack([lookup[tree.apply(x)] for tree,lookup in members])
        forecast[ids] = np.median(draws,axis=1).reshape(-1,13) * size[ids,None]
    assert np.isfinite(forecast).all()
    gb.to_long(m, forecast, pd.date_range(cut,periods=13,freq="7D")).to_parquet(OUT/f"{name}.parquet",index=False)
    print(name, cut.date(), flush=True)


def main(stage):
    if stage == "target":
        for fold in TARGET:
            m, cut = history("test",fold)
            predict(fit(*data(m)), m, cut, f"test_{fold}")
    else:
        shops, samples = {}, []
        for shop in "ABCD":
            m, cut = history(shop,0)
            s = data(m)
            predict(fit(*s),m,cut,f"{shop}_own")
            shops[shop] = m, cut
            samples.append(s)
        model = fit(*(np.concatenate(a) for a in zip(*samples)))
        for shop,(m,cut) in shops.items():
            predict(model,m,cut,f"{shop}_pooled")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage",choices=["target","shops"])
    main(ap.parse_args().stage)
