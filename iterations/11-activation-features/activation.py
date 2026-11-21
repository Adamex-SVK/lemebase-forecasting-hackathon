"""Opt-in activation features; original feature builder remains unchanged."""
import warnings
import numpy as np
import features as f

EXTRA = ['activation_frequency13', 'positive_median13_scaled']
NAMES = f.FEATURES[:-1] + EXTRA + ['h']


def at(M, o):
    assert f.MIN_HIST <= o <= M.W
    U = M.U[:, :o].astype(np.float64)
    positive = U > 0
    age = np.where(positive.any(axis=1), o - positive.argmax(axis=1), 0)
    denominator = np.minimum(13, age)
    frequency = np.divide(positive[:, -13:].sum(axis=1), denominator,
                          out=np.zeros(M.n), where=denominator > 0)
    scale = np.abs(np.diff(U, axis=1)).mean(axis=1)
    recent = U[:, -13:]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(np.where(recent > 0, recent, np.nan), axis=1)
    median = median / np.where(scale > 0, scale, 1.)
    return np.column_stack([frequency, median]).astype(np.float32)


def augment(win, M):
    extra = np.empty((len(win['y']), 2), dtype=np.float32)
    for o in np.unique(win['origin']):
        mask = win['origin'] == o
        extra[mask] = at(M, int(o))[win['series'][mask]]
    return {**win, 'X': np.column_stack([win['X'][:, :-1], extra, win['X'][:, -1]])}


def inference(M):
    X, scale, valid = f.inference_features(M)
    extra = np.repeat(at(M, M.W), f.H, axis=0)
    return np.column_stack([X[:, :-1], extra, X[:, -1]]), scale, valid
