"""Opt-in training-only recency weights, preserving total base weight."""
import numpy as np

HALF_LIFE = 104.


def apply(win):
    origins = np.asarray(win['origin'])
    base = np.asarray(win['w'], dtype=np.float64)
    if len(base) == 0 or base.shape != origins.shape:
        raise ValueError('Require aligned nonempty origin and weight arrays')
    if not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError('Require finite positive base weights')
    factor = np.exp2((origins-origins.max())/HALF_LIFE)
    weights = base*factor
    weights *= base.sum()/weights.sum()
    return {**win, 'w':weights.astype(np.float32)}
