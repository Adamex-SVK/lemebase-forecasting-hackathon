"""Monotone distribution regression for direct weekly demand medians."""
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
import features as f

FLOW_NAMES = ["flow_coverage", "negative_flow_rate", "positive_flow_rate",
              "positive_flow_cover", "signed_flow_cover", "since_positive_flow"]
NAMES = f.FEATURES[:-1] + FLOW_NAMES + f.FEATURES[-1:] + ["log_threshold", "scaled_log_threshold", "cdf13", "cdf26"]


def features_at(matrix, origin):
    base, scale, valid = f.features_at(matrix, origin)
    start = max(0, origin - 14)
    stock = matrix.ledger[:, start:origin]
    units = matrix.U[:, start + 1:origin].astype(float)
    observed = np.isfinite(stock[:, :-1]) & np.isfinite(stock[:, 1:])
    signed = np.where(observed, np.diff(stock, axis=1) + units, 0.0)
    count = observed.sum(axis=1)
    denominator = np.where(observed, units, 0.0).sum(axis=1) + 1.0
    cover = signed.sum(axis=1) / denominator
    positive = signed > 1e-6
    rate_denominator = np.maximum(count, 1)
    recency = np.where(positive.any(axis=1), positive[:, ::-1].argmax(axis=1), 13)
    summaries = np.column_stack(((signed < -1e-6).sum(axis=1) / rate_denominator,
                                 positive.sum(axis=1) / rate_denominator,
                                 np.log1p(np.maximum(signed, 0).sum(axis=1) / denominator),
                                 np.sign(cover) * np.log1p(np.abs(cover)), recency))
    summaries[count == 0] = np.nan
    return np.column_stack((base, count / 13, summaries)), scale, valid


def threshold_features(base, scale, recent, h, threshold, series):
    """All four threshold summaries increase weakly with the test quantity."""
    history = recent[series]
    below = history <= threshold[:, None]
    return np.column_stack((base[series], h, np.log1p(threshold),
                            np.log1p(threshold / scale[series]),
                            below[:, -13:].mean(axis=1), below.mean(axis=1))).astype("float32")


def examples(matrix):
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for origin in f.make_origins(matrix.W):
        base, scale, valid = features_at(matrix, origin)
        idx = np.flatnonzero(valid)
        if not len(idx):
            continue
        recent = matrix.U[:, max(0, origin - 26):origin]
        positive_size = recent.sum(axis=1) / np.maximum(1, (recent > 0).sum(axis=1))
        series = np.repeat(idx, 6)
        h = rng.integers(1, 14, size=len(series))
        mix = rng.random(len(series))
        observed = recent[series, rng.integers(recent.shape[1], size=len(series))]
        upper = 4 * positive_size[series] + 1
        sampled = np.floor(np.expm1(rng.random(len(series)) * np.log1p(upper + 1)))
        threshold = np.where(mix < .2, 0, np.where(mix < .5, observed, sampled))
        xs.append(threshold_features(base, scale, recent, h, threshold, series))
        ys.append(matrix.U[series, origin + h - 1] <= threshold)
    return np.concatenate(xs), np.concatenate(ys)


def fit(samples):
    x = np.concatenate([s[0] for s in samples]) if len(samples) > 1 else samples[0][0]
    y = np.concatenate([s[1] for s in samples]) if len(samples) > 1 else samples[0][1]
    params = dict(objective="binary", learning_rate=.05, num_leaves=31, min_data_in_leaf=200,
                  lambda_l2=1, num_threads=3, seed=0, verbosity=-1, force_col_wise=True,
                  deterministic=True, monotone_constraints=[0] * (len(f.FEATURES) + len(FLOW_NAMES)) + [1] * 4)
    return lgb.train(params, lgb.Dataset(x, label=y, feature_name=NAMES), num_boost_round=240)


def support(matrices):
    seen = np.unique(np.concatenate([np.unique(m.U) for m in matrices]))
    return np.union1d(np.arange(np.floor(seen.max()) + 1), seen)


def medians(model, matrix, levels):
    base, scales, valid = features_at(matrix, matrix.W)
    ids = np.flatnonzero(valid)
    out = np.zeros((matrix.n, 13))
    recent = matrix.U[:, -26:]
    for start in range(0, len(ids), 1024):
        chunk = ids[start:start + 1024]
        series = np.repeat(chunk, 13)
        h = np.tile(np.arange(1, 14), len(chunk))
        lo = np.zeros(len(h), dtype=int)
        hi = np.full(len(h), len(levels) - 1, dtype=int)
        # CDF at the upper bound equals one under the bounded-support convention.
        while np.any(lo < hi):
            mid = (lo + hi) // 2
            x = threshold_features(base, scales, recent, h, levels[mid], series)
            above = model.predict(x, num_threads=3) >= .5
            hi = np.where(above, mid, hi)
            lo = np.where((lo < hi) & ~above, mid + 1, lo)
        out[chunk] = levels[lo].reshape(-1, 13)
    assert np.isfinite(out).all() and (out >= 0).all()
    return out
