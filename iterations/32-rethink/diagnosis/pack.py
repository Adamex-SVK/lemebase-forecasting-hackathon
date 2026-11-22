"""Project eligible nonnegative article forecasts onto historical sales increments."""
import numpy as np


def project(units, forecast):
    """Rows are articles. Columns are pre-origin weeks or future forecast horizons."""
    history = np.asarray(units)
    values = np.asarray(forecast)
    if history.ndim != 2 or values.ndim != 2 or len(history) != len(values):
        raise ValueError("Expected two article-by-week arrays with matching row counts")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative forecast values")
    projected = values.copy()
    recent = history[:, -52:]
    if recent.shape[1] < 5:
        return projected
    valid = (np.isfinite(recent) & (recent >= 0) & (recent == np.rint(recent))).all(axis=1)
    indices = np.flatnonzero(valid & ((recent > 0).sum(axis=1) >= 5))
    if len(indices):
        quantum = np.gcd.reduce(recent[indices].astype(np.int64), axis=1)
        selected = quantum > 1
        articles = indices[selected]
        sizes = quantum[selected, None]
        projected[articles] = np.rint(values[articles] / sizes) * sizes
    return projected
