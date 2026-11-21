"""Optional AutoGluon boundary: stable IDs, strict alignment, and D26 MASE."""
import json

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer

import validate as v


class ChallengeMASE(TimeSeriesScorer):
    """Negative MASE inside AutoGluon; report (1-MASE)*100 externally.

    Padding is done before conversion. Always use adjacent-week differences,
    excluding zero/undefined scales, regardless of seasonal frequency defaults.
    Clip negatives identically during selection and final evaluation.
    """
    greater_is_better_internal = False
    optimum = 0.0
    optimized_by_median = True
    equivalent_tabular_regression_metric = "mean_absolute_error"

    def save_past_metrics(self, data_past, target="target", **kwargs):
        past = data_past[target].sort_index()
        self.scales = past.groupby(level="item_id", sort=False).diff().abs().groupby(level="item_id").mean()

    def clear_past_metrics(self):
        self.scales = None

    def compute_metric(self, data_future, predictions, target="target", **kwargs):
        if not predictions.index.is_unique or not data_future.index.is_unique:
            raise ValueError("Duplicate forecast/actual keys")
        if len(predictions) != len(data_future) or not data_future.index.isin(predictions.index).all():
            raise ValueError("Forecast keys do not match actual keys")
        p = predictions["mean"].reindex(data_future.index)
        if not np.isfinite(p).all():
            raise ValueError("Non-finite predictions")
        ae = (data_future[target] - p.clip(lower=0)).abs().groupby(level="item_id").mean()
        scales = self.scales.reindex(ae.index)
        ok = np.isfinite(scales) & (scales > 0)
        if not ok.any():
            raise ValueError("No series has a nonzero history scale")
        return float((ae[ok] / scales[ok]).mean())


def to_timeseries(train, shop):
    """Units-only challenger. No holdout targets or future stock are accepted."""
    mapping = train[v.KEYS].drop_duplicates().copy()
    mapping["item_id"] = [json.dumps([shop, str(s), str(a)], separators=(",", ":"))
                          for s, a in mapping.itertuples(index=False, name=None)]
    d = train[v.KEYS + ["week", "units"]].merge(mapping, on=v.KEYS, validate="many_to_one")
    if d.duplicated(["item_id", "week"]).any():
        raise ValueError("Duplicate training keys")
    d = d.rename(columns={"week": "timestamp", "units": "target"})
    ts = TimeSeriesDataFrame.from_data_frame(d[["item_id", "timestamp", "target"]])
    return ts.sort_index(), mapping


def from_predictions(predictions, mapping, hold):
    """Reject omissions/extra rows rather than silently filling failed forecasts."""
    p = predictions.reset_index().rename(columns={"timestamp": "week", "mean": "forecast"})
    if p.duplicated(["item_id", "week"]).any() or not np.isfinite(p.forecast).all():
        raise ValueError("Duplicate or non-finite predictions")
    p = p.merge(mapping, on="item_id", how="left", validate="many_to_one")
    if p[v.KEYS].isna().any().any():
        raise ValueError("Unknown forecast IDs")
    keys = v.KEYS + ["week"]
    expected = hold[keys]
    m = expected.merge(p[keys + ["forecast"]], on=keys, how="outer",
                       validate="one_to_one", indicator=True)
    if not (m["_merge"] == "both").all():
        raise ValueError("Forecast must cover exactly the expected series and 13 weeks")
    m["forecast"] = m.forecast.clip(lower=0)
    return m[keys + ["forecast"]]


def validation_windows(train_weeks, requested):
    """AG discards series <= (windows+1)*13 weeks; never touch outer holdout."""
    return min(requested, max(0, (train_weeks - 1) // v.HORIZON - 1))
