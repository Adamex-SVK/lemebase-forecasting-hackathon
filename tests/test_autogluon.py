"""Run in the optional environment: .venv-autogluon/bin/python tests/test_autogluon.py"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from autogluon_adapter import ChallengeMASE, from_predictions, to_timeseries, validation_windows
import validate as v


class AdapterTests(unittest.TestCase):
    def setUp(self):
        rows = []
        # Leading zeros, zero scale, and constant positive histories, plus active series.
        for store, article, units in [("s", "a", [0, 0, 2, 0, 4, 1]),
                                      ("s", "b", [1, 3, 1, 3, 2, 4]),
                                      ("s", "cold", [0, 0, 0, 0, 1, 2]),
                                      ("other", "a", [5, 5, 5, 5, 8, 7])]:
            for week, u in zip(pd.date_range("2024-01-01", periods=6, freq="7D"), units):
                rows.append(dict(store_id=store, article_id=article, week=week, units=float(u)))
        self.df = pd.DataFrame(rows)
        self.train, self.hold, _ = v.split(self.df, horizon=2)
        self.ts, self.mapping = to_timeseries(self.train, "test")
        future, _ = to_timeseries(self.hold, "test")
        self.future = future

    def test_metric_matches_project_and_excludes_constant_histories(self):
        pred = self.future.copy()
        pred["mean"] = pred.target + 1
        fc = from_predictions(pred, self.mapping, self.hold)
        expected = v.score(self.train, self.hold, fc, horizon=2)
        metric = ChallengeMASE()
        metric.save_past_metrics(self.ts, seasonal_period=52)
        actual = metric.compute_metric(self.future, pred)
        self.assertAlmostEqual(actual, expected["mase"])
        self.assertEqual(expected["n_scored"], 2)
        # Internal AG call must also match, including its sign convention.
        full, _ = to_timeseries(self.df, "test")
        metric.prediction_length = 2
        self.assertAlmostEqual(metric(full, pred), -expected["mase"])

    def test_prediction_alignment_and_clipping(self):
        pred = self.future.copy()
        pred["mean"] = -2.0
        pred = pred.iloc[::-1]
        fc = from_predictions(pred, self.mapping, self.hold)
        self.assertTrue((fc.forecast == 0).all())
        with self.assertRaises(ValueError):
            from_predictions(pred.iloc[:-1], self.mapping, self.hold)
        pred["mean"] = np.inf
        with self.assertRaises(ValueError):
            from_predictions(pred, self.mapping, self.hold)

    def test_same_article_in_different_stores_does_not_collide(self):
        self.assertEqual(self.mapping.item_id.nunique(), 4)
        self.assertEqual(self.ts.shape[1], 1)
        self.assertLess(self.ts.index.get_level_values("timestamp").max(), self.hold.week.min())

    def test_short_history_window_budget(self):
        self.assertEqual(validation_windows(130, 3), 3)
        self.assertEqual(validation_windows(29, 3), 1)
        self.assertEqual(validation_windows(33, 3), 1)
        self.assertEqual(validation_windows(26, 3), 0)
        self.assertEqual(validation_windows(16, 3), 0)

    def test_saved_reference_metric_parity(self):
        df = v.load_weekly("test")
        train, hold, _ = v.split(df)
        path = v.ROOT / "logs/chronos_four/LightGBM_0.parquet"
        if not path.exists():
            self.skipTest("Historical saved predictions are not present on this checkout")
        fc = pd.read_parquet(path)
        ts, mapping = to_timeseries(train, "test")
        future, _ = to_timeseries(hold, "test")
        pred = fc.merge(mapping, on=v.KEYS, validate="many_to_one").rename(
            columns={"week": "timestamp", "forecast": "mean"})
        pred = TimeSeriesDataFrame.from_data_frame(pred[["item_id", "timestamp", "mean"]])
        metric = ChallengeMASE()
        metric.save_past_metrics(ts)
        self.assertAlmostEqual(metric.compute_metric(future, pred), v.score(train, hold, fc)["mase"], places=10)


if __name__ == "__main__":
    unittest.main()
