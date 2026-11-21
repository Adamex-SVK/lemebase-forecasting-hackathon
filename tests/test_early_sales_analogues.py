import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'iterations/16-early-sales-analogues'))
from analogues import peer_forecast


class EarlySalesTests(unittest.TestCase):
    def test_equal_age_cutoff_alignment_and_ties(self):
        history = np.zeros((18, 33))
        history[:16, 16:20] = [4, 0, 2, 1]
        history[:16, 20:] = np.arange(1, 14)
        history[16, 17:21] = [4, 0, 2, 1]  # Continuation is one week short.
        history[16, 21:] = 1000
        history[17, -4:] = [4, 0, 2, 1]
        champion = np.full((18, 13), 41)
        forecast, records = peer_forecast(history, champion)
        self.assertEqual(records, [dict(target=17, age=4, eligible=16, used=16)])
        np.testing.assert_array_equal(forecast[17], np.arange(1, 14))
        np.testing.assert_array_equal(forecast[:17], champion[:17])
        # Make two donors incomparable; 14 remaining triggers exact fallback.
        history[:2, 16] = 999
        forecast, records = peer_forecast(history, champion)
        self.assertEqual(records[0]['used'], 0)
        np.testing.assert_array_equal(forecast, champion)

    def test_weekly_shape_and_selling_frequency_affect_selection(self):
        history = np.zeros((31, 40))
        history[:15, 13:17] = [3, 0, 0, 3]
        history[15:30, 13:17] = [1, 2, 3, 0]
        history[:15, 17:30] = 2
        history[15:30, 17:30] = 99
        history[-1,-4:] = [3, 0, 0, 3]
        forecast, records = peer_forecast(history, np.zeros((31,13)))
        self.assertEqual(records[0]['used'], 15)
        np.testing.assert_array_equal(forecast[-1], np.full(13, 2))


if __name__ == '__main__':
    unittest.main()
