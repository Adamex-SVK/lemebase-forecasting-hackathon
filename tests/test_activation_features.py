import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'iterations/11-activation-features')]
import activation as a
import features as f


class ActivationTests(unittest.TestCase):
    def matrix(self):
        U = np.zeros((3, 30), np.float32)
        U[0, 11:13] = [2, 4]
        U[1, 0] = 2
        U[1, 12] = 4
        return SimpleNamespace(U=U, n=3, W=30, OH=np.full_like(U, np.nan), OOS=np.zeros_like(U, dtype=bool))

    def test_frequency_and_positive_median(self):
        M = self.matrix()
        x = a.at(M, 13)
        self.assertAlmostEqual(x[0,0], 1)
        self.assertAlmostEqual(x[1,0], 2/13)
        self.assertEqual(x[2,0], 0)
        scale = np.abs(np.diff(M.U[:,:13].astype(float), axis=1)).mean(axis=1)
        self.assertAlmostEqual(x[0,1], 3/scale[0], places=5)
        self.assertTrue(np.isnan(x[2,1]))

    def test_future_invariance(self):
        M = self.matrix()
        before = a.at(M,13)
        M.U[:,13:] = 9999
        np.testing.assert_equal(before, a.at(M,13))

    def test_row_alignment_and_existing_columns(self):
        M = self.matrix()
        w = f.build_windows(M, [13,15])
        new = a.augment(w,M)
        np.testing.assert_equal(new['X'][:,:-3], w['X'][:,:-1])
        np.testing.assert_equal(new['X'][:,-1], w['X'][:,-1])
        for i in range(len(w['y'])):
            np.testing.assert_equal(new['X'][i,-3:-1], a.at(M,int(w['origin'][i]))[w['series'][i]])
        X, _, _ = a.inference(M)
        np.testing.assert_equal(X[:,-3:-1], np.repeat(a.at(M,30), 13, axis=0))


if __name__ == '__main__':
    unittest.main()
