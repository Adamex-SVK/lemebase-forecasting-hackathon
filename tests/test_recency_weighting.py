import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'iterations/13-recency-weighting'))
import recency


class RecencyTests(unittest.TestCase):
    def test_half_life_and_total_weight(self):
        win={'origin':np.array([13,117]),'w':np.ones(2,dtype=np.float32),'X':np.ones((2,1))}
        weighted=recency.apply(win)
        self.assertAlmostEqual(weighted['w'][1]/weighted['w'][0],2.)
        self.assertAlmostEqual(float(weighted['w'].sum()),2.)
        np.testing.assert_array_equal(win['w'],np.ones(2))
        self.assertIs(weighted['X'],win['X'])

    def test_equal_origin_rows_and_translation_invariance(self):
        win={'origin':np.array([13,13,39,39,65,65]),'w':np.ones(6)}
        weights=recency.apply(win)['w']
        self.assertEqual(weights[0],weights[1])
        self.assertLess(weights[0],weights[2])
        self.assertLess(weights[2],weights[4])
        shifted={**win,'origin':win['origin']+100}
        np.testing.assert_array_equal(weights,recency.apply(shifted)['w'])

    def test_preserves_nonuniform_base_weight_sum(self):
        win={'origin':np.array([13,39,65]),'w':np.array([10.,1.,1.])}
        weights=recency.apply(win)['w']
        self.assertAlmostEqual(float(weights.sum()),12.,places=5)
        self.assertAlmostEqual((weights[2]/win['w'][2])/(weights[0]/win['w'][0]),2**.5,places=5)


if __name__=='__main__':
    unittest.main()
