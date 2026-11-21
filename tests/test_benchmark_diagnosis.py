"""Boundary and leakage-contract checks for phase-1 diagnostic groups and fold purge."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


d=module("diagnosis",ROOT/"iterations/10-benchmark-diagnosis/diagnose.py")
p=module("protocol",ROOT/"iterations/10-benchmark-diagnosis/check_protocol.py")


class DiagnosisTests(unittest.TestCase):
    def test_activation_age_boundaries_and_never_sold(self):
        U=np.zeros((4,60))
        U[0,47]=1  # 13 weeks including first sale
        U[1,46]=1  # 14 weeks
        U[2,0]=1
        groups=d.history_groups(U,np.ones_like(U),[np.zeros((4,13))]*4,np.ones(4))
        self.assertEqual(groups["observed_activation_age"].tolist(),
                         ["first_sale_le13weeks","first_sale_14to26weeks","older","never_sold"])
        self.assertEqual(groups["activity13"].tolist(),["rare_1to3","inactive","inactive","inactive"])
        for labels in groups.values():
            self.assertEqual(len(labels),4)

    def test_future_changes_do_not_change_prefix_groups(self):
        rng=np.random.default_rng(0)
        sales=rng.integers(0,4,size=(5,80)).astype(float)
        stock=sales+3
        F=[np.ones((5,13))]*4
        a=d.history_groups(sales[:,:60],stock[:,:60],F,np.ones(5))
        sales[:,60:]=9999
        stock[:,60:]=np.nan
        b=d.history_groups(sales[:,:60],stock[:,:60],F,np.ones(5))
        for name in a:
            np.testing.assert_array_equal(a[name],b[name])

    def test_purge_boundary_includes_touching_nonoverlapping_weeks(self):
        self.assertEqual(p.eligible_offsets([0,4,12,13,16],0),[13,16])
        self.assertEqual(p.eligible_offsets([16,28,29,32,48],16),[29,32,48])
        self.assertEqual(p.eligible_offsets([0,4,8],0),[])


if __name__=="__main__":
    unittest.main()
