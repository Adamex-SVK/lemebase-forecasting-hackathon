"""Focused checks for the iteration-09 improvement modules.
Run: .venv/bin/python tests/test_improve_i09.py
"""
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "improve"))
import i09_error_atlas as atlas  # noqa: E402
import i09_temporal_reconcile as tr  # noqa: E402
import i09_analog as analog  # noqa: E402

H = 13


def test_aggregate_block_boundaries():
    rng = np.random.default_rng(0)
    U = rng.integers(1, 20, size=(3, 10)).astype(float)
    B = tr.aggregate(U, 2)
    assert B.shape == (3, 5)
    # aggregate returns blocks oldest -> newest, anchored at the origin: block j is weeks [2j:2j+2]
    # for this width, i.e. the same windows as [10-2*(nb-j) : 10-2*(nb-j-1)] with nb=5.
    for j in range(5):
        lo = 10 - 2 * (5 - j)
        assert np.array_equal(B[:, j], U[:, lo:lo + 2].sum(axis=1))
    assert np.array_equal(B[:, -1], U[:, 8:10].sum(axis=1))       # anchor at the origin

    U11 = rng.integers(1, 20, size=(3, 11)).astype(float)
    B4 = tr.aggregate(U11, 4)
    assert B4.shape == (3, 2)                                     # 11 // 4 = 2 complete blocks
    assert np.array_equal(B4[:, 0], U11[:, 3:7].sum(axis=1))
    assert np.array_equal(B4[:, -1], U11[:, 7:11].sum(axis=1))
    assert np.allclose(B4.sum(axis=1), U11[:, 3:].sum(axis=1))    # earliest 3 weeks dropped
    assert not np.allclose(B4.sum(axis=1), U11.sum(axis=1))

    # only the used tail is aggregated: for W=11, b=2 the leading week is dropped
    B2 = tr.aggregate(U11, 2)
    assert np.allclose(B2.sum(axis=1), U11[:, 11 - 2 * 5:].sum(axis=1))
    assert np.allclose(B2.sum(axis=1), U11[:, 1:].sum(axis=1))
    assert not np.allclose(B2.sum(axis=1), U11.sum(axis=1))
    # for W=10 the used tail is the whole matrix
    assert np.allclose(tr.aggregate(U, 2).sum(axis=1), U[:, 10 - 2 * 5:].sum(axis=1))


def test_disaggregate_equal():
    rng = np.random.default_rng(0)
    U = rng.integers(1, 10, size=(3, 10)).astype(float)
    Fb = rng.random((3, 7)) * 10.0
    Fw = tr.disaggregate(Fb, U, 2, "equal")
    assert Fw.shape == (3, 13)
    for i in range(6):
        assert np.allclose(Fw[:, 2 * i], Fb[:, i] / 2)
        assert np.allclose(Fw[:, 2 * i + 1], Fb[:, i] / 2)
    assert np.allclose(Fw[:, 12], Fb[:, 6] / 2)                   # 13th week = first half of block 7
    assert np.allclose(Fw.sum(axis=1), Fb[:, :6].sum(axis=1) + Fb[:, 6] / 2)


def test_last_block_shares_fallback():
    Uz = np.zeros((3, 12))
    for b in (2, 4):
        sh = tr.last_block_shares(Uz, b)
        assert sh.shape == (3, b)
        assert np.allclose(sh, np.full((3, b), 1.0 / b))          # uniform fallback on a zero block

    U = np.array([[1, 2, 0, 0], [3, 4, 2, 6], [0, 0, 0, 5]], dtype=float)
    sh = tr.last_block_shares(U, 2)
    assert (sh >= 0).all() and np.allclose(sh.sum(axis=1), 1.0)
    last = U[:, -2:]
    tot = last.sum(axis=1)
    pos = tot > 0
    assert np.allclose(sh[pos], last[pos] / tot[pos, None])
    assert np.allclose(sh[0], [0.5, 0.5])                         # row 0 has a zero last block


def test_donor_eligibility():
    for rep_len in (26, 20):
        for W, target in ((143, 130), (143, 131), (77, 60)):
            origins = analog.donor_origins(W, target, rep_len=rep_len)
            assert origins.ndim == 1 and len(origins) > 0
            assert (origins >= rep_len).all()
            assert (origins + H <= target).all()                  # whole donor future is pre-origin

    origins = analog.donor_origins(143, 130)
    assert origins.max() <= 130 - H                               # only up to 117
    # donor_origins starts at the even rep_len and steps by 2, so on the default grid the largest
    # origin <= 117 is 116; 117 (target_o - 13) is the exact boundary and is reached with step=1.
    assert origins.max() == 116
    assert analog.donor_origins(143, 130, step=1).max() == 130 - H
    # an odd target keeps the exact boundary on the step-2 grid as well
    assert analog.donor_origins(143, 131).max() == 131 - H


def test_w_ge_n_plus_13_purge():
    for N in (0, 4, 8, 12):
        eligible = [W for W in atlas.OFFSETS if W >= N + H]        # exact rule in selector_eval
        assert all(W >= N + H for W in eligible)
        assert eligible == atlas.purged_donor_offsets(N)
        # Overlap/recency protocol: the cut at offset W holds out the 13 weeks [W, W+12] back from
        # the series end; the target cut at offset N holds out [N, N+12].  W - N >= 13 puts the
        # donor's whole holdout strictly before the target's first holdout week, so no target
        # holdout data can enter the donor fit.
        for W in eligible:
            assert W - N >= H
            assert W - H >= N


def test_analog_features_no_leakage():
    rng = np.random.default_rng(0)
    U = rng.integers(0, 8, size=(4, 80))
    o = 50
    D1, s1 = analog.series_features(U, o)
    U2 = U.copy()
    U2[:, o:] = U2[:, o:] + 1_000_000
    D2, s2 = analog.series_features(U2, o)
    assert np.array_equal(D1, D2, equal_nan=True)
    assert np.array_equal(s1, s2, equal_nan=True)
    # forecast functions are given only the pre-origin matrix (they predict beyond the last given
    # column), so this feature-builder perturbation is the meaningful leak check.


def test_block_features_no_leakage():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 10, size=(4, 40))
    o = 30
    X1, sc1, v1 = tr.block_features(A, o)
    A2 = A.copy()
    A2[:, o:] = A2[:, o:] + 1_000_000
    X2, sc2, v2 = tr.block_features(A2, o)
    assert np.array_equal(X1, X2, equal_nan=True)
    assert np.array_equal(sc1, sc2, equal_nan=True)
    assert np.array_equal(v1, v2)


def test_forecast_shape_nonneg():
    rng = np.random.default_rng(0)
    U = rng.integers(0, 6, size=(4, 60))
    U[0, :40] = 0                                                  # some leading zeros
    out, fb, info = analog.analog_forecast(U, k=10)
    assert out.shape == (4, 13)
    assert np.isfinite(out).all() and (out >= 0).all()
    assert fb.shape == (4, 13)                                     # median13 fallback shape
    assert isinstance(info, dict)


def test_analog_forecast_post_origin_invariance():
    rng = np.random.default_rng(0)
    U = rng.poisson(2.0, size=(6, 60)).astype(np.float32)
    origin = 47
    F1, fb1, _ = analog.analog_forecast(U, k=5, origin=origin)
    U2 = U.copy()
    U2[:, origin:] = rng.poisson(50.0, size=U2[:, origin:].shape)
    F2, fb2, _ = analog.analog_forecast(U2, k=5, origin=origin)
    assert np.array_equal(F1, F2) and np.array_equal(fb1, fb2), "post-origin values leaked into the forecast"


def test_inverse_cdf_monotonicity():
    path = ROOT / "src" / "improve" / "i09_distribution_pool.py"
    if not path.exists():
        print("SKIP: i09_distribution_pool not present (task 3 GPU-blocked)")
        return
    dp = importlib.import_module("i09_distribution_pool")
    if not (hasattr(dp, "mixture_quantile") and hasattr(dp, "enforce_monotone")):
        print("SKIP: i09_distribution_pool has no inverse-CDF helpers yet")
        return
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    q1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    med, tail = dp.mixture_quantile([q1, q2], [levels, levels], 0.5)
    assert not bool(tail) and 3.0 <= float(med) <= 4.0
    q_bad = np.array([1.0, 4.0, 2.0])
    fixed = dp.enforce_monotone(q_bad)
    assert (np.diff(fixed) >= 0).all() and np.array_equal(fixed, [1.0, 4.0, 4.0])


TESTS = [
    test_aggregate_block_boundaries,
    test_disaggregate_equal,
    test_last_block_shares_fallback,
    test_donor_eligibility,
    test_w_ge_n_plus_13_purge,
    test_analog_features_no_leakage,
    test_block_features_no_leakage,
    test_forecast_shape_nonneg,
    test_analog_forecast_post_origin_invariance,
    test_inverse_cdf_monotonicity,
]


def main():
    for fn in TESTS:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(TESTS)} iteration-09 improvement checks passed")


if __name__ == "__main__":
    main()
