"""The optional C++ kernels must give the same answers as the numpy reference."""
import numpy as np
import pytest

from grasp2show._accel import HAVE_CPP
from grasp2show.franka_ik import Q_MAX, Q_MIN, fk, ik
from grasp2show.stability import StabilityParams, utilisation_wrench

pytestmark = pytest.mark.skipif(not HAVE_CPP, reason="C++ extension not built (pip install ./cpp)")


def test_ik_cpp_matches_numpy():
    rng = np.random.default_rng(1)
    Q = Q_MIN + (Q_MAX - Q_MIN) * (0.02 + 0.96 * rng.random((400, 7)))
    T = fk(Q)
    q7 = np.linspace(Q_MIN[6] + 0.01, Q_MAX[6] - 0.01, 9)
    a, b = ik(T, q7, use_cpp=True), ik(T, q7, use_cpp=False)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], atol=1e-10)
    assert (~np.isnan(a[..., 0])).sum() > 1000


def test_stability_cpp_matches_numpy():
    rng = np.random.default_rng(2)
    n = 5000
    y = rng.normal(size=(n, 3))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    centre = rng.normal(scale=0.02, size=(n, 3))
    w = rng.uniform(0.01, 0.07, n)[:, None]
    c1, c2 = centre - 0.5 * w * y, centre + 0.5 * w * y
    F = rng.normal(scale=4.0, size=(n, 3))
    Tq = rng.normal(scale=0.05, size=(n, 3))
    com = np.array([0.003, -0.002, 0.01])
    p = StabilityParams()
    a = utilisation_wrench(c1, c2, y, com, F, Tq, p, use_cpp=True)
    b = utilisation_wrench(c1, c2, y, com, F, Tq, p, use_cpp=False)
    assert np.array_equal(np.isinf(a), np.isinf(b))
    np.testing.assert_allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-9)
