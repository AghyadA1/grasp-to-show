"""Learned scorer: feature definitions and model shapes (skipped without PyTorch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grasp2show.learn.data import N_GLOBAL, global_features
from grasp2show.learn.model import N_REASONS, Scorer
from grasp2show.transforms import frame_from_axes, make_T, rot_z


def test_gravity_features_in_gripper_frame():
    # Top-down grasp on an upright object; at the target the object is turned 90 deg about x.
    T = make_T(frame_from_axes(y=[0, 1, 0], z=[0, 0, -1]), [0, 0, 0.05])[None]
    start = make_T(rot_z(0.7), [0.5, 0, 0.06])                   # yaw does not change gravity
    turned = make_T(frame_from_axes(x=[1, 0, 0], z=[0, -1, 0]), [0.4, 0.2, 0.4])[None]
    f = global_features(T, start, turned, np.array([0.04]), np.array([0.2]), np.zeros(3), np.array([1.0, 0, 0]))
    assert f.shape == (1, N_GLOBAL)
    np.testing.assert_allclose(f[0, :3], [0, 0, 1], atol=1e-6)          # gravity along the approach
    g_target_obj = turned[0, :3, :3].T @ [0, 0, -1]
    np.testing.assert_allclose(f[0, 3:6], T[0, :3, :3].T @ g_target_obj, atol=1e-6)


def test_scorer_shapes_and_frame():
    clouds = torch.randn(3, 64, 7)
    m = Scorer(clouds)
    obj = torch.tensor([0, 2])
    R = torch.eye(3).repeat(2, 1, 1)
    t = torch.tensor([[0.1, 0, 0], [0, 0, 0]])
    logit, reason = m(obj, R, t, torch.zeros(2, N_GLOBAL), n_points=32)
    assert logit.shape == (2,) and reason.shape == (2, N_REASONS)
    pf = m.point_features(obj[:1], R[:1], t[:1])
    torch.testing.assert_close(pf[0, :3].T, (clouds[0, :, :3] - t[0]) * 10)
