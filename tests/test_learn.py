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


def test_heatmap_paint_takes_best_nearby_grasp():
    from grasp2show.heatmap import paint

    pts = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.5, 0, 0]])
    c1 = np.array([[0.001, 0, 0], [0.099, 0, 0]])
    c2 = np.array([[0.002, 0, 0], [np.nan, np.nan, np.nan]])
    h = paint(pts, c1, c2, np.array([0.3, 0.8]), radius=0.01)
    assert h[0] == 0.3 and h[1] == 0.8 and np.isnan(h[2])
    h2 = paint(pts, c1, c2, np.array([[0.3, 0.0], [0.8, 0.1]]), radius=0.01)
    assert h2.shape == (3, 2) and h2[1, 1] == 0.1


def test_dense_labels_and_heatmap_net():
    import torch

    from grasp2show.learn.dense import HeatmapNet, grasp_from_prediction, point_labels, split_outputs

    pts = np.array([[0.0, 0, 0], [0.05, 0, 0], [0.5, 0, 0]])
    T = np.tile(np.eye(4), (2, 1, 1))
    T[1, :3, 3] = [0.05, 0.01, 0]
    d = {"T_obj_tcp": T, "S": np.array([[0.2, 0.0], [0.9, 0.0]]), "c1": np.array([[0.001, 0, 0], [0.051, 0, 0]]),
         "c2": np.full((2, 3), np.nan), "width": np.array([0.03, 0.04]), "lift_margin": np.array([0.5, 0.1])}
    score, params, stab = point_labels(pts, d, radius=0.01)
    assert score.shape == (2, 3) and score[0, 0] == np.float32(0.2) and score[0, 1] == np.float32(0.9)
    assert np.isnan(score[0, 2]) and score[1, 1] == 0.0
    assert np.allclose(params[0, 1, 6:9], [0.0, 0.01, 0]) and np.isclose(params[0, 1, 9], 0.04)
    assert np.isclose(stab[0], 0.5)
    Tg = grasp_from_prediction(pts[1], params[0, 1, 0:3], params[0, 1, 3:6], params[0, 1, 6:9])
    assert np.allclose(Tg, T[1])
    net = HeatmapNet(torch.randn(2, 64, 7), k=8).eval()
    out = split_outputs(net(torch.tensor([0, 1]), torch.randn(2, 25)))
    assert out["score"].shape == (2, 64) and torch.allclose(out["approach"].norm(dim=-1), torch.ones(2, 64))
