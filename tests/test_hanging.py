"""Hanging presentation (grasp2show/hanging.py): the show pose of a hooked object."""
import itertools
from types import SimpleNamespace

import numpy as np

from grasp2show.config import CameraConfig, SceneConfig
from grasp2show.hanging import show_targets
from grasp2show.transforms import inv_T, make_T, rot_z


def _rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def test_show_targets_face_the_scanner_and_keep_the_hanging():
    rng = np.random.default_rng(0)
    cfg = SceneConfig(camera=CameraConfig(pos=(0.30, 0.55, 0.60), look_at=(0.45, 0.20, 0.44)))
    T_obj_qr = make_T(_rot([0.3, 1, 0.2], 0.7), [0.05, 0.01, 0.06])
    obj = SimpleNamespace(T_obj_qr=T_obj_qr)
    T_obj_tcp = make_T(_rot(rng.normal(size=3), 1.2), [0.08, 0.0, 0.06])      # hand at the grasp (object upright)
    T_tcp_obj = make_T(_rot(rng.normal(size=3), 0.9), [0.0, 0.05, -0.04])    # where the object hangs from the hand
    cam, at = np.asarray(cfg.camera.pos), np.asarray(cfg.camera.look_at)
    (extra, T_tcp, T_obj), *others = show_targets(cfg, obj, T_obj_tcp[:3, :3], T_tcp_obj)
    assert extra == 0.0 and len(others) == 2
    # the object hangs from the hand exactly as found
    np.testing.assert_allclose(inv_T(T_tcp) @ T_obj, T_tcp_obj, atol=1e-12)
    # the hand only turned about the vertical: same tilt as it carries with
    dR = T_tcp[:3, :3] @ T_obj_tcp[:3, :3].T
    np.testing.assert_allclose(dR[2], [0, 0, 1], atol=1e-12)
    # QR centre at the look-at point, its normal's horizontal part pointing at the camera
    T_qr = T_obj @ T_obj_qr
    np.testing.assert_allclose(T_qr[:3, 3], at, atol=1e-12)
    n, c = T_qr[:3, 0], cam - at
    assert np.isclose(np.arctan2(n[1], n[0]), np.arctan2(c[1], c[0]), atol=1e-9)
    # the extra turns (used when the exact one is out of reach) turn the hand further about the vertical
    for extra_deg, T_tcp_k, _ in others:
        np.testing.assert_allclose(T_tcp_k[:3, :3], rot_z(np.radians(extra_deg)) @ T_tcp[:3, :3], atol=1e-12)


def test_carry_pose_raises_the_fingers():
    from grasp2show.hanging import carry_pose

    T = make_T(_rot([1.0, 0.3, 0.0], 1.4), [0.08, 0.0, 0.06])      # fingers about level
    C = carry_pose(T, 0.15, 25.0)
    assert np.isclose(np.degrees(np.arcsin(C[2, 2])), 25.0)
    np.testing.assert_allclose(C[:3, 3], T[:3, 3] + [0, 0, 0.15])
    h0, h1 = T[:2, 2] / np.linalg.norm(T[:2, 2]), C[:2, 2] / np.linalg.norm(C[:2, 2])
    np.testing.assert_allclose(h0, h1, atol=1e-12)                   # same heading, only raised
    up = make_T(_rot([1.0, 0.0, 0.0], -0.9), [0, 0, 0])             # already pointing up: unchanged
    np.testing.assert_allclose(carry_pose(up, 0.1)[:3, :3], up[:3, :3])


def test_rrt_connect_goes_around_an_obstacle():
    from grasp2show.episode import rrt_connect
    from grasp2show.franka_ik import Q_MAX, Q_MIN

    mid = 0.5 * (Q_MIN + Q_MAX)
    q0, g = mid.copy(), mid.copy()
    q0[0], g[0] = mid[0] - 1.0, mid[0] + 1.0

    def blocked(q):                                   # a wall across joint 0, open only at joint 1 > mid + 0.5
        return abs(q[0] - mid[0]) < 0.2 and q[1] < mid[1] + 0.5

    def edge_ok(a, b):
        return not any(blocked((1 - s) * a + s * b) for s in np.linspace(0, 1, 20))

    path = rrt_connect(q0, [g], edge_ok, np.random.default_rng(0), iters=3000)
    assert path is not None
    assert np.allclose(path[0], q0) and np.allclose(path[-1], g)
    assert all(edge_ok(a, b) for a, b in itertools.pairwise(path))
