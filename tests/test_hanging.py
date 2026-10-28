"""Hanging presentation (grasp2show/hanging.py): the show pose of a hooked object."""
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
    (extra, T_tcp, T_obj), *others = show_targets(cfg, obj, T_obj_tcp, T_tcp_obj)
    assert extra == 0.0 and len(others) == 2
    # the object hangs from the hand exactly as found
    np.testing.assert_allclose(inv_T(T_tcp) @ T_obj, T_tcp_obj, atol=1e-12)
    # the hand only turned about the vertical: same tilt as at the grasp
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
