"""Analytic IK and checks 3-5."""
import numpy as np
import pytest

from grasp2show.config import SceneConfig
from grasp2show.feasibility import FeasibilityChecker, twin_of
from grasp2show.franka_ik import Q_MAX, Q_MIN, fk, ik, limit_margin
from grasp2show.scene import build_scene
from grasp2show.task import scripted_grasps


def test_fk_matches_mujoco():
    import mujoco

    from grasp2show.scene import robot_spec

    m = robot_spec(SceneConfig()).compile()
    d = mujoco.MjData(m)
    q = np.array([0.3, -0.5, 0.2, -2.0, 0.4, 1.8, 0.7])
    d.qpos[:7] = q
    mujoco.mj_kinematics(m, d)
    T = fk(q)
    np.testing.assert_allclose(T[:3, 3], d.site("tcp").xpos, atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], d.site("tcp").xmat.reshape(3, 3), atol=1e-6)


def test_ik_recovers_configurations_in_its_domain():
    rng = np.random.default_rng(3)
    Q = Q_MIN + (Q_MAX - Q_MIN) * (0.05 + 0.9 * rng.random((300, 7)))
    Q = Q[Q[:, 3] < -0.5]                       # outside the uncovered straight-elbow band
    S = ik(fk(Q), Q[:, 6:7])[:, 0]              # (N, 4, 7)
    err = np.nanmin(np.where(np.isnan(S).any(-1), np.inf, np.abs(S - Q[:, None]).max(-1)), axis=-1)
    assert np.all(err < 1e-8)
    assert np.all(limit_margin(Q) > 0)


def test_twin_is_a_half_turn_about_the_approach():
    g = scripted_grasps(SceneConfig())["clear_qr"].T_obj_tcp
    t = twin_of(g)
    np.testing.assert_allclose(t[:3, 2], g[:3, 2])
    np.testing.assert_allclose(t[:3, 1], -g[:3, 1])
    np.testing.assert_allclose(twin_of(t), g)


@pytest.fixture(scope="module")
def checker():
    cfg = SceneConfig()
    return FeasibilityChecker(build_scene(cfg), cfg), cfg


def test_side_grasp_feasible_face_grasp_blocked(checker):
    """Same verdicts as the full simulation of day 1-2."""
    fc, cfg = checker
    g = scripted_grasps(cfg)
    rolls = np.radians([0, 90, 180, 270])
    clear = fc.evaluate(g["clear_qr"].T_obj_tcp, g["clear_qr"].width, rolls)
    cover = fc.evaluate(g["covers_qr"].T_obj_tcp, g["covers_qr"].width, rolls)
    assert clear.ok
    assert not cover.ok
    t = cover.twins[0].targets[0]
    assert t.reach.ok and t.stability_margin > 0                 # it fails on visibility only
    assert t.visibility.fraction < 0.9
    assert any("finger" in b for b in t.visibility.blocked_by)


def test_grasp_from_below_the_table_is_unreachable(checker):
    fc, cfg = checker
    g = scripted_grasps(cfg)["clear_qr"]
    T = g.T_obj_tcp.copy()
    T[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])           # approach from below
    T[:3, 3] = [0, 0, -0.02]
    rep = fc.evaluate(T, g.width, np.radians([0.0]))
    assert not any(tw.start_ok for tw in rep.twins)
