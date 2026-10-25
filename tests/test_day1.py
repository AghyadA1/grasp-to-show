"""Fast checks of the day 1-2 building blocks. Run: pytest -q"""
import numpy as np
import pytest

from grasp2show.config import SceneConfig
from grasp2show.ik import PandaIK
from grasp2show.qr import decode_qr, make_qr_texture
from grasp2show.scene import build_scene
from grasp2show.task import flipped, scripted_grasps, show_pose
from grasp2show.transforms import make_T


@pytest.fixture(scope="module")
def cfg():
    return SceneConfig()


def test_scene_compiles(cfg):
    model = build_scene(cfg).model
    assert model.nu == 8
    assert model.actuator("gripper").ctrlrange[1] == pytest.approx(cfg.gripper.max_width)
    assert model.opt.impratio == pytest.approx(cfg.impratio)


def test_qr_roundtrip(tmp_path):
    import cv2

    path = make_qr_texture("hello", tmp_path / "qr.png")
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    assert decode_qr(rgb)[0] == "hello"


def test_ik_reaches_a_reachable_pose(cfg):
    ik = PandaIK(cfg)
    import mujoco

    d = mujoco.MjData(ik.model)
    q_true = np.array([0.3, -0.2, 0.1, -2.0, 0.1, 1.9, 0.5])
    d.qpos[:7] = q_true
    mujoco.mj_forward(ik.model, d)
    T = make_T(d.site("tcp").xmat.reshape(3, 3), d.site("tcp").xpos)
    r = ik.solve(T)
    assert r.success, (r.pos_err, r.rot_err)


def test_flipped_grasp_touches_same_points(cfg):
    g = scripted_grasps(cfg)["clear_qr"]
    f = flipped(g)
    np.testing.assert_allclose(f.T_obj_tcp[:3, 3], g.T_obj_tcp[:3, 3])           # same TCP
    np.testing.assert_allclose(f.T_obj_tcp[:3, 2], g.T_obj_tcp[:3, 2])           # same approach
    np.testing.assert_allclose(f.T_obj_tcp[:3, 1], -g.T_obj_tcp[:3, 1])          # fingers swapped


@pytest.mark.parametrize("roll_deg", [0, 45, -90])
def test_show_pose_faces_camera(cfg, roll_deg):
    obj = build_scene(cfg).obj
    T = show_pose(cfg, obj, np.radians(roll_deg))
    qr_world = T[:3, :3] @ [cfg.obj.half_size[0], 0, cfg.obj.qr_center_z] + T[:3, 3]
    np.testing.assert_allclose(qr_world, cfg.camera.look_at, atol=1e-9)
    to_cam = np.subtract(cfg.camera.pos, cfg.camera.look_at)
    cos = T[:3, 0] @ to_cam / np.linalg.norm(to_cam)
    assert cos == pytest.approx(1.0)


@pytest.mark.slow
def test_episode_clear_succeeds_covering_fails(cfg):
    """The day 1-2 result: two equally stable grasps, only one shows the QR."""
    from grasp2show.episode import PickAndShow

    scene = build_scene(cfg)
    ep = PickAndShow(scene, cfg)
    grasps = scripted_grasps(cfg)
    clear = ep.run(grasps["clear_qr"], show_pose(cfg, scene.obj))
    cover = ep.run(grasps["covers_qr"], show_pose(cfg, scene.obj))
    assert clear.success, clear.summary()
    assert cover.held and not cover.qr_ok, cover.summary()
