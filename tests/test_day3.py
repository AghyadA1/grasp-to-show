"""Day 3: objects, QR placement, antipodal sampling, shake-test labels."""
import numpy as np
import pytest
import trimesh

from grasp2show.config import ObjectConfig, SceneConfig
from grasp2show.grasp_sampler import candidates_from_pairs, sample_antipodal
from grasp2show.gripper_sim import GripperSim
from grasp2show.objects import load_object_set, make_box, object_from_config, scanned_objects_dir
from grasp2show.qr_placement import find_qr_face
from grasp2show.task import scripted_grasps
from grasp2show.transforms import frame_from_axes, make_T


@pytest.fixture(scope="module")
def cfg():
    return SceneConfig()


def test_qr_goes_on_a_side_face_of_a_box():
    mesh = trimesh.creation.box(extents=[0.05, 0.05, 0.12])
    mesh.apply_translation([0, 0, 0.06])                     # resting on z = 0
    qr = find_qr_face(mesh)
    assert qr is not None
    n = qr.T_obj_qr[:3, 0]
    assert abs(n[2]) < 1e-6                                   # side face, not top or bottom
    assert qr.side == pytest.approx(0.04)                     # 80 % of the 5 cm face
    assert qr.flatness < 1e-6


def test_antipodal_pairs_are_antipodal():
    mesh = trimesh.creation.box(extents=[0.04, 0.06, 0.10])
    rng = np.random.default_rng(0)
    p1, p2 = sample_antipodal(mesh, 50, rng)
    assert len(p1) > 20
    w = np.linalg.norm(p2 - p1, axis=1)
    assert np.all(w < 0.075)
    # On a box every antipodal pair spans two opposite faces: width = one of the extents,
    # up to the 30 deg cone (1 / cos 30 = 1.155).
    assert np.all(w > 0.04 - 1e-6) and np.all(w < 0.06 * 1.155 + 1e-6)
    cand = candidates_from_pairs(p1, p2, 4, rng)
    R = cand.T_obj_tcp[:, :3, :3]
    np.testing.assert_allclose(np.einsum("nij,nik->njk", R, R), np.tile(np.eye(3), (len(R), 1, 1)), atol=1e-9)


def test_shake_test_holds_the_day1_grasp(cfg):
    sim = GripperSim(cfg, object_from_config(cfg.obj))
    g = scripted_grasps(cfg)["clear_qr"]
    r = sim.shake_test(g.T_obj_tcp, g.width)
    assert r.success and r.slip_mm < 5.0


def test_shake_test_detects_a_miss(cfg):
    sim = GripperSim(cfg, object_from_config(cfg.obj))
    g = scripted_grasps(cfg)["clear_qr"]
    T = g.T_obj_tcp.copy()
    T[:3, 3] += [0.0, 0.0, 0.20]                              # 20 cm above the box: nothing to grip
    r = sim.shake_test(T, g.width)
    assert not r.success and not r.closed_on_object


def test_torque_about_the_finger_axis_breaks_an_offcentre_grasp(cfg):
    """A heavy bar held near one end spins about the finger axis: gravity torque
    m*g*d = 0.8*9.81*0.10 = 0.78 N m exceeds the torsional friction ~ 2*mu_t*F = 0.2 N m.
    The same grasp at the centre (d = 0) holds. This is the physics behind check 2."""
    bar = make_box("obj", (0.02, 0.02, 0.15), 0.8, cfg.obj.friction, cfg.obj.condim, 0.03, 0.0,
                   (0.5, 0.5, 0.5, 1))
    sim = GripperSim(cfg, bar)
    R = frame_from_axes(y=[0, 1, 0], z=[1, 0, 0])             # top-down approach, fingers along world y
    centre = sim.shake_test(make_T(R, [0, 0, 0.0]), 0.04, R_world_obj=_lay_down())
    end = sim.shake_test(make_T(R, [0, 0, 0.10]), 0.04, R_world_obj=_lay_down())
    assert centre.success
    assert not end.success and end.slip_deg > 20


def _lay_down():
    """Bar lying horizontally: obj z (bar axis) -> world x, obj x -> world -z (so the grasp's
    obj-x approach is top-down), obj y -> world y (the finger axis, horizontal)."""
    return frame_from_axes(x=[0, 0, -1], z=[1, 0, 0])


def _first_downloaded():
    names = [n for n in load_object_set()["objects"] if (scanned_objects_dir() / n / "model.obj").exists()]
    return names[0] if names else None


@pytest.mark.skipif(_first_downloaded() is None, reason="run scripts/get_objects.py first")
def test_scanned_object_loads_with_qr_and_mass():
    obj = object_from_config(ObjectConfig(asset=_first_downloaded()))
    assert 0.05 <= obj.mass <= 1.5
    assert len(obj.collision_files) > 0
    assert obj.qr.side >= 0.03


def test_box_inertia_matches_closed_form():
    b = make_box("obj", (0.025, 0.025, 0.06), 0.15, (0.5, 0.005, 1e-4), 4, 0.045, 0.02, (1, 1, 1, 1))
    m = trimesh.creation.box(extents=[0.05, 0.05, 0.12])
    m.density = 0.15 / m.volume
    np.testing.assert_allclose(b.inertia, m.moment_inertia, rtol=1e-9)
