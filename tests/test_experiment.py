"""Policies and the analytic episode executor, on the day-1 box."""
import numpy as np
import pytest

from grasp2show.config import SceneConfig
from grasp2show.episode import PickAndShow
from grasp2show.experiment import POLICIES, choose
from grasp2show.feasibility import FeasibilityChecker
from grasp2show.qr import decode_qr
from grasp2show.scene import build_scene
from grasp2show.task import Grasp, scripted_grasps, show_pose


@pytest.fixture(scope="module")
def setup():
    cfg = SceneConfig()
    scene = build_scene(cfg)
    fc = FeasibilityChecker(scene, cfg)
    g = scripted_grasps(cfg)
    rolls = np.radians([0, 90, 180, 270])
    reports = {0: fc.evaluate(g["clear_qr"].T_obj_tcp, g["clear_qr"].width, rolls),
               1: fc.evaluate(g["covers_qr"].T_obj_tcp, g["covers_qr"].width, rolls)}
    return cfg, scene, fc, g, rolls, reports


def test_baseline_prefers_the_more_stable_grasp_even_if_it_hides_the_qr(setup):
    *_, reports = setup
    lift = {0: 0.80, 1: 0.95}                       # pretend the QR-covering grasp is more stable
    base = choose("stability_only", reports, lift)
    aware = choose("aware", reports, lift)
    no_vis = choose("aware_no_vis", reports, lift)
    assert base[0].grasp == 1                       # stability only: picks the covering grasp
    assert aware[0].grasp == 0 and aware[0].predicted_ok
    assert no_vis[0].grasp == 1                     # without check 5 the difference disappears
    assert set(POLICIES) >= {"stability_only", "aware"}


def test_analytic_plan_and_execute_side_grasp(setup):
    cfg, scene, fc, g, rolls, reports = setup
    c = choose("aware", reports, {0: 0.8, 1: 0.9})[0]
    tw = reports[c.grasp].twins[c.twin]
    t = tw.targets[c.roll_idx]
    ep = PickAndShow(scene, cfg)
    T = g["clear_qr"].T_obj_tcp if c.twin == 0 else tw_twin(g["clear_qr"].T_obj_tcp)
    paths, why = ep.plan_analytic(T, show_pose(cfg, scene.obj, rolls[c.roll_idx]), tw.start.q, t.reach.q,
                                  checker=fc, width=g["clear_qr"].width)
    assert paths is not None, why
    r = ep.execute(Grasp("clear", T, g["clear_qr"].width), paths)
    assert r.success, r.summary()


def tw_twin(T):
    from grasp2show.feasibility import twin_of

    return twin_of(T)


def test_decoder_cascade_reads_a_rendered_code(tmp_path):
    import cv2

    from grasp2show.qr import make_qr_texture

    img = cv2.cvtColor(cv2.imread(str(make_qr_texture("abc", tmp_path / "q.png", 256))), cv2.COLOR_BGR2RGB)
    small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    canvas = np.full((240, 240, 3), 40, np.uint8)               # dark surroundings
    canvas[70:170, 70:170] = small
    assert decode_qr(canvas)[0] == "abc"
