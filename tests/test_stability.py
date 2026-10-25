"""Analytic stability model (checks 1 and 2): closed forms and physical behaviour."""
import numpy as np
import pytest
import trimesh

from grasp2show.objects import make_box
from grasp2show.stability import (
    StabilityParams,
    check_margin,
    finger_contacts,
    stability_margin,
    utilisation,
    utilisation_shake,
)
from grasp2show.transforms import frame_from_axes, make_T

PRM = StabilityParams(mu=0.5, mu_t=0.005, finger_force=20.0)


def test_closed_form_top_grasp():
    """Upright box, top grasp through the centre: u = m g / (2 mu F)."""
    box = trimesh.creation.box(extents=[0.05, 0.05, 0.12])
    T = make_T(frame_from_axes(y=[0, 1, 0], z=[0, 0, -1]), [0, 0, 0.025])
    m = stability_margin(box, T, np.zeros(3), 0.15, np.eye(3), PRM)
    assert m[0] == pytest.approx(1 - 0.15 * 9.81 / (2 * 0.5 * 20), rel=1e-9)


def test_contacts_on_the_finger_axis():
    box = trimesh.creation.box(extents=[0.05, 0.04, 0.12])
    T = make_T(frame_from_axes(y=[0, 1, 0], z=[0, 0, -1]), [0.01, 0, 0.02])
    c1, c2, ok = finger_contacts(box, T)
    assert ok[0]
    np.testing.assert_allclose(c1[0], [0.01, -0.02, 0.02], atol=1e-9)
    np.testing.assert_allclose(c2[0], [0.01, 0.02, 0.02], atol=1e-9)


def test_bar_threshold_matches_simulation():
    """Same bar as tests/test_day3.py: MuJoCo holds at d = 0 and fails at d = 5 and 10 cm."""
    bar = trimesh.creation.box(extents=[0.04, 0.04, 0.30])
    R_lay = frame_from_axes(x=[0, 0, -1], z=[1, 0, 0])
    Rg = frame_from_axes(y=[0, 1, 0], z=[1, 0, 0])
    m = [stability_margin(bar, make_T(Rg, [0, 0, d]), np.zeros(3), 0.8, R_lay, PRM)[0] for d in (0, 0.05, 0.1)]
    assert m[0] > 0 and m[1] < 0 and m[2] < 0


def test_check2_same_grasp_fails_after_turning_the_object():
    """Grasp near the top of a heavy upright bar: gravity passes through the finger axis, so it
    holds. Turn the object 90 deg about its own y axis (as the 'show' pose might): the same
    grasp now carries m g d about the finger axis and fails. This is what check 2 catches."""
    bar = make_box("obj", (0.02, 0.02, 0.15), 0.8, (0.5, 0.005, 1e-4), 4, 0.03, 0.0, (1, 1, 1, 1))
    T = make_T(frame_from_axes(y=[0, 1, 0], z=[1, 0, 0]), [0, 0, 0.10])   # finger axis = obj y
    upright = check_margin(bar, T, np.eye(3), PRM)
    turned = check_margin(bar, T, frame_from_axes(x=[0, 0, 1], y=[0, 1, 0]), PRM)
    assert upright[0] > 0.5
    assert turned[0] < 0


def test_shake_is_harsher_than_static():
    box = trimesh.creation.box(extents=[0.04, 0.08, 0.16])
    rng = np.random.default_rng(0)
    Ts = []
    for _ in range(30):
        z = rng.normal(size=3)
        z[1] = 0
        z /= np.linalg.norm(z)
        Ts.append(make_T(frame_from_axes(y=[0, 1, 0], z=z), rng.uniform(-0.01, 0.01, 3) * [1, 0, 4]))
    T = np.stack(Ts)
    c1, c2, ok = finger_contacts(box, T)
    I = np.diag([1e-3, 1e-3, 5e-4])
    us = utilisation(c1[ok], c2[ok], T[ok, :3, 1], np.zeros(3), 0.4, np.eye(3), PRM)
    ud = utilisation_shake(c1[ok], c2[ok], T[ok], np.zeros(3), 0.4, I, PRM)
    assert np.all(ud >= us - 1e-9)
