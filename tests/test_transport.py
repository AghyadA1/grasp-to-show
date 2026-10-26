import numpy as np

from grasp2show.config import ObjectConfig
from grasp2show.objects import object_from_config
from grasp2show.transforms import make_T
from grasp2show.transport import count_stops, friction_params, inertial_load, sparc
from grasp2show.transport_opt import TransportProblem


def _problem():
    obj = object_from_config(ObjectConfig())
    T_obj_tcp = make_T(np.column_stack([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]]), [0, 0, 0.0])
    q0 = np.array([0.0, 0.2, 0.0, -2.0, 0.0, 2.2, 0.8])
    return TransportProblem(obj, T_obj_tcp, 0.05, q0, q0 + 0.4, friction_params(0.5), checker=None, n=60)


def test_trajectory_starts_and_ends_at_rest_for_any_shape():
    pr = _problem()
    c = np.random.default_rng(0).normal(0, 0.3, (pr.K, 7))
    q, qd, qdd, _ = pr.derivs(c, 2.0)
    assert np.allclose(q[0], pr.q0) and np.allclose(q[-1], pr.q1)
    assert np.allclose(qd[[0, -1]], 0, atol=1e-9) and np.allclose(qdd[[0, -1]], 0, atol=1e-9)
    traj = pr.sample(c, 2.0, 0.002)
    assert np.allclose(traj[-1], pr.q1)


def test_static_load_is_the_weight_and_faster_is_harder():
    pr = _problem()
    T = np.tile(np.eye(4), (50, 1, 1))
    F, Tq = inertial_load(T, 0.01, 0.2, np.zeros(3), np.eye(3) * 1e-4)
    assert np.allclose(F, [0, 0, -0.2 * 9.81]) and np.allclose(Tq, 0)
    slow, fast = pr.evaluate(np.zeros((pr.K, 7)), 5.0, False), pr.evaluate(np.zeros((pr.K, 7)), 0.5, False)
    assert fast["u_max"] > slow["u_max"] and fast["limit"] > slow["limit"]


def test_smoothness_measures():
    t = np.linspace(0, 1, 500)
    bell = np.sin(np.pi * t) ** 2
    two = np.sin(2 * np.pi * t) ** 2
    assert sparc(bell, 500) > sparc(two, 500)
    assert count_stops(bell) == 0 and count_stops(two) == 1
