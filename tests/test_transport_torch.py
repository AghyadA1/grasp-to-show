"""The PyTorch constraints must agree with TransportProblem.evaluate (the judge)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grasp2show.config import ObjectConfig
from grasp2show.franka_ik import fk
from grasp2show.learn.transport_torch import Constraints, fk_torch, pack
from grasp2show.objects import object_from_config
from grasp2show.transport import friction_params
from grasp2show.transport_opt import TransportProblem


def _problem(seed):
    rng = np.random.default_rng(seed)
    obj = object_from_config(ObjectConfig())                 # the default box
    q0 = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785]) + rng.normal(0, 0.1, 7)
    q1 = q0 + rng.normal(0, 0.6, 7)
    q1[3] = min(q1[3], -0.3)
    return dict(obj=obj, q0=q0, q1=q1, T_obj_tcp=np.eye(4), mu=float(rng.uniform(0.15, 0.6)))  # centred grasp


def test_fk_matches():
    q = np.random.default_rng(0).uniform(-2, 2, (5, 7))
    assert np.allclose(fk_torch(torch.as_tensor(q)).numpy(), fk(q), atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_constraints_match_numpy(seed):
    p = _problem(seed)
    P = pack([p])
    assert len(P["index"]) == 1
    pr = TransportProblem(p["obj"], p["T_obj_tcp"], 0.05, p["q0"], p["q1"], friction_params(p["mu"]))
    con = Constraints()
    rng = np.random.default_rng(seed + 10)
    for T in (0.6, 1.5, 4.0):
        c = rng.normal(0, 0.2, (4, 7))
        e = pr.evaluate(c, T, collisions=False)
        out = con(torch.as_tensor(c)[None], torch.tensor([T], dtype=torch.float64), P)
        u = out["u"][0].numpy()
        fin = e["u"] < 10
        assert np.allclose(u[fin], e["u"][fin], rtol=1e-6, atol=1e-6)
        assert np.all(u[~fin] > 1)
        assert np.isclose(out["lim"][0].max().item(), e["limit"], rtol=1e-9)
        assert np.isclose(out["jl"][0].max().item(), e["joint_limit"], atol=1e-12)
        assert np.isclose(out["jerk_rel"][0].item(), e["jerk_rel"], rtol=1e-9)
    assert np.isclose(con.static_u(P)[0].item(), pr.u_static, rtol=1e-3, atol=1e-3)


def test_gradient_flows():
    p = _problem(1)
    P = pack([p])
    con = Constraints()
    c = torch.zeros(1, 4, 7, dtype=torch.float64, requires_grad=True)
    T = torch.tensor([0.8], dtype=torch.float64, requires_grad=True)
    out = con(c, T, P)
    (out["u"].amax() + out["lim"].amax()).backward()
    assert torch.isfinite(c.grad).all() and torch.isfinite(T.grad).all() and c.grad.abs().sum() > 0
