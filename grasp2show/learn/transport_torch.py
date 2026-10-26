"""The transport constraints in PyTorch: differentiable, batched, and identical to
grasp2show/transport_opt.py's TransportProblem.evaluate (same trajectory basis, same samples,
same finite differences, same soft-finger model, same leverage worst case, same limits).

This lets a network be trained *on the constraints themselves* (self-supervised: minimise
duration subject to grasp, leverage and robot limits) instead of imitating the optimiser's
solutions. Collisions are not differentiable and are not included (checked in full episodes).

The friction utilisation u of the soft-finger model is the smallest u at which the two
contacts can supply the needed force and the torque about the finger axis. Where the torque
binds, numpy finds u by bisection; here the bisection runs without gradients and the gradient
comes from the implicit function theorem (one Newton step on the binding equation), which is
exact at the solution.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..franka_ik import A4, A7, D1, D3, D5, D7E, Q_MAX, Q_MIN
from ..transport import QD_MAX, QDD_MAX
from ..transport_opt import S_MINJERK, _basis

G = (0.0, 0.0, -9.81)


def fk_torch(q: torch.Tensor) -> torch.Tensor:
    """TCP pose (..., 4, 4) for joint angles (..., 7): same parameters as franka_ik.fk."""
    a = [0, 0, 0, A4, -A4, 0, A7]
    d = [D1, 0, D3, 0, D5, 0, 0]
    alpha = [0, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2]
    T = torch.eye(4, dtype=q.dtype, device=q.device).expand(*q.shape[:-1], 4, 4)
    z, o = torch.zeros_like(q[..., 0]), torch.ones_like(q[..., 0])
    for i in range(7):
        ct, st = torch.cos(q[..., i]), torch.sin(q[..., i])
        ca, sa = math.cos(alpha[i]), math.sin(alpha[i])
        A = torch.stack([torch.stack([ct, -st, z, z + a[i]], -1),
                         torch.stack([st * ca, ct * ca, z - sa, z - sa * d[i]], -1),
                         torch.stack([st * sa, ct * sa, z + ca, z + ca * d[i]], -1),
                         torch.stack([z, z, z, o], -1)], -2)
        T = T @ A
    c, s = math.cos(-math.pi / 4), math.sin(-math.pi / 4)
    F = torch.tensor([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, D7E], [0, 0, 0, 1.0]], dtype=q.dtype, device=q.device)
    return T @ F


def _grad_t(x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """np.gradient along dim 1 (central inside, one-sided at the ends), spacing dt (B,)."""
    g = torch.cat([x[:, 1:2] - x[:, 0:1], 0.5 * (x[:, 2:] - x[:, :-2]), x[:, -1:] - x[:, -2:-1]], 1)
    return g / dt.reshape(-1, *([1] * (x.dim() - 1)))


def _norm(v: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((v * v).sum(-1) + 1e-18)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1)


def inertial_load(T_obj: torch.Tensor, dt: torch.Tensor, mass, com, inertia):
    """F_ext, T_ext (B, n, 3) in the object frame; T_obj (B, n, 4, 4), mass (B,), com (B, 3),
    inertia (B, 3, 3). Same as grasp2show.transport.inertial_load."""
    R, p = T_obj[..., :3, :3], T_obj[..., :3, 3]
    c = p + torch.einsum("bnij,bj->bni", R, com)
    a = _grad_t(_grad_t(c, dt), dt)
    W = _grad_t(R, dt) @ R.transpose(-1, -2)
    w = 0.5 * torch.stack([W[..., 2, 1] - W[..., 1, 2], W[..., 0, 2] - W[..., 2, 0], W[..., 1, 0] - W[..., 0, 1]], -1)
    al = _grad_t(w, dt)
    Rt = R.transpose(-1, -2)
    w_o, al_o = torch.einsum("bnij,bnj->bni", Rt, w), torch.einsum("bnij,bnj->bni", Rt, al)
    g = torch.tensor(G, dtype=T_obj.dtype, device=T_obj.device)
    F = mass[:, None, None] * torch.einsum("bnij,bnj->bni", Rt, g - a)
    Iw = torch.einsum("bij,bnj->bni", inertia, w_o)
    T = -(torch.einsum("bij,bnj->bni", inertia, al_o) + torch.cross(w_o, Iw, dim=-1))
    return F, T


def utilisation(c1, c2, y, com, F, T, mu, finger_force: float, axial_cap: float, iters: int = 40):
    """Soft-finger friction utilisation (B, n), as stability.utilisation_wrench. c1, c2, y, com
    (B, 3); F, T (B, n, 3); mu (B,) (torsional coefficient 0.01 mu). Where a finger would be
    pulled off (axial load above 2F; inf in numpy) it returns 3 + |f_y| / axial_cap, which is
    above any limit and still has a gradient."""
    mu = mu[:, None]
    mu_t = 0.01 * mu
    c, w = 0.5 * (c1 + c2), _norm(c2 - c1)
    y_ = y[:, None, :]
    f = -F
    tau = -(torch.cross((com - c)[:, None, :].expand_as(F), F, dim=-1) + T)
    f_y = _dot(f, y_)
    f_t = f - f_y[..., None] * y_
    tau_y = _dot(tau, y_)
    tau_p = tau - tau_y[..., None] * y_
    d = (2.0 / w)[:, None, None] * torch.cross(tau_p, y_.expand_as(tau_p), dim=-1)
    t1, t2 = 0.5 * (f_t - d), 0.5 * (f_t + d)
    n1, n2 = finger_force + 0.5 * f_y, finger_force - 0.5 * f_y
    a1, a2 = _norm(t1) / mu, _norm(t2) / mu
    valid = (n1 > 0) & (n2 > 0)
    n1s, n2s = torch.where(valid, n1, torch.ones_like(n1)), torch.where(valid, n2, torch.ones_like(n2))
    lo = torch.maximum(a1 / n1s, a2 / n2s)
    ty = torch.abs(tau_y)

    def g(u, n1, n2, a1, a2, ty):                       # torque capacity - need (>= 0: feasible)
        r1 = mu_t * torch.sqrt(torch.clamp((u * n1) ** 2 - a1**2, min=0.0))
        r2 = mu_t * torch.sqrt(torch.clamp((u * n2) ** 2 - a2**2, min=0.0))
        return r1 + r2 - ty

    with torch.no_grad():
        args = (n1s.detach(), n2s.detach(), a1.detach(), a2.detach(), ty.detach())
        lo_d = lo.detach()
        lo_ok = g(lo_d, *args) >= 0
        a_, b_ = lo_d.clone(), lo_d + 1.0 + ty.detach() / (mu_t * torch.minimum(n1s, n2s)).detach()
        for _ in range(iters):
            mid = 0.5 * (a_ + b_)
            fm = g(mid, *args) >= 0
            b_ = torch.where(fm, mid, b_)
            a_ = torch.where(fm, a_, mid)
        us = b_
        # dg/du at the solution, for the implicit gradient
        s1 = torch.sqrt(torch.clamp((us * args[0]) ** 2 - args[2] ** 2, min=1e-18))
        s2 = torch.sqrt(torch.clamp((us * args[1]) ** 2 - args[3] ** 2, min=1e-18))
        dgdu = (mu_t * (us * args[0] ** 2 / s1 + us * args[1] ** 2 / s2)).clamp_min(1e-12)
    u_tor = us - g(us, n1s, n2s, a1, a2, ty) / dgdu    # value us (g(us) ~ 0), gradient -dg/dθ / dg/du
    u = torch.where(lo_ok, lo, u_tor)
    return torch.where(valid, u, 3.0 + torch.abs(f_y) / axial_cap)


class Constraints(torch.nn.Module):
    """Batched TransportProblem.evaluate (no collisions). Buffers: the trajectory basis."""

    def __init__(self, K: int = 4, n: int = 80, limit_ratio: float = 0.6, com_margin: float = 0.005,
                 axial_frac: float = 0.35, finger_force: float = 20.0, u_lim: float = 0.7, w_smooth: float = 0.05,
                 dtype=torch.float64):
        super().__init__()
        self.K, self.n, self.r, self.m = K, n, limit_ratio, com_margin
        self.F, self.axial_cap, self.u_lim0, self.w_smooth = finger_force, axial_frac * 2 * finger_force, u_lim, w_smooth
        tau = np.linspace(0.0, 1.0, n)
        polys = _basis(K)
        B = np.stack([np.stack([p.deriv(dd)(tau) if dd else p(tau) for p in polys]) for dd in range(4)])
        S = np.stack([S_MINJERK.deriv(dd)(tau) if dd else S_MINJERK(tau) for dd in range(4)])
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype)
        self.register_buffer("B", t(B))                 # (4, K, n)
        self.register_buffer("S", t(S))                 # (4, n)
        self.register_buffer("qd_max", t(QD_MAX))
        self.register_buffer("qdd_max", t(QDD_MAX))
        self.register_buffer("q_min", t(Q_MIN))
        self.register_buffer("q_max", t(Q_MAX))

    def derivs(self, c, T, q0, dq):
        out = []
        for dd in range(4):
            x = self.S[dd][None, :, None] * dq[:, None, :] + torch.einsum("kn,bkj->bnj", self.B[dd], c)
            if dd == 0:
                x = x + q0[:, None, :]
            out.append(x / T[:, None, None] ** dd)
        return out

    def jerk(self, c, dq):
        j = self.S[3][None, :, None] * dq[:, None, :] + torch.einsum("kn,bkj->bnj", self.B[3], c)
        return (j**2).sum(-1).mean(-1)

    def grasp_u(self, T_obj, dt, P):
        """Robust utilisation with the axial cap, (B, n)."""
        F, Tq = inertial_load(T_obj, dt, P["mass"], P["com"], P["inertia"])
        R = P["T_obj_tcp"][:, :3, :3]
        y = R[:, :, 1]
        shifts = [torch.zeros_like(P["com"])] + [s * self.m * R[:, :, k] for k in range(3) for s in (-1, 1)]
        u = torch.stack([utilisation(P["c1"], P["c2"], y, P["com"] + dc, F, Tq, P["mu"], self.F, self.axial_cap)
                         for dc in shifts]).amax(0)
        return torch.maximum(u, torch.abs(_dot(F, y[:, None, :])) / self.axial_cap)

    def static_u(self, P):
        """Utilisation at rest at the start and end poses, (B,): gravity only."""
        ends = torch.stack([P["q0"], P["q0"] + P["dq"]], 1)                  # (B, 2, 7)
        T_obj = fk_torch(ends) @ P["T_tcp_obj"][:, None]
        R = T_obj[..., :3, :3]
        g = torch.tensor(G, dtype=R.dtype, device=R.device)
        F = P["mass"][:, None, None] * torch.einsum("bnji,j->bni", R, g)
        y = P["T_obj_tcp"][:, :3, 1]
        Rg = P["T_obj_tcp"][:, :3, :3]
        shifts = [torch.zeros_like(P["com"])] + [s * self.m * Rg[:, :, k] for k in range(3) for s in (-1, 1)]
        u = torch.stack([utilisation(P["c1"], P["c2"], y, P["com"] + dc, F, torch.zeros_like(F), P["mu"], self.F,
                                     self.axial_cap) for dc in shifts]).amax(0)
        return torch.maximum(u, torch.abs(_dot(F, y[:, None, :])) / self.axial_cap).amax(1)

    def forward(self, c, T, P):
        """Per-sample constraint values for coefficients c (B, K, 7) and durations T (B,).
        P: dict of problem tensors (q0, dq, mass, com, inertia, T_obj_tcp, T_tcp_obj, c1, c2, mu,
        u_lim). Returns u (B, n), lim (B, n), jl (B, n), jerk_rel (B,)."""
        q, qd, qdd, _ = self.derivs(c, T, P["q0"], P["dq"])
        T_obj = fk_torch(q) @ P["T_tcp_obj"][:, None]
        u = self.grasp_u(T_obj, T / (self.n - 1), P)
        lim = torch.maximum((torch.abs(qd) / (self.r * self.qd_max)).amax(-1), (torch.abs(qdd) / (self.r * self.qdd_max)).amax(-1))
        jl = torch.maximum(self.q_min - q, q - self.q_max).amax(-1)
        jerk_ref = self.jerk(torch.zeros_like(c), P["dq"]).clamp_min(1e-9)
        return {"u": u, "lim": lim, "jl": jl, "jerk_rel": self.jerk(c, P["dq"]) / jerk_ref - 1.0}

    def feasible(self, out, u_lim):
        return (out["u"].amax(1) <= u_lim + 1e-3) & (out["lim"].amax(1) <= 1 + 1e-3) & (out["jl"].amax(1) <= 0)


def pack(problems: list[dict], dtype=torch.float64, device="cpu") -> dict:
    """Problem tensors from dicts with q0, q1, T_obj_tcp, mu and an object model `obj`
    (mass, com, inertia, mesh). Problems whose finger rays miss the object are dropped
    (see the returned "index")."""
    from ..stability import finger_contacts

    keep, c1s, c2s = [], [], []
    for i, p in enumerate(problems):
        c1, c2, ok = finger_contacts(p["obj"].mesh, np.asarray(p["T_obj_tcp"], float))
        if ok[0]:
            keep.append(i)
            c1s.append(c1[0])
            c2s.append(c2[0])
    sel = [problems[i] for i in keep]
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=dtype, device=device)
    T_obj_tcp = t([p["T_obj_tcp"] for p in sel])
    q0 = t([p["q0"] for p in sel])
    return {"index": torch.as_tensor(keep), "q0": q0, "dq": t([p["q1"] for p in sel]) - q0,
            "mass": t([p["obj"].mass for p in sel]), "com": t([p["obj"].com for p in sel]),
            "inertia": t([p["obj"].inertia for p in sel]), "T_obj_tcp": T_obj_tcp,
            "T_tcp_obj": torch.linalg.inv(T_obj_tcp), "c1": t(c1s), "c2": t(c2s), "mu": t([p["mu"] for p in sel])}


def subset(P: dict, idx) -> dict:
    return {k: v[idx] for k, v in P.items()}
