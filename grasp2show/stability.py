"""Analytic grasp stability under gravity (check 1 at the start pose, check 2 at the target).

Model (derivation in docs/physics.md):
  - Two soft-finger contacts on the finger axis y: c1 (finger pushing +y) and c2 (pushing -y),
    width w = |c2 - c1|, reference point c = (c1 + c2) / 2.
  - Force-limited gripper: the two normal forces sum to 2F (F per finger); the coupled finger
    mechanism takes any load along y, so n1 = F + f_y/2, n2 = F - f_y/2.
  - Each contact carries a tangential force t_i (2-D, perpendicular to y) and a torsional
    moment m_i about y, limited by MuJoCo's elliptic soft-finger cone (condim 4):
        (|t_i| / mu)^2 + (m_i / mu_t)^2 <= n_i^2
  - The contacts must cancel the object's weight W = s*m*g (s = safety factor for the
    accelerations of the motion), applied at the centre of mass:
        force  f = -W,   torque about c  tau = -(com - c) x W.
    Splitting along the finger axis y:
        f_y   -> normal forces (above)
        f_t   = t1 + t2                         (tangential force)
        tau_p = (w/2) y x (t2 - t1)             (torque perpendicular to y: a couple of t's)
        tau_y = m1 + m2                         (torque about y: torsional friction ONLY)
    so t1, t2 are fixed:  t2 - t1 = (2/w) tau_p x y,   t_{1,2} = (f_t -+ d) / 2,
    and only the split of tau_y between m1 and m2 is free.
  - Utilisation u = the smallest u such that every contact satisfies
        (|t_i| / mu)^2 + (m_i / mu_t)^2 <= (u n_i)^2   for some m1 + m2 = tau_y.
    For a given u, |m_i| <= r_i(u) = mu_t sqrt(u^2 n_i^2 - (|t_i|/mu)^2), so a split exists
    iff r_1(u) + r_2(u) >= |tau_y|. That is monotone in u: we bisect.
  - margin = 1 - u. margin > 0: the grasp holds; margin = 0.3: the load can grow by 1/(1-0.3).

Why torque about the finger axis is the weak point: perpendicular torques are carried by a
couple of tangential forces with lever w (centimetres), but tau_y only by torsional friction,
whose "lever" mu_t is the contact-patch radius (millimetres).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ._accel import HAVE_CPP, cpp

G = np.array([0.0, 0.0, -9.81])


@dataclass
class StabilityParams:
    mu: float = 0.5            # sliding friction
    mu_t: float = 0.005        # torsional friction coefficient (m): effective patch radius
    finger_force: float = 20.0  # N per finger
    safety: float = 1.0        # load factor on gravity (dynamic loads during motion)


def finger_contacts(mesh: trimesh.Trimesh, T_obj_tcp: np.ndarray, max_half_width: float = 0.04):
    """Where the closing fingers touch the object: cast rays from the TCP along -y and +y.

    The finger pads close along the TCP's y axis, so the contacts are where that line meets
    the surface (not necessarily the sampled antipodal points: the TCP depth is randomised).
    Returns c1, c2 (N, 3) and a bool mask `ok` (both rays hit within the gripper opening).
    The rays start outside the object and travel inwards, so the first hit is the surface.
    """
    T = np.asarray(T_obj_tcp).reshape(-1, 4, 4)
    p, y = T[:, :3, 3], T[:, :3, 1]
    out = []
    for sign in (-1.0, 1.0):
        origins = p + sign * y * max_half_width
        locs, idx, _ = mesh.ray.intersects_location(origins, -sign * y, multiple_hits=False)
        c = np.full((len(T), 3), np.nan)
        if len(idx):
            c[idx] = np.asarray(locs).reshape(-1, 3)
        out.append(c)
    c1, c2 = out
    ok = ~np.isnan(c1).any(1) & ~np.isnan(c2).any(1)
    # Both contacts must lie on the finger side of the TCP they belong to.
    ok &= np.einsum("ij,ij->i", c2 - c1, y) > 0
    return c1, c2, ok


def utilisation_wrench(c1, c2, y, com, F_ext, T_ext, prm: StabilityParams, iters: int = 40,
                       use_cpp: bool = True) -> np.ndarray:
    """Friction utilisation u (N,) for an arbitrary external load on the object.

    F_ext (N, 3): total external force on the object (gravity, inertial forces);
    T_ext (N, 3): external torque about the centre of mass (e.g. inertial torque);
    everything in the object frame. u <= 1: the contacts can cancel the load. inf: they cannot.
    Runs the C++ kernel when it is installed (see _accel.py), numpy otherwise.
    """
    if use_cpp and HAVE_CPP:
        f64 = lambda a: np.ascontiguousarray(np.broadcast_to(a, np.shape(np.atleast_2d(c1))), np.float64)
        return cpp.utilisation_wrench(f64(c1), f64(c2), f64(y), np.ascontiguousarray(com, np.float64),
                                      f64(F_ext), f64(T_ext), prm.mu, prm.mu_t, prm.finger_force, iters)
    c1, c2, y = np.atleast_2d(c1), np.atleast_2d(c2), np.atleast_2d(y)
    c = 0.5 * (c1 + c2)
    w = np.linalg.norm(c2 - c1, axis=1)
    f = -F_ext                                                            # contact force needed
    tau = -(np.cross(com - c, F_ext) + T_ext)                             # contact torque about c
    f_y = np.einsum("ij,ij->i", f, y)
    f_t = f - f_y[:, None] * y
    tau_y = np.einsum("ij,ij->i", tau, y)
    tau_p = tau - tau_y[:, None] * y
    d = (2.0 / np.maximum(w, 1e-9))[:, None] * np.cross(tau_p, y)
    t1, t2 = 0.5 * (f_t - d), 0.5 * (f_t + d)
    n1, n2 = prm.finger_force + 0.5 * f_y, prm.finger_force - 0.5 * f_y
    a1 = np.linalg.norm(t1, axis=1) / prm.mu                               # |t_i| / mu
    a2 = np.linalg.norm(t2, axis=1) / prm.mu
    u = np.full(len(c1), np.inf)
    valid = (n1 > 0) & (n2 > 0) & (w > 1e-6)
    n1s, n2s = np.where(valid, n1, 1.0), np.where(valid, n2, 1.0)
    lo = np.where(valid, np.maximum(a1 / n1s, a2 / n2s), np.inf)
    hi = lo + 1.0 + np.abs(tau_y) / (prm.mu_t * np.minimum(n1s, n2s))

    def feasible(uu):
        r1 = prm.mu_t * np.sqrt(np.maximum((uu * n1) ** 2 - a1**2, 0.0))
        r2 = prm.mu_t * np.sqrt(np.maximum((uu * n2) ** 2 - a2**2, 0.0))
        return r1 + r2 >= np.abs(tau_y)

    lo_ok = valid & feasible(lo)
    u[lo_ok] = lo[lo_ok]
    todo = valid & ~lo_ok
    a, b = lo.copy(), hi.copy()
    for _ in range(iters):
        mid = 0.5 * (a + b)
        fm = feasible(mid)
        b = np.where(todo & fm, mid, b)
        a = np.where(todo & ~fm, mid, a)
    u[todo] = b[todo]
    return u


def utilisation(c1, c2, y, com, mass, R_world_obj, prm: StabilityParams, iters: int = 40,
                use_cpp: bool = True) -> np.ndarray:
    """Static case: the load is the weight s*m*g, with the object at orientation R_world_obj.

    c1, c2, y: (N, 3) in the object frame; com (3,); R_world_obj (3, 3) or (N, 3, 3).
    """
    c1 = np.atleast_2d(c1)
    R = np.asarray(R_world_obj)
    g_obj = (R.T @ G) if R.ndim == 2 else np.einsum("nji,j->ni", R, G)
    W = prm.safety * mass * np.broadcast_to(g_obj, c1.shape)
    return utilisation_wrench(c1, c2, y, com, W, np.zeros_like(W), prm, iters, use_cpp)


def _rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrices (N, 3, 3) about unit axes (N, 3) by one angle (Rodrigues)."""
    K = np.zeros((len(axis), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K = K - K.transpose(0, 2, 1)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def utilisation_shake(c1, c2, T_obj_tcp, com, mass, inertia, prm: StabilityParams,
                      lift_acc: float = 2.31, amp: float = 0.02, rock: float = 0.25, hz: float = 2.0,
                      samples: int = 16) -> np.ndarray:
    """Worst-case utilisation over the shake test's motion (object upright at the start).

    Quasi-static with d'Alembert inertial loads: at each sampled instant the object, rigidly
    held, has the hand's acceleration; the contacts must supply m*(a - g) and the torque for
    the angular acceleration. Phases (as in GripperSim.shake_test):
      lift   : vertical acceleration +-lift_acc (min-jerk 10 cm in 0.5 s peaks at 2.31 m/s^2)
      shake  : translation amp*sin(wt) along the approach axis z
      rock   : rotation theta = rock*sin(wt) about the finger axis y, pivot at the TCP:
               alpha = -rock w^2 sin(wt), omega = rock w cos(wt); gravity rotates with the hand.
    Returns max u over all instants (N,).
    """
    T = np.asarray(T_obj_tcp)
    y, z, p = T[:, :3, 1], T[:, :3, 2], T[:, :3, 3]
    N = len(T)
    g = np.broadcast_to(G, (N, 3))
    zero = np.zeros((N, 3))
    w = 2 * np.pi * hz
    loads = [(mass * (g - s * lift_acc * np.array([0, 0, 1.0])), zero) for s in (1.0, -1.0)]
    for k in range(samples):
        ph = 2 * np.pi * k / samples
        a = -amp * w**2 * np.sin(ph)
        loads.append((mass * (g - a * z), zero))
    r = com - p                                                    # TCP -> COM
    for k in range(samples):
        ph = 2 * np.pi * k / samples
        th, om, al = rock * np.sin(ph), rock * w * np.cos(ph), -rock * w**2 * np.sin(ph)
        R = _rot_about(y, th)                                      # hand rotation in the world
        g_obj = np.einsum("nji,nj->ni", R, g)                      # gravity seen in the object frame
        W, A = om * y, al * y
        a_com = np.cross(A, r) + np.cross(W, np.cross(W, r))
        F = mass * (g_obj - a_com)                                 # gravity + inertial force
        Tq = -(A @ inertia.T + np.cross(W, W @ inertia.T))         # inertial torque about the COM
        loads.append((F, Tq))
    u = np.zeros(N)
    for F, Tq in loads:
        u = np.maximum(u, utilisation_wrench(c1, c2, y, com, F, Tq, prm))
    return u


def stability_margin(mesh, T_obj_tcp, com, mass, R_world_obj, prm: StabilityParams | None = None):
    """margin = 1 - u (N,); -inf where the fingers find no contact. Positive = predicted to hold."""
    prm = prm or StabilityParams()
    T = np.asarray(T_obj_tcp).reshape(-1, 4, 4)
    c1, c2, ok = finger_contacts(mesh, T)
    m = np.full(len(T), -np.inf)
    if ok.any():
        u = utilisation(c1[ok], c2[ok], T[ok, :3, 1], com, mass, R_world_obj, prm)
        m[ok] = 1.0 - u
    return m


def check_margin(obj, T_obj_tcp, R_world_obj, prm: StabilityParams | None = None,
                 load_factor: float = 1.0) -> np.ndarray:
    """Checks 1 and 2: static margin with the object at orientation(s) R_world_obj.

    check 1: R_world_obj = start orientation (load_factor 1.0, lifting slowly);
    check 2: R_world_obj = target orientation (load_factor 1.2 for the transit).
    obj is an objects.ObjectModel. T_obj_tcp (N, 4, 4) or (4, 4); R_world_obj (3, 3) or
    (M, 3, 3) for M orientations of the same grasp(s). Returns margin = 1 - u, shape (N,) or
    (N, M); -inf where the fingers find no contact.
    """
    p = prm or StabilityParams()
    p = StabilityParams(p.mu, p.mu_t, p.finger_force, load_factor)
    R = np.asarray(R_world_obj)
    if R.ndim == 2:
        return stability_margin(obj.mesh, T_obj_tcp, obj.com, obj.mass, R, p)
    T = np.asarray(T_obj_tcp).reshape(-1, 4, 4)
    N, M = len(T), len(R)
    c1, c2, ok = finger_contacts(obj.mesh, T)
    out = np.full((N, M), -np.inf)
    if ok.any():
        n = int(ok.sum())
        rep = lambda a: np.repeat(a[ok], M, axis=0)
        u = utilisation(rep(c1), rep(c2), rep(T[:, :3, 1]), obj.com, obj.mass, np.tile(R, (n, 1, 1)), p)
        out[ok] = 1.0 - u.reshape(n, M)
    return out
