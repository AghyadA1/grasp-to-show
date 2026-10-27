"""The grip model with every contact (part 3): how much load a closed grasp can hold.

The point-contact model (stability.py) knows two finger pads, each touching at one point.
Real grasps of curved or irregular objects touch at many points: along a mug's rim, on the side
of a finger, at the palm. Here the contacts come from closing the simulated gripper on the
object (gripper_sim.close_contacts), and a linear program finds out whether contact forces
exist that hold a given load, and how many times that load they could hold:

    maximise alpha  over the contact forces f_k (and torsional moments) such that
      sum_k f_k + alpha F_ext = 0,   sum_k (p_k - com) x f_k + n_k m_k + alpha T_ext = 0
      f_k in its friction cone (8-sided pyramid, MuJoCo's mu per contact)
      |m_k| <= mu_t,k * (normal force of k)                       (torsional friction)
      sum over finger contacts of f_k . c_k = 2 F                 (the gripper's actuator)

The last line is the Franka hand: one motor drives both fingers through a coupling, so the two
fingers together push with exactly 2F along their closing directions (each finger F when the
grasp is symmetric), and the coupling and the hand's structure carry everything else: a load
that presses the object into a finger's side or into the palm costs no grip force and no
friction. That is what makes hooking a handle strong.

utilisation u = 1 / alpha: u <= 1 holds (u = 0: the structure alone carries the load;
inf: the contacts cannot even balance the squeeze, e.g. no opposing pads).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .gripper_sim import ContactSet

N_EDGES = 8
ALPHA_MAX = 1e3


def _tangents(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return t1, np.cross(n, t1)


def capacity(cs: ContactSet, com: np.ndarray, F_ext: np.ndarray, T_ext: np.ndarray | None = None,
             finger_force: float = 20.0) -> float:
    """alpha: how many times the external load (F_ext, T_ext about the COM; object frame, the
    load the contacts must cancel, e.g. m g) the grasp can hold. 0: not even the squeeze."""
    K = len(cs)
    if K == 0:
        return 0.0
    T_ext = np.zeros(3) if T_ext is None else np.asarray(T_ext, float)
    t1, t2 = _tangents(cs.n)
    ang = 2 * np.pi * np.arange(N_EDGES) / N_EDGES
    # edge directions of each friction pyramid: n + mu (cos t1 + sin t2)   (K, E, 3)
    E = (cs.n[:, None, :] + cs.mu[:, None, None] * (np.cos(ang)[None, :, None] * t1[:, None, :]
                                                    + np.sin(ang)[None, :, None] * t2[:, None, :]))
    r = cs.p - np.asarray(com, float)
    M = np.cross(r[:, None, :], E)                                     # moments of the edges (K, E, 3)
    nv = K * N_EDGES
    # variables: lambda (K*E) >= 0, torsion plus/minus (K each) >= 0, alpha >= 0
    A_eq = np.zeros((7, nv + 2 * K + 1))
    A_eq[0:3, :nv] = E.reshape(nv, 3).T
    A_eq[3:6, :nv] = M.reshape(nv, 3).T
    A_eq[3:6, nv:nv + K] = cs.n.T
    A_eq[3:6, nv + K:nv + 2 * K] = -cs.n.T
    A_eq[0:3, -1] = F_ext
    A_eq[3:6, -1] = T_ext
    fing = cs.part == 0
    A_eq[6, :nv] = (np.einsum("ked,kd->ke", E, cs.c) * fing[:, None]).reshape(nv)
    b_eq = np.array([0, 0, 0, 0, 0, 0, 2 * finger_force], float)
    # torsion: m+ + m- <= mu_t * normal force (the normal component of each edge is 1)
    A_ub = np.zeros((K, nv + 2 * K + 1))
    for k in range(K):
        A_ub[k, k * N_EDGES:(k + 1) * N_EDGES] = -cs.mu_t[k]
        A_ub[k, nv + k] = A_ub[k, nv + K + k] = 1.0
    c = np.zeros(nv + 2 * K + 1)
    c[-1] = -1.0
    bounds = [(0, None)] * (nv + 2 * K) + [(0, ALPHA_MAX)]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(K), A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return float(res.x[-1]) if res.status == 0 else 0.0


def utilisation_static(cs: ContactSet, com: np.ndarray, mass: float, g_obj: np.ndarray,
                       finger_force: float = 20.0) -> float:
    """u for the object's weight with gravity g_obj (object frame, m/s^2): 1 / capacity."""
    a = capacity(cs, com, mass * np.asarray(g_obj, float), None, finger_force)
    return np.inf if a <= 0 else 1.0 / a


def apply_to_reports(reports, contacts, obj, target_poses: np.ndarray, u_ok: float = 0.8,
                     finger_force: float = 20.0) -> None:
    """Replace the point-model stability margin of every reachable, visible target in feasibility
    reports with this model's: margin = (u_ok - u) / u_ok, positive when u < u_ok (in validation,
    70 % of grasps with u in 0.75-0.8 held, 98 % below 0.5). contacts[i]: the ContactSet of
    grasp i (None: the fingers did not close on it -> every target fails). A grasp and its twin
    (the hand turned 180 deg about the approach) touch the object at the same places, so one
    contact set serves both. target_poses: (R, 4, 4) world poses of the object at the targets."""
    g_world = np.array([0.0, 0.0, -9.81])
    cache = {}
    for i, rep in enumerate(reports):
        cs = contacts[i]
        for tw in rep.twins:
            if not tw.start_ok:
                continue
            for j, t in enumerate(tw.targets):
                if not t.reach.ok or t.visibility is None or not t.visibility.ok():
                    continue
                if cs is None:
                    t.stability_margin = -1.0
                    continue
                if (i, j) not in cache:
                    g = target_poses[j][:3, :3].T @ g_world
                    cache[i, j] = utilisation_static(cs, obj.com, obj.mass, g, finger_force)
                u = cache[i, j]
                t.stability_margin = -1.0 if not np.isfinite(u) else (u_ok - u) / u_ok
