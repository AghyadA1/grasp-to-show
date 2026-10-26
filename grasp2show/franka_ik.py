"""Analytic inverse kinematics of the Franka Panda, vectorised over poses and wrist angles.

The Panda has 7 joints for a 6-DoF task, so its IK has a one-parameter family of solutions.
Fixing the last joint q7 removes the redundancy; the remaining six joints then follow from
geometry in closed form, with up to 4 solutions (two "elbow" branches for q6 x two
"shoulder" branches for q1, q2). Sampling q7 on a grid covers the redundancy.

Geometric solver of He & Liu, "Analytical Inverse Kinematics for Franka Emika Panda",
ICCMA 2021 (doi:10.1109/ICCMA54375.2021.9646185). This is a numpy port of the C++
implementation in https://github.com/CoMMALab/frantik (Apache-2.0, itself based on
https://github.com/ffall007/franka_analytical_ik), vectorised so a whole batch of poses x
q7 values is solved in one call. Frame: the end effector is the Franka Hand's TCP
(d7e = 0.107 flange + 0.1034 hand = 0.2104 m), which is our "tcp" site.

All solutions returned are within joint limits and reproduce the pose (checked by forward
kinematics); invalid entries are NaN.

Known limitation (shared with the reference implementation): the elbow triangle formula
gives q4 = theta246 - 3.609 with theta246 in [0, pi], i.e. q4 <= -0.468 rad. The almost
straight elbow, q4 in (-0.468, -0.0698], about 13 % of the q4 range at the edge of the
workspace, is not covered. Reachability judged with this IK is therefore conservative there.
Two fixes relative to the reference: the shoulder-singularity shortcut (q2 := 0) is applied
only when joint 2's axis is exactly vertical (the reference used a 2.6 deg band, returning
poses up to metres off), and every solution is verified by forward kinematics.
"""
from __future__ import annotations

import numpy as np

from ._accel import HAVE_CPP, cpp

D1, D3, D5, D7E, A4, A7 = 0.3330, 0.3160, 0.3840, 0.2104, 0.0825, 0.0880
LL24, LL46 = A4**2 + D3**2, A4**2 + D5**2
L24, L46 = np.sqrt(LL24), np.sqrt(LL46)
THETA_H46 = np.arctan(D5 / A4)
THETA_342 = np.arctan(D3 / A4)
THETA_46H = np.pi / 2 - np.arctan(D5 / A4)
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
NEAR_ONE = 1.0 - 1e-10
FK_TOL = 1e-6  # m / rad-ish: max abs difference of the 4x4 pose


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _within(q, j):
    return (q > Q_MIN[j]) & (q < Q_MAX[j])


def ik(T: np.ndarray, q7: np.ndarray, q1_singular: float = 0.0, use_cpp: bool = True) -> np.ndarray:
    """All IK solutions for TCP poses T (N, 4, 4) and wrist angles q7.

    q7: (K,) a grid shared by all poses, or (N, K) per pose.
    Returns q (N, K, 4, 7); NaN rows are invalid (unreachable or outside joint limits).
    q1_singular: q1 to use when the shoulder is singular (joint-2 axis vertical).
    Runs the C++ kernel when it is installed (see _accel.py), numpy otherwise.
    """
    T = np.asarray(T, float).reshape(-1, 4, 4)
    N = len(T)
    q7 = np.asarray(q7, float)
    q7b = np.broadcast_to(q7[None, :] if q7.ndim == 1 else q7, (N, q7.shape[-1]))
    if use_cpp and HAVE_CPP:
        return cpp.ik(np.ascontiguousarray(T), np.ascontiguousarray(q7b), q1_singular)
    K = q7b.shape[1]
    R = np.broadcast_to(T[:, None, :3, :3], (N, K, 3, 3))
    z_ee = R[..., :, 2]
    p_ee = np.broadcast_to(T[:, None, :3, 3], (N, K, 3))

    with np.errstate(invalid="ignore", divide="ignore"):
        # p_6: wrist centre, from the flange back along x_6.
        p_7 = p_ee - D7E * z_ee
        x_ee_6 = np.stack([np.cos(q7b - np.pi / 4), -np.sin(q7b - np.pi / 4), np.zeros_like(q7b)], -1)
        x_6 = _unit(np.einsum("nkij,nkj->nki", R, x_ee_6))
        p_6 = p_7 - A7 * x_6

        # q4: triangle (shoulder 2, elbow 4, wrist 6).
        p_2 = np.array([0.0, 0.0, D1])
        V26 = p_6 - p_2
        LL26 = np.sum(V26 * V26, -1)
        L26 = np.sqrt(LL26)
        tri_ok = (L24 + L46 >= L26) & (L24 + L26 >= L46) & (L26 + L46 >= L24)
        theta246 = np.arccos(np.clip((LL24 + LL46 - LL26) / (2 * L24 * L46), -1, 1))
        q4 = theta246 + THETA_H46 + THETA_342 - 2 * np.pi
        ok = tri_ok & _within(q4, 3) & _within(q7b, 6)

        # q6: two branches.
        theta462 = np.arccos(np.clip((LL26 + LL46 - LL24) / (2 * L26 * L46), -1, 1))
        theta26H = THETA_46H + theta462
        D26 = -L26 * np.cos(theta26H)
        Z_6 = np.cross(z_ee, x_6)
        Y_6 = np.cross(Z_6, x_6)
        R_6 = np.stack([x_6, _unit(Y_6), _unit(Z_6)], -1)                      # columns
        V_6_62 = np.einsum("nkji,nkj->nki", R_6, -V26)                          # R_6^T (-V26)
        Phi6 = np.arctan2(V_6_62[..., 1], V_6_62[..., 0])
        Theta6 = np.arcsin(np.clip(D26 / np.hypot(V_6_62[..., 0], V_6_62[..., 1]), -1, 1))
        q6 = np.stack([np.pi - Theta6 - Phi6, Theta6 - Phi6], -1)               # (N, K, 2)
        q6 = np.where(q6 <= Q_MIN[5], q6 + 2 * np.pi, np.where(q6 >= Q_MAX[5], q6 - 2 * np.pi, q6))
        ok6 = _within(q6, 5) & ok[..., None]

        # q1, q2: from the point P on the joint-5 axis.
        thetaP26 = 3 * np.pi / 2 - theta462 - theta246 - THETA_342
        thetaP = np.pi - thetaP26 - theta26H
        LP6 = L26 * np.sin(thetaP26) / np.sin(thetaP)
        z_6_5 = np.stack([np.sin(q6), np.cos(q6), np.zeros_like(q6)], -1)       # (N, K, 2, 3)
        z_5 = np.einsum("nkij,nkbj->nkbi", R_6, z_6_5)
        V2P = p_6[:, :, None] - LP6[..., None, None] * z_5 - p_2
        L2P = np.linalg.norm(V2P, axis=-1)
        singular = np.abs(V2P[..., 2] / L2P) > NEAR_ONE
        q1a = np.where(singular, q1_singular, np.arctan2(V2P[..., 1], V2P[..., 0]))
        q2a = np.where(singular, 0.0, np.arccos(np.clip(V2P[..., 2] / L2P, -1, 1)))
        q1b = np.where(singular, q1_singular, q1a + np.where(q1a < 0, np.pi, -np.pi))
        q2b = np.where(singular, 0.0, -q2a)

        # Assemble the 4 branches: index 2*i + j, i = q6 branch, j = shoulder branch.
        q = np.full((N, K, 4, 7), np.nan)
        q1 = np.stack([q1a[..., 0], q1b[..., 0], q1a[..., 1], q1b[..., 1]], -1)
        q2 = np.stack([q2a[..., 0], q2b[..., 0], q2a[..., 1], q2b[..., 1]], -1)
        q6s = np.repeat(q6, 2, axis=-1)
        z5s = np.repeat(z_5, 2, axis=2)
        V2Ps = np.repeat(V2P, 2, axis=2)
        valid = np.repeat(ok6, 2, axis=-1) & _within(q1, 0) & _within(q2, 1)

        # q3
        z_3 = _unit(V2Ps)
        y_3 = _unit(-np.cross(V26[:, :, None], V2Ps))
        x_3 = np.cross(y_3, z_3)
        c1, s1, c2, s2 = np.cos(q1), np.sin(q1), np.cos(q2), np.sin(q2)
        # R_2 = Rz(q1) @ [[c2, -s2, 0], [0, 0, 1], [-s2, -c2, 0]]; x_2_3 = R_2^T x_3
        R_2 = np.zeros(q1.shape + (3, 3))
        R_2[..., 0, 0], R_2[..., 0, 1], R_2[..., 0, 2] = c1 * c2, -c1 * s2, -s1
        R_2[..., 1, 0], R_2[..., 1, 1], R_2[..., 1, 2] = s1 * c2, -s1 * s2, c1
        R_2[..., 2, 0], R_2[..., 2, 1], R_2[..., 2, 2] = -s2, -c2, 0.0
        x_2_3 = np.einsum("...ji,...j->...i", R_2, x_3)
        q3 = np.arctan2(x_2_3[..., 2], x_2_3[..., 0])
        valid &= _within(q3, 2)

        # q5
        VH4 = p_2 + D3 * z_3 + A4 * x_3 - p_6[:, :, None] + D5 * z5s
        c6, s6 = np.cos(q6s), np.sin(q6s)
        R_5_6 = np.zeros(q6s.shape + (3, 3))
        R_5_6[..., 0, 0], R_5_6[..., 0, 1] = c6, -s6
        R_5_6[..., 1, 2] = -1.0
        R_5_6[..., 2, 0], R_5_6[..., 2, 1] = s6, c6
        R_5 = np.einsum("nkij,nkbmj->nkbim", R_6, R_5_6)                        # R_6 @ R_5_6^T
        V_5_H4 = np.einsum("...ji,...j->...i", R_5, VH4)
        q5 = -np.arctan2(V_5_H4[..., 1], V_5_H4[..., 0])
        valid &= _within(q5, 4)

    q[..., 0], q[..., 1], q[..., 2] = q1, q2, q3
    q[..., 3] = q4[..., None]
    q[..., 4], q[..., 5] = q5, q6s
    q[..., 6] = q7b[..., None]
    q[~valid] = np.nan
    # Safety net: keep only solutions that reproduce the pose.
    v = ~np.isnan(q[..., 0])
    if v.any():
        Tn = np.broadcast_to(T[:, None, None], (N, K, 4, 4, 4))[v]
        bad = np.abs(fk(q[v]) - Tn).max(axis=(-1, -2)) > FK_TOL
        idx = np.argwhere(v)[bad]
        q[tuple(idx.T)] = np.nan
    return q


def fk(q: np.ndarray) -> np.ndarray:
    """TCP pose (..., 4, 4) for joint angles q (..., 7), Craig (modified DH) parameters."""
    q = np.asarray(q, float)
    a = [0, 0, 0, A4, -A4, 0, A7]
    d = [D1, 0, D3, 0, D5, 0, 0]
    alpha = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
    T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    for i in range(7):
        ct, st = np.cos(q[..., i]), np.sin(q[..., i])
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        A = np.zeros(q.shape[:-1] + (4, 4))
        A[..., 0, 0], A[..., 0, 1], A[..., 0, 3] = ct, -st, a[i]
        A[..., 1, 0], A[..., 1, 1], A[..., 1, 2], A[..., 1, 3] = st * ca, ct * ca, -sa, -sa * d[i]
        A[..., 2, 0], A[..., 2, 1], A[..., 2, 2], A[..., 2, 3] = st * sa, ct * sa, ca, ca * d[i]
        A[..., 3, 3] = 1.0
        T = T @ A
    # flange -> TCP: 0.2104 m along z, rotated -45 deg about z (Franka Hand mounting)
    c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    F = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, D7E], [0, 0, 0, 1.0]])
    return T @ F


def limit_margin(q: np.ndarray) -> np.ndarray:
    """Smallest distance (rad) of any joint to its limit, (...,). NaN where q is NaN."""
    return np.min(np.minimum(q - Q_MIN, Q_MAX - q), axis=-1)
