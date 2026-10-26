"""Part 2: shaping the transport motion so the grasp carries the load where it is strong.

Fixed: the grasp, the start (after the lift) and the end (presentation) joint configurations.
Free: the path between them and its duration T. The trajectory is one smooth piece, so there are
no stops on the way:

    q(tau) = q0 + (q1 - q0) s(tau) + sum_k c_k phi_k(tau),     tau = t / T in [0, 1]
    s      = minimum-jerk progress 10 tau^3 - 15 tau^4 + 6 tau^5
    phi_k  = 64 (tau (1 - tau))^3 T_k(2 tau - 1)   (Chebyshev polynomials; peak ~1 rad per unit c)

Every phi_k has zero value, velocity and acceleration at both ends, so the start and end are at
rest whatever the coefficients. All terms are polynomials: velocities, accelerations and jerk
are exact.

Problem (defaults from the plan: friction utilisation <= 0.7, <= 60 % of the Franka limits):
    minimise   T * (1 + w_s * extra_jerk)                    (fast, and no needless wiggles)
    subject to u(t) <= u_lim                                 (grasp: soft-finger model + d'Alembert,
                                                              worst case over +-5 mm COM errors: leverage)
               |qd| <= r QD_MAX,  |qdd| <= r QDD_MAX,  joint limits
               no collision (arm and object vs table, robot)
Two solvers:
    retime_only     c = 0 (the straight joint-space path), smallest T that satisfies everything:
                    the "slow down" answer (as in Pham & Pham 2019, here with uniform scaling)
    shape_and_time  c and T together, by a cross-entropy search (diagonal Gaussian, elites)
                    started from the retimed solution: the "move differently" answer
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

from .franka_ik import Q_MAX, Q_MIN, fk
from .stability import StabilityParams, finger_contacts
from .transport import QD_MAX, QDD_MAX, inertial_load, robust_utilisation

S_MINJERK = Polynomial([0, 0, 0, 10, -15, 6])


def _basis(K: int) -> list[Polynomial]:
    bump = Polynomial([0, 1, -1]) ** 3 * 64.0                        # 64 (tau (1 - tau))^3
    shift = Polynomial([-1, 2])                                      # 2 tau - 1
    return [bump * Chebyshev.basis(k).convert(kind=Polynomial)(shift) for k in range(K)]


@dataclass
class Solution:
    T: float
    c: np.ndarray            # (K, 7)
    feasible: bool
    info: dict


class TransportProblem:
    def __init__(self, obj, T_obj_tcp: np.ndarray, width: float, q0: np.ndarray, q1: np.ndarray,
                 prm: StabilityParams, checker=None, u_lim: float = 0.7, limit_ratio: float = 0.6,
                 K: int = 4, n: int = 80, w_smooth: float = 0.05, com_margin: float = 0.005,
                 axial_frac: float = 0.35):
        self.obj, self.T_obj_tcp, self.width = obj, np.asarray(T_obj_tcp, float), float(width)
        self.q0, self.q1, self.prm = np.asarray(q0, float), np.asarray(q1, float), prm
        self.checker, self.u_lim, self.r, self.K, self.n, self.w_smooth = checker, u_lim, limit_ratio, K, n, w_smooth
        self.com_margin = com_margin   # leverage robustness: the grasp must hold with the COM this far off
        # Load along the finger axis is capped at axial_frac x 2F. The ideal model allows up to 2F,
        # but in simulation a box balanced on the lower finger tips over (an inverted pendulum with
        # lever = half the grasp width): a 1.5 kg box (14.7 N axial) held with 9 mm slip, 2 kg was
        # lost (scripts/hold_tilt_test.py). 0.35 x 40 N = 14 N matches that.
        self.axial_cap = axial_frac * 2 * prm.finger_force
        self.tau = np.linspace(0.0, 1.0, n)
        self.dq = self.q1 - self.q0
        polys = _basis(K)
        # basis values and derivatives at the samples: B[d] (K, n) for derivative order d
        self.B = [np.stack([p.deriv(dd)(self.tau) if dd else p(self.tau) for p in polys]) for dd in range(4)]
        self.S = [S_MINJERK.deriv(dd)(self.tau) if dd else S_MINJERK(self.tau) for dd in range(4)]
        c1, c2, ok = finger_contacts(obj.mesh, self.T_obj_tcp)
        self.contacts = (c1[0], c2[0]) if ok[0] else None
        self.T_tcp_obj = np.linalg.inv(self.T_obj_tcp)
        self.jerk_ref = self._jerk_integral(np.zeros((K, 7)))
        # Gravity alone at the two ends (the object at rest): no motion can do better than this.
        # If an end already needs more than u_lim, the limit is relaxed to that end's need + 0.05
        # (the task is then only possible slowly); above 0.95 it is impossible (static_ok False).
        rest = self.evaluate(np.zeros((K, 7)), 1e3, collisions=False)["u"]
        self.u_static = float(max(rest[0], rest[-1]))
        self.static_ok = self.u_static < 0.95
        self.u_lim = max(u_lim, self.u_static + 0.05)   # never ask more of the motion than of the ends

    # -- trajectory --------------------------------------------------------------------------
    def derivs(self, c: np.ndarray, T: float):
        """q, qd, qdd, qddd at the samples (n, 7) for coefficients c (K, 7) and duration T."""
        out = []
        for dd in range(4):
            q_tau = np.outer(self.S[dd], self.dq) + self.B[dd].T @ c + (self.q0 if dd == 0 else 0.0)
            out.append(q_tau / T**dd)
        return out

    def _jerk_integral(self, c: np.ndarray) -> float:
        """Duration-free jerk measure: integral over tau of |d^3 q / d tau^3|^2."""
        j = self.derivs(c, 1.0)[3]
        return float(np.mean(np.sum(j**2, axis=1)))

    def sample(self, c: np.ndarray, T: float, dt: float) -> np.ndarray:
        """Joint trajectory sampled every dt (for execution)."""
        tau = np.arange(1, max(round(T / dt), 1) + 1) * dt / T
        tau = np.clip(tau, 0, 1)
        B = np.stack([p(tau) for p in _basis(self.K)])
        return self.q0 + np.outer(S_MINJERK(tau), self.dq) + B.T @ c

    # -- evaluation ----------------------------------------------------------------------------
    def evaluate(self, c: np.ndarray, T: float, collisions: bool = True) -> dict:
        q, qd, qdd, _ = self.derivs(c, T)
        T_tcp = fk(q)
        T_obj = T_tcp @ self.T_tcp_obj
        if self.contacts is None:
            u = np.full(self.n, np.inf)
        else:
            F, Tq = inertial_load(T_obj, T / (self.n - 1), self.obj.mass, self.obj.com, self.obj.inertia)
            u = robust_utilisation(self.contacts[0], self.contacts[1], self.T_obj_tcp, self.obj.com, F, Tq,
                                   self.prm, self.com_margin)
            u = np.maximum(u, np.abs(F @ self.T_obj_tcp[:3, 1]) / self.axial_cap)
        lim = max(float(np.max(np.abs(qd) / (self.r * QD_MAX))), float(np.max(np.abs(qdd) / (self.r * QDD_MAX))))
        jl = float(np.max(np.maximum(Q_MIN - q, q - Q_MAX)))
        hits = 0
        if collisions and self.checker is not None:
            for k in range(0, self.n, 2):
                self.checker._set(q[k], self.width, T_obj[k])
                hits += bool(self.checker._bad_contacts(holding=True))
        u_f = np.where(np.isfinite(u), u, 10.0)
        return {"T": T, "u_max": float(u_f.max()), "u": u_f, "limit": lim, "joint_limit": jl, "hits": hits,
                "jerk_rel": self._jerk_integral(c) / max(self.jerk_ref, 1e-9) - 1.0,
                "feasible": bool(u_f.max() <= self.u_lim + 1e-3 and lim <= 1.0 + 1e-3 and jl <= 0 and hits == 0)}

    def cost(self, c: np.ndarray, T: float) -> tuple[float, dict]:
        e = self.evaluate(c, T)
        viol = (max(e["u_max"] - self.u_lim, 0) * 10) ** 2 + max(e["limit"] - 1, 0) ** 2 * 10 \
            + max(e["joint_limit"], 0) * 100 + e["hits"] * 2.0
        return T * (1 + self.w_smooth * max(e["jerk_rel"], 0)) + 100 * viol, e

    # -- solvers ---------------------------------------------------------------------------------
    def retime_only(self, T_lo: float = 0.3, T_hi: float = 20.0, iters: int = 20) -> Solution:
        c = np.zeros((self.K, 7))
        e = self.evaluate(c, T_hi)
        if not e["feasible"]:
            return Solution(T_hi, c, False, e)
        for _ in range(iters):
            mid = np.sqrt(T_lo * T_hi)
            if self.evaluate(c, mid, collisions=False)["feasible"]:
                T_hi = mid
            else:
                T_lo = mid
        return Solution(T_hi, c, True, self.evaluate(c, T_hi))

    def shape_and_time(self, start: Solution | None = None, iters: int = 60, pop: int = 32, elite: int = 6,
                       sigma_c: float = 0.3, seed: int = 0) -> Solution:
        rng = np.random.default_rng(seed)
        start = start or self.retime_only()
        dim = self.K * 7 + 1
        mean = np.concatenate([start.c.ravel(), [np.log(start.T)]])
        std = np.concatenate([np.full(self.K * 7, sigma_c), [0.3]])
        best_x, (best_f, _) = mean.copy(), self.cost(start.c, start.T)
        for _ in range(iters):
            X = mean + std * rng.standard_normal((pop, dim))
            X[0] = best_x
            F = []
            for x in X:
                f, _ = self.cost(x[:-1].reshape(self.K, 7), float(np.exp(x[-1])))
                F.append(f)
                if f < best_f:
                    best_f, best_x = f, x.copy()
            idx = np.argsort(F)[:elite]
            mean = 0.7 * X[idx].mean(0) + 0.3 * mean
            std = np.maximum(0.7 * X[idx].std(0) + 0.3 * std, 1e-3)
        c, T = best_x[:-1].reshape(self.K, 7), float(np.exp(best_x[-1]))
        e = self.evaluate(c, T)
        if not e["feasible"] and not e["hits"] and e["joint_limit"] <= 0:
            # The search ended just outside the constraints: slow the same path down until it fits.
            for T2 in T * 1.05 ** np.arange(1, 40):
                e2 = self.evaluate(c, float(T2), collisions=False)
                if e2["feasible"]:
                    T, e = float(T2), self.evaluate(c, float(T2))
                    break
        if start.feasible and (not e["feasible"] or T >= start.T):
            return start                       # shaping did not help: keep the retimed straight motion
        return Solution(T, c, e["feasible"], e)
