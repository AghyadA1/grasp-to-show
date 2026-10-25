"""Inverse kinematics for the Panda TCP with mink (differential IK solved as a QP).

Each iteration solves   min_v  sum_i || J_i v + K_i e_i ||^2_{W_i} + damping ||v||^2
subject to joint-limit constraints, then integrates q <- q + v*dt. The frame task drives
the TCP to the target pose; a low-weight posture task picks a natural solution among the
redundant ones (the Panda has 7 DoF for a 6-DoF task) and keeps us away from limits.

The IK uses its own robot-only model, so the object's free joint is not part of the
optimisation. Collision checks against the scene come later (checks 3 and 4).
"""
from __future__ import annotations

from dataclasses import dataclass

import mink
import numpy as np

from .config import SceneConfig
from .scene import HOME_Q, robot_spec


@dataclass
class IKResult:
    q: np.ndarray          # 7 arm joint angles
    success: bool
    pos_err: float         # m
    rot_err: float         # rad
    iters: int
    limit_margin: float    # smallest distance to a joint limit (rad); used by checks 3/4


class PandaIK:
    def __init__(self, cfg: SceneConfig | None = None, dt: float = 0.05, max_iters: int = 400,
                 pos_tol: float = 1e-3, rot_tol: float = 1e-2):
        cfg = cfg or SceneConfig()
        self.model = robot_spec(cfg).compile()
        self.configuration = mink.Configuration(self.model)
        self.frame_task = mink.FrameTask("tcp", "site", position_cost=1.0, orientation_cost=1.0,
                                         lm_damping=1e-6)
        self.posture_task = mink.PostureTask(self.model, cost=1e-2)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.dt, self.max_iters, self.pos_tol, self.rot_tol = dt, max_iters, pos_tol, rot_tol
        self.lower = self.model.jnt_range[:7, 0].copy()
        self.upper = self.model.jnt_range[:7, 1].copy()

    def _full_q(self, q_arm: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[:7] = q_arm
        q[7:9] = 0.02
        return q

    def solve(self, T_world_tcp: np.ndarray, q_seed: np.ndarray | None = None,
              q_posture: np.ndarray | None = None) -> IKResult:
        q_seed = HOME_Q if q_seed is None else q_seed
        q_posture = HOME_Q if q_posture is None else q_posture
        self.configuration.update(self._full_q(q_seed))
        self.posture_task.set_target(self._full_q(q_posture))
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(T_world_tcp[:3, :3]), T_world_tcp[:3, 3])
        self.frame_task.set_target(target)

        tasks = [self.frame_task, self.posture_task]
        for it in range(1, self.max_iters + 1):
            v = mink.solve_ik(self.configuration, tasks, self.dt, "daqp", damping=1e-3,
                              limits=self.limits)
            self.configuration.integrate_inplace(v, self.dt)
            err = self.frame_task.compute_error(self.configuration)
            pos_err, rot_err = np.linalg.norm(err[:3]), np.linalg.norm(err[3:])
            if pos_err < self.pos_tol and rot_err < self.rot_tol:
                break
        q = self.configuration.q[:7].copy()
        margin = float(np.min(np.minimum(q - self.lower, self.upper - q)))
        ok = pos_err < self.pos_tol and rot_err < self.rot_tol
        return IKResult(q, bool(ok), float(pos_err), float(rot_err), it, margin)

    def solve_path(self, T_list: list[np.ndarray], q_seed: np.ndarray) -> list[IKResult]:
        """Solve a sequence of nearby poses, each seeded with the previous solution.

        Seeding (and using the previous solution as the posture target) keeps the arm on
        one IK branch, so a straight Cartesian line maps to a smooth joint-space path.
        """
        out, q = [], q_seed
        for T in T_list:
            r = self.solve(T, q_seed=q, q_posture=q)
            out.append(r)
            q = r.q
        return out
