"""Checks 2-5 for one grasp: everything except lift stability (check 1, gripper_sim / stability).

For a grasp T_obj_tcp (and its 180-degree twin, which is a different arm configuration):
  check 3  start  : the TCP pose T_start_obj @ T_obj_tcp and the pre-grasp pose (10 cm back
                    along the approach) have an IK solution within joint limits, with the open
                    gripper and the arm free of collision with the table and the object;
  check 4  target : for each pose in the target set (rotations about the viewing axis), an IK
                    solution exists with the closed gripper holding the object, and no contact
                    except fingers-object (arm-object, arm-table, object-table, self, scanner);
  check 2  target : static friction margin at the target orientation (stability.check_margin);
  check 5  target : the QR is visible from the scanner camera: rays from the camera to a grid
                    of points on the patch reach the patch first (not a finger, the hand or the
                    arm), including all four corner zones (the finder patterns live in three of
                    them); plus the viewing angle and the QR's size in pixels.
The same twin must pass at the start and at the target: the robot cannot regrasp.

IK: the analytic solver (franka_ik) over a grid of q7 values; candidates are tried in order of
joint-limit margin until one is collision-free. Collision: MuJoCo kinematics + collision only
(7 microseconds per query), penetration tolerance 1 mm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import SceneConfig
from .franka_ik import Q_MAX, Q_MIN, ik, limit_margin
from .scene import PANDA_BODIES, Scene, start_pose
from .stability import StabilityParams, check_margin
from .task import show_pose
from .transforms import make_T


@dataclass
class PoseCheck:
    ok: bool
    reachable: bool                     # an IK solution within joint limits exists
    q: np.ndarray | None = None         # the collision-free configuration found
    limit_margin: float = np.nan        # rad
    contacts: list[str] = field(default_factory=list)  # offending contacts of the best candidate


@dataclass
class Visibility:
    fraction: float                     # of sample rays reaching the QR
    corners_ok: bool                    # all four corner zones visible
    angle_deg: float                    # between the QR normal and the direction to the camera
    pixels: float                       # QR side length in the image
    blocked_by: list[str] = field(default_factory=list)

    def ok(self, min_fraction: float = 0.9, max_angle: float = 45.0, min_pixels: float = 50.0) -> bool:
        return (self.corners_ok and self.fraction >= min_fraction and self.angle_deg <= max_angle
                and self.pixels >= min_pixels)


@dataclass
class TargetCheck:
    roll: float
    reach: PoseCheck
    stability_margin: float
    visibility: Visibility | None

    @property
    def ok(self) -> bool:
        return self.reach.ok and self.stability_margin > 0 and self.visibility is not None and self.visibility.ok()


@dataclass
class TwinReport:
    twin: int                           # 0: the grasp as given, 1: rotated 180 deg about the approach
    start: PoseCheck
    pregrasp: PoseCheck
    targets: list[TargetCheck]

    @property
    def start_ok(self) -> bool:
        return self.start.ok and self.pregrasp.ok

    @property
    def ok(self) -> bool:
        return self.start_ok and any(t.ok for t in self.targets)


@dataclass
class GraspReport:
    twins: list[TwinReport]

    @property
    def ok(self) -> bool:
        return any(t.ok for t in self.twins)

    def best(self) -> tuple[TwinReport, TargetCheck] | None:
        """The passing (twin, target) with the largest min(joint margin, stability margin)."""
        best, key = None, -np.inf
        for tw in self.twins:
            if not tw.start_ok:
                continue
            for t in tw.targets:
                if t.ok:
                    k = min(tw.start.limit_margin, t.reach.limit_margin, t.stability_margin)
                    if k > key:
                        best, key = (tw, t), k
        return best


def twin_of(T_obj_tcp: np.ndarray) -> np.ndarray:
    T = T_obj_tcp.copy()
    T[:3, :3] = T[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
    return T


class FeasibilityChecker:
    def __init__(self, scene: Scene, cfg: SceneConfig, n_q7: int = 16, max_tries: int = 8,
                 pen_tol: float = 0.001, pregrasp_dist: float = 0.10, n_vis: int = 7,
                 stability: StabilityParams | None = None, target_load_factor: float = 1.2):
        self.m, self.cfg, self.obj = scene.model, cfg, scene.obj
        self.d = mujoco.MjData(self.m)
        self.q7 = np.linspace(Q_MIN[6] + 0.02, Q_MAX[6] - 0.02, n_q7)
        self.max_tries, self.pen_tol, self.pregrasp_dist = max_tries, pen_tol, pregrasp_dist
        self.stab = stability or StabilityParams(mu=cfg.obj.friction[0], mu_t=cfg.obj.friction[1],
                                                 finger_force=cfg.gripper.finger_force)
        self.target_load_factor = target_load_factor
        m = self.m
        self.robot = {m.body(b).id for b in PANDA_BODIES}
        self.fingers = {m.body("left_finger").id, m.body("right_finger").id}
        self.hand = m.body("hand").id
        self.hooked = False      # plan_hook: the held object hangs on a finger and may lean on the hand
        self.engaging = False    # a hook's fingers turning up in the handle, the object still standing
        self.obj_body = m.body(self.obj.name).id
        self.obj_qadr = m.jnt_qposadr[m.joint(f"{self.obj.name}_free").id]
        self.qr_geom = m.geom(f"{self.obj.name}_qr").id
        self.T_start = start_pose(cfg, self.obj)
        # QR sample points (QR frame): a grid inside the plate, plus the four corner zones.
        s = self.obj.qr.side
        a = np.linspace(-0.45, 0.45, n_vis) * s
        Y, Z = np.meshgrid(a, a)
        grid = np.stack([np.full(Y.size, 0.0005), Y.ravel(), Z.ravel()], 1)
        c = (0.5 - 5.5 / 25) * s                     # finder-pattern centres: 5.5 modules in of 25
        corners = np.array([[0.0005, sy * c, sz * c] for sy in (-1, 1) for sz in (-1, 1)])
        self.vis_pts = np.vstack([grid, corners])
        self.n_grid = len(grid)
        cam = cfg.camera
        self.cam_pos = np.asarray(cam.pos, float)
        self.focal_px = 0.5 * cam.height / np.tan(np.radians(cam.fovy_deg) / 2)
        self._geomgroup = np.array([1, 1, 1, 0, 0, 0], np.uint8)   # world/table, QR, visual meshes

    # -- state and contacts ---------------------------------------------------------------
    def _set(self, q: np.ndarray, finger_gap: float, T_world_obj: np.ndarray) -> None:
        d = self.d
        d.qpos[:7] = q
        d.qpos[7:9] = finger_gap / 2.0
        a = self.obj_qadr
        d.qpos[a:a + 3] = T_world_obj[:3, 3]
        mujoco.mju_mat2Quat(d.qpos[a + 3:a + 7], np.ascontiguousarray(T_world_obj[:3, :3]).ravel())
        mujoco.mj_kinematics(self.m, d)

    def _bad_contacts(self, holding: bool) -> list[str]:
        m, d = self.m, self.d
        mujoco.mj_collision(m, d)
        bad = []
        for c in d.contact[: d.ncon]:
            if c.dist > -self.pen_tol:
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            pair = {b1, b2}
            if self.obj_body in pair:
                other = (pair - {self.obj_body}).pop() if len(pair) == 2 else self.obj_body
                if holding and other in self.fingers:
                    continue                                   # the grasp itself
                if holding and self.hooked and other == self.hand:
                    continue                                   # a hooked object rests on the hand too
                if self.engaging and (other in self.fingers or other == self.hand or other == 0):
                    continue                                   # a hook engaging: object still on the table
                if not holding and other == 0:
                    continue                                   # object resting on the table
            name = f"{m.body(b1).name}-{m.body(b2).name}"
            if name not in bad:
                bad.append(name)
        return bad

    # -- checks ----------------------------------------------------------------------------
    def _reach(self, sols: np.ndarray, finger_gap: float, T_world_obj: np.ndarray, holding: bool,
               q_ref: np.ndarray | None = None, min_margin: float = 0.1) -> PoseCheck:
        """sols (K, 4, 7) IK candidates for one pose. Candidates are tried in order of joint-limit
        margin, or, if q_ref is given, of closeness to q_ref among those with margin >= min_margin
        (so the arm does not, e.g., unwind its wrist by a full turn between start and target)."""
        q = sols.reshape(-1, 7)
        q = q[~np.isnan(q[:, 0])]
        if len(q) == 0:
            return PoseCheck(False, False)
        lm = limit_margin(q)
        if q_ref is None:
            order = np.argsort(-lm)[: self.max_tries]
        else:
            dist = np.abs(q - q_ref).max(axis=1) + np.where(lm >= min_margin, 0.0, 100.0)
            order = np.argsort(dist)[: self.max_tries]
        first_contacts = []
        for i in order:
            self._set(q[i], finger_gap, T_world_obj)
            bad = self._bad_contacts(holding)
            if not bad:
                return PoseCheck(True, True, q[i].copy(), float(lm[i]))
            if not first_contacts:
                first_contacts = bad
        return PoseCheck(False, True, None, float(lm[order[0]]), first_contacts)

    def visibility(self, q: np.ndarray, finger_gap: float, T_world_obj: np.ndarray) -> Visibility:
        self._set(q, finger_gap, T_world_obj)
        T_world_qr = T_world_obj @ self.obj.T_obj_qr
        pts = (T_world_qr[:3, :3] @ self.vis_pts.T).T + T_world_qr[:3, 3]
        hits, blocked = np.zeros(len(pts), bool), []
        gid = np.zeros(1, np.int32)
        for i, p in enumerate(pts):
            v = p - self.cam_pos
            # Start 5 cm in front of the lens: the camera sits inside the scanner's housing geom.
            start = self.cam_pos + 0.05 * v / np.linalg.norm(v)
            mujoco.mj_ray(self.m, self.d, start, p - start, self._geomgroup, 1, -1, gid)
            hits[i] = gid[0] == self.qr_geom
            if not hits[i] and gid[0] >= 0:
                name = self.m.body(self.m.geom_bodyid[gid[0]]).name
                if name not in blocked:
                    blocked.append(name)
        n = T_world_qr[:3, 0]
        to_cam = self.cam_pos - T_world_qr[:3, 3]
        dist = np.linalg.norm(to_cam)
        cosang = float(np.clip(n @ to_cam / dist, -1, 1))
        return Visibility(float(hits[: self.n_grid].mean()), bool(hits[self.n_grid:].all()),
                          float(np.degrees(np.arccos(cosang))), float(self.obj.qr.side * self.focal_px / dist * max(cosang, 0)),
                          blocked)

    def evaluate(self, T_obj_tcp: np.ndarray, width: float, rolls=(0.0,)) -> GraspReport:
        open_gap = min(width + 0.02, self.cfg.gripper.max_width)
        targets_obj = [show_pose(self.cfg, self.obj, r) for r in rolls]
        R_targets = [T[:3, :3] for T in targets_obj]
        twins = []
        for k, Tg in enumerate((T_obj_tcp, twin_of(T_obj_tcp))):
            T_grasp = self.T_start @ Tg
            T_pre = T_grasp @ make_T(np.eye(3), [0, 0, -self.pregrasp_dist])
            poses = [T_grasp, T_pre] + [To @ Tg for To in targets_obj]
            S = ik(np.stack(poses), self.q7)                               # (P, K, 4, 7)
            start = self._reach(S[0], open_gap, self.T_start, holding=False)
            pre = self._reach(S[1], open_gap, self.T_start, holding=False,
                              q_ref=start.q if start.ok else None)
            tcs = []
            if start.ok and pre.ok:                                        # no point otherwise
                margins = check_margin(self.obj, Tg, np.stack(R_targets), self.stab, self.target_load_factor)[0]
                for j, (r, To) in enumerate(zip(rolls, targets_obj)):
                    reach = self._reach(S[2 + j], width, To, holding=True, q_ref=start.q)
                    vis = self.visibility(reach.q, width, To) if reach.ok else None
                    tcs.append(TargetCheck(r, reach, float(margins[j]), vis))
            twins.append(TwinReport(k, start, pre, tcs))
        return GraspReport(twins)
