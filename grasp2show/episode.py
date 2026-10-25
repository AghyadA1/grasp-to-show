"""The scripted pick -> move -> show episode, and its ground-truth evaluation (check 6).

Every segment is a straight Cartesian line for the TCP, solved by IK at evenly spaced
points and timed with a single min-jerk profile. (A plain joint-space interpolation
between far-apart configurations can sweep the hand through the table: the first
version of this script did exactly that.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import SceneConfig
from .ik import IKResult, PandaIK
from .motion import cartesian_line, path_trajectory
from .qr import decode_qr
from .scene import HOME_Q, PANDA_BODIES, reset_to_home
from .task import Grasp, body_pose, flipped, site_pose, start_pose
from .transforms import inv_T, rotation_angle


@dataclass
class EpisodeResult:
    grasp: str
    ik_ok: bool
    decoded: str | None
    qr_ok: bool
    held: bool
    slip_mm: float
    slip_deg: float
    success: bool
    collisions: list[str] = field(default_factory=list)  # unexpected contacts, first time each
    notes: list[str] = field(default_factory=list)
    scanner_rgb: np.ndarray | None = field(default=None, repr=False)

    def summary(self) -> str:
        verdict = "SUCCESS" if self.success else "FAIL"
        qr = f"decoded '{self.decoded}'" if self.qr_ok else "QR NOT decoded"
        s = (f"[{verdict}] grasp={self.grasp}: {qr}, held={self.held}, "
             f"slip={self.slip_mm:.1f} mm / {self.slip_deg:.1f} deg")
        if self.collisions:
            s += f", collisions: {', '.join(self.collisions)}"
        if self.notes:
            s += f" ({'; '.join(self.notes)})"
        return s


class PickAndShow:
    """home -> pre-grasp -> grasp -> close -> lift -> transit -> show -> hold."""

    # (segment name, number of IK points, duration in s)
    SEGMENTS = (("approach", 25, 2.0), ("descend", 10, 1.2), ("lift", 10, 1.2),
                ("transit", 30, 2.5), ("present", 8, 1.0))

    def __init__(self, model: mujoco.MjModel, cfg: SceneConfig, ik: PandaIK | None = None,
                 slip_tol_mm: float = 5.0, slip_tol_deg: float = 5.0):
        self.model, self.cfg = model, cfg
        self.ik = ik or PandaIK(cfg)
        self.data = mujoco.MjData(model)
        self.slip_tol_mm, self.slip_tol_deg = slip_tol_mm, slip_tol_deg
        self.renderer = mujoco.Renderer(model, cfg.camera.height, cfg.camera.width)
        self._robot_bodies = {model.body(b).id for b in PANDA_BODIES}
        self._fingers = {model.body("left_finger").id, model.body("right_finger").id}
        self._obj = model.body(cfg.obj.name).id

    # -- planning ---------------------------------------------------------------------
    def key_poses(self, grasp: Grasp, T_target_obj: np.ndarray) -> dict[str, np.ndarray]:
        """TCP poses at the segment boundaries."""
        reset_to_home(self.model, self.data)
        T_home = site_pose(self.data, "tcp")
        T_grasp = start_pose(self.cfg) @ grasp.T_obj_tcp
        T_pre = T_grasp.copy()
        T_pre[:3, 3] -= 0.10 * T_grasp[:3, 2]              # 10 cm back along the approach
        T_lift = T_grasp.copy()
        T_lift[:3, 3] += [0, 0, 0.15]
        T_show = T_target_obj @ grasp.T_obj_tcp
        away = T_show[:3, 3] - np.asarray(self.cfg.camera.pos)
        T_preshow = T_show.copy()
        T_preshow[:3, 3] += 0.08 * away / np.linalg.norm(away)   # 8 cm short of the target
        return {"home": T_home, "pre": T_pre, "grasp": T_grasp, "lift": T_lift,
                "preshow": T_preshow, "show": T_show}

    def _plan_one(self, grasp: Grasp, T_target_obj: np.ndarray):
        P = self.key_poses(grasp, T_target_obj)
        order = ["home", "pre", "grasp", "lift", "preshow", "show"]
        q, paths, results = HOME_Q.copy(), {}, []
        for (name, n, _), a, b in zip(self.SEGMENTS, order[:-1], order[1:]):
            rs: list[IKResult] = self.ik.solve_path(cartesian_line(P[a], P[b], n), q)
            paths[name] = [q] + [r.q for r in rs]
            results += rs
            q = rs[-1].q
        n_failed = sum(not r.success for r in results)
        margin = min(r.limit_margin for r in results)
        return paths, n_failed, margin

    def plan(self, grasp: Grasp, T_target_obj: np.ndarray):
        """Plan the grasp and its 180-degree twin; keep the one that tracks best.

        Ranking: fewest IK failures along the path, then the largest joint-limit margin.
        """
        best = None
        for g in (grasp, flipped(grasp)):
            paths, n_failed, margin = self._plan_one(g, T_target_obj)
            key = (n_failed, -margin)
            if best is None or key < best[0]:
                best = (key, g, paths, n_failed, margin)
        _, g, paths, n_failed, margin = best
        note = f"{n_failed} IK points failed" if n_failed else None
        return g, paths, note, margin

    # -- execution ----------------------------------------------------------------------
    def _unexpected_contacts(self, holding: bool) -> set[str]:
        """Contacts that should never happen: robot-table, and (once lifted) object-anything
        except the two fingers. Their presence invalidates the episode."""
        m, d, out = self.model, self.data, set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            pair = {b1, b2}
            if self._obj in pair:
                other = (pair - {self._obj}) or {self._obj}
                o = other.pop()
                if holding and o not in self._fingers or not holding and o in self._robot_bodies - self._fingers:
                    out.add(f"object-{m.body(o).name}")
            elif pair & self._robot_bodies and 0 in pair:   # body 0 = world (table, floor)
                out.add(f"{m.body((pair - {0}).pop()).name}-table")
        return out

    def run(self, grasp: Grasp, T_target_obj: np.ndarray, on_frame=None, fps: int = 30) -> EpisodeResult:
        m, d, dt = self.model, self.data, self.model.opt.timestep
        grasp, paths, ik_note, _ = self.plan(grasp, T_target_obj)
        reset_to_home(m, d)
        g_open = min(grasp.width + 0.025, self.cfg.gripper.max_width)
        render_every = max(round(1.0 / (fps * dt)), 1)
        state = {"step": 0, "holding": False, "collisions": []}

        def follow(q_traj, width):
            for q in q_traj:
                d.ctrl[:7], d.ctrl[7] = q, width
                mujoco.mj_step(m, d)
                state["step"] += 1
                for c in sorted(self._unexpected_contacts(state["holding"])):
                    if c not in state["collisions"]:
                        state["collisions"].append(c)
                if on_frame is not None and state["step"] % render_every == 0:
                    on_frame(d)

        def hold(q, width, T):
            follow(np.repeat(np.asarray(q)[None], max(int(T / dt), 1), axis=0), width)

        seg = {name: T for name, _, T in self.SEGMENTS}
        hold(HOME_Q, g_open, 0.3)
        follow(path_trajectory(paths["approach"], seg["approach"], dt), g_open)
        follow(path_trajectory(paths["descend"], seg["descend"], dt), g_open)
        hold(paths["descend"][-1], 0.0, 0.6)          # close: force-limited squeeze
        follow(path_trajectory(paths["lift"], seg["lift"], dt), 0.0)
        hold(paths["lift"][-1], 0.0, 0.3)
        state["holding"] = True                       # from here only finger contacts allowed
        T_tcp_obj_ref = inv_T(site_pose(d, "tcp")) @ body_pose(m, d, self.cfg.obj.name)
        follow(path_trajectory(paths["transit"], seg["transit"], dt), 0.0)
        follow(path_trajectory(paths["present"], seg["present"], dt), 0.0)
        hold(paths["present"][-1], 0.0, 1.0)

        # -- ground truth: slip in the hand, still held, QR decoded by the scanner ---------
        T_tcp_obj = inv_T(site_pose(d, "tcp")) @ body_pose(m, d, self.cfg.obj.name)
        dT = inv_T(T_tcp_obj_ref) @ T_tcp_obj
        slip_mm = float(np.linalg.norm(dT[:3, 3]) * 1e3)
        slip_deg = float(np.degrees(rotation_angle(dT[:3, :3])))
        finger_contacts = {
            m.geom_bodyid[c.geom1] ^ m.geom_bodyid[c.geom2] ^ self._obj
            for c in d.contact[: d.ncon]
            if self._obj in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
        } & self._fingers
        held = len(finger_contacts) == 2
        self.renderer.update_scene(d, camera="scanner")
        rgb = self.renderer.render().copy()
        text, _ = decode_qr(rgb)
        qr_ok = text == self.cfg.obj.qr_payload
        ok = (qr_ok and held and slip_mm < self.slip_tol_mm and slip_deg < self.slip_tol_deg
              and not state["collisions"])
        return EpisodeResult(grasp.name, ik_note is None, text, qr_ok, held, slip_mm, slip_deg,
                             ok, state["collisions"], [ik_note] if ik_note else [], rgb)
