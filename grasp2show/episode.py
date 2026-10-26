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
from .franka_ik import Q_MAX, Q_MIN, fk, ik
from .ik import IKResult, PandaIK
from .motion import cartesian_line, path_trajectory
from .qr import decode_qr
from .scene import HOME_Q, PANDA_BODIES, Scene, reset_to_home, start_pose
from .task import Grasp, body_pose, flipped, site_pose
from .transforms import inv_T, make_T, rotation_angle


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

    def __init__(self, scene: Scene, cfg: SceneConfig, ik: PandaIK | None = None,
                 slip_tol_mm: float = 20.0, slip_tol_deg: float = 15.0):
        """slip_tol: gross slip, as in the lift test (20 mm). MuJoCo's soft contacts creep a few
        millimetres even for a 70 g box with 96 % friction reserve, so a tighter tolerance would
        measure the simulator, not the grasp. The exact slip is always reported."""
        model = scene.model
        self.model, self.cfg, self.obj = model, cfg, scene.obj
        self.ik = ik or PandaIK(cfg)
        self.data = mujoco.MjData(model)
        self.slip_tol_mm, self.slip_tol_deg = slip_tol_mm, slip_tol_deg
        self.renderer = mujoco.Renderer(model, cfg.camera.height, cfg.camera.width)
        self._robot_bodies = {model.body(b).id for b in PANDA_BODIES}
        self._fingers = {model.body("left_finger").id, model.body("right_finger").id}
        self._obj = model.body(scene.obj.name).id
        self._hand = model.body("hand").id

    # -- planning ---------------------------------------------------------------------
    def key_poses(self, grasp: Grasp, T_target_obj: np.ndarray) -> dict[str, np.ndarray]:
        """TCP poses at the segment boundaries."""
        reset_to_home(self.model, self.data)
        T_home = site_pose(self.data, "tcp")
        T_grasp = start_pose(self.cfg, self.obj) @ grasp.T_obj_tcp
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

    # -- analytic planning (for evaluating a chosen twin, target and configurations) --------
    @staticmethod
    def nearest_ik(T: np.ndarray, q_ref: np.ndarray, span: float = 1.0, n: int = 41) -> np.ndarray | None:
        """The IK solution closest to q_ref (max joint difference), q7 searched near q_ref[6]."""
        q7 = np.clip(q_ref[6] + np.linspace(-span, span, n), Q_MIN[6] + 1e-3, Q_MAX[6] - 1e-3)
        S = ik(T[None], q7)[0].reshape(-1, 7)
        S = S[~np.isnan(S[:, 0])]
        if len(S) == 0:
            return None
        return S[np.argmin(np.abs(S - q_ref).max(axis=1))]

    def track(self, poses: list[np.ndarray], q0: np.ndarray, max_step: float = 0.3) -> list[np.ndarray] | None:
        """Follow a Cartesian path, each point's IK nearest the previous configuration.
        None if a point has no solution or the arm would jump (branch switch)."""
        path, q = [q0], q0
        for T in poses:
            qn = self.nearest_ik(T, q)
            if qn is None or np.abs(qn - q).max() > max_step:
                return None
            path.append(qn)
            q = qn
        return path

    def plan_analytic(self, T_obj_tcp: np.ndarray, T_target_obj: np.ndarray, q_start: np.ndarray,
                      q_target: np.ndarray, lift: float = 0.15, checker=None, width: float = 0.0):
        """Joint paths for one twin and one target, seeded with the configurations found by the
        feasibility checks (q_start at the grasp, q_target at the show pose).
        approach: joint-space home -> pre-grasp; descend, lift, present: tracked Cartesian lines;
        transit: joint-space lift -> pre-show. With a FeasibilityChecker, the joint-space segments
        are collision-checked (with the object held during transit); if the direct transit
        collides, it goes via a pose raised 25 cm higher. Returns (paths, None) or (None, reason)."""
        T_start = start_pose(self.cfg, self.obj)
        T_grasp = T_start @ T_obj_tcp
        T_pre = T_grasp @ make_T(np.eye(3), [0, 0, -0.10])
        T_lift = T_grasp.copy()
        T_lift[:3, 3] += [0, 0, lift]
        T_show = T_target_obj @ T_obj_tcp
        away = T_show[:3, 3] - np.asarray(self.cfg.camera.pos)
        T_preshow = T_show.copy()
        T_preshow[:3, 3] += 0.08 * away / np.linalg.norm(away)
        T_tcp_obj = inv_T(T_obj_tcp)

        def free(qa, qb, holding, n=30):
            if checker is None:
                return True
            gap = width if holding else min(width + 0.02, self.cfg.gripper.max_width)
            for s in np.linspace(0, 1, n):
                q = (1 - s) * qa + s * qb
                T_obj = fk(q) @ T_tcp_obj if holding else T_start
                checker._set(q, gap, T_obj)
                if checker._bad_contacts(holding):
                    return False
            return True

        q_pre = self.nearest_ik(T_pre, q_start)
        if q_pre is None:
            return None, "pre-grasp unreachable"
        approach = [HOME_Q.copy(), q_pre]
        if not free(HOME_Q, q_pre, holding=False):          # detour via a point 20 cm higher
            T_via = T_pre.copy()
            T_via[:3, 3] += [0, 0, 0.20]
            q_via = self.nearest_ik(T_via, q_pre)
            if q_via is None or not free(HOME_Q, q_via, False) or not free(q_via, q_pre, False):
                return None, "approach collides"
            approach = [HOME_Q.copy(), q_via, q_pre]
        descend = self.track(cartesian_line(T_pre, T_grasp, 15), q_pre)
        if descend is None:
            return None, "descent not trackable"
        up = self.track(cartesian_line(T_grasp, T_lift, 20), descend[-1])
        if up is None:                                       # joint-space lift if it stays clear
            q_lift = self.nearest_ik(T_lift, descend[-1])
            if q_lift is None or not free(descend[-1], q_lift, holding=True):
                return None, "lift not trackable"
            up = [descend[-1], q_lift]
        q_preshow = self.nearest_ik(T_preshow, q_target)
        if q_preshow is None:
            return None, "pre-show unreachable"
        present = self.track(cartesian_line(T_preshow, T_show, 8), q_preshow)
        if present is None:
            return None, "presentation not trackable"
        transit = [up[-1], q_preshow]
        if not free(up[-1], q_preshow, holding=True):
            T_high = T_lift.copy()
            T_high[:3, 3] += [0, 0, 0.25]
            high = self.track(cartesian_line(T_lift, T_high, 15), up[-1])
            if high is None or not free(high[-1], q_preshow, holding=True):
                return None, "transit collides"
            up = up + high[1:]
            transit = [high[-1], q_preshow]
        return {"approach": approach, "descend": descend, "lift": up,
                "transit": transit, "present": present}, None

    # -- execution ----------------------------------------------------------------------
    def _unexpected_contacts(self, holding: bool) -> set[str]:
        """Contacts that should never happen: robot-table, and (once lifted) object-anything
        except the fingers and the palm. Their presence invalidates the episode."""
        m, d, out = self.model, self.data, set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            pair = {b1, b2}
            if self._obj in pair:
                other = (pair - {self._obj}) or {self._obj}
                o = other.pop()
                if o == self._hand:
                    continue                                 # touching the palm is harmless
                if holding and o not in self._fingers or not holding and o in self._robot_bodies - self._fingers:
                    out.add(f"object-{m.body(o).name}")
            elif pair & self._robot_bodies and 0 in pair:   # body 0 = world (table, floor)
                out.add(f"{m.body((pair - {0}).pop()).name}-table")
        return out

    def run(self, grasp: Grasp, T_target_obj: np.ndarray, on_frame=None, fps: int = 30) -> EpisodeResult:
        """Plan with mink (both twins, keep the better) and execute."""
        grasp, paths, ik_note, _ = self.plan(grasp, T_target_obj)
        return self.execute(grasp, paths, [ik_note] if ik_note else [], on_frame, fps)

    def execute(self, grasp: Grasp, paths: dict, notes: list[str] | None = None, on_frame=None,
                fps: int = 30) -> EpisodeResult:
        """Run joint paths (keys approach, descend, lift, transit, present) in the simulator and
        judge the outcome: QR decoded, object held without slipping, no unexpected contact."""
        m, d, dt = self.model, self.data, self.model.opt.timestep
        notes = notes or []
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
        # Large joint-space moves get more time: peak joint speed ~ 1.875 * dq / T <= ~1.1 rad/s.
        for name in ("approach", "transit"):
            dq = float(np.abs(np.diff(np.asarray(paths[name]), axis=0)).max())
            seg[name] = max(seg[name], 1.7 * dq)
        hold(HOME_Q, g_open, 0.3)
        follow(path_trajectory(paths["approach"], seg["approach"], dt), g_open)
        follow(path_trajectory(paths["descend"], seg["descend"], dt), g_open)
        hold(paths["descend"][-1], 0.0, 0.6)          # close: force-limited squeeze
        follow(path_trajectory(paths["lift"], seg["lift"], dt), 0.0)
        hold(paths["lift"][-1], 0.0, 0.3)
        state["holding"] = True                       # from here only finger contacts allowed
        T_tcp_obj_ref = inv_T(site_pose(d, "tcp")) @ body_pose(m, d, self.obj.name)
        follow(path_trajectory(paths["transit"], seg["transit"], dt), 0.0)
        follow(path_trajectory(paths["present"], seg["present"], dt), 0.0)
        hold(paths["present"][-1], 0.0, 1.0)

        # -- ground truth: slip in the hand, still held, QR decoded by the scanner ---------
        T_tcp_obj = inv_T(site_pose(d, "tcp")) @ body_pose(m, d, self.obj.name)
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
        return EpisodeResult(grasp.name, not notes, text, qr_ok, held, slip_mm, slip_deg,
                             ok, state["collisions"], notes, rgb)
