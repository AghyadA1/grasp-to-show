"""The scripted pick -> move -> show episode, and its ground-truth evaluation (check 6).

Every segment is a straight Cartesian line for the TCP, solved by IK at evenly spaced
points and timed with a single min-jerk profile. (A plain joint-space interpolation
between far-apart configurations can sweep the hand through the table: the first
version of this script did exactly that.)
"""
from __future__ import annotations

import contextlib
import itertools
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

_ACTIVE: dict = {"token": None, "renderer": None}


def _acquire_renderer(model: mujoco.MjModel, height: int, width: int, token: object) -> mujoco.Renderer:
    """One live offscreen renderer per process.

    Freeing a MuJoCo renderer deletes its OpenGL objects in whichever context is current; when an
    old episode object was garbage-collected after a new one had created its renderer, that
    deleted the *new* renderer's textures and the scanner saw noise (OSMesa) or a wrong image.
    Here the previous renderer is closed explicitly, with its own context current, before a new
    one is created, and the renderer stays owned by this registry (not by the episode object).
    """
    if _ACTIVE["token"] is token:
        return _ACTIVE["renderer"]
    old = _ACTIVE["renderer"]
    if old is not None:
        with contextlib.suppress(Exception):   # best effort: the context may already be gone
            old._gl_context.make_current()
            old.close()
    _ACTIVE.update(token=token, renderer=mujoco.Renderer(model, height, width))
    return _ACTIVE["renderer"]


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
    transport: object | None = field(default=None, repr=False)   # transport.TransportLog

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
        self._render_token = object()
        grip = self._fingers_and_object(model, scene.obj.name)
        self._fric_geoms, self._fric_default = grip, model.geom_friction[grip].copy()
        self._robot_bodies = {model.body(b).id for b in PANDA_BODIES}
        self._fingers = {model.body("left_finger").id, model.body("right_finger").id}
        self._obj = model.body(scene.obj.name).id
        self._hand = model.body("hand").id

    @property
    def renderer(self) -> mujoco.Renderer:
        """This episode's offscreen renderer; at most one is alive per process (see _acquire)."""
        return _acquire_renderer(self.model, self.cfg.camera.height, self.cfg.camera.width, self._render_token)

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
                      q_target: np.ndarray, lift: float = 0.15, checker=None, width: float = 0.0,
                      T_show: np.ndarray | None = None, T_tcp_held: np.ndarray | None = None,
                      open_width: float | None = None, finger_floor: float | None = None,
                      T_lift: np.ndarray | None = None, T_start: np.ndarray | None = None):
        """Joint paths for one twin and one target, seeded with the configurations found by the
        feasibility checks (q_start at the grasp, q_target at the show pose).
        approach: joint-space home -> pre-grasp; descend, lift, present: tracked Cartesian lines;
        transit: joint-space lift -> pre-show. With a FeasibilityChecker, the joint-space segments
        are collision-checked (with the object held during transit); if the direct transit
        collides, it goes via a pose raised 25 cm higher. Returns (paths, None) or (None, reason).
        Hooks (grasp2show/hanging.py): T_show = the hand's show pose (instead of T_target_obj @
        T_obj_tcp), T_tcp_held = where the object hangs from the hand once lifted (for the
        collision checks), open_width = the finger gap before the grasp (the hook's own gap),
        finger_floor (hooks): the fingers must not point further down than this during the carry
        (lowest allowed vertical component of the TCP's z axis), or the handle slides off them. A
        joint-space transit that breaks it is replaced by a tracked Cartesian line that only turns
        the hand about the vertical (a hook's show pose keeps the carry's tilt). T_lift: the hand's
        pose at the end of the lift (default: 15 cm above the grasp, same orientation; a hook
        raises its fingers on the way up, hanging.carry_pose). T_start: where the object really lies
        (default: its nominal start pose; a hook needs the settled one, see resting_pose)."""
        T_start = start_pose(self.cfg, self.obj) if T_start is None else T_start
        T_grasp = T_start @ T_obj_tcp
        T_pre = T_grasp @ make_T(np.eye(3), [0, 0, -0.10])
        if T_lift is None:
            T_lift = T_grasp.copy()
            T_lift[:3, 3] += [0, 0, lift]
        if T_show is None:
            T_show = T_target_obj @ T_obj_tcp
        away = T_show[:3, 3] - np.asarray(self.cfg.camera.pos)
        T_preshow = T_show.copy()
        T_preshow[:3, 3] += 0.08 * away / np.linalg.norm(away)
        T_tcp_obj = inv_T(T_obj_tcp) if T_tcp_held is None else T_tcp_held
        if open_width is None:
            open_width = min(width + 0.02, self.cfg.gripper.max_width)

        def free(qa, qb, holding, n=30, T_held=None):
            if checker is None:
                return True
            gap = width if holding else open_width
            T_held = T_tcp_obj if T_held is None else T_held
            for s in np.linspace(0, 1, n):
                q = (1 - s) * qa + s * qb
                T_obj = fk(q) @ T_held if holding else T_start
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
                q_via = self._approach_via(T_pre, q_pre, free) if finger_floor is not None else None
                if q_via is None:
                    return None, "approach collides"
            approach = [HOME_Q.copy(), q_via, q_pre]
        descend = self.track(cartesian_line(T_pre, T_grasp, 15), q_pre)
        if descend is None:
            return None, "descent not trackable"
        if np.allclose(T_lift[:3, :3], T_grasp[:3, :3]):
            up = self.track(cartesian_line(T_grasp, T_lift, 20), descend[-1])
        else:      # straight up until the object hangs, then turn into the lift pose's orientation
            T_mid = T_grasp.copy()
            T_mid[:3, 3] += [0, 0, 0.5 * (T_lift[2, 3] - T_grasp[2, 3])]
            up = self.track(cartesian_line(T_grasp, T_mid, 10), descend[-1])
            rest = None if up is None else self.track(cartesian_line(T_mid, T_lift, 12), up[-1])
            up = None if rest is None else up + rest[1:]
        if up is None:                                       # joint-space lift if it stays clear
            q_lift = self.nearest_ik(T_lift, descend[-1])
            if q_lift is None or not free(descend[-1], q_lift, holding=True, T_held=inv_T(T_obj_tcp)):
                return None, "lift not trackable"
            up = [descend[-1], q_lift]
        q_preshow = self.nearest_ik(T_preshow, q_target)
        if q_preshow is None:
            return None, "pre-show unreachable"
        present = self.track(cartesian_line(T_preshow, T_show, 8), q_preshow)
        if present is None:
            return None, "presentation not trackable"
        if finger_floor is not None:
            return self._hook_transit(approach, descend, up, T_lift, T_preshow, T_show, q_preshow, free, finger_floor)
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

    def _approach_via(self, T_pre, q_pre, free):
        """A hook's pre-grasp is low and to the side of the handle, which a straight joint move from
        home often sweeps through the table or the object. Search one via configuration above or
        behind the pre-grasp (same hand orientation, any IK branch) that clears both legs."""
        vias = []
        for back in (0.0, 0.10, 0.20):
            for up in (0.10, 0.20, 0.30):
                T = T_pre @ make_T(np.eye(3), [0, 0, -back])
                T[:3, 3] += [0, 0, up]
                vias.append(T)
        V = ik(np.stack(vias), np.linspace(Q_MIN[6] + 0.02, Q_MAX[6] - 0.02, 16)).reshape(-1, 7)
        V = V[~np.isnan(V[:, 0])]
        for qv in V[np.argsort(np.abs(V - q_pre).max(axis=1))][:60]:
            if free(HOME_Q, qv, False) and free(qv, q_pre, False):
                return qv
        return None

    def _hook_transit(self, approach, descend, up, T_lift, T_preshow, T_show, q_preshow, free, finger_floor):
        """Carry for a hook: the fingers must not point further down than finger_floor (or the
        handle slides off them). Tried in order: a joint-space move to one of the pre-show IK
        solutions (nearest first) that keeps the fingers up and is collision-free, from the lift
        pose or from 25 cm higher; then a tracked Cartesian line that only turns the hand about
        the vertical."""
        def droops(qa, qb, n=30):
            return any(fk((1 - s) * qa + s * qb)[2, 2] < finger_floor for s in np.linspace(0, 1, n))

        S = ik(T_preshow[None], np.linspace(Q_MIN[6] + 0.02, Q_MAX[6] - 0.02, 32))[0].reshape(-1, 7)
        S = np.vstack([q_preshow[None], S[~np.isnan(S[:, 0])]])
        starts = [up]
        T_high = T_lift.copy()
        T_high[:3, 3] += [0, 0, 0.25]
        high = self.track(cartesian_line(T_lift, T_high, 15), up[-1])
        if high is not None:
            starts.append(up + high[1:])
        why = "transit droops or collides"
        for lift_path in starts:
            q0 = lift_path[-1]
            for qc in S[np.argsort(np.abs(S - q0).max(axis=1))][:12]:
                if droops(q0, qc) or not free(q0, qc, holding=True):
                    continue
                present = self.track(cartesian_line(T_preshow, T_show, 8), qc)
                if present is None:
                    why = "presentation not trackable"
                    continue
                return {"approach": approach, "descend": descend, "lift": lift_path,
                        "transit": [q0, qc], "present": present}, None
        # one via configuration: the hand above the middle of the carry, fingers level or up,
        # facing any way (a hanging object only needs the fingers not to point down)
        q0 = up[-1]
        mid = 0.5 * (fk(q0)[:3, 3] + T_preshow[:3, 3]) + [0, 0, 0.12]
        vias = []
        for yaw in np.radians(np.arange(0, 360, 30)):
            for pitch in np.arcsin(np.clip(finger_floor, -1, 1)) + np.radians([10.0, 30.0]):
                z = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
                for roll in np.radians(np.arange(0, 180, 45)):
                    x0 = np.cross([0.0, 0.0, 1.0], z)
                    x0 /= np.linalg.norm(x0)
                    x = np.cos(roll) * x0 + np.sin(roll) * np.cross(z, x0)
                    vias.append(make_T(np.column_stack([x, np.cross(z, x), z]), mid))
        V = ik(np.stack(vias), np.linspace(Q_MIN[6] + 0.02, Q_MAX[6] - 0.02, 8)).reshape(-1, 7)
        V = V[~np.isnan(V[:, 0])]
        for qv in V[np.argsort(np.abs(V - q0).max(axis=1) + np.abs(V - q_preshow).max(axis=1))][:150]:
            for qc in S[np.argsort(np.abs(S - qv).max(axis=1))][:3]:
                if droops(q0, qv) or droops(qv, qc) or not free(q0, qv, True) or not free(qv, qc, True):
                    continue
                present = self.track(cartesian_line(T_preshow, T_show, 8), qc)
                if present is not None:
                    return {"approach": approach, "descend": descend, "lift": up,
                            "transit": [q0, qv, qc], "present": present}, None
        transit = self.track(cartesian_line(fk(up[-1]), T_preshow, 40), up[-1], max_step=0.5)
        if transit is None or not all(free(a, b, holding=True, n=4) for a, b in itertools.pairwise(transit)):
            return None, why
        present = self.track(cartesian_line(T_preshow, T_show, 8), transit[-1])
        if present is None:
            return None, "presentation not trackable"
        return {"approach": approach, "descend": descend, "lift": up, "transit": transit, "present": present}, None

    def resting_pose(self, t: float = 0.6) -> np.ndarray:
        """The object's pose once it has settled on the table (the arm at home): the nominal start
        pose is 1 mm high and assumes a flat base, so a scanned object ends up a millimetre lower
        and a fraction of a degree tilted. What a perception system would report."""
        m, d = self.model, self.data
        reset_to_home(m, d)
        for _ in range(max(round(t / m.opt.timestep), 1)):
            d.ctrl[:7], d.ctrl[7] = HOME_Q, self.cfg.gripper.max_width
            mujoco.mj_step(m, d)
        return body_pose(m, d, self.obj.name)

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

    @staticmethod
    def _fingers_and_object(model: mujoco.MjModel, obj_name: str) -> np.ndarray:
        bodies = {model.body(b).id for b in ("left_finger", "right_finger", obj_name)}
        return np.array([g for g in range(model.ngeom)
                         if model.geom_bodyid[g] in bodies and model.geom_contype[g] | model.geom_conaffinity[g]])

    def set_friction(self, mu: float | None) -> None:
        """Friction between fingers and object (MuJoCo takes the max of the two geoms, so both
        are set). Torsional coefficient = 1 cm x mu (5 mm at mu = 0.5). None: the defaults."""
        m = self.model
        m.geom_friction[self._fric_geoms] = self._fric_default
        if mu is not None:
            m.geom_friction[self._fric_geoms, 0] = mu
            m.geom_friction[self._fric_geoms, 1] = 0.01 * mu

    def _displace_object(self, dx: float, dy: float, dyaw: float) -> None:
        """Shift the object on the table and turn it about the vertical through its origin."""
        m, d = self.model, self.data
        adr = m.jnt_qposadr[m.body_jntadr[self._obj]]
        d.qpos[adr:adr + 2] += (dx, dy)
        q = d.qpos[adr + 3:adr + 7].copy()
        rz = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
        mujoco.mju_mulQuat(d.qpos[adr + 3:adr + 7], rz, q)
        mujoco.mj_forward(m, d)

    def execute(self, grasp: Grasp, paths: dict, notes: list[str] | None = None, on_frame=None,
                fps: int = 30, obj_offset: tuple[float, float, float] | None = None, speed: float = 1.0,
                mu: float | None = None, log_every: int = 0) -> EpisodeResult:
        """Run joint paths (keys approach, descend, lift, transit, present) in the simulator and
        judge the outcome: QR decoded, object held without slipping, no unexpected contact.

        obj_offset = (dx, dy, dyaw) [m, m, rad]: the object actually lies this far from the pose
        the paths were planned for (pose-estimation error); the robot does not know.
        speed: the transport (lift, transit, present) runs `speed` times faster than planned.
        mu: finger-object friction for this episode (None: the default 0.5).
        paths["transport"] (optional): a joint trajectory sampled every timestep that replaces the
        transit and present segments (the optimised transport); it must end at paths["present"][-1].
        log_every > 0: record the transport every log_every steps (result.transport).
        """
        m, d, dt = self.model, self.data, self.model.opt.timestep
        notes = notes or []
        self.set_friction(mu)
        reset_to_home(m, d)
        if obj_offset is not None:
            self._displace_object(*obj_offset)
        hook = grasp.kind == "hook"
        # a hook keeps its finger gap all along (through the loop); a pinch opens, then squeezes (ctrl 0)
        g_open = grasp.width if hook else min(grasp.width + 0.025, self.cfg.gripper.max_width)
        g_hold = grasp.width if hook else 0.0
        render_every = max(round(1.0 / (fps * dt)), 1)
        state = {"step": 0, "holding": False, "collisions": [], "log": None}
        tcp_id, obj_id = m.site("tcp").id, self._obj

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
                if state["log"] is not None and state["step"] % log_every == 0:
                    tau = d.qfrc_actuator[:7] + d.qfrc_gravcomp[:7]
                    state["log"].append((d.qpos[:7].copy(), d.qvel[:7].copy(), d.qacc[:7].copy(), tau.copy(),
                                         make_T(d.site_xmat[tcp_id].reshape(3, 3), d.site_xpos[tcp_id]),
                                         make_T(d.xmat[obj_id].reshape(3, 3), d.xpos[obj_id])))

        def hold(q, width, T):
            follow(np.repeat(np.asarray(q)[None], max(int(T / dt), 1), axis=0), width)

        seg = {name: T for name, _, T in self.SEGMENTS}
        # Large joint-space moves get more time: peak joint speed ~ 1.875 * dq / T <= ~1.1 rad/s.
        for name in ("approach", "transit"):
            if name not in paths:
                continue
            dq = float(np.abs(np.diff(np.asarray(paths[name]), axis=0)).max())
            seg[name] = max(seg[name], 1.7 * dq)
        hold(HOME_Q, g_open, 0.3)
        follow(path_trajectory(paths["approach"], seg["approach"], dt), g_open)
        if hook:       # threading a fingertip through a handle with 1-3 mm to spare: settle, go slowly
            hold(paths["descend"][0], g_open, 0.5)
        follow(path_trajectory(paths["descend"], seg["descend"] * (2.5 if hook else 1.0), dt), g_open)
        hold(paths["descend"][-1], g_hold, 0.6)       # close: force-limited squeeze (a hook: stays)
        if log_every > 0:
            state["log"] = []
        follow(path_trajectory(paths["lift"], seg["lift"] / speed, dt), g_hold)
        hold(paths["lift"][-1], g_hold, (1.0 if hook else 0.3) / speed)   # a hanging object settles
        state["holding"] = True                       # from here only finger contacts allowed
        T_tcp_obj_ref = inv_T(site_pose(d, "tcp")) @ body_pose(m, d, self.obj.name)
        if "transport" in paths:      # one optimised trajectory (grasp2show/transport_opt.py), sampled every dt
            traj = np.asarray(paths["transport"])
            if speed != 1.0:
                t_src = np.arange(1, len(traj) + 1) * dt
                t_new = np.arange(1, max(round(len(traj) / speed), 1) + 1) * dt * speed
                traj = np.column_stack([np.interp(t_new, t_src, traj[:, j]) for j in range(traj.shape[1])])
            follow(traj, g_hold)
        else:
            follow(path_trajectory(paths["transit"], seg["transit"] / speed, dt), g_hold)
            follow(path_trajectory(paths["present"], seg["present"] / speed, dt), g_hold)
        log, state["log"] = state["log"], None
        hold(paths["present"][-1], g_hold, 2.0 if hook else 1.0)   # a hanging object's swing dies down

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
        held = len(finger_contacts) >= (1 if hook else 2)   # a one-finger hook touches one finger
        self.renderer.update_scene(d, camera="scanner")
        rgb = self.renderer.render().copy()
        # Guard: a broken offscreen GL context returns noise, which would silently count as an
        # unreadable QR. A real scanner image is smooth (mean neighbour difference ~2-3 levels).
        if np.abs(np.diff(rgb.astype(np.int16), axis=1)).mean() > 25:
            raise RuntimeError("scanner image is noise: the offscreen renderer is broken in this process")
        text, _ = decode_qr(rgb)
        qr_ok = text == self.cfg.obj.qr_payload
        # A hooked object hangs and swings: its motion relative to the hand is not a grip failing,
        # so a hook is judged by the scanner (and held, no collision); the slip is still reported.
        slipped = not hook and (slip_mm >= self.slip_tol_mm or slip_deg >= self.slip_tol_deg)
        ok = qr_ok and held and not slipped and not state["collisions"]
        transport = None
        if log:
            from .transport import TransportLog

            q, qd, qdd, tau, Tt, To = (np.array(x) for x in zip(*log))
            transport = TransportLog(log_every * dt, q, qd, qdd, tau, Tt, To)
        return EpisodeResult(grasp.name, not notes, text, qr_ok, held, slip_mm, slip_deg,
                             ok, state["collisions"], notes, rgb, transport)
