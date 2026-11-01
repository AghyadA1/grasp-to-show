"""Part 3: carrying the grasped object so it is not lost - the problems, the motion, the simulator
rollouts and their score, for learning the carry (learn/carry_net.py, scripts/train_carry.py).

Where objects are lost today (scripts/diagnose_drops.py, 692 analytic pinches, 274 hooks): most
pinches are lost 0.3 s into the lift, as the table lets go; grasps the checks expected to hold
are also lost during the carry (8 % of them; way B's: 14 %), and a quarter of the hooks' failures
are drops during the carry. The motion between grasp and scanner is today's fixed planner:
straight up 15 cm in 1.2 s, then a joint-space move (min-jerk, its duration from the largest
joint step) and a straight line to the scanner.

A carry here:
  lift   pinch: the planner's straight-up path, its duration T_lift free (0.4-4 s); hook: the
         planner's tilt-then-lift, at the hook speed (hanging.HOOK_SPEED)
  settle a short hold (pinch 0.3 s; hook 1 s / HOOK_SPEED: the object comes to hang)
  carry  one smooth joint trajectory from the lift's end to the show pose (part 2's form,
         transport_opt): q(tau) = q0 + (q1 - q0) s(tau) + sum_k c_k phi_k(tau), duration T; every
         phi_k is zero with zero velocity and acceleration at both ends
  hold   at the scanner (pinch 1 s, hook 2 s)
so a carry is theta = (c (K x 7), log T, log T_lift).

A rollout runs the robot to the closed grip once per problem and pose error (PickAndShow, the
same simulator as the episodes) and saves the simulator's state; every candidate carry starts
from that state. Its score, in order of importance:
  1. the object is not lost (a pinch: never 20 mm / 15 deg of slip in the hand, both fingers on
     it at the end; a hook: it never falls, a finger still holds it): else the score is below 0
     (-1 + 0.5 x how much of the motion was done before it was lost);
  2. no unexpected contact (arm, hand or object against the table, robot or scanner), every joint
     within its limits, joint speed and acceleration within the Franka's: else as 1;
  3. then, for a success, 1 minus: 0.1 per second of motion (lift + carry), the in-hand slip
     beyond 5 mm (0.5 at 20 mm), extra jerk over the straight path, tracking error.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from .config import CameraConfig, ObjectConfig, SceneConfig
from .franka_ik import Q_MAX, Q_MIN
from .motion import path_trajectory
from .scene import HOME_Q, build_scene, reset_to_home
from .transforms import inv_T, rotation_angle
from .transport import QD_MAX, QDD_MAX
from .transport_opt import S_MINJERK, _basis

K = 4                         # path coefficients per joint (as part 2)
T_MIN, T_MAX = 0.4, 12.0      # carry duration (s)
TL_MIN, TL_MAX = 0.4, 4.0     # pinch lift duration (s)
SETTLE = {"pinch": 0.3, "hook": 1.0}
HOLD = {"pinch": 1.0, "hook": 2.0}
SLIP_TOL = (20.0, 15.0)       # mm, deg: lost (as PickAndShow.slip_tol)
CHECK_EVERY = 5               # contact and slip checks every this many steps (10 ms)


@dataclass
class CarryProblem:
    """One grasped object to carry from the table to the scanner (what the robot knows, plus the
    pose error it does not)."""
    asset: str
    kind: str                       # "pinch" or "hook"
    yaw_deg: float
    start_xy: tuple
    camera: CameraConfig
    mu: float
    mass_scale: float
    offset: tuple                   # the unknown pose error of the object (dx, dy, dyaw)
    T_obj_tcp: np.ndarray
    width: float
    open_width: float
    paths: dict                     # approach, descend, lift (joint waypoints, from the planner)
    q_show: np.ndarray              # the show pose (end of the carry)
    T_tcp_obj: np.ndarray           # the object in the hand while carried (a hook: hanging)
    baseline: dict = field(default_factory=dict)   # the planner's transit and present (for comparison)
    split: str = "train"

    @property
    def q0(self) -> np.ndarray:
        return np.asarray(self.paths["lift"][-1], float)


def straight_T(q0: np.ndarray, q1: np.ndarray) -> float:
    """The planner's transit time for this joint move (episode: 1.7 x the largest joint step, at
    least 2.5 s) plus its 1 s presentation: the duration the network's T starts from."""
    return max(2.5, 1.7 * float(np.abs(np.asarray(q1) - q0).max())) + 1.0


def carry_trajectory(q0, q1, c: np.ndarray, T: float, dt: float) -> np.ndarray:
    """Joint positions every dt (the first sample after q0, the last at q1)."""
    n = max(round(T / dt), 1)
    tau = np.arange(1, n + 1) / n
    B = np.stack([p(tau) for p in _basis(K)])
    return np.asarray(q0) + np.outer(S_MINJERK(tau), np.asarray(q1) - q0) + B.T @ c


def motion_limits(q0, q1, c: np.ndarray, T: float, n: int = 80) -> dict:
    """The commanded carry against the robot: largest joint-speed and -acceleration ratios to the
    Franka's limits, worst joint-limit excess (rad, > 0: outside), and the extra jerk over the
    straight path (duration-free, relative)."""
    tau = np.linspace(0, 1, n)
    polys = _basis(K)
    dq = np.asarray(q1) - q0
    d = []
    for k in range(4):
        B = np.stack([p.deriv(k)(tau) if k else p(tau) for p in polys])
        s = S_MINJERK.deriv(k)(tau) if k else S_MINJERK(tau)
        d.append((np.outer(s, dq) + B.T @ c + (q0 if k == 0 else 0.0)) / T**k)
    jerk = lambda cc: float(np.mean(np.sum((np.outer(S_MINJERK.deriv(3)(tau), dq) + np.stack(
        [p.deriv(3)(tau) for p in polys]).T @ cc) ** 2, 1)))
    return {"qd": float(np.max(np.abs(d[1]) / QD_MAX)), "qdd": float(np.max(np.abs(d[2]) / QDD_MAX)),
            "joint_limit": float(np.max(np.maximum(Q_MIN - d[0], d[0] - Q_MAX))),
            "jerk_rel": jerk(c) / max(jerk(np.zeros_like(c)), 1e-9) - 1.0}


class CarrySim:
    """The full robot and object of one problem in MuJoCo; the grip is run once per pose error and
    every carry starts from the saved state."""

    def __init__(self, p: CarryProblem):
        from .episode import PickAndShow
        from .objects import object_from_config

        self.p = p
        extra = {"friction": (float(p.mu), 0.01 * float(p.mu), 0.0001)}
        cfg = SceneConfig(obj=ObjectConfig(asset=p.asset, start_yaw=np.radians(p.yaw_deg),
                                           start_xy=tuple(p.start_xy), **extra), camera=p.camera)
        obj = object_from_config(cfg.obj)
        if p.mass_scale != 1.0:
            obj = replace(obj, mass=obj.mass * p.mass_scale, inertia=obj.inertia * p.mass_scale)
        self.cfg, self.obj = cfg, obj
        self.ep = PickAndShow(build_scene(cfg, obj), cfg)
        self.m, self.d = self.ep.model, self.ep.data
        self.dt = self.m.opt.timestep
        self.hook = p.kind == "hook"
        self._states: dict = {}

    def _step(self, q, width) -> None:
        self.d.ctrl[:7], self.d.ctrl[7] = q, width
        mujoco.mj_step(self.m, self.d)

    def _in_hand(self) -> np.ndarray:
        from .task import body_pose, site_pose
        return inv_T(site_pose(self.d, "tcp")) @ body_pose(self.m, self.d, self.obj.name)

    def grip(self, offset) -> tuple:
        """Home -> approach -> descend -> close, with the object displaced by offset; returns the
        saved state (cached per offset)."""
        key = tuple(np.round(offset, 6))
        if key in self._states:
            return self._states[key]
        m, d, dt, p, ep = self.m, self.d, self.dt, self.p, self.ep
        ep.set_friction(p.mu)
        reset_to_home(m, d)
        ep._displace_object(*offset)
        g_open = p.width if self.hook else min(p.width + 0.025, self.cfg.gripper.max_width)
        g_hold = p.width if self.hook else 0.0
        seg = {name: T for name, _, T in ep.SEGMENTS}
        dq = float(np.abs(np.diff(np.asarray(p.paths["approach"]), axis=0)).max())
        seg["approach"] = max(seg["approach"], 1.7 * dq)
        for _ in range(int(0.3 / dt)):
            self._step(HOME_Q, g_open)
        for q in path_trajectory(p.paths["approach"], seg["approach"], dt):
            self._step(q, g_open)
        if self.hook:
            for _ in range(int(0.5 / dt)):
                self._step(p.paths["descend"][0], g_open)
        for q in path_trajectory(p.paths["descend"], seg["descend"] * (2.5 if self.hook else 1.0), dt):
            self._step(q, g_open)
        for _ in range(int(0.6 / dt)):
            self._step(p.paths["descend"][-1], g_hold)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        st = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, st, spec)
        self._states[key] = (st, self._in_hand(), g_hold)
        return self._states[key]

    def rollout(self, offset, lift_T: float | None, carry: np.ndarray, hold: float | None = None) -> dict:
        """One carry from the saved grip: lift (pinch: lift_T s; hook: the planner's, at hook
        speed), settle, carry (joint positions every dt, ending at the show pose), hold. Returns
        what happened (see score)."""
        from .hanging import HOOK_SPEED

        m, d, dt, ep = self.m, self.d, self.dt, self.ep
        st, ref, g_hold = self.grip(offset)
        mujoco.mj_setState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(m, d)
        kind = "hook" if self.hook else "pinch"
        lift_T = (ep.SEGMENTS[2][2] / HOOK_SPEED) if self.hook else lift_T
        segs = [("lift", path_trajectory(self.p.paths["lift"], lift_T, dt)),
                ("settle", None), ("carry", carry), ("hold", None)]
        lengths = {"lift": None, "settle": SETTLE[kind] / (HOOK_SPEED if self.hook else 1.0),
                   "hold": HOLD[kind] if hold is None else hold}
        out = {"lost": None, "collisions": [], "slip_max": (0.0, 0.0), "track_err": 0.0, "progress": 1.0}
        total = len(segs[0][1]) + round(lengths["settle"] / dt) + len(carry) + round(lengths["hold"] / dt)
        step, holding, no_touch, ref2 = 0, False, 0, None
        q_last = segs[0][1][-1]
        for name, traj in segs:
            n = len(traj) if traj is not None else round(lengths[name] / dt)
            if name == "carry":
                holding = True
                ref2 = self._in_hand()
            for i in range(n):
                q = traj[i] if traj is not None else q_last
                self._step(q, g_hold)
                step += 1
                if self.hook:
                    touch = any(ep._obj in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) for c in d.contact[: d.ncon])
                    no_touch = 0 if touch else no_touch + 1
                    if no_touch * dt >= 0.05 and out["lost"] is None:
                        out["lost"], out["progress"] = name, step / total
                if step % CHECK_EVERY:
                    continue
                if name == "carry":
                    out["track_err"] = max(out["track_err"], float(np.abs(d.qpos[:7] - q).max()))
                if not self.hook:
                    dT = inv_T(ref) @ self._in_hand()
                    s = (float(np.linalg.norm(dT[:3, 3]) * 1e3), float(np.degrees(rotation_angle(dT[:3, :3]))))
                    out["slip_max"] = (max(out["slip_max"][0], s[0]), max(out["slip_max"][1], s[1]))
                    if out["lost"] is None and (s[0] >= SLIP_TOL[0] or s[1] >= SLIP_TOL[1]):
                        out["lost"], out["progress"] = name, step / total
                for c in ep._unexpected_contacts(holding):
                    if c not in out["collisions"]:
                        out["collisions"].append(c)
                        out.setdefault("collision_phase", name)
                        out.setdefault("collision_progress", step / total)
                if out["lost"] is not None and not self.hook:
                    break
            if traj is not None:
                q_last = traj[-1]
            if out["lost"] is not None and not self.hook:
                break
        fingers = {m.geom_bodyid[c.geom1] ^ m.geom_bodyid[c.geom2] ^ ep._obj for c in d.contact[: d.ncon]
                   if ep._obj in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])} & ep._fingers
        out["held"] = len(fingers) >= (1 if self.hook else 2)
        if ref2 is not None and not self.hook:
            dT = inv_T(ref2) @ self._in_hand()
            out["slip_end"] = (float(np.linalg.norm(dT[:3, 3]) * 1e3), float(np.degrees(rotation_angle(dT[:3, :3]))))
        out["seconds"] = (len(segs[0][1]) + len(carry)) * dt
        return out


def score(r: dict, lim: dict, jerk_w: float = 0.05) -> tuple[float, bool]:
    """(score, success) of one rollout (see the module docstring)."""
    collided = bool(r["collisions"])
    bad_limits = lim["joint_limit"] > 0 or lim["qd"] > 1.0 or lim["qdd"] > 1.0
    ok = r["lost"] is None and r["held"] and not collided and not bad_limits
    if not ok:
        prog = min(r["progress"], r.get("collision_progress", 1.0))
        if bad_limits:
            prog = 0.0
        return -1.0 + 0.5 * prog, False
    slip = max(r["slip_max"][0] - 5.0, 0.0) / 15.0 + max(r["slip_max"][1] - 5.0, 0.0) / 10.0
    s = 1.0 - 0.1 * r["seconds"] - 0.5 * min(slip, 1.0) - jerk_w * min(max(lim["jerk_rel"], 0.0), 10.0) \
        - 0.5 * min(max(r["track_err"] - 0.05, 0.0) / 0.05, 1.0)
    return float(s), True


def features(p: CarryProblem, obj) -> np.ndarray:
    """The carry network's inputs for one problem (N_IN): start and end joints and their
    difference, friction, mass, the object's centre of mass and inertia in the hand's frame (the
    leverage), the grip width, where the fingers touch it (in the hand's frame; zeros for a hook),
    pinch or hook."""
    from .stability import finger_contacts

    q0, q1 = p.q0, np.asarray(p.q_show, float)
    T = np.asarray(p.T_tcp_obj, float)                   # object in the hand's frame
    com = T[:3, :3] @ obj.com + T[:3, 3]
    I_t = T[:3, :3] @ obj.inertia @ T[:3, :3].T
    cs = np.zeros(6)
    if p.kind == "pinch":
        c1, c2, ok = finger_contacts(obj.mesh, np.asarray(p.T_obj_tcp)[None])
        if ok[0]:
            cs = np.concatenate([T[:3, :3] @ c1[0] + T[:3, 3], T[:3, :3] @ c2[0] + T[:3, 3]])
    return np.concatenate([q0, q1, q1 - q0, [p.mu, np.log(obj.mass)], com * 10, I_t[np.triu_indices(3)] * 1e3,
                           [p.width * 10], cs * 10, [float(p.kind == "hook")]]).astype(np.float32)


N_IN = 7 * 3 + 2 + 3 + 6 + 1 + 6 + 1
N_THETA = K * 7 + 2           # c, log T (relative to straight_T), log T_lift (relative to 1.2 s)


def decode(p: CarryProblem, theta: np.ndarray) -> tuple[np.ndarray, float, float]:
    """theta (N_THETA,) -> (c (K, 7), T, T_lift). T is relative to the planner's duration for the
    straight move (a hook's at the hook speed); T_lift to the planner's 1.2 s (pinch only)."""
    from .hanging import HOOK_SPEED

    c = np.asarray(theta[: K * 7], float).reshape(K, 7)
    T0 = straight_T(p.q0, p.q_show) / (HOOK_SPEED if p.kind == "hook" else 1.0)
    T = float(np.clip(T0 * np.exp(theta[K * 7]), T_MIN, T_MAX))
    TL = float(np.clip(1.2 * np.exp(theta[K * 7 + 1]), TL_MIN, TL_MAX))
    return c, T, TL


def evaluate_theta(sim: CarrySim, theta: np.ndarray, offsets: list) -> dict:
    """Score of one carry, run with each pose error: the mean score, success with all of them."""
    p = sim.p
    c, T, TL = decode(p, theta)
    lim = motion_limits(p.q0, p.q_show, c, T)
    traj = carry_trajectory(p.q0, p.q_show, c, T, sim.dt)
    runs = []
    for off in offsets:
        if lim["joint_limit"] > 0.05:                  # far outside the joints: not worth simulating
            r = {"lost": None, "held": False, "collisions": [], "progress": 0.0, "slip_max": (0, 0),
                 "track_err": 0.0, "seconds": T + TL}
        else:
            r = sim.rollout(off, TL, traj)
        runs.append((r, *score(r, lim)))
    return {"score": float(np.mean([x[1] for x in runs])), "success": all(x[2] for x in runs),
            "T": T, "T_lift": TL, "lost": [x[0]["lost"] for x in runs],
            "collisions": [x[0]["collisions"][:2] for x in runs], "limits": lim}


def evaluate_baseline(sim: CarrySim, offsets: list) -> dict:
    """The planner's own carry (lift 1.2 s, transit and present as planned), same score."""
    p = sim.p
    ep = sim.ep
    from .hanging import HOOK_SPEED

    speed = HOOK_SPEED if p.kind == "hook" else 1.0
    tr = np.asarray(p.baseline["transit"])
    T_tr = max(ep.SEGMENTS[3][2], 1.7 * float(np.abs(np.diff(tr, axis=0)).max())) / speed
    traj = np.vstack([path_trajectory(p.baseline["transit"], T_tr, sim.dt),
                      path_trajectory(p.baseline["present"], ep.SEGMENTS[4][2] / speed, sim.dt)])
    lim = {"qd": 0.0, "qdd": 0.0, "joint_limit": -1.0, "jerk_rel": 0.0}
    runs = []
    for off in offsets:
        r = sim.rollout(off, ep.SEGMENTS[2][2], traj)
        runs.append((r, *score(r, lim)))
    return {"score": float(np.mean([x[1] for x in runs])), "success": all(x[2] for x in runs),
            "T": len(traj) * sim.dt, "lost": [x[0]["lost"] for x in runs]}


def carry_paths(net, kind: str, paths: dict, T_obj_tcp: np.ndarray, width: float, T_tcp_obj: np.ndarray, mu: float,
                obj, dt: float) -> dict:
    """The planner's paths with the carry network's carry: its lift duration (a pinch) and one
    trajectory from the lift's end to the show pose (episode.PickAndShow.execute: "transport",
    already timed: not rescaled by the hook speed)."""
    p = CarryProblem(obj.name, kind, 0.0, (0.0, 0.0), None, mu, 1.0, (0.0, 0.0, 0.0), np.asarray(T_obj_tcp),
                     float(width), float(width), {k: paths[k] for k in ("approach", "descend", "lift")},
                     np.asarray(paths["present"][-1]), np.asarray(T_tcp_obj))
    theta, head, _ = net.best(features(p, obj))
    c, T, TL = decode(p, theta)
    out = dict(paths)
    out["transport"] = carry_trajectory(p.q0, p.q_show, c, T, dt)
    out["transport_timed"] = True
    out["carry_head"] = head
    if kind == "pinch":
        out["lift_T"] = TL
    return out
