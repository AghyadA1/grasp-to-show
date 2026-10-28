"""Showing a hooked object (part 3): where it hangs, and how to turn the hand so the QR faces the scanner.

A pinched object moves with the hand, so the planner picks the object's show pose and the hand
follows (task.show_pose). A hooked object hangs: once lifted, gravity decides how it sits on the
fingers. The planner therefore works the other way round:

1. settle: the hook lift in the floating-gripper simulator, held until the swing has died down,
   gives the hanging pose relative to the hand, T_tcp_obj. It holds as long as the hand keeps its
   tilt: turning the hand about the vertical turns the hanging object with it and leaves T_tcp_obj
   unchanged (gravity is symmetric about the vertical).
2. target: the hand keeps the tilt it had at the grasp and turns about the vertical by the angle
   that points the QR's horizontal direction at the scanner. The tilt of the QR (up or down) is
   whatever the hanging gives; the visibility check tells whether the scanner can still read it.
   The QR's centre goes to the scanner's look-at point, the hand is placed from there.
3. checks: IK and collisions at the grasp, the pre-grasp (10 cm back along the hole) and the show
   pose with the object hanging, and the visibility of the QR, as for a pinch (feasibility.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .feasibility import FeasibilityChecker, PoseCheck, Visibility, twin_of
from .franka_ik import ik
from .gripper_sim import GripperSim
from .motion import cartesian_line, min_jerk
from .scene import start_pose
from .transforms import inv_T, make_T, rot_z, rotation_angle

PITCH_UP = 25.0   # deg: fingers raised this far above the horizontal while a hook carries (see carry_pose)


@dataclass
class Hang:
    T_tcp_obj: np.ndarray | None  # settled hanging pose relative to the hand (None: it fell)
    swing_deg: float               # remaining swing over the last 0.5 s (peak-to-peak, deg)
    tilt_deg: float                # how far the object turned from its upright start
    T_carry: np.ndarray | None = None   # the hand while carrying (object's upright start frame)


def carry_pose(T_obj_tcp: np.ndarray, lift: float, pitch_deg: float = PITCH_UP) -> np.ndarray:
    """The hand lifted `lift` above the grasp with its fingers raised to `pitch_deg` above the
    horizontal (turned about the horizontal axis across the fingers; if they already point
    higher, unchanged). The Franka's fingertips taper, so a hook only passes them a few
    millimetres through the handle: carried level, a swing slides the handle off the tip. Raised,
    the handle slides the other way, down the finger to where it thickens, and wedges there."""
    R = T_obj_tcp[:3, :3]
    z = R[:, 2]
    elev = np.arcsin(np.clip(z[2], -1, 1))
    T = T_obj_tcp.copy()
    T[:3, 3] += [0, 0, lift]
    h = np.array([z[0], z[1], 0.0])
    if np.linalg.norm(h) < 1e-6 or elev >= np.radians(pitch_deg):
        return T
    a = np.cross(h / np.linalg.norm(h), [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    th = np.radians(pitch_deg) - elev
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T[:3, :3] = (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K) @ R
    return T


def settle(sim: GripperSim, T_obj_tcp: np.ndarray, width: float, mass_scale: float = 1.0,
           lift: float = 0.15, hold: float = 2.5, pitch_deg: float = PITCH_UP) -> Hang:
    """The hook lift (hooks.lift_test) as the robot will do it: fingers at `width` through the
    loop, gravity on, the hand lifted 7.5 cm straight up (the object now hangs), then another 7.5 cm
    while its fingers turn up to the carry pose (carry_pose), then held until the swing has settled. The object starts upright (its frame =
    the world frame), the hand at T_obj_tcp."""
    m, d, dt = sim.model, sim.data, sim.model.opt.timestep
    body = sim._obj_body
    mass0, inertia0 = m.body_mass[body], m.body_inertia[body].copy()
    m.body_mass[body], m.body_inertia[body] = mass0 * mass_scale, inertia0 * mass_scale
    try:
        sim._reset(np.eye(3), T_obj_tcp, width)
        d.ctrl[0] = width
        for _ in range(max(round(sim.T_CLOSE / dt), 1)):
            mujoco.mj_step(m, d)
        m.opt.gravity[:] = [0, 0, -9.81]
        T_carry = carry_pose(T_obj_tcp, lift, pitch_deg)
        T_mid = T_obj_tcp.copy()
        T_mid[:3, 3] += [0, 0, lift / 2]          # straight up until it hangs, then raise the fingers
        path = cartesian_line(T_obj_tcp, T_mid, 25) + cartesian_line(T_mid, T_carry, 25)
        n = max(round(1.2 / dt), 1)
        for k in range(1, n + 1):
            sim._set_hand(path[min(int(min_jerk(k / n) * 50), 49)])
            mujoco.mj_step(m, d)
        n_hold, tail = max(round(hold / dt), 1), max(round(0.5 / dt), 1)
        Rs = []
        for k in range(n_hold):
            mujoco.mj_step(m, d)
            if k >= n_hold - tail and k % 10 == 0:
                Rs.append((inv_T(sim._tcp_pose()) @ sim._obj_pose())[:3, :3])
        touching, _ = sim._contacts()
        T_obj = sim._obj_pose()
        if not touching or T_obj[2, 3] < 0.5 * lift:
            return Hang(None, np.nan, np.nan)
        swing = max(np.degrees(rotation_angle(Rs[0].T @ R)) for R in Rs) * 2 if Rs else 0.0
        return Hang(inv_T(sim._tcp_pose()) @ T_obj, float(swing), float(np.degrees(rotation_angle(T_obj[:3, :3]))),
                    T_carry)
    finally:
        m.body_mass[body], m.body_inertia[body] = mass0, inertia0
        m.opt.gravity[:] = 0.0


def max_penetration(sim: GripperSim, T_obj_tcp: np.ndarray, width: float) -> float:
    """Deepest gripper-object penetration (m, >= 0) with the object upright and the hand at T_obj_tcp."""
    sim._reset(np.eye(3), T_obj_tcp, width)
    m, d = sim.model, sim.data
    deepest = 0.0
    for c in d.contact[: d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if sim._obj_body in (b1, b2) and (b2 if b1 == sim._obj_body else b1) in sim._gripper_bodies:
            deepest = max(deepest, -c.dist)
    return deepest


def nudge(sim: GripperSim, T_obj_tcp: np.ndarray, width: float, tol: float = 0.001,
          step: float = 0.001, reach: int = 2, depths=(-0.004, -0.002, 0.0, 0.002, 0.004)) -> np.ndarray | None:
    """The 2-D fit (hooks.best_fits) places the fingertip section in one cut through the hole; the
    real finger is 3-D and the simulator's handle is a set of convex pieces, a little fatter than
    the scan, so a fit can still dig 1-2 mm into the handle. Search small shifts (+-2 mm across the
    hole, a few mm along it) for the pose with the least penetration; None if none is within tol."""
    best, key = None, np.inf
    R = T_obj_tcp[:3, :3]
    for dz in depths:
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                T = T_obj_tcp.copy()
                T[:3, 3] += R @ np.array([dx * step, dy * step, dz])
                pen = max_penetration(sim, T, width)
                if pen == 0.0 and dx == dy == 0 and dz == 0.0:
                    return T                                  # the fit itself is clear
                if pen < key - 1e-5:
                    best, key = T, pen
    return best if key <= tol else None


def usable_hooks(obj, sim: GripperSim, turns: int = 12, limit: int = 40, seed: int = 0) -> list[dict]:
    """Hook grasps for an object, ready for the robot: the fitted fingertip placements
    (hooks.fit_hook_poses, `turns` turns per hole), nudged out of the handle (nudge), that lift the
    object in the floating-gripper test. dicts T, width, kind (as fit_hook_poses)."""
    from .hooks import find_loops, fit_hook_poses, lift_test

    out = []
    for h in fit_hook_poses(obj, find_loops(obj.mesh, np.random.default_rng(seed)), turns=turns):
        T = nudge(sim, h["T"], h["width"])
        if T is not None and lift_test(sim, T, h["width"], close=False)["success"]:
            out.append({**h, "T": T})
            if len(out) == limit:
                break
    return out


def show_targets(cfg, obj, R_hand0: np.ndarray, T_tcp_obj: np.ndarray,
                 extra_turns_deg=(0.0, -25.0, 25.0)) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Show poses for a hooked object: (extra turn, T_world_tcp, T_world_obj). The hand keeps the
    tilt it carries with (R_hand0, in the object's upright start frame) and turns about the
    vertical so the QR's horizontal direction points at the scanner (extra turns: the same, a
    little off, when the exact one is out of reach)."""
    cam, at = np.asarray(cfg.camera.pos, float), np.asarray(cfg.camera.look_at, float)
    R_obj0 = R_hand0 @ T_tcp_obj[:3, :3]
    n0 = R_obj0 @ obj.T_obj_qr[:3, 0]
    psi = np.arctan2(*(cam - at)[[1, 0]]) - np.arctan2(n0[1], n0[0])
    out = []
    for extra in extra_turns_deg:
        Rz = rot_z(psi + np.radians(extra))
        R_obj = Rz @ R_obj0
        T_world_obj = make_T(R_obj, at - R_obj @ obj.T_obj_qr[:3, 3])
        out.append((float(extra), T_world_obj @ inv_T(T_tcp_obj), T_world_obj))
    return out


@dataclass
class HookOption:
    """One hook grasp, one twin, one show pose, with its checks."""
    hook: int                    # index into the hook list
    twin: int
    T_obj_tcp: np.ndarray
    width: float
    hang: Hang
    extra_deg: float = np.nan
    T_show: np.ndarray | None = None
    T_world_obj_show: np.ndarray | None = None
    start: PoseCheck | None = None
    pre: PoseCheck | None = None
    show: PoseCheck | None = None
    vis: Visibility | None = None
    why: str = ""
    T_start: np.ndarray | None = None   # where the object lies (world)
    pitch_deg: float = PITCH_UP          # fingers above the horizontal while carrying

    @property
    def ok(self) -> bool:
        return not self.why and self.vis is not None and self.vis.ok()


def hook_options(fc: FeasibilityChecker, sim: GripperSim, hooks: list[dict], mass_scale: float = 1.0,
                 pregrasp: float = 0.10, T_start: np.ndarray | None = None,
                 pitches=(15.0, PITCH_UP, 40.0)) -> list[HookOption]:
    """Every hook (dicts T, width from hooks.fit_hook_poses), both twins: reach and collisions at
    the grasp; then for each carry pitch (how far the fingers are raised; it changes where the
    handle rests on the finger, so how the object hangs and which way its QR faces) where it
    hangs, the show pose and the reach / collision / visibility checks; the best pitch and turn
    are kept. T_start: where the object lies (a hook has 1-3 mm to spare, so pass the settled
    pose, PickAndShow.resting_pose; default: the nominal start pose). Sorted: feasible first, then
    by the QR's size in the scanner image."""
    cfg, obj = fc.cfg, fc.obj
    T_start = start_pose(cfg, obj) if T_start is None else T_start
    out = []
    for i, h in enumerate(hooks):
        for k, Tg in enumerate((h["T"], twin_of(h["T"]))):
            w = float(h["width"])
            T_grasp = T_start @ Tg
            T_pre = T_grasp @ make_T(np.eye(3), [0, 0, -pregrasp])
            S = ik(np.stack([T_grasp, T_pre]), fc.q7)
            start = fc._reach(S[0], w, T_start, holding=False)
            pre = fc._reach(S[1], w, T_start, holding=False, q_ref=start.q if start.ok else None)
            if not (start.ok and pre.ok):
                out.append(HookOption(i, k, Tg, w, Hang(None, np.nan, np.nan), start=start, pre=pre,
                                      why="grasp unreachable" if not start.ok else "pre-grasp unreachable"))
                continue
            best, hang = None, None
            for pitch in pitches:
                hang = settle(sim, Tg, w, mass_scale, pitch_deg=pitch)
                if hang.T_tcp_obj is None:
                    continue
                # show_targets works in the object's upright frame: the object's start yaw is a turn
                # about the vertical, which changes nothing in the hanging
                for extra, T_show, T_obj in show_targets(cfg, obj, hang.T_carry[:3, :3], hang.T_tcp_obj):
                    Ss = ik(T_show[None], fc.q7)[0]
                    reach = fc._reach(Ss, w, T_obj, holding=True, q_ref=start.q)
                    if not reach.ok:
                        continue
                    vis = fc.visibility(reach.q, w, T_obj)
                    opt = HookOption(i, k, Tg, w, hang, extra, T_show, T_obj, start, pre, reach, vis,
                                     T_start=T_start, pitch_deg=pitch)
                    if best is None or (vis.ok(), vis.pixels) > (best.vis.ok(), best.vis.pixels):
                        best = opt
            if best is None:
                why = "falls off" if hang is None or hang.T_tcp_obj is None else "show pose unreachable"
                best = HookOption(i, k, Tg, w, hang, start=start, pre=pre, why=why)
            out.append(best)
    return sorted(out, key=lambda o: (not o.ok, -(o.vis.pixels if o.vis else 0.0)))


HOOK_SPEED = 0.6   # hooks carry at 0.6 x the pinch speed: the object swings on a few mm of fingertip
SWING = 12.0       # deg: collision margin for the swing, about the finger axis


def _rot_axis(axis, a: float) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def swing_period(obj, T_tcp_obj: np.ndarray, g: float = 9.81) -> float:
    """Period (s) of the hanging object swinging about the finger (the TCP's z axis): a physical
    pendulum, T = 2 pi sqrt(I_axis / (m g r)), r = distance from the COM to the finger axis."""
    com = T_tcp_obj[:3, :3] @ np.asarray(obj.com) + T_tcp_obj[:3, 3]
    r = float(np.linalg.norm(com[:2]))
    k = T_tcp_obj[:3, :3].T @ np.array([0.0, 0.0, 1.0])            # the finger axis, object frame
    I_axis = float(k @ np.asarray(obj.inertia) @ k) + obj.mass * r ** 2
    return float(2 * np.pi * np.sqrt(I_axis / (obj.mass * g * max(r, 1e-3))))


def plan_hook(ep, fc: FeasibilityChecker, o: HookOption, shape: bool = False):
    """Joint paths for a hook option (episode.PickAndShow.plan_analytic): the lift ends in the carry
    pose (fingers raised), the carry keeps them at most 15 deg below that pitch (and not down), the
    held object hangs as found by settle() (and is checked swung by +-SWING deg). shape: the carry
    is input-shaped against the swing (episode: paths["shaper_delay"]); off by default: on 16
    scenes it changed neither the successes nor the object's deviation at the scanner (the swing
    has died down after the 2 s hold; what remains is a static offset of the hanging pose). Returns (paths, None) or
    (None, reason)."""
    # the object swings on the finger while carried: check it hanging as found and swung +-SWING deg
    swung = [make_T(_rot_axis([0.0, 0.0, 1.0], np.radians(a)), [0, 0, 0]) @ o.hang.T_tcp_obj for a in (-SWING, SWING)]
    paths, why = ep.plan_analytic(o.T_obj_tcp, None, o.start.q, o.show.q, checker=fc, width=o.width, T_show=o.T_show,
                                  T_tcp_held=[o.hang.T_tcp_obj, *swung], open_width=o.width,
                                  finger_floor=float(np.sin(np.radians(max(o.pitch_deg - 15.0, 0.0)))),
                                  T_lift=o.T_start @ o.hang.T_carry, T_start=o.T_start)
    if paths is not None and shape:
        paths["shaper_delay"] = 0.5 * swing_period(fc.obj, o.hang.T_tcp_obj)
    return paths, why
