"""Hook grasps (part 3): closed fingers through a handle, the object hanging on them.

A pinch holds an object with friction, which the grip force (2 x 20 N) limits. A hook holds it by
shape: the closed fingers pass through a loop (a mug's handle, a basket's grip) and the weight
presses the loop onto the fingers, so the load goes into the hand's structure. Friction and grip
force barely matter, and the object can be much heavier.

Finding a loop on a scanned mesh (not watertight, so no inside/outside test): a point is a
"through-hole" for a finger rod along direction d when, in the plane perpendicular to d, rays in
almost every direction hit the object within reach (it is surrounded), none of them closer than
the rod's radius (the closed fingers fit), and the rod's own line along +d and -d is free for a
finger length (the hole goes through: not the inside of a cup).

The simulated hook test is the ground truth: fingers closed through the loop, gravity on, lift the
hand 10 cm and hold; the object must come along (its height rises with the hand).

Finding (part 3): with the Franka Hand this does not work on the scanned mugs. The closed fingers
form a block of 2.1 x 5.3 cm (each finger link is about 2.6 cm deep), one finger alone 2.1 x 2.6
cm, while the mugs' handle holes are about 2 x 5 cm in the visual meshes and about 1.8 cm across
in the convex collision parts the simulator uses. Hooking needs thinner fingertips (a hook tool).
"""
from __future__ import annotations

import mujoco
import numpy as np
import trimesh

from .gripper_sim import GripperSim
from .motion import min_jerk
from .transforms import make_T, rotation_angle

ROD_RADIUS = 0.012      # m: half the thickness of the two closed fingers, plus clearance
REACH = 0.06            # m: the loop must be within this distance all around
FREE_ALONG = 1.0        # m: the rod's line must be free on both sides (the hole goes through: not a cup)


def _fibonacci_dirs(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def _basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def collision_mesh(obj) -> trimesh.Trimesh:
    """The object as the simulator sees it: the union of its convex collision parts (a
    decomposition can partly fill a handle's hole, so loops are searched on this, not on the
    visual mesh)."""
    if not obj.collision_files:
        return obj.mesh
    return trimesh.util.concatenate([trimesh.load(f, force="mesh") for f in obj.collision_files])


def find_loops(mesh: trimesh.Trimesh, rng: np.random.Generator, n_points: int = 1500, n_dirs: int = 24,
               n_ring: int = 16, min_ring_hits: int = 15) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Through-holes a closed-finger rod fits in: list of (point, axis d (unit), ring hits)."""
    lo, hi = mesh.bounds[0] - 0.01, mesh.bounds[1] + 0.01
    pts = rng.uniform(lo, hi, (n_points, 3))
    _, dist, _ = trimesh.proximity.closest_point(mesh, pts)
    pts = pts[(dist > ROD_RADIUS) & (dist < 0.04)]                 # near the surface, with room for the rod
    dirs = _fibonacci_dirs(2 * n_dirs)
    dirs = dirs[dirs[:, 2] >= 0]                                   # d and -d are the same rod
    ang = 2 * np.pi * np.arange(n_ring) / n_ring
    out = []
    for d in dirs:
        u, v = _basis(d)
        ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v                # (R, 3)
        # axial rays first (cheap rejection): the line of the rod must be free on both sides
        o = np.repeat(pts, 2, axis=0)
        ax = np.tile(np.stack([d, -d]), (len(pts), 1))
        loc, idx, _ = mesh.ray.intersects_location(o, ax, multiple_hits=False)
        blocked = np.zeros(len(o), bool)
        blocked[idx[np.linalg.norm(loc - o[idx], axis=1) < FREE_ALONG]] = True
        ok = ~blocked.reshape(-1, 2).any(1)
        cand = pts[ok]
        if len(cand) == 0:
            continue
        o = np.repeat(cand, n_ring, axis=0)
        r = np.tile(ring, (len(cand), 1))
        loc, idx, _ = mesh.ray.intersects_location(o, r, multiple_hits=False)
        dd = np.full(len(o), np.inf)
        dd[idx] = np.linalg.norm(loc - o[idx], axis=1)
        dd = dd.reshape(len(cand), n_ring)
        hits = (dd < REACH).sum(1)
        fits = dd.min(1) >= ROD_RADIUS
        for p, h in zip(cand[(hits >= min_ring_hits) & fits], hits[(hits >= min_ring_hits) & fits]):
            out.append((p, d.copy(), int(h)))
    return out


def hook_poses(loops, rng: np.random.Generator, n_roll: int = 8, max_loops: int = 20,
               depths: tuple[float, ...] = (-0.005, 0.005, 0.015)) -> np.ndarray:
    """TCP poses (N, 4, 4) in the object frame: approach along the loop's axis (both ways), the
    loop at a few depths along the fingers (TCP this far past the loop; the fingertips reach
    about 1 cm past the TCP: shallow keeps the palm away from the object, deep holds better),
    and several rolls about the axis (the palm is wide: some rolls clear the object's body)."""
    if not loops:
        return np.zeros((0, 4, 4))
    sel = rng.permutation(len(loops))[:max_loops]
    out = []
    for i in sel:
        p, d, _ = loops[i]
        for z in (d, -d):
            u, _ = _basis(z)
            for a in 2 * np.pi * np.arange(n_roll) / n_roll + rng.uniform(0, np.pi / n_roll):
                y = np.cos(a) * u + np.sin(a) * np.cross(z, u)
                for t in depths:
                    out.append(make_T(np.column_stack([np.cross(y, z), y, z]), p + t * z))
    return np.array(out)


def hook_test(sim: GripperSim, T_obj_tcp: np.ndarray, mass_scale: float = 1.0, lift: float = 0.10,
              settle: float = 0.3, t_lift: float = 0.6, hold: float = 0.6) -> dict:
    """Fingers closed (no object between them) through the loop, object upright and free, gravity
    on: settle, lift the hand straight up, hold. Success: the object rose with the hand (at least
    70 % of the lift) and still touches the gripper. Also returns how far it swung (deg)."""
    m, d, dt = sim.model, sim.data, sim.model.opt.timestep
    body = sim._obj_body
    mass0, inertia0 = m.body_mass[body], m.body_inertia[body].copy()
    m.body_mass[body], m.body_inertia[body] = mass0 * mass_scale, inertia0 * mass_scale
    try:
        T0 = T_obj_tcp
        sim._reset(np.eye(3), T0, 0.0)
        _, deep = sim._contacts(min_depth=0.001)
        if deep:                                    # the closed fingers would cut through the object
            return {"success": False, "reason": "collision", "rise": 0.0, "swing_deg": np.nan}
        d.ctrl[0] = 0.0
        z0 = sim._obj_pose()[2, 3]
        R0 = sim._obj_pose()[:3, :3]
        m.opt.gravity[:] = [0, 0, -9.81]
        for _ in range(max(round(settle / dt), 1)):
            mujoco.mj_step(m, d)
        n = max(round(t_lift / dt), 1)
        for k in range(1, n + 1):
            T = T0.copy()
            T[:3, 3] += [0, 0, lift * min_jerk(k / n)]
            sim._set_hand(T)
            mujoco.mj_step(m, d)
        for _ in range(max(round(hold / dt), 1)):
            mujoco.mj_step(m, d)
        touching, _ = sim._contacts()
        Tn = sim._obj_pose()
        rise = float(Tn[2, 3] - z0)
        ok = bool(rise > 0.7 * lift and touching)
        return {"success": ok, "reason": "" if ok else "fell", "rise": rise,
                "swing_deg": float(np.degrees(rotation_angle(R0.T @ Tn[:3, :3])))}
    finally:
        m.body_mass[body], m.body_inertia[body] = mass0, inertia0
        m.opt.gravity[:] = 0.0


def pinch_hold(sim: GripperSim, T_obj_tcp: np.ndarray, width: float, mass_scale: float = 1.0) -> bool:
    """The same lift for an ordinary pinch (gripper_sim.shake_test without the shake): close on
    the object, gravity on, lift 10 cm, hold; at mass x mass_scale."""
    from .gripper_sim import hold_test

    m = sim.model
    body = sim._obj_body
    mass0, inertia0 = m.body_mass[body], m.body_inertia[body].copy()
    m.body_mass[body], m.body_inertia[body] = mass0 * mass_scale, inertia0 * mass_scale
    try:
        held, _, _ = hold_test(sim, T_obj_tcp, width, np.eye(3), duration=1.2)
        return held
    finally:
        m.body_mass[body], m.body_inertia[body] = mass0, inertia0
        m.opt.gravity[:] = 0.0

