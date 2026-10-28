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

The fingers taper: 5 mm in front of the TCP the two closed fingertips are 1.8 x 3.1 cm (at the
base 2.1 x 5.3 cm), so only the tips go through, turned to match the slot (fit_hook_poses: the
hole is cut by a plane, the fingertip cross-section from the simulator's own finger geometry is
placed and turned for the most clearance; media/hook_fit.png). One finger alone (the other stays
outside, gripper open) fits even tighter handles.
"""
from __future__ import annotations

import mujoco
import numpy as np
import trimesh

from .gripper_sim import GripperSim
from .motion import min_jerk
from .transforms import make_T, rotation_angle

ROD_RADIUS = 0.008      # m: a candidate hole is at least this far from the object all around (the
                        #    exact fit of the fingertips is checked on the hole's cross-section)
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
              settle: float = 0.3, t_lift: float = 0.6, hold: float = 0.6, width: float = 0.0) -> dict:
    """Fingers through the loop (width 0: closed; > 0: held open at that gap, one finger through),
    object upright and free, gravity on: settle, lift the hand straight up, hold. Success: the
    object rose with the hand (at least 70 % of the lift) and still touches the gripper. Also
    returns how far it swung (deg)."""
    m, d, dt = sim.model, sim.data, sim.model.opt.timestep
    body = sim._obj_body
    mass0, inertia0 = m.body_mass[body], m.body_inertia[body].copy()
    m.body_mass[body], m.body_inertia[body] = mass0 * mass_scale, inertia0 * mass_scale
    try:
        T0 = T_obj_tcp
        sim._reset(np.eye(3), T0, width)
        _, deep = sim._contacts(min_depth=0.001)
        if deep:                                    # the fingers or the hand would cut through the object
            return {"success": False, "reason": "collision", "rise": 0.0, "swing_deg": np.nan}
        d.ctrl[0] = width
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



# -- placing the fingertips in a hole -------------------------------------------------------------
def finger_sections(z: float):
    """Cross-sections (shapely, TCP frame x-y, metres) of (both fingers closed, one finger alone)
    at height z along the approach axis, from the simulator's finger geometry (convex hulls)."""
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    from .config import ObjectConfig, SceneConfig
    from .objects import object_from_config

    sim = GripperSim(SceneConfig(), object_from_config(ObjectConfig()))
    m, d = sim.model, sim.data
    sim._reset(np.eye(3), np.eye(4), 0.0)
    mujoco.mj_forward(m, d)
    tcp = sim._tcp_pose()
    R, t = tcp[:3, :3], tcp[:3, 3]
    per = {}
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b not in sim._fingers or not m.geom_contype[g]:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        else:
            v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_size[g]
        P = ((d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g] - t) @ R
        sec = trimesh.convex.convex_hull(P).section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is not None:
            per.setdefault(int(b), []).append(MultiPoint(sec.vertices[:, :2]).convex_hull)
    left = m.body("g_left_finger").id
    fingers = {b: unary_union(v) for b, v in per.items()}
    return unary_union(list(fingers.values())), fingers[left]


def hole_section(obj, centre, axis, half=0.045):
    """The hole cut by the plane through centre perpendicular to axis, in that plane's (u, v)
    coordinates: (solid: the simulator's collision parts, hole: the free region around centre)."""
    from shapely.geometry import MultiPoint, Point
    from shapely.ops import unary_union

    u, v = _basis(axis)
    parts = []
    for f in obj.collision_files:
        sec = trimesh.load(f, force="mesh").convex_hull.section(plane_origin=centre, plane_normal=axis)
        if sec is not None and len(sec.vertices) >= 3:
            parts.append(MultiPoint(np.stack([(sec.vertices - centre) @ u, (sec.vertices - centre) @ v], 1)).convex_hull)
    solid = unary_union(parts)
    free = Point(0, 0).buffer(half).difference(solid)
    geoms = list(getattr(free, "geoms", [free]))
    hole = geoms[int(np.argmin([g.distance(Point(0, 0)) for g in geoms]))]
    return solid, hole


def best_fits(poly, hole, solid, step_deg: int = 10, span: float = 0.012, n: int = 13, k: int = 4,
              min_clearance: float = 0.0005):
    """Up to k fits of poly in the hole at distinct turns: for each turn theta (deg, a full turn:
    the hand is not symmetric), the shift (dx, dy) keeping the most clearance from the solid; the k
    turns with the most clearance, at least 30 deg apart. [(theta, dx, dy, clearance m)]."""
    from shapely import affinity

    c0 = poly.centroid
    per = []
    for a in range(0, 360, step_deg):
        pr = affinity.rotate(poly, a, origin=c0)
        best, key = None, -np.inf
        for dx in np.linspace(-span, span, n):
            for dy in np.linspace(-span, span, n):
                pt = affinity.translate(pr, dx - c0.x, dy - c0.y)
                if not pt.centroid.within(hole):
                    continue
                cl = pt.distance(solid) if not pt.intersects(solid) else -pt.intersection(solid).area ** 0.5
                if cl > key:
                    best, key = (a, dx, dy), cl
        if best is not None and key >= min_clearance:
            per.append((*best, key))
    per.sort(key=lambda f: -f[3])
    out = []
    for f in per:
        if all(min(abs(f[0] - g[0]), 360 - abs(f[0] - g[0])) >= 30 for g in out):
            out.append(f)
        if len(out) == k:
            break
    return out


def best_fit(poly, hole, solid, **kw):
    """The single best fit (see best_fits), or None."""
    fits = best_fits(poly, hole, solid, k=1, min_clearance=-np.inf, **kw)
    return fits[0] if fits else None


def _placed_pose(poly, fit, centre, axis, z_sec: float, tcp_in_poly=(0.0, 0.0)):
    """T_obj_tcp for a fit of a fingertip section cut at height z_sec: the point tcp_in_poly of the
    section (TCP frame) maps to the TCP, the section plane is z_sec in front of it."""
    a, dx, dy, _ = fit
    c0 = poly.centroid
    th = np.radians(a)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q = rot @ (np.asarray(tcp_in_poly) - [c0.x, c0.y]) + [c0.x, c0.y] + [dx - c0.x, dy - c0.y]
    u, v = _basis(axis)
    x = rot[0, 0] * u + rot[1, 0] * v
    y = rot[0, 1] * u + rot[1, 1] * v
    origin = centre + q[0] * u + q[1] * v - z_sec * axis
    return make_T(np.column_stack([x, y, np.cross(x, y)]), origin)


def fit_hook_poses(obj, loops, z_secs=(0.005, -0.004), open_width: float = 0.08, max_holes: int = 2,
                   turns: int = 4):
    """Hook poses that fit, per distinct hole: both fingers closed through it, and one finger
    through it with the gripper open (the other outside), from both sides, at a few depths and
    several turns (up to `turns` per hole, side, depth and kind; the palm, wide and behind the
    fingertips, must also clear the object: the simulated test decides; and, on the table, the
    table: more turns give the arm more ways to reach a handle that sits low). Each: dict(T, width,
    kind, depth, clearance)."""
    if not loops:
        return []
    P = np.array([lp[0] for lp in loops])
    holes, used = [], np.zeros(len(P), bool)
    for i in np.argsort([-lp[2] for lp in loops]):          # cluster: one hole per 2 cm
        if used[i]:
            continue
        near = np.linalg.norm(P - P[i], axis=1) < 0.02
        used |= near
        holes.append((P[near].mean(0), loops[i][1]))
        if len(holes) == max_holes:
            break
    secs = {z: finger_sections(z) for z in z_secs}
    out = []
    for centre, d in holes:
        for axis in (d, -d):
            solid, hole = hole_section(obj, centre, axis)
            for z, (two, one) in secs.items():
                for kind, poly, w, tcp_pt in (("two", two, 0.0, (0.0, 0.0)),
                                              ("one", one, open_width, (0.0, -open_width / 2))):
                    for fit in best_fits(poly, hole, solid, k=turns):
                        out.append({"T": _placed_pose(poly, fit, centre, axis, z, tcp_pt), "width": w, "kind": kind,
                                    "depth": z, "clearance": fit[3]})
    return out


def lift_test(sim: GripperSim, T_obj_tcp: np.ndarray, width: float, close: bool, mass_scale: float = 1.0,
              lift: float = 0.10, frames: list | None = None, renderer=None, camera=None) -> dict:
    """One lift for both kinds of grasp, so they compare fairly: a pinch (close=True: fingers open
    at width + 2 cm, then closed on the object) or a hook (close=False: fingers kept at `width`,
    0 = closed, through the loop). Gravity on, lift the hand 10 cm, hold. Success: a pinch rose
    with the hand (70 % of the lift) and is held; a hook hangs on the gripper, not falling.
    frames: if given, a rendered image is appended every 50 ms."""
    m, d, dt = sim.model, sim.data, sim.model.opt.timestep
    body = sim._obj_body
    mass0, inertia0 = m.body_mass[body], m.body_inertia[body].copy()
    m.body_mass[body], m.body_inertia[body] = mass0 * mass_scale, inertia0 * mass_scale
    every = max(round(0.05 / dt), 1)
    step = [0]

    def advance():
        mujoco.mj_step(m, d)
        step[0] += 1
        if frames is not None and step[0] % every == 0:
            renderer.update_scene(d, camera=camera)
            frames.append(renderer.render().copy())

    try:
        open_w = min(width + 0.02, sim.cfg.gripper.max_width) if close else width
        sim._reset(np.eye(3), T_obj_tcp, open_w)
        _, deep = sim._contacts(min_depth=0.001)
        if deep:
            return {"success": False, "reason": "collision", "rise": 0.0, "swing_deg": np.nan}
        z0, R0 = sim._obj_pose()[2, 3], sim._obj_pose()[:3, :3]
        d.ctrl[0] = 0.0 if close else width
        for _ in range(max(round(sim.T_CLOSE / dt), 1)):
            advance()
        m.opt.gravity[:] = [0, 0, -9.81]
        for _ in range(max(round(0.3 / dt), 1)):
            advance()
        n = max(round(0.6 / dt), 1)
        for k in range(1, n + 1):
            T = T_obj_tcp.copy()
            T[:3, 3] += [0, 0, lift * min_jerk(k / n)]
            sim._set_hand(T)
            advance()
        for _ in range(max(round(0.8 / dt), 1)):
            advance()
        touching, _ = sim._contacts()
        Tn = sim._obj_pose()
        rise = float(Tn[2, 3] - z0)
        if close:
            ok = bool(rise > 0.7 * lift and touching)
        else:
            # A hooked object turns until its centre of mass is under the finger, so its frame can
            # end up lower than at the start (a tall jug tips by 75 deg) although it hangs fine;
            # what matters is that it hangs on the gripper and is not falling (no table here: a
            # dropped object is metres down by now).
            com = np.asarray(sim.obj.com)
            com_rise = float((Tn[:3, :3] @ com)[2] - (R0 @ com)[2]) + rise
            ok = bool(touching and com_rise > -0.05)
        return {"success": ok, "reason": "" if ok else "fell", "rise": rise,
                "swing_deg": float(np.degrees(rotation_angle(R0.T @ Tn[:3, :3])))}
    finally:
        m.body_mass[body], m.body_inertia[body] = mass0, inertia0
        m.opt.gravity[:] = 0.0
