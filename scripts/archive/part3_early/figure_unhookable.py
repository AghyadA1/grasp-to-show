"""Pictures: the handled objects the robot cannot hook, each with the Franka hand at its best try.

For each handle candidate of object_set_v2 that has no usable hook from the table, the hook
pipeline is run step by step and the hand is drawn where it came closest:
  - no through-hole found          -> the hand (open) next to the object, for scale
  - holes found, no fingertip fit  -> the hand at the best fit (overlapping the handle by the
                                      printed amount)
  - fits in the floating test, but -> the hand at a fitting hook, with the table drawn: the palm
    the palm goes through the table   goes through it (or, if it does not, too few hooks for the
                                      arm to reach and carry it in the evaluation scenes)
Two views per object: through the hole (along its axis) and from the side. The table top (z = 0)
is the translucent grey plane.

Usage:  $env:G2S_OBJECT_SET="v2"; python scripts/figure_unhookable.py
Output: media/unhookable.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
from grasp2show.config import ObjectConfig, SceneConfig
from grasp2show.gripper_sim import GripperSim
from grasp2show.hanging import hand_min_z, max_penetration, nudge
from grasp2show.hooks import (
    _placed_pose,
    best_fit,
    find_loops,
    finger_sections,
    fit_hook_poses,
    hole_section,
    lift_test,
)
from grasp2show.objects import load_object_set, object_from_config
from grasp2show.transforms import make_T

S = 360


def diagnose(name: str):
    """(verdict text, hand pose T_obj_tcp, finger gap, hole axis or None)."""
    obj = object_from_config(ObjectConfig(asset=name))
    sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj)
    loops = find_loops(obj.mesh, np.random.default_rng(0))
    if not loops:
        lo, hi = obj.mesh.bounds
        T = make_T(np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),           # fingers along +y, toward it
                   [0.5 * (lo[0] + hi[0]), lo[1] - 0.06, 0.5 * (lo[2] + hi[2])])
        return obj, sim, "no through-hole found", T, 0.08, None
    fits = fit_hook_poses(obj, loops, turns=12)
    nudged = [(h, nudge(sim, h["T"], h["width"])) for h in fits]
    clear = [(h, T) for h, T in nudged if T is not None]
    lifting = [(h, T) for h, T in clear if lift_test(sim, T, h["width"], close=False)["success"]]
    P = np.array([lp[0] for lp in loops])
    i = int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))
    centre, axis = loops[i][0], loops[i][1]
    if lifting:
        h, T = lifting[0]
        sim._reset(np.eye(3), T, h["width"])
        zmin = hand_min_z(sim)
        if zmin < 0:
            return obj, sim, (f"{len(lifting)} hooks fit and lift it, but with the object on the table\n"
                              f"the hand reaches {zmin * 100:.1f} cm below the table top"), T, h["width"], axis
        return obj, sim, (f"{len(lifting)} hook(s) fit and lift it (floating test), but too few: in the\n"
                          "scenes no reachable approach + carry was found (scripts/hook_episodes.py)"), T, h["width"], axis
    # best attempt in the hole: least overlap over both finger options
    secs = finger_sections(0.005)
    solid, hole = hole_section(obj, centre, axis)
    best = None
    for poly, w, tcp_pt in ((secs[0], 0.0, (0.0, 0.0)), (secs[1], 0.08, (0.0, -0.04))):
        f = best_fit(poly, hole, solid)
        if f is not None and (best is None or f[3] > best[0][3]):
            best = (f, poly, w, tcp_pt)
    if best is None:
        return obj, sim, "hole too small for even one fingertip", np.eye(4), 0.08, axis
    f, poly, w, tcp_pt = best
    T = _placed_pose(poly, f, centre, axis, 0.005, tcp_pt)
    if clear:                                       # show one that clears the handle (and fails the lift)
        h, T = clear[0]
        w = h["width"]
    why = (f"best fingertip fit: {f[3] * 1000:+.1f} mm clearance in the 2-D cut" if f[3] < 0.0005 else
           f"fingertip fits the cut ({f[3] * 1000:.1f} mm), but in 3-D it digs "
           f"{max_penetration(sim, T, w) * 1000:.1f} mm into the handle" if not clear else
           f"{len(clear)} fits clear the handle, but none lifts the object")
    return obj, sim, why, T, w, axis


def render(sim, T, w, axis, obj) -> list[np.ndarray]:
    m, d = sim.model, sim.data
    m.vis.headlight.ambient[:] = 0.5
    m.vis.headlight.diffuse[:] = 0.7
    sim._reset(np.eye(3), T, w)
    r = mujoco.Renderer(m, S, S)
    centre = obj.mesh.bounds.mean(0)
    views = []
    dirs = [axis if axis is not None else np.array([1.0, 0, 0]), None]
    for k, dv in enumerate(dirs):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = 0.5 * (centre + T[:3, 3])
        cam.distance = 0.42
        if k == 0:
            dv = np.asarray(dv) * (1 if dv[2] <= 0.3 else -1)
            cam.azimuth = float(np.degrees(np.arctan2(dv[1], dv[0])))
            cam.elevation = -10.0
        else:
            cam.azimuth = float(np.degrees(np.arctan2(T[1, 3] - centre[1], T[0, 3] - centre[0]))) + 90.0
            cam.elevation = -30.0
        r.update_scene(d, camera=cam)
        scn = r.scene
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([0.25, 0.25, 0.0008]), np.array([*centre[:2], -0.0008]),
                            np.eye(3).ravel(), np.array([0.6, 0.6, 0.6, 0.45], np.float32))
        scn.ngeom += 1
        views.append(r.render().copy())
    r.close()
    return views


def main() -> int:
    oset = load_object_set()
    rows = []
    for name in oset["handle_candidates"]:
        if name.startswith("ACE_Coffee_Mug"):
            continue                                  # the one that can be hooked
        obj, sim, why, T, w, axis = diagnose(name)
        print(f"{name}: {why}", flush=True)
        a, b = render(sim, T, w, axis, obj)
        panel = np.concatenate([a, b], 1)
        bar = np.full((70, panel.shape[1], 3), 245, np.uint8)
        cv2.putText(bar, name.replace("_", " ")[:58], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                    cv2.LINE_AA)
        for j, line in enumerate(why.split("\n")):
            cv2.putText(bar, line, (8, 44 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 160), 1, cv2.LINE_AA)
        rows.append(np.concatenate([bar, panel], 0))
    grid = []
    for i in range(0, len(rows), 2):
        pair = rows[i:i + 2]
        if len(pair) == 1:
            pair.append(np.full_like(pair[0], 255))
        grid.append(np.concatenate(pair, 1))
    out = ROOT / "media" / "unhookable.png"
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(grid, 0), cv2.COLOR_RGB2BGR))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
