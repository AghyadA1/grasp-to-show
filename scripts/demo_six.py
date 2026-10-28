"""Video: six objects of different shapes carried to the scanner, side by side, with the hand's path.

Five are pinched (way B's grasp, the analytic motion planner) and one is hooked by its handle
(grasp2show/hanging.py). Each object gets the first of its evaluation scenes (seeds as in
eval_way_b_v2.py / hook_episodes.py) where the episode succeeds. The planning time of each method
is measured and printed (and written next to the video).

Usage:  $env:G2S_OBJECT_SET="v2"; python scripts/demo_six.py
Outputs: media/six_objects.mp4, .gif, media/six_objects_timing.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from demo_hook import scene
from hook_episodes import hooks_for

from grasp2show.experiment import Scenario
from grasp2show.feasibility import twin_of
from grasp2show.gripper_sim import GripperSim
from grasp2show.hanging import HOOK_SPEED, hook_options, plan_hook
from grasp2show.task import Grasp, show_pose

OBJECTS = [("Perricone_MD_The_Crease_Cure_Duo", "pinch"),
           ("Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack", "pinch"),
           ("Pennington_Electric_Pot_Cabana_4", "pinch"),
           ("Aroma_Stainless_Steel_Milk_Frother_2_Cup", "pinch"),
           ("Squirt_Strain_Fruit_Basket", "pinch"),
           ("ACE_Coffee_Mug_Kristen_16_oz_cup", "hook")]
PW, PH = 320, 240
EMPTY = {"T_obj_tcp": np.zeros((0, 4, 4)), "width": np.zeros(0), "success": np.zeros(0, bool)}


def pinch(name, k, model, clouds):
    """Way B + analytic motion. Returns (scenario, grasp, paths, timing) or None."""
    from eval_way_b import kinematic_choices, way_b_grasps

    yaw, xy, cam = scene(name, k)
    t0 = time.perf_counter()
    T, W, R, _ = way_b_grasps(name, yaw, model, clouds, 10, start_xy=xy, camera=cam)
    t1 = time.perf_counter()
    sc = Scenario(name, yaw, {"T_obj_tcp": T, "width": W, "success": np.ones(len(T), bool)}, start_xy=xy, camera=cam)
    t2 = time.perf_counter()
    for c in kinematic_choices(sc, list(zip(range(len(T)), R.tolist()))):
        tw = sc.reports[c.grasp].twins[c.twin]
        G = sc.T[c.grasp] if c.twin == 0 else twin_of(sc.T[c.grasp])
        paths, _ = sc.ep.plan_analytic(G, show_pose(sc.cfg, sc.obj, float(sc.rolls[c.roll_idx])), tw.start.q,
                                       tw.targets[c.roll_idx].reach.q, checker=sc.fc, width=float(sc.W[c.grasp]))
        if paths is not None:
            t3 = time.perf_counter()
            timing = {"network_s": t1 - t0, "checks_s": t2 - t1, "motion_s": t3 - t2, "total_s": t3 - t0}
            return sc, Grasp("pinch", G, float(sc.W[c.grasp])), paths, 1.0, timing
    return None


def hook(name, k):
    yaw, xy, cam = scene(name, k)
    hooks = hooks_for(name)                                       # cached per object
    t0 = time.perf_counter()
    sc = Scenario(name, yaw, EMPTY, start_xy=xy, camera=cam)
    opts = [o for o in hook_options(sc.fc, GripperSim(sc.cfg, sc.obj), hooks, T_start=sc.ep.resting_pose()) if o.ok]
    t1 = time.perf_counter()
    for o in opts[:5]:
        paths, _ = plan_hook(sc.ep, sc.fc, o)
        if paths is not None:
            t2 = time.perf_counter()
            timing = {"hook_options_s": t1 - t0, "motion_s": t2 - t1, "total_s": t2 - t0}
            return sc, Grasp("hook", o.T_obj_tcp, o.width, "hook"), paths, HOOK_SPEED, timing
    return None


def record(sc, grasp, paths, speed, fps):
    m = sc.ep.model
    tcp = m.site("tcp").id
    qs, pts = [], []

    def on_frame(d):
        qs.append(d.qpos.copy())
        pts.append(d.site_xpos[tcp].copy())

    res = sc.ep.execute(grasp, paths, on_frame=on_frame, fps=fps, speed=speed)
    return res, qs, np.array(pts)


def render(sc, qs, pts, res, title):
    m = sc.ep.model
    m.vis.headlight.ambient[:] = 0.4
    d = mujoco.MjData(m)
    r = sc.ep.renderer
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    at = np.asarray(sc.cfg.camera.look_at)
    start = np.array([*sc.cfg.obj.start_xy, 0.05])
    cam.lookat[:] = 0.5 * (at + start) + [0, 0, 0.05]
    to_scanner = np.asarray(sc.cfg.camera.pos) - at
    cam.azimuth = float(np.degrees(np.arctan2(to_scanner[1], to_scanner[0]))) + 70.0
    cam.distance, cam.elevation = 1.2, -20.0
    path = pts[::3]
    out = []
    for i, q in enumerate(qs):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        scn = r.scene
        for j, p in enumerate(path):                       # the whole path faint, the part done bright
            if scn.ngeom >= scn.maxgeom:
                break
            done = j * 3 <= i
            rgba = np.array([1.0, 0.55, 0.1, 1.0] if done else [0.3, 0.6, 1.0, 0.35], np.float32)
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.004 if done else 0.003),
                                p, np.eye(3).ravel(), rgba)
            scn.ngeom += 1
        img = cv2.resize(r.render(), (PW, PH), interpolation=cv2.INTER_AREA)
        cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if i >= len(qs) - 30:
            ok = res.success
            cv2.putText(img, "QR read" if ok else "failed", (8, PH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (90, 230, 90) if ok else (90, 90, 240), 1, cv2.LINE_AA)
        out.append(img)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--way-b", type=Path, default=ROOT / "data" / "models" / "heatmap_net_v2.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset_v2" / "clouds.npz")
    p.add_argument("--max-scenes", type=int, default=8)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()
    panels, timing = [], {}
    for name, method in OBJECTS:
        for k in range(args.max_scenes):
            got = pinch(name, k, args.way_b, args.clouds) if method == "pinch" else hook(name, k)
            if got is None:
                continue
            sc, g, paths, speed, t = got
            res, qs, pts = record(sc, g, paths, speed, args.fps)
            print(f"{name[:30]:30s} scene {k} {method}: {res.summary()}  planning {t['total_s']:.1f} s", flush=True)
            if res.success:
                short = name.replace("_", " ")[:26]
                panels.append(render(sc, qs, pts, res, f"{short} ({method})"))
                timing[name] = {"method": method, "scene": k, **{kk: round(v, 2) for kk, v in t.items()}}
                break
        else:
            print(f"{name}: no successful scene in {args.max_scenes}")
    n = max(len(f) for f in panels)
    panels = [f + [f[-1]] * (n - len(f)) for f in panels]
    while len(panels) < 6:
        panels.append([np.zeros((PH, PW, 3), np.uint8)] * n)
    frames = [np.concatenate([np.concatenate([panels[r * 3 + c][i] for c in range(3)], 1) for r in range(2)], 0)
              for i in range(n)]
    stem = ROOT / "media" / "six_objects"
    imageio.mimwrite(f"{stem}.mp4", frames, fps=args.fps, quality=7, macro_block_size=8)
    imageio.mimwrite(f"{stem}.gif", [cv2.resize(f, (f.shape[1] * 2 // 3, f.shape[0] * 2 // 3)) for f in frames[::3]],
                     duration=1000 * 3 / args.fps, loop=0)
    (ROOT / "media" / "six_objects_timing.json").write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing, indent=1))
    print(f"wrote {stem}.mp4 and .gif ({n / args.fps:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
