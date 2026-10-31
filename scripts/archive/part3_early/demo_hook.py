"""Video: a heavy mug, pinched (way B's choice) vs hooked by the handle and shown hanging (part 3).

Both runs are the same scene (start pose, scanner pose) as scripts/hook_episodes.py, with the
object made `--mass` times heavier. Left: way B's pinch; right: the hook planner
(grasp2show/hanging.py). The scanner's last image is inset in each view.

Usage:  $env:G2S_OBJECT_SET="v2"; python scripts/demo_hook.py --scene 13 --mass 8 --scanner-scale 2
Outputs: media/hook_vs_pinch_<object>.mp4 and .gif
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
from hook_episodes import GRASPS, hooks_for

from grasp2show.config import random_camera
from grasp2show.experiment import Scenario
from grasp2show.feasibility import twin_of
from grasp2show.gripper_sim import GripperSim
from grasp2show.hanging import HOOK_SPEED, hook_options, plan_hook
from grasp2show.objects import load_object_set
from grasp2show.task import Grasp, show_pose

W, H = 640, 480


def scene(name: str, k: int):
    oset = load_object_set()
    r = np.random.default_rng(9000 + oset["objects"].index(name))
    for _ in range(k + 1):
        yaw, xy, cam = float(r.uniform(0, 360)), (float(r.uniform(0.40, 0.60)), float(r.uniform(-0.30, 0.0))), \
            random_camera(r)
    return yaw, xy, cam


def run_pinch(name, yaw, xy, cam, mass, model, clouds, frames):
    from eval_way_b import kinematic_choices, way_b_grasps

    T, Wd, R, _ = way_b_grasps(name, yaw, model, clouds, 10, start_xy=xy, camera=cam)
    sc = Scenario(name, yaw, {"T_obj_tcp": T, "width": Wd, "success": np.ones(len(T), bool)}, start_xy=xy,
                  camera=cam, mass_scale=mass)
    for c in kinematic_choices(sc, list(zip(range(len(T)), R.tolist()))):
        tw = sc.reports[c.grasp].twins[c.twin]
        t = tw.targets[c.roll_idx]
        G = sc.T[c.grasp] if c.twin == 0 else twin_of(sc.T[c.grasp])
        paths, _ = sc.ep.plan_analytic(G, show_pose(sc.cfg, sc.obj, float(sc.rolls[c.roll_idx])), tw.start.q,
                                       t.reach.q, checker=sc.fc, width=float(sc.W[c.grasp]))
        if paths is not None:
            res = sc.ep.execute(Grasp("pinch", G, float(sc.W[c.grasp])), paths,
                                on_frame=lambda d: frames.append(d.qpos.copy()))
            return sc, res
    return sc, None


def run_hook(name, yaw, xy, cam, mass, frames):
    z = np.load(GRASPS / f"{name}.npz")
    sc = Scenario(name, yaw, {kk: z[kk][:0] for kk in ("T_obj_tcp", "width", "success")}, start_xy=xy, camera=cam,
                  mass_scale=mass)
    hooks = hooks_for(name)
    for o in [o for o in hook_options(sc.fc, GripperSim(sc.cfg, sc.obj), hooks, T_start=sc.ep.resting_pose()) if o.ok][:5]:
        paths, _ = plan_hook(sc.ep, sc.fc, o)
        if paths is not None:
            res = sc.ep.execute(Grasp("hook", o.T_obj_tcp, o.width, "hook"), paths, speed=HOOK_SPEED,
                                on_frame=lambda d: frames.append(d.qpos.copy()))
            return sc, res
    return sc, None


def render(sc, qs, res, title):
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
    cam.distance, cam.elevation = 1.15, -18.0
    out = []
    inset = None
    if res is not None and res.scanner_rgb is not None:
        inset = cv2.resize(res.scanner_rgb, (W // 3, H // 3), interpolation=cv2.INTER_AREA)
    for i, q in enumerate(qs):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        img = cv2.resize(r.render(), (W, H), interpolation=cv2.INTER_AREA)
        cv2.putText(img, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if i >= len(qs) - 45 and inset is not None:          # the last 1.5 s: what the scanner saw
            img[H - inset.shape[0] - 8:H - 8, W - inset.shape[1] - 8:W - 8] = inset
            ok = res.success
            why = "dropped" if not res.held else "collision" if res.collisions else "QR not read" if not res.qr_ok \
                else "slipped"
            cv2.putText(img, "success: QR read, still held" if ok else f"failed: {why}",
                        (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (90, 230, 90) if ok else (90, 90, 240), 2,
                        cv2.LINE_AA)
        out.append(img)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", default="ACE_Coffee_Mug_Kristen_16_oz_cup")
    p.add_argument("--scene", type=int, default=0)
    p.add_argument("--mass", type=float, default=8)
    p.add_argument("--way-b", type=Path, default=ROOT / "data" / "models" / "heatmap_net_v2.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset_v2" / "clouds.npz")
    p.add_argument("--scanner-scale", type=float, default=1.0, help="scanner resolution x this")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    yaw, xy, cam = scene(args.object, args.scene)
    if args.scanner_scale != 1:
        cam = replace(cam, width=int(cam.width * args.scanner_scale), height=int(cam.height * args.scanner_scale))
    fp, fh = [], []
    sp, rp = run_pinch(args.object, yaw, xy, cam, args.mass, args.way_b, args.clouds, fp)
    kg = f"{sp.obj.mass:.1f} kg"
    left = render(sp, fp, rp, f"pinch (way B), {kg}") if rp else []
    print("pinch:", rp.summary() if rp else "no plan")
    sh, rh = run_hook(args.object, yaw, xy, cam, args.mass, fh)
    right = render(sh, fh, rh, f"hook by the handle, {kg}") if rh else []
    print("hook:", rh.summary() if rh else "no plan")
    n = max(len(left), len(right))
    blank = np.zeros((H, W, 3), np.uint8)
    pad = lambda f: f + [f[-1] if f else blank] * (n - len(f))
    frames = [np.concatenate([a, b], 1) for a, b in zip(pad(left), pad(right))]
    stem = ROOT / "media" / f"hook_vs_pinch_{args.object[:24]}"
    imageio.mimwrite(f"{stem}.mp4", frames, fps=args.fps, quality=8, macro_block_size=8)
    small = [cv2.resize(f, (f.shape[1] * 2 // 5, f.shape[0] * 2 // 5), interpolation=cv2.INTER_AREA)
             for f in frames[::5]]                     # README-sized (~3 MB)
    imageio.mimwrite(f"{stem}.gif", small, duration=1000 * 5 / args.fps, loop=0)
    print(f"wrote {stem}.mp4 and .gif ({n / args.fps:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
