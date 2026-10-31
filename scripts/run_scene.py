"""Run the pipeline (grasp2show/pipeline.py) on one scene and film it.

The scene is one of the evaluation's (an object and a scene number: pipeline.sample_scene) or
your own (--yaw, --xy, --mu, --mass-scale; the scanner where config.CameraConfig puts it). Prints
the plan (hook or pinch, the chooser's predicted success of each, the candidates), runs it and
writes a video: the robot from the side, and the scanner's view.

Usage:  python scripts/run_scene.py ACE_Coffee_Mug_Kristen_16_oz_cup --scene 1
        python scripts/run_scene.py ACE_Coffee_Mug_Kristen_16_oz_cup --mu 0.2 --mass-scale 4 --yaw 30
Outputs: media/run_<object>_<scene>.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("G2S_OBJECT_SET", "v2")


def main() -> int:
    import cv2
    import imageio.v2 as imageio
    import mujoco

    from grasp2show.config import CameraConfig
    from grasp2show.pipeline import SCANNER_SCALE, Pipeline, Scene, sample_scene

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("asset")
    p.add_argument("--scene", type=int, default=None, help="the evaluation's scene k of this object")
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--xy", type=float, nargs=2, default=(0.5, -0.15))
    p.add_argument("--mu", type=float, default=0.5, help="finger-object friction")
    p.add_argument("--mass-scale", type=float, default=1.0)
    p.add_argument("--pose-error", action="store_true", help="with --scene: also the evaluation's unknown pose error")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    off = None
    if args.scene is not None:
        scene, err = sample_scene(args.asset, args.scene)
        off = err if args.pose_error else None
        tag = f"{args.scene}"
    else:
        cam = CameraConfig()
        cam = replace(cam, width=int(cam.width * SCANNER_SCALE), height=int(cam.height * SCANNER_SCALE))
        scene = Scene(args.asset, args.yaw, tuple(args.xy), cam, args.mu, args.mass_scale)
        tag = f"mu{args.mu}_x{args.mass_scale}"
    pipe = Pipeline()
    plan = pipe.plan(scene)
    m = plan.checked.sc.ep.model
    print(f"{args.asset}: mu {scene.mu}, {plan.checked.sc.obj.mass:.2f} kg -> {plan.method} (predicted success: "
          + ", ".join(f"{s} {p:.2f}" for s, p in plan.probs.items()) + "; candidates: "
          + ", ".join(f"{s} {len(c)}" for s, c in plan.ranked.items()) + ")")
    side = mujoco.Renderer(m, 480, 640)
    scan = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.45, -0.05, 0.25]
    cam.distance, cam.azimuth, cam.elevation = 1.5, 150.0, -22.0
    frames = []

    def on_frame(d):
        side.update_scene(d, camera=cam)
        scan.update_scene(d, camera="scanner")
        a, b = side.render().copy(), scan.render().copy()
        cv2.putText(a, f"t = {d.time:4.1f} s", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(b, "scanner", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frames.append(np.concatenate([a, b], 1))

    res = pipe.run(scene, obj_offset=off, on_frame=on_frame, plan=plan)
    print(f"ran {res.method}: {'success' if res.success else 'failed: ' + res.reason} ({res.seconds:.0f} s)")
    if frames:
        out = ROOT / "media" / f"run_{args.asset[:40]}_{tag}.mp4"
        imageio.mimwrite(out, frames, fps=args.fps, quality=7, macro_block_size=8)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
