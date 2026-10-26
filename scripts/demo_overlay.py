"""One view, two robots: the baseline's motion and ours overlaid at half opacity, each drawing
its gripper path in 3D as it moves.

Both episodes run in full simulation exactly as in the experiments (same planner, same judge);
the joint states are recorded, then replayed and rendered from the same camera. Each layer has
its robot tinted (baseline orange, ours blue) and both 3D paths drawn; the two layers are
blended 50/50, so the static scene looks normal, each robot is translucent and the paths stay
at full strength. Insets: what each scanner camera sees.

Usage:
  python scripts/demo_overlay.py --object Crayola_Crayons_24_count --yaw 90
Writes media/overlay_<object>_<yaw>.mp4 and .gif.
"""
from __future__ import annotations

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from demo_comparison import find_scenario, outcome_text, plan

from grasp2show.experiment import Scenario
from grasp2show.scene import PANDA_BODIES

POLICIES = {  # name: (label, RGB tint of the robot and its path)
    "stability_only": ("baseline: most stable grasp", (235, 120, 40)),
    "aware": ("target-aware grasp (ours)", (40, 120, 230)),
}


def record(sc: Scenario, policy: str, fps: int):
    """Run the episode; return the joint state and gripper (TCP) position at every frame."""
    grasp, paths = plan(sc, policy)
    m = sc.ep.model
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    qpos, path = [], []

    def on_frame(d):
        qpos.append(d.qpos.copy())
        path.append(d.site_xpos[tcp].copy())

    res = sc.ep.execute(grasp, paths, on_frame=on_frame, fps=fps)
    print(f"{policy}: {res.summary()}")
    return np.array(qpos), np.array(path), res


def robot_geoms(m: mujoco.MjModel) -> np.ndarray:
    bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ["link0", *PANDA_BODIES]}
    return np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.geom_group[g] < 3])


def add_path(scn: mujoco.MjvScene, pts: np.ndarray, rgb, radius: float = 0.0035) -> None:
    """Capsules along the path so far, plus a ball at the current gripper position."""
    rgba = np.array([*np.asarray(rgb) / 255, 1.0], dtype=np.float32)
    for a, b in pairwise(pts):
        if scn.ngeom >= scn.maxgeom or np.linalg.norm(b - a) < 1e-5:
            continue
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
        scn.ngeom += 1
    if len(pts) and scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 2.2 * radius), pts[-1],
                            np.eye(3).ravel(), rgba)
        scn.ngeom += 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", default=None)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()
    obj, yaw = (args.object, args.yaw) if args.object else find_scenario()
    z = np.load(ROOT / "data" / "grasps" / f"{obj}.npz")
    sc = Scenario(obj, yaw, {k: z[k] for k in ("T_obj_tcp", "width", "success")})
    runs = {pol: record(sc, pol, args.fps) for pol in POLICIES}

    m, r = sc.ep.model, sc.ep.renderer
    d = mujoco.MjData(m)
    geoms = robot_geoms(m)
    # Tint through the geom colour: take each robot geom's paint (white / black) as a brightness,
    # then drop its material so the geom colour is used.
    base_rgba, base_mat = m.geom_rgba[geoms].copy(), m.geom_matid[geoms].copy()
    paint = np.where((base_mat >= 0)[:, None], m.mat_rgba[np.maximum(base_mat, 0), :3], base_rgba[:, :3])
    shade = np.clip(paint.mean(1, keepdims=True) * 1.3, 0.25, 1.0)
    m.geom_matid[geoms] = -1
    n = max(len(q) for q, _, _ in runs.values())
    frames = []
    for i in range(n):
        layers, insets = [], []
        for pol, (q, _, _) in runs.items():
            rgb = POLICIES[pol][1]
            m.geom_rgba[geoms, :3] = shade * np.asarray(rgb) / 255
            d.qpos[:] = q[min(i, len(q) - 1)]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera="scanner")
            insets.append(r.render().copy())
            r.update_scene(d, camera="overview")
            # Both paths in both layers, so they stay at full strength after blending.
            for pol2, (_, path, _) in runs.items():
                add_path(r.scene, path[:min(i, len(path) - 1) + 1], POLICIES[pol2][1])
            layers.append(r.render().astype(np.float32))
        frame = (0.5 * layers[0] + 0.5 * layers[1]).astype(np.uint8)
        frames.append(decorate(frame, insets, runs, i, done=False))
    for _ in range(int(2.5 * args.fps)):
        frames.append(decorate(frame, insets, runs, n - 1, done=True))
    m.geom_rgba[geoms], m.geom_matid[geoms] = base_rgba, base_mat

    stem = ROOT / "media" / f"overlay_{obj[:30]}_{int(yaw)}"
    imageio.mimwrite(f"{stem}.mp4", frames, fps=args.fps, quality=8, macro_block_size=8)
    write_gif(frames, args.fps, Path(f"{stem}.gif"))
    print(f"wrote {stem}.mp4 and .gif")
    return 0


def decorate(frame, insets, runs, i, done):
    out = frame.copy()
    h, w = out.shape[:2]
    iw, ih = w // 4, h // 4
    for k, ((pol, (q, _, res)), inset) in enumerate(zip(runs.items(), insets)):
        label, rgb = POLICIES[pol]
        y0 = 8 + k * (ih + 22)
        out[y0:y0 + ih, w - iw - 8:w - 8] = cv2.resize(inset, (iw, ih), interpolation=cv2.INTER_AREA)
        cv2.rectangle(out, (w - iw - 10, y0 - 2), (w - 6, y0 + ih + 1), rgb, 2)
        # legend
        cv2.rectangle(out, (12, 14 + 26 * k), (30, 30 + 26 * k), rgb, -1)
        cv2.putText(out, label, (38, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                    cv2.LINE_AA)
        if done:
            text, ok = outcome_text(res)
            col = (40, 170, 70) if ok else (200, 50, 50)
            x0, x1 = (0, w // 2) if k == 0 else (w // 2, w)
            cv2.rectangle(out, (x0, h - 40), (x1, h), col, -1)
            cv2.rectangle(out, (x0, h - 44), (x1, h - 40), rgb, -1)
            cv2.putText(out, text, (x0 + 10, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)
    cv2.putText(out, "scanner views", (w - iw - 8, 2 * (ih + 22) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_gif(frames, fps, path, gif_fps=10, width=480):
    from PIL import Image

    step = max(fps // gif_fps, 1)
    h, w = frames[0].shape[:2]
    small = [cv2.resize(f, (width, round(h * width / w)), interpolation=cv2.INTER_AREA) for f in frames[::step]]
    sample = np.vstack([small[0], small[len(small) // 2], small[-1]])
    palette = Image.fromarray(sample).quantize(128, dither=Image.Dither.NONE)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in small]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / gif_fps), loop=0,
                 optimize=True)


if __name__ == "__main__":
    sys.exit(main())
