"""Goal-conditioned grasp heatmap, way A (no training): dense candidates -> checks -> paint.

For one object at one start pose:
  1. sample dense antipodal grasp candidates (more than the labelled set) and drop those where
     the open gripper intersects the object;
  2. run the analytic checks for every candidate, both twins, and --rolls target poses (the QR
     faces the scanner, rolled about the viewing axis);
  3. paint onto the surface: each point gets the best score of the grasps whose finger
     contact lies within --radius of it (grasp2show/heatmap.py).
Lift stability uses the analytic margin (check 1's model, AUC 0.92-0.94 vs simulation), so
no simulation is needed: this is the checks made visible.

Usage:
  python scripts/heatmap_way_a.py --object Crayola_Crayons_24_count --yaw 90
Outputs: data/heatmaps/<object>_<yaw>.npz, media/heatmap_<object>_<yaw>.gif (+ .png: one frame)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
from grasp2show.config import ObjectConfig, SceneConfig
from grasp2show.experiment import lift_margins
from grasp2show.feasibility import FeasibilityChecker
from grasp2show.grasp_sampler import candidates_from_pairs, sample_antipodal
from grasp2show.gripper_sim import GripperSim
from grasp2show.heatmap import aware_scores, paint, stability_scores
from grasp2show.objects import object_from_config
from grasp2show.scene import build_scene
from grasp2show.stability import finger_contacts
from grasp2show.task import show_pose

_ctx = {}


def _init(asset: str, yaw: float):
    cfg = SceneConfig(obj=ObjectConfig(asset=asset, start_yaw=np.radians(yaw)))
    obj = object_from_config(cfg.obj)
    _ctx.update(cfg=cfg, obj=obj, fc=FeasibilityChecker(build_scene(cfg, obj), cfg), sim=GripperSim(cfg, obj))


def _collides(args):
    T, w = args
    return _ctx["sim"].in_collision(T, w)


def _evaluate(args):
    T, w, rolls = args
    return _ctx["fc"].evaluate(T, w, rolls)


def compute(asset: str, yaw: float, n_pairs: int, n_rolls: int, workers: int, seed: int = 0) -> dict:
    cfg = SceneConfig(obj=ObjectConfig(asset=asset, start_yaw=np.radians(yaw)))
    obj = object_from_config(cfg.obj)
    rng = np.random.default_rng(seed)
    p1, p2 = sample_antipodal(obj.mesh, n_pairs, rng)
    cand = candidates_from_pairs(p1, p2, 8, rng)
    rolls = np.linspace(0, 2 * np.pi, n_rolls, endpoint=False)
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init, initargs=(asset, yaw)) as pool:
        hit = np.array(pool.map(_collides, list(zip(cand.T_obj_tcp, cand.width)), chunksize=32))
        cand = cand.subset(~hit)
        reports = pool.map(_evaluate, [(T, w, rolls) for T, w in zip(cand.T_obj_tcp, cand.width)], chunksize=16)
    lm = lift_margins(obj, cand.T_obj_tcp, FeasibilityChecker(build_scene(cfg, obj), cfg).stab)
    c1, c2, _ = finger_contacts(obj.mesh, cand.T_obj_tcp)
    S_aware = aware_scores(reports, lm, n_rolls)
    S_stab = stability_scores(reports, lm)
    print(f"{len(cand)} collision-free candidates, checks in {time.perf_counter() - t0:.0f} s; "
          f"feasible for some target: {(S_aware.max(1) > 0).mean():.0%}")
    return {"T_obj_tcp": cand.T_obj_tcp, "width": cand.width, "c1": c1, "c2": c2, "lift_margin": lm,
            "rolls": rolls, "S_aware": S_aware, "S_stab": S_stab}


def render(asset: str, yaw: float, res: dict, radius: float, out_gif: Path, fps: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    cfg = SceneConfig(obj=ObjectConfig(asset=asset, start_yaw=np.radians(yaw)))
    obj = object_from_config(cfg.obj)
    mesh = obj.mesh
    tri = mesh.vertices[mesh.faces]
    cent = tri.mean(1)
    H_stab = paint(cent, res["c1"], res["c2"], res["S_stab"], radius)
    H_aware = paint(cent, res["c1"], res["c2"], res["S_aware"], radius)

    cmap = LinearSegmentedColormap.from_list("score", ["#eef3f8", "#9ecae1", "#3182bd", "#08306b"])
    no_info = np.array([0.62, 0.60, 0.57])
    T_qr = obj.qr.T_obj_qr
    n_qr = T_qr[:3, 0]
    view_dirs = [n_qr, -n_qr]            # look at the QR face, and at the opposite face

    def colours(h, light):
        c = np.where(np.isnan(h)[:, None], no_info, cmap(np.nan_to_num(h))[:, :3])
        shade = 0.6 + 0.4 * np.clip(mesh.face_normals @ light, 0, 1)
        return np.clip(c * shade[:, None], 0, 1)

    qr_corners = np.array([[0, sy, sz] for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]) * obj.qr.side / 2
    qr_world = (T_qr[:3, :3] @ qr_corners.T).T + T_qr[:3, 3] + 0.002 * n_qr
    c0 = mesh.bounds.mean(0)
    L = 0.5 * np.ptp(mesh.bounds, axis=0).max()
    side = np.cross([0, 0, 1.0], n_qr)                   # where the gravity arrow is drawn
    side /= np.linalg.norm(side)

    frames = []
    for j, roll in enumerate(res["rolls"]):
        R_t = show_pose(cfg, obj, roll)[:3, :3]
        g_obj = R_t.T @ np.array([0, 0, -1.0])           # gravity at the target, object frame
        fig = plt.figure(figsize=(8.4, 8.0), dpi=90)
        for r, vd in enumerate(view_dirs):
            az = np.degrees(np.arctan2(vd[1], vd[0])) + 35       # three-quarter view
            light = vd / np.linalg.norm(vd) + np.array([0, 0, 0.6])
            light /= np.linalg.norm(light)
            for k, (h, title) in enumerate(((H_stab, "stability only (target-blind)"),
                                            (H_aware[:, j], "target-aware (this target)"))):
                ax = fig.add_subplot(2, 2, 2 * r + k + 1, projection="3d", computed_zorder=False)
                ax.add_collection3d(Poly3DCollection(tri, facecolors=colours(h, light), edgecolors="none",
                                                     linewidths=0, antialiased=False, zorder=1))
                if r == 0:                                    # the QR outline, on the visible side
                    ax.plot(*qr_world.T, c="#d62728", lw=1.8, zorder=5)
                if k == 1:                                    # gravity at the target, beside the object
                    a = c0 + 0.95 * L * side - 0.45 * L * g_obj
                    ax.quiver(*a, *(0.9 * L * g_obj), color="#111111", lw=2.2, arrow_length_ratio=0.3,
                              zorder=6)
                ax.set_xlim(c0[0] - L, c0[0] + L)
                ax.set_ylim(c0[1] - L, c0[1] + L)
                ax.set_zlim(c0[2] - L, c0[2] + L)
                ax.set_box_aspect((1, 1, 1), zoom=1.45)
                ax.view_init(elev=20, azim=az)
                ax.set_axis_off()
                if r == 0:
                    ax.set_title(title, fontsize=11, pad=0)
        fig.text(0.03, 0.97, f"target: QR facing the scanner, rolled {np.degrees(roll):3.0f}° about the viewing axis",
                 fontsize=11)
        fig.text(0.015, 0.72, "QR side", rotation=90, fontsize=10, va="center")
        fig.text(0.015, 0.30, "opposite side", rotation=90, fontsize=10, va="center")
        fig.text(0.52, 0.02, "arrow: gravity at the target   red: QR patch   grey: no grasp touches here",
                 fontsize=8.5, color="0.3")
        cax = fig.add_axes([0.08, 0.025, 0.3, 0.015])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_label("best grasp score touching here (0 = fails a check)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        fig.subplots_adjust(0.03, 0.07, 0.99, 0.93, 0.0, 0.0)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        if j == 0:
            Image.fromarray(frames[-1]).save(out_gif.with_suffix(".png"))
    sample = np.vstack([frames[0], frames[len(frames) // 2]])
    pal = Image.fromarray(sample).quantize(160, dither=Image.Dither.NONE)
    imgs = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", default="Crayola_Crayons_24_count")
    p.add_argument("--yaw", type=float, default=90.0)
    p.add_argument("--pairs", type=int, default=400, help="antipodal pairs (x 8 approach angles)")
    p.add_argument("--rolls", type=int, default=24)
    p.add_argument("--radius", type=float, default=0.008, help="m: how far a finger contact paints")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--render-only", action="store_true")
    args = p.parse_args()
    out = ROOT / "data" / "heatmaps"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{args.object}_{int(args.yaw)}.npz"
    if args.render_only:
        res = dict(np.load(f))
    else:
        res = compute(args.object, args.yaw, args.pairs, args.rolls, args.workers)
        np.savez_compressed(f, **res)
    gif = ROOT / "media" / f"heatmap_{args.object[:30]}_{int(args.yaw)}.gif"
    render(args.object, args.yaw, res, args.radius, gif)
    print(f"wrote {gif}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
