"""Pictures of way B on unseen objects: the network's heatmap and its best grasp, next to way A.

For each object (at one start pose): the network predicts, for each target pose, a score and a
grasp at every surface point. Shown side by side for the target the network likes best:
  left   way A: the checks on ~1,000 dense candidates, painted on the surface (slow, the teacher)
  right  way B: the network's prediction (one forward pass) and its top grasp, drawn as a gripper
The surface colour at a point is interpolated from the 1,024 cloud points the network sees.
Also a GIF of one object over all 12 targets.

Usage:  python scripts/visualize_way_b.py
Outputs: media/way_b_objects.png, media/way_b_targets.gif
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
import generate_dense_labels as gdl
from eval_way_b import _load

from grasp2show.config import CameraConfig, ObjectConfig, SceneConfig
from grasp2show.gripper_sim import GripperSim
from grasp2show.learn.dense import condition, grasp_from_prediction, point_labels, split_outputs
from grasp2show.objects import load_object_set, object_from_config
from grasp2show.scene import start_pose
from grasp2show.task import show_pose

# All four are test objects: the network never saw them in training.
OBJECTS = ["Nestle_Skinny_Cow_Dreamy_Clusters_Candy_Dark_Chocolate_6_pack_1_oz_pouches",
           "Office_Depot_Canon_PGI5BK_Remanufactured_Ink_Cartridge_Black",
           "Persona_Q_Shadow_of_the_Labyrinth_Nintendo_3DS", "Phillips_Caplets_Size_24"]


def predict(asset: str, yaw: float, model: Path, clouds_path: Path) -> dict:
    """Network output for all 12 targets + way-A labels on the same points + the top grasps."""
    import torch

    net, clouds = _load(model, clouds_path)
    o = clouds.names.index(asset)
    cfg = SceneConfig(obj=ObjectConfig(asset=asset, start_yaw=np.radians(yaw)))
    obj = object_from_config(cfg.obj)
    rolls = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    cond = np.stack([condition(start_pose(cfg, obj), show_pose(cfg, obj, r), CameraConfig().pos, clouds.mass[o],
                               clouds.com[o]) for r in rolls])
    with torch.no_grad():
        pr = {k: v.numpy() for k, v in split_outputs(net(torch.full((12,), o), torch.as_tensor(cond))).items()}
    q = 1 / (1 + np.exp(-pr["logit"])) * pr["score"]
    pts = clouds.feats[o, :, :3]
    with tempfile.TemporaryDirectory() as tmp:
        gdl.OUT = Path(tmp)
        gdl.process((asset, [yaw], 400, 12, 1000 + load_object_set()["objects"].index(asset)))
        d = dict(np.load(Path(tmp) / f"{asset}__yaw{int(yaw)}.npz"))
    labels, _, _ = point_labels(pts, d)
    sim = GripperSim(cfg, obj)
    best = []
    for r in range(12):                          # top collision-free predicted grasp per target
        for pi in np.argsort(-q[r]):
            T = grasp_from_prediction(pts[pi], pr["approach"][r, pi], pr["closing"][r, pi], pr["offset"][r, pi])
            w = float(np.clip(pr["width"][r, pi], 0, 0.08))
            if not sim.in_collision(T, w):
                best.append((float(q[r, pi]), T, w, pi))
                break
    return {"obj": obj, "cfg": cfg, "pts": pts, "q": q, "labels": labels, "best": best, "rolls": rolls}


def gripper_lines(T: np.ndarray, w: float) -> list[np.ndarray]:
    """Franka hand as line segments in the object frame: two fingers, palm, wrist."""
    h = w / 2
    segs = [[(0, -h, 0), (0, -h, -0.045)], [(0, h, 0), (0, h, -0.045)],
            [(0, -h - 0.01, -0.045), (0, h + 0.01, -0.045)], [(0, 0, -0.045), (0, 0, -0.10)]]
    return [(T[:3, :3] @ np.array(s, float).T).T + T[:3, 3] for s in segs]


def draw(ax, obj, pts, values, view_dir, cmap, T=None, w=None, g=None, title=""):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import cKDTree

    mesh = obj.mesh
    tri = mesh.vertices[mesh.faces]
    dist, idx = cKDTree(pts).query(tri.mean(1), k=3)
    wgt = 1 / np.maximum(dist, 1e-4)
    vals = values[idx]
    known = ~np.isnan(vals)
    v = np.where(known.any(1), np.nansum(np.nan_to_num(vals) * wgt * known, 1) / np.maximum((wgt * known).sum(1), 1e-9),
                 np.nan)
    col = np.where(np.isnan(v)[:, None], [0.62, 0.60, 0.57], cmap(np.nan_to_num(v))[:, :3])
    light = view_dir / np.linalg.norm(view_dir) + np.array([0, 0, 0.6])
    light /= np.linalg.norm(light)
    col = np.clip(col * (0.6 + 0.4 * np.clip(mesh.face_normals @ light, 0, 1))[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolors="none", linewidths=0, antialiased=False,
                                         zorder=1))
    T_qr = obj.qr.T_obj_qr
    sq = np.array([[0, a, b] for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]) * obj.qr.side / 2
    ax.plot(*((T_qr[:3, :3] @ sq.T).T + T_qr[:3, 3] + 0.002 * T_qr[:3, 0]).T, c="#d62728", lw=1.5, zorder=5)
    c0 = mesh.bounds.mean(0)
    L = 0.55 * np.ptp(mesh.bounds, axis=0).max() + 0.03
    if T is not None:
        for s in gripper_lines(T, w):
            ax.plot(*s.T, c="#ff7f0e", lw=2.6, zorder=7, solid_capstyle="round")
    if g is not None:
        side = np.cross([0, 0, 1.0], view_dir)
        side /= np.linalg.norm(side) + 1e-9
        a = c0 - 0.9 * L * side - 0.4 * L * g
        ax.quiver(*a, *(0.8 * L * g), color="#111111", lw=2, arrow_length_ratio=0.3, zorder=6)
    ax.set_xlim(c0[0] - L, c0[0] + L)
    ax.set_ylim(c0[1] - L, c0[1] + L)
    ax.set_zlim(c0[2] - L, c0[2] + L)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.view_init(elev=22, azim=np.degrees(np.arctan2(view_dir[1], view_dir[0])) + 35)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", nargs="+", default=OBJECTS)
    p.add_argument("--yaw", type=float, default=90.0)
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--gif-object", default=OBJECTS[1])
    args = p.parse_args()
    test = set(load_object_set()["test"])
    assert all(n in test for n in args.objects + [args.gif_object]), "show unseen (test) objects only"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    cmap = LinearSegmentedColormap.from_list("score", ["#eef3f8", "#9ecae1", "#3182bd", "#08306b"])
    G = np.array([0, 0, -1.0])
    res = {n: predict(n, args.yaw, args.model, args.clouds) for n in dict.fromkeys(args.objects + [args.gif_object])}

    n_obj = len(args.objects)
    fig = plt.figure(figsize=(3.2 * n_obj, 6.6), dpi=100)
    for i, n in enumerate(args.objects):
        r = res[n]
        j = int(np.argmax([b[0] for b in r["best"]]))          # the target the network likes best
        _, T, w, _ = r["best"][j]
        g = show_pose(r["cfg"], r["obj"], r["rolls"][j])[:3, :3].T @ G
        vd = r["obj"].qr.T_obj_qr[:3, 0]
        name = " ".join(n.split("_")[:4])
        draw(fig.add_subplot(2, n_obj, i + 1, projection="3d", computed_zorder=False), r["obj"], r["pts"],
             r["labels"][j], vd, cmap, g=g, title=f"{name}\n(target roll {np.degrees(r['rolls'][j]):.0f}\u00b0)")
        draw(fig.add_subplot(2, n_obj, n_obj + i + 1, projection="3d", computed_zorder=False), r["obj"], r["pts"],
             r["q"][j], vd, cmap, T, w, g)
    fig.text(0.006, 0.73, "way A: checks on\n~1,000 candidates (~25 s)", rotation=90, va="center", fontsize=9)
    fig.text(0.006, 0.28, "way B: network (7 ms)\n+ its top grasp", rotation=90, va="center", fontsize=9)
    fig.text(0.5, 0.008, "unseen objects   blue: good for this target   pale: fails a check   grey: no candidate "
             "(way A)   red: QR   orange: gripper   arrow: gravity at the target", ha="center", fontsize=8, color="0.3")
    fig.subplots_adjust(0.03, 0.03, 0.99, 0.95, 0.0, 0.0)
    fig.savefig(ROOT / "media" / "way_b_objects.png")
    plt.close(fig)

    r = res[args.gif_object]
    vd = r["obj"].qr.T_obj_qr[:3, 0]
    frames = []
    for j in range(12):
        _, T, w, _ = r["best"][j]
        g = show_pose(r["cfg"], r["obj"], r["rolls"][j])[:3, :3].T @ G
        fig = plt.figure(figsize=(7.2, 3.9), dpi=90)
        draw(fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False), r["obj"], r["pts"], r["labels"][j], vd,
             cmap, g=g, title="way A (checks, ~25 s)")
        draw(fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False), r["obj"], r["pts"], r["q"][j], vd,
             cmap, T, w, g, title="way B (network, 7 ms) + top grasp")
        fig.text(0.02, 0.95, f"target: QR to the scanner, rolled {np.degrees(r['rolls'][j]):3.0f}°", fontsize=10)
        fig.subplots_adjust(0.0, 0.0, 1.0, 0.9, 0.0, 0.0)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    pal = Image.fromarray(np.vstack([frames[0], frames[6]])).quantize(160, dither=Image.Dither.NONE)
    imgs = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(ROOT / "media" / "way_b_targets.gif", save_all=True, append_images=imgs[1:], duration=700, loop=0,
                 optimize=True)
    print("wrote media/way_b_objects.png and media/way_b_targets.gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
