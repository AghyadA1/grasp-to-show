"""Do the Franka fingertips fit through a mug's handle? A picture, not a number.

For each handled object: the handle cut by the plane perpendicular to its hole, as the visual
mesh draws it (black outline) and as the simulator sees it (orange: the convex collision parts).
In the hole, the cross-section of the fingertips, from the simulator's own finger geometry (the
convex hull of each finger link plus its pads), placed and turned to keep the most clearance:
  blue   both fingers closed, cut 5 mm behind the fingertip point (TCP + 5 mm)
  green  one finger (the other stays outside the handle), same cut
Clearance > 0: fits with that much room; < 0: overlaps by that much.
Usage:  $env:G2S_OBJECT_SET="v2"; python scripts/figure_hook_fit.py      -> media/hook_fit.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import grasp2show.hooks as H
from grasp2show.config import ObjectConfig, SceneConfig
from grasp2show.gripper_sim import GripperSim
from grasp2show.objects import object_from_config

MUGS = ["Cole_Hardware_Mug_Classic_Blue", "ACE_Coffee_Mug_Kristen_16_oz_cup",
        "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White"]


def finger_sections(z: float):
    """(two closed fingers, one finger) cross-sections at height z in the TCP frame (x, y)."""
    o = object_from_config(ObjectConfig())
    sim = GripperSim(SceneConfig(), o)
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
        w = (d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g]
        P = (w - t) @ R
        s = trimesh.convex.convex_hull(P).section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if s is not None:
            per.setdefault(b, []).append(MultiPoint(s.vertices[:, :2]).convex_hull)
    fingers = [unary_union(v) for v in per.values()]
    return unary_union(fingers), fingers[0]


def handle_section(obj, centre, axis, half=0.045):
    u, v = H._basis(axis)

    def to2d(V):
        return np.stack([(V - centre) @ u, (V - centre) @ v], 1)

    vis = obj.mesh.section(plane_origin=centre, plane_normal=axis)
    vis_lines = [to2d(vis.vertices[e]) for e in vis.vertex_nodes] if vis is not None else []
    parts = []
    for f in obj.collision_files:
        s = trimesh.load(f, force="mesh").convex_hull.section(plane_origin=centre, plane_normal=axis)
        if s is not None and len(s.vertices) >= 3:
            parts.append(MultiPoint(to2d(s.vertices)).convex_hull)
    solid = unary_union(parts) if parts else Polygon()
    window = Point(0, 0).buffer(half)
    free = window.difference(solid)
    hole = free.geoms[np.argmin([g.distance(Point(0, 0)) for g in free.geoms])] if free.geom_type == "MultiPolygon" \
        else free
    return vis_lines, solid, hole


def best_fit(poly, hole, solid):
    """Place and turn poly to keep the most clearance from the solid; returns (placed, clearance m)."""
    best, key = None, -np.inf
    c0 = poly.centroid
    xs = np.linspace(-0.012, 0.012, 13)
    for a in range(0, 180, 6):
        pr = affinity.rotate(poly, a, origin=c0)
        for dx in xs:
            for dy in xs:
                pt = affinity.translate(pr, dx - c0.x, dy - c0.y)
                if not pt.centroid.within(hole):
                    continue
                cl = pt.distance(solid) if not pt.intersects(solid) else -pt.intersection(solid).area ** 0.5
                if cl > key:
                    best, key = pt, cl
    return best, key


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    two, one = finger_sections(0.005)
    fig, axes = plt.subplots(1, len(MUGS), figsize=(4.2 * len(MUGS), 5.2))
    for ax, name in zip(axes, MUGS):
        obj = object_from_config(ObjectConfig(asset=name))
        H.ROD_RADIUS = 0.008
        loops = H.find_loops(obj.mesh, np.random.default_rng(0))
        if not loops:
            ax.set_title(f"{name[:28]}\nno hole found", fontsize=8)
            continue
        P = np.array([lp[0] for lp in loops])
        c = P[np.argmin(np.linalg.norm(P - P.mean(0), axis=1))]
        axis = loops[int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))][1]
        vis, solid, hole = handle_section(obj, c, axis)
        for g in getattr(solid, "geoms", [solid]):
            if not g.is_empty:
                ax.fill(*np.array(g.exterior.coords).T * 100, color="#e07a2f", alpha=0.45, lw=0)
        for seg in vis:
            ax.plot(*seg.T * 100, c="k", lw=1)
        lines = []
        for poly, col, lab in ((two, "#1d4e6e", "both fingers closed"), (one, "#3f8f6b", "one finger")):
            placed, cl = best_fit(poly, hole, solid)
            if placed is not None:
                ax.fill(*np.array(placed.exterior.coords).T * 100, color=col, alpha=0.75, lw=0)
            lines.append(f"{lab}: {'fits, ' if cl > 0 else 'overlaps, '}{cl * 1000:+.0f} mm")
        ax.set_title(f"{name.replace('_', ' ')[:30]}\n" + "\n".join(lines), fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel("cm")
    fig.text(0.5, 0.01, "Handle cut across its hole: black = visual mesh, orange = what the simulator collides with. "
             f"Fingertips 5 mm in front of the TCP: blue = both fingers closed ({(two.bounds[2] - two.bounds[0]) * 100:.1f}"
             f" x {(two.bounds[3] - two.bounds[1]) * 100:.1f} cm), green = one finger", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = ROOT / "media" / "hook_fit.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
