"""Generated objects with handles (part 3): many shapes, many kinds of handle, exact collision.

Real handled objects are few (one of the 112 scanned objects can be hooked from the table) and a
scan's convex decomposition can partly fill a handle's hole. So objects are generated here, from
convex pieces only: the simulator's collision geometry is then exactly the object, and a hole is
a hole.

Body: 1-3 convex pieces (stacked or side by side), each one of
  - a superellipsoid (box, cylinder, sphere, octahedron and everything in between),
  - a frustum with 3-40 sides (prisms, pyramids, cones, cylinders; tapered either way),
  - a random convex polytope (12-40 points on a jittered ellipsoid: faceted pebbles),
cut flat at the bottom so it stands, base at z = 0.
Handle: a chain of convex segments (round or flat cross-section, 3-8 mm), one of
  - loop:     on the side, a C-, D- or ear-shaped loop between two points of the body,
  - bail:     over the top, like a bucket's handle,
  - ring:     a small loop on top, like a lid's ring,
  - bar:      two posts and a bar, on the top (suitcase, drawer),
  - tab:      a flat plate sticking out of the side with a hole in it, vertical or horizontal.
The opening is drawn from a range that straddles what the Franka's fingertips need (the
fingertips at 5 mm: 1.8 x 3.1 cm closed, about 0.9 x 1.8 cm for one finger), so some objects
are hookable and some are just too tight: both are useful. Whether an object is really hookable
is decided afterwards by the same pipeline as for scans (scripts/generate_hookable.py): fingertip
fit, lift test, hanging, and the hand clearing the table.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from .qr_placement import find_qr_patch

MIN_THICK, MAX_THICK = 0.003, 0.008      # handle cross-section half-size (m)


@dataclass
class Generated:
    name: str
    body: list[trimesh.Trimesh]
    handle: list[trimesh.Trimesh]
    params: dict = field(default_factory=dict)
    rgba: tuple = (0.7, 0.7, 0.7, 1.0)

    @property
    def pieces(self) -> list[trimesh.Trimesh]:
        return self.body + self.handle

    @property
    def mesh(self) -> trimesh.Trimesh:
        return trimesh.util.concatenate(self.pieces)


# -- body pieces --------------------------------------------------------------------------------
def _hull(points: np.ndarray) -> trimesh.Trimesh:
    """Convex hull as a closed mesh. Densely sampled curved shapes have many nearly coplanar
    points, which can leave qhull's output with slivers that are not watertight; points are
    merged to 0.01 mm and, if needed, jittered by a micrometre."""
    P = np.unique(np.round(points, 5), axis=0)
    rng = np.random.default_rng(0)
    for _ in range(6):
        h = trimesh.convex.convex_hull(P)
        if h.is_watertight:
            return h
        P = P + rng.normal(scale=1e-6, size=P.shape)
    return h


def _cut_bottom(points: np.ndarray, frac: float) -> np.ndarray:
    """Flatten the lowest part: z clipped at frac of the way up from the bottom."""
    lo, hi = points[:, 2].min(), points[:, 2].max()
    p = points.copy()
    p[:, 2] = np.maximum(p[:, 2], lo + frac * (hi - lo))
    return p


def superellipsoid(rng, size) -> np.ndarray:
    a, b, c = size
    e1, e2 = rng.uniform(0.1, 1.8, 2)              # <= 2: convex; 0.1 box-like, 1 round, 1.8 pinched
    u, v = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, 16), np.linspace(-np.pi, np.pi, 24))

    def f(w, e):
        return np.sign(np.cos(w)) * np.abs(np.cos(w)) ** e, np.sign(np.sin(w)) * np.abs(np.sin(w)) ** e

    cu, su = f(u, e1)
    cv, sv = f(v, e2)
    pts = np.stack([a * cu * cv, b * cu * sv, c * su], -1).reshape(-1, 3)
    return _cut_bottom(pts, rng.uniform(0.05, 0.3) if e1 > 0.6 else 0.0)


def frustum(rng, size) -> np.ndarray:
    a, b, c = size
    n = int(rng.choice([3, 4, 5, 6, 8, 12, 40]))
    taper = rng.uniform(0.4, 1.3)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, np.pi)
    ring = np.stack([a * np.cos(ang), b * np.sin(ang)], 1)
    bot = np.column_stack([ring, np.full(n, -c)])
    top = np.column_stack([ring * taper, np.full(n, c)])
    return np.vstack([bot, top])


def pebble(rng, size) -> np.ndarray:
    k = int(rng.integers(12, 41))
    d = rng.normal(size=(k, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pts = d * np.asarray(size) * rng.uniform(0.8, 1.15, (k, 1))
    return _cut_bottom(pts, rng.uniform(0.1, 0.3))


SHAPES = {"superellipsoid": superellipsoid, "frustum": frustum, "pebble": pebble}


def make_body(rng) -> tuple[list[trimesh.Trimesh], dict]:
    n = int(rng.choice([1, 1, 1, 2, 2, 3]))
    pieces, kinds = [], []
    z = 0.0
    for i in range(n):
        kind = str(rng.choice(list(SHAPES)))
        if i == 0:
            size = rng.uniform([0.03, 0.03, 0.03], [0.09, 0.09, 0.12])
        else:
            size = rng.uniform(0.4, 1.1, 3) * pieces[0].extents / 2
        P = SHAPES[kind](rng, size)
        P[:, 2] -= P[:, 2].min()
        if i == 0 or rng.random() < 0.7:            # stacked on top (overlapping 1 cm)
            P[:, 2] += max(z - 0.01, 0.0)
        else:                                       # beside the first piece, overlapping it
            side = rng.normal(size=2)
            side /= np.linalg.norm(side)
            P[:, :2] += side * (pieces[0].extents[:2].min() / 2) * rng.uniform(0.5, 0.9)
        piece = _hull(P)
        pieces.append(piece)
        kinds.append(kind)
        z = max(z, piece.bounds[1, 2])
    return pieces, {"pieces": kinds}


# -- handles ----------------------------------------------------------------------------------
def _frame_points(p0, p1, normal, half_w, half_t, round_: bool) -> np.ndarray:
    """Points of a convex segment from p0 to p1: a capsule-ish tube (round) or a flat bar whose
    thickness is along `normal`."""
    t = p1 - p0
    t /= max(np.linalg.norm(t), 1e-9)
    n = normal - (normal @ t) * t
    n /= max(np.linalg.norm(n), 1e-9)
    b = np.cross(t, n)
    if round_:
        ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        ring = np.cos(ang)[:, None] * n * half_t + np.sin(ang)[:, None] * b * half_t
    else:
        ring = np.array([n * sn * half_t + b * sb * half_w for sn in (-1, 1) for sb in (-1, 1)])
    ext = 0.4 * half_t * t                           # a little overlap between segments
    return np.vstack([p0 - ext + ring, p1 + ext + ring])


def _tube(path: np.ndarray, normal, half_w, half_t, round_: bool) -> list[trimesh.Trimesh]:
    return [_hull(_frame_points(a, b, normal, half_w, half_t, round_)) for a, b in itertools.pairwise(path)]


def _surface_radius(body: trimesh.Trimesh, z: float, direction: np.ndarray) -> float | None:
    """Distance from the vertical axis (x = y = 0) to the body's surface along a horizontal
    direction, at height z (None if the ray misses)."""
    o = np.array([[0.0, 0.0, z]]) + 0.0
    far = o + direction * 1.0
    loc, _, _ = body.ray.intersects_location(far, -direction[None], multiple_hits=False)
    if len(loc) == 0:
        return None
    return float(np.linalg.norm(loc[0, :2]))


def make_handle(rng, body: trimesh.Trimesh) -> tuple[list[trimesh.Trimesh], dict]:
    kind = str(rng.choice(["loop", "loop", "loop", "bail", "ring", "bar", "tab"]))
    round_ = bool(rng.random() < 0.6)
    half_t = rng.uniform(MIN_THICK, MAX_THICK)
    half_w = half_t if round_ else rng.uniform(1.5, 3.0) * half_t
    phi = rng.uniform(0, 2 * np.pi)
    ex = np.array([np.cos(phi), np.sin(phi), 0.0])            # outward, horizontal
    ey = np.array([-np.sin(phi), np.cos(phi), 0.0])           # tangential
    ez = np.array([0.0, 0.0, 1.0])
    top = body.bounds[1, 2]
    # the opening: inner size drawn around what the fingertips need (see the module docstring)
    gap = rng.uniform(0.012, 0.05)                            # inner width (off the body / across)
    span = rng.uniform(0.025, 0.08)                           # inner height (along the handle)
    p = {"kind": kind, "round": round_, "thickness": round(2 * half_t, 4), "gap": round(gap, 4),
         "span": round(span, 4)}
    pieces: list[trimesh.Trimesh] = []
    if kind == "loop":
        shape = str(rng.choice(["C", "D", "ear"]))
        p["shape"] = shape
        zc = rng.uniform(0.35, 0.8) * top
        z1, z2 = zc + (span / 2 + half_t), zc - (span / 2 + half_t)
        if z2 < half_t + 0.004 or z1 > top - 0.003:
            span = max(min(2 * (top - 0.003 - zc), 2 * (zc - 0.004)) - 2 * half_t, 0.015)
            z1, z2 = zc + span / 2 + half_t, zc - span / 2 - half_t
        r1, r2 = _surface_radius(body, z1, ex), _surface_radius(body, z2, ex)
        if r1 is None or r2 is None:
            return [], {**p, "failed": "no surface"}
        out = max(r1, r2) + gap + half_t
        if shape == "D":
            path = np.array([[r1 - 0.004, z1], [out, z1], [out, z2], [r2 - 0.004, z2]])
            path = np.vstack([np.linspace(path[i], path[i + 1], 4, endpoint=False) for i in range(3)] + [path[-1:]])
        else:
            bulge = 1.0 if shape == "C" else 1.6
            t = np.linspace(-np.pi / 2, np.pi / 2, 14)
            cz, hz = (z1 + z2) / 2, (z1 - z2) / 2 * (1.0 if shape == "C" else 0.8)
            base = min(r1, r2) - 0.004
            path = np.column_stack([base + (out - base) * np.cos(t) ** (1 / bulge), cz - hz * np.sin(t)])
            path = np.vstack([[r1 - 0.004, z1], path[1:-1], [r2 - 0.004, z2]])   # t runs from z1 down to z2
        P3 = path[:, :1] * ex + path[:, 1:] * ez
        pieces = _tube(P3, ey, half_w, half_t, round_)
    elif kind == "bail":
        zt = top - rng.uniform(0.0, 0.02)
        r1, r2 = _surface_radius(body, zt, ex), _surface_radius(body, zt, -ex)
        if r1 is None or r2 is None:
            return [], {**p, "failed": "no surface"}
        h = top - zt + span + half_t
        t = np.linspace(0, np.pi, 16)
        a = 0.5 * (r1 + r2) + half_t
        cx = 0.5 * (r1 - r2)
        P3 = (cx + a * np.cos(t))[:, None] * ex + (zt + h * np.sin(t))[:, None] * ez
        pieces = _tube(P3, ey, half_w, half_t, round_)
    elif kind == "ring":
        R = gap / 2 + half_t + 0.006
        h = span
        t = np.linspace(0, np.pi, 12)
        c0 = rng.uniform(-0.3, 0.3) * body.extents[0] * ex
        z0 = _top_at(body, c0) - 0.004
        P3 = c0 + (R * np.cos(t))[:, None] * ex + (z0 + h * np.sin(t))[:, None] * ez
        pieces = _tube(P3, ey, half_w, half_t, round_)
    elif kind == "bar":
        L = max(gap, 0.02) + 0.04
        c0 = np.zeros(3)
        za, zb = _top_at(body, c0 - L / 2 * ex), _top_at(body, c0 + L / 2 * ex)
        hb = max(za, zb) + span * 0.6 + half_t
        P3 = np.array([c0 - L / 2 * ex + (za - 0.004) * ez, c0 - L / 2 * ex + hb * ez,
                       c0 + L / 2 * ex + hb * ez, c0 + L / 2 * ex + (zb - 0.004) * ez])
        pieces = _tube(P3, ey, half_w, half_t, round_)
    else:                                                      # tab with a hole
        vertical = bool(rng.random() < 0.6)
        p["vertical"] = vertical
        zc = rng.uniform(0.3, 0.8) * top
        r = _surface_radius(body, zc, ex)
        if r is None:
            return [], {**p, "failed": "no surface"}
        hw, hh = gap / 2, min(span, 0.05) / 2
        rim = rng.uniform(0.006, 0.014)
        up = ez if vertical else ey                            # in-plane "height" of the hole
        n = ey if vertical else ez                             # plate normal (the hole's axis)
        c = (r + 0.004 + rim + hw) * ex + zc * ez
        corners = [c + sx * (hw + rim / 2) * ex + sy * (hh + rim / 2) * up for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        frame = np.array(corners + corners[:1])
        root = np.array([(r - 0.006) * ex + zc * ez, c - (hw + rim / 2) * ex])
        pieces = _tube(frame, n, rim / 2, half_t, False) + _tube(root, n, hh, half_t, False)
    return pieces, p


def _top_at(body: trimesh.Trimesh, xy: np.ndarray) -> float:
    o = np.array([[xy[0], xy[1], body.bounds[1, 2] + 0.5]])
    loc, _, _ = body.ray.intersects_location(o, np.array([[0, 0, -1.0]]), multiple_hits=False)
    return float(loc[0, 2]) if len(loc) else float(body.bounds[1, 2])


# -- the whole object ---------------------------------------------------------------------------
def generate(seed: int) -> Generated | None:
    """One object from a seed; None if it came out unusable (no handle could be attached, it
    would tip over, or no spot for the QR sticker)."""
    rng = np.random.default_rng(seed)
    body_pieces, bp = make_body(rng)
    body = trimesh.util.concatenate(body_pieces)
    handle, hp = make_handle(rng, body)
    if not handle:
        return None
    name = f"gen_{seed:05d}"
    hue = rng.uniform(0, 1)
    rgba = tuple(float(x) for x in (*_hsv(hue, rng.uniform(0.35, 0.8), rng.uniform(0.55, 0.95)), 1.0))
    g = Generated(name, body_pieces, handle, {"seed": seed, "body": bp, "handle": hp}, rgba)
    lo = g.mesh.bounds[0]
    if lo[2] < -1e-6:                                         # nothing below the base
        return None
    return g


def _hsv(h, s, v):
    import colorsys

    return colorsys.hsv_to_rgb(h, s, v)


def mass_properties(g: Generated, density: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, centre of mass and inertia (about the COM) of the pieces as solids (overlaps between
    pieces are counted twice: a few percent)."""
    m_tot, c_acc, parts = 0.0, np.zeros(3), []
    for pc in g.pieces:
        pc = pc.copy()
        pc.density = density
        m_tot += pc.mass
        c_acc += pc.mass * pc.center_mass
        parts.append((pc.mass, pc.center_mass, pc.moment_inertia))
    com = c_acc / m_tot
    inertia = np.zeros((3, 3))
    for m, c, I in parts:
        d = c - com
        inertia += I + m * (d @ d * np.eye(3) - np.outer(d, d))
    return m_tot, com, inertia


def stands(g: Generated, com: np.ndarray, margin: float = 0.005) -> bool:
    """Does it stand on its base? The COM's projection must be inside the footprint (the hull of
    the points within 2 mm of the table) with a margin."""
    from shapely.geometry import MultiPoint, Point

    v = g.mesh.vertices
    foot = v[v[:, 2] < 0.002][:, :2]
    if len(foot) < 3:
        return False
    poly = MultiPoint(foot).convex_hull
    return poly.area > 4e-4 and poly.buffer(-margin).contains(Point(com[:2]))


def write(g: Generated, root: Path, density: float, mass_range=(0.05, 1.5)) -> dict | None:
    """Write the object as an asset folder (model.obj, convex model_collision_i.obj, model.xml,
    props.json with mass, COM, inertia, QR placement, colour). Returns props, or None if it does
    not stand or has no QR spot."""
    m, com, inertia = mass_properties(g, density)
    if not stands(g, com):
        return None
    scale = float(np.clip(m, *mass_range) / m)                 # keep the mass in the usual range
    m, inertia = m * scale, inertia * scale
    body = trimesh.util.concatenate(g.body)
    qr = find_qr_patch(body, n_candidates=60, seed=g.params["seed"])
    if qr is None:
        return None
    d = root / g.name
    d.mkdir(parents=True, exist_ok=True)
    g.mesh.export(d / "model.obj")
    files = []
    for i, pc in enumerate(g.pieces):
        f = f"model_collision_{i}.obj"
        pc.export(d / f)
        files.append(f)
    (d / "model.xml").write_text("<mujoco>\n" + "".join(f'  <mesh file="{f}"/>\n' for f in files) + "</mujoco>\n")
    props = {"mass": m, "com": com.tolist(), "inertia": inertia.tolist(), "qr": qr.to_dict(), "rgba": list(g.rgba),
             "params": g.params, "density": density * scale}
    (d / "props.json").write_text(json.dumps(props, indent=1))
    return props
