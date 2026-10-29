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


def spike(rng, base: np.ndarray, normal: np.ndarray, r: float, length: float) -> np.ndarray:
    """A cone: a small disc on the surface and an apex sticking out along the normal."""
    u, v = _basis(normal)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    disc = base - 0.004 * normal + r * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
    tip = base + length * (normal + rng.normal(scale=0.25, size=3))
    return np.vstack([disc, tip])


def _basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = d / np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


CONTAINERS_FROM_SEED = 700   # seeds below (the first 500 objects) were generated before containers existed: unchanged


def make_container(rng, profile: str | None = None) -> tuple[list[trimesh.Trimesh], dict]:
    """An open container (cup, mug, bowl, bucket, vase): a thin wall around a vertical axis, open
    at the top, on a solid bottom. Real mugs are hollow, and a cloud of a cup's inside is unlike
    any solid body, so the network must see some. The wall is cut into convex pieces (angular
    sectors x height bands), each the hull of its inner and outer corners."""
    h = rng.uniform(0.06, 0.18)
    r_bot = rng.uniform(0.03, 0.07)
    profile = profile or str(rng.choice(["straight", "taper", "flare", "bowl", "belly"]))
    r_top = {"straight": r_bot, "taper": r_bot * rng.uniform(0.7, 0.95), "flare": r_bot * rng.uniform(1.1, 1.5),
             "bowl": r_bot * rng.uniform(1.3, 1.8), "belly": r_bot * rng.uniform(0.8, 1.0)}[profile]
    wall = rng.uniform(0.003, 0.008)
    bottom = rng.uniform(0.004, 0.012)
    n_ang = int(rng.choice([12, 16, 20, 24]))
    n_z = 1 if profile in ("straight", "taper", "flare") else 4
    ell = rng.uniform(0.8, 1.0)                                   # slightly oval, sometimes

    def r_at(z):
        t = z / h
        r = r_bot + (r_top - r_bot) * t
        if profile == "bowl":
            r = r_bot + (r_top - r_bot) * np.sqrt(t)
        if profile == "belly":
            r = r + 0.25 * r_bot * np.sin(np.pi * t)
        return r

    pieces = []
    zs = np.linspace(bottom * 0.5, h, n_z + 1)
    ang = np.linspace(0, 2 * np.pi, n_ang + 1)
    for z0, z1 in itertools.pairwise(zs):
        for a0, a1 in itertools.pairwise(ang):
            pts = []
            for z in (z0, z1):
                for a in (a0, a1):
                    for r in (r_at(z), r_at(z) - wall):
                        pts.append([r * np.cos(a), ell * r * np.sin(a), z])
            pieces.append(_hull(np.array(pts)))
    disc = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    rb = r_at(0.0)
    pieces.append(_hull(np.array([[rb * np.cos(a), ell * rb * np.sin(a), z] for a in disc for z in (0.0, bottom)])))
    return pieces, {"style": "container", "profile": profile, "height": round(h, 3), "wall": round(wall, 4),
                    "rim_radius": round(float(r_top), 3)}


def make_body(rng) -> tuple[list[trimesh.Trimesh], dict]:
    """1-4 convex pieces: stacked, side by side or clustered (an organic lump), or a twisted stack
    of slabs; sometimes with spikes."""
    style = str(rng.choice(["single", "stack", "cluster", "twist", "single", "stack"]))
    pieces, kinds = [], []
    if style == "twist":
        n_side = int(rng.choice([3, 4, 5, 6]))
        a, b = rng.uniform(0.03, 0.07, 2)
        h = rng.uniform(0.08, 0.2)
        k = int(rng.integers(3, 7))
        turn = rng.uniform(0.3, 1.2)
        for i in range(k):
            ang = np.linspace(0, 2 * np.pi, n_side, endpoint=False) + turn * i / k
            sc = 1.0 + 0.25 * np.sin(np.pi * i / (k - 1)) * rng.choice([-1, 1])
            ring = np.stack([a * sc * np.cos(ang), b * sc * np.sin(ang)], 1)
            z0, z1 = h * i / k, h * (i + 1) / k + 0.004
            pieces.append(_hull(np.vstack([np.column_stack([ring, np.full(n_side, z0)]),
                                           np.column_stack([ring, np.full(n_side, z1)])])))
            kinds.append("slab")
    else:
        n = {"single": 1, "stack": int(rng.integers(2, 4)), "cluster": int(rng.integers(2, 5))}[style]
        z = 0.0
        for i in range(n):
            kind = str(rng.choice(list(SHAPES)))
            if i == 0:
                size = rng.uniform([0.03, 0.03, 0.03], [0.09, 0.09, 0.12])
            else:
                size = rng.uniform(0.45, 1.1, 3) * pieces[0].extents / 2
            P = SHAPES[kind](rng, size)
            P[:, 2] -= P[:, 2].min()
            if style == "stack" or i == 0:
                P[:, 2] += max(z - 0.01, 0.0)
            else:                                   # clustered around the first piece, overlapping it
                side = rng.normal(size=2)
                side /= np.linalg.norm(side)
                P[:, :2] += side * (pieces[0].extents[:2].min() / 2) * rng.uniform(0.4, 0.9)
                P[:, 2] += rng.uniform(0, 0.5) * pieces[0].extents[2]
            pieces.append(_hull(P))
            kinds.append(kind)
            z = max(z, pieces[-1].bounds[1, 2])
    info = {"style": style, "pieces": kinds}
    if rng.random() < 0.25:                         # spikes, not on the bottom
        body = trimesh.util.concatenate(pieces)
        pts, fid = trimesh.sample.sample_surface(body, 40, seed=int(rng.integers(1 << 31)))
        nrm = body.face_normals[fid]
        sel = [i for i in range(len(pts)) if nrm[i, 2] > -0.3 and pts[i, 2] > 0.01][: int(rng.integers(3, 10))]
        for i in sel:
            pieces.append(_hull(spike(rng, pts[i], nrm[i], rng.uniform(0.004, 0.01), rng.uniform(0.01, 0.025))))
        info["spikes"] = len(sel)
    return pieces, info


# -- handles ------------------------------------------------------------------------------------
# A handle is a curve from one attachment point to another, in a plane: the base shape (arc,
# rectangle, triangle, random polygon, superellipse, pointed arch) times a perturbation
# (none, squiggle, zigzag, doodle = random low-frequency wobble, a little 3-D twist), swept with a
# cross-section (round, flat strap, square, triangle; the thickness can swell and taper), with
# optional spikes on its outer side. The base shape's opening is what the fingers need; the
# perturbations are kept to a fraction of it, and the pipeline has the last word anyway.
BASES = ["arc", "rect", "triangle", "polygon", "superellipse", "arch"]
WIGGLES = ["none", "none", "squiggle", "zigzag", "doodle", "twist"]
SECTIONS = ["round", "round", "strap", "square", "triangle"]


def base_curve(rng, kind: str, L: float, H: float, n: int = 40) -> np.ndarray:
    """Points (s, h) from (0, 0) to (L, 0), bulging to about h = H (h > 0 is away from the body)."""
    t = np.linspace(0, 1, n)
    if kind == "arc":
        a = np.pi * t
        return np.column_stack([L / 2 - L / 2 * np.cos(a), H * np.sin(a)])
    if kind == "superellipse":
        e = rng.uniform(0.25, 0.8)                                 # < 1: boxy, ~1: elliptic
        a = np.pi * t
        c, s_ = np.cos(a), np.sin(a)
        return np.column_stack([L / 2 - L / 2 * np.sign(c) * np.abs(c) ** e, H * np.abs(s_) ** e])
    if kind == "arch":                                             # pointed (gothic) arch
        a = np.pi * t
        return np.column_stack([L / 2 - L / 2 * np.cos(a), H * np.sin(a) ** 0.6 * (1 + 0.25 * (1 - np.abs(np.cos(a))))])
    if kind == "rect":
        V = np.array([[0, 0], [0, H], [L, H], [L, 0]], float)
    elif kind == "triangle":
        V = np.array([[0, 0], [rng.uniform(0.2, 0.8) * L, H * rng.uniform(1.2, 1.6)], [L, 0]], float)
    else:                                                          # random polygon over the chord
        k = int(rng.integers(3, 7))
        ang = np.sort(rng.uniform(0.15, np.pi - 0.15, k))[::-1]
        V = np.vstack([[0, 0], np.column_stack([L / 2 + L / 2 * np.cos(ang), H * np.sin(ang) * rng.uniform(0.8, 1.3, k)]),
                       [L, 0]])
    seg = np.linalg.norm(np.diff(V, axis=0), axis=1)
    u = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
    return np.column_stack([np.interp(t, u, V[:, 0]), np.interp(t, u, V[:, 1])])


def wiggle(rng, kind: str, P: np.ndarray, amp: float) -> tuple[np.ndarray, np.ndarray]:
    """Offset the curve along its in-plane normal (and, for "twist", out of the plane).
    Returns (2-D points, out-of-plane offsets). The ends stay put."""
    n = len(P)
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * t) ** 0.5                                 # nothing at the attachments
    tang = np.gradient(P, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    off, out = np.zeros(n), np.zeros(n)
    if kind == "squiggle":
        off = amp * np.sin(2 * np.pi * rng.uniform(2, 6) * t + rng.uniform(0, 6.3))
    elif kind == "zigzag":
        k = int(rng.integers(3, 9))
        off = amp * (2 * np.abs(((t * k) % 1) - 0.5) * 2 - 1)
    elif kind == "doodle":
        for f in range(1, 5):
            off += amp / f * rng.normal() * np.sin(np.pi * f * t + rng.uniform(0, 6.3))
    elif kind == "twist":
        out = amp * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t + rng.uniform(0, 6.3))
    return P + (off * env)[:, None] * nrm, out * env


def section_points(kind: str, half_t: float, half_w: float) -> np.ndarray:
    """Cross-section outline in (a, b): a = along the plane normal (the hole's axis), b = in plane."""
    if kind == "round":
        ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        return np.column_stack([half_t * np.cos(ang), half_t * np.sin(ang)])
    if kind == "strap":                                            # wide along the hole's axis
        return np.array([[sa * half_w, sb * half_t] for sa in (-1, 1) for sb in (-1, 1)])
    if kind == "square":
        return np.array([[sa * half_t, sb * half_t] for sa in (-1, 1) for sb in (-1, 1)])
    return np.array([[half_t, -half_t], [-half_t, -half_t], [0.0, half_t * 1.2]])   # triangle


def sweep(path3: np.ndarray, plane_n: np.ndarray, sec: np.ndarray, scale: np.ndarray) -> list[trimesh.Trimesh]:
    """Convex segments along a 3-D path; the section's a axis is the plane normal."""
    out = []
    for i in range(len(path3) - 1):
        p0, p1 = path3[i], path3[i + 1]
        t = p1 - p0
        t /= max(np.linalg.norm(t), 1e-9)
        a = plane_n - (plane_n @ t) * t
        a /= max(np.linalg.norm(a), 1e-9)
        b = np.cross(t, a)
        ext = 0.002 * t
        r0 = sec[:, :1] * a * scale[i] + sec[:, 1:] * b * scale[i]
        r1 = sec[:, :1] * a * scale[i + 1] + sec[:, 1:] * b * scale[i + 1]
        out.append(_hull(np.vstack([p0 - ext + r0, p1 + ext + r1])))
    return out


def _surface_radius(body: trimesh.Trimesh, z: float, direction: np.ndarray) -> float | None:
    """Distance from the vertical axis (x = y = 0) to the body's surface along a horizontal
    direction, at height z (None if the ray misses)."""
    o = np.array([[0.0, 0.0, z]]) + direction * 1.0
    loc, _, _ = body.ray.intersects_location(o, -direction[None], multiple_hits=False)
    if len(loc) == 0:
        return None
    return float(np.linalg.norm(loc[0, :2]))


def _top_at(body: trimesh.Trimesh, xy: np.ndarray) -> float | None:
    """Height of the body's top surface above (x, y); None if (x, y) is beside the body."""
    o = np.array([[xy[0], xy[1], body.bounds[1, 2] + 0.5]])
    loc, _, _ = body.ray.intersects_location(o, np.array([[0, 0, -1.0]]), multiple_hits=False)
    return float(loc[0, 2]) if len(loc) else None


def make_handle(rng, body: trimesh.Trimesh, phi: float | None = None, rim: float | None = None,
                where: str | None = None, base: str | None = None, wig: str | None = None,
                thick: tuple[float, float] | None = None) -> tuple[list[trimesh.Trimesh], dict]:
    """One handle: where (side, top, ring on a stem; on an open container: side, or a bail from
    rim to rim, rim = the rim's radius), base shape, wiggle, section, spikes."""
    where = where or (str(rng.choice(["side", "side", "side", "top", "top", "ring"])) if rim is None else
                      str(rng.choice(["side", "side", "side", "bail"])))
    base = base or str(rng.choice(BASES))
    wig = wig or str(rng.choice(WIGGLES))
    sec_kind = str(rng.choice(SECTIONS))
    half_t = rng.uniform(*thick) if thick else rng.uniform(MIN_THICK, MAX_THICK)
    half_w = rng.uniform(1.5, 3.0) * half_t
    phi = rng.uniform(0, 2 * np.pi) if phi is None else phi
    ex = np.array([np.cos(phi), np.sin(phi), 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    top = body.bounds[1, 2]
    # opening: biased towards what the fingertips need (two closed fingers: 1.8 x 3.1 cm)
    hookable_size = rng.random() < 0.75
    gap = rng.uniform(0.028, 0.06) if hookable_size else rng.uniform(0.012, 0.03)   # away from the body
    span = rng.uniform(0.04, 0.09) if hookable_size else rng.uniform(0.025, 0.045)  # along the body
    tilt = rng.uniform(-0.5, 0.5) if rng.random() < 0.4 else 0.0     # the handle's plane leans
    p = {"where": where, "base": base, "wiggle": wig, "section": sec_kind, "thickness": round(2 * half_t, 4),
         "gap": round(gap, 4), "span": round(span, 4), "tilt": round(tilt, 3)}
    if where == "side":
        zc = rng.uniform(0.45, 0.85) * top
        z1, z2 = min(zc + span / 2 + half_t, top - 0.004), max(zc - span / 2 - half_t, half_t + 0.004)
        r1, r2 = _surface_radius(body, z1, ex), _surface_radius(body, z2, ex)
        if r1 is None or r2 is None or z1 - z2 < 0.02:
            return [], {**p, "failed": "no surface"}
        A, B = (r1 - 0.004) * ex + z1 * ez, (r2 - 0.004) * ex + z2 * ez
        out_dir = ex
    elif where == "bail":                                            # bucket handle, rim to rim
        zr = top - 0.004
        A, B = -(rim - 0.002) * ex + zr * ez, (rim - 0.002) * ex + zr * ez
        out_dir = ez
    elif where == "top":
        L0 = span + 2 * half_t
        a, b = -L0 / 2 * ex, L0 / 2 * ex
        za, zb = _top_at(body, a), _top_at(body, b)
        if za is None or zb is None:
            return [], {**p, "failed": "top too small"}
        A, B = a + (za - 0.004) * ez, b + (zb - 0.004) * ez
        out_dir = ez
    else:                                                           # a ring on a short stem on top
        c0 = rng.uniform(-0.25, 0.25) * body.extents[0] * ex
        z0 = _top_at(body, c0)
        if z0 is None:
            return [], {**p, "failed": "top too small"}
        stem = rng.uniform(0.005, 0.02)
        A = c0 + (z0 - 0.004) * ez
        B = c0 + (z0 + stem) * ez
        out_dir = ez
    chord = B - A
    L = float(np.linalg.norm(chord))
    s_dir = chord / L
    h_dir = out_dir - (out_dir @ s_dir) * s_dir
    h_dir = h_dir / np.linalg.norm(h_dir) if np.linalg.norm(h_dir) > 1e-9 else ex   # (ring: set below)
    if where == "ring":                                             # closed loop above the stem
        t = np.linspace(0, 2 * np.pi, 48)
        R1, R2 = span / 2 + half_t, gap / 2 + half_t
        P2 = np.column_stack([R1 * np.sin(t), half_t + R2 * (1 - np.cos(t))])   # (s, h): bottom sits on the stem
        s_dir, h_dir = ex, ez
        A = B
    else:
        P2 = base_curve(rng, base, L, gap + 2 * half_t)
    P2, out = wiggle(rng, wig, P2, amp=min(gap, span) * rng.uniform(0.08, 0.2))
    n_plane = np.cross(s_dir, h_dir)
    if tilt:                                                        # lean the plane about the chord
        c, s_ = np.cos(tilt), np.sin(tilt)
        h_dir, n_plane = c * h_dir + s_ * n_plane, -s_ * h_dir + c * n_plane
    path = A + P2[:, :1] * s_dir + P2[:, 1:] * h_dir + out[:, None] * n_plane
    if where == "ring":
        stem_path = np.array([c0 + (z0 - 0.004) * ez, A + 1.5 * half_t * ez])
    tt = np.linspace(0, 1, len(path))
    swell = 1.0 + (rng.uniform(-0.35, 0.5) * np.sin(np.pi * tt) if rng.random() < 0.4 else 0.0)
    pieces = sweep(path, n_plane, section_points(sec_kind, half_t, half_w), np.broadcast_to(swell, tt.shape))
    if where == "ring":
        pieces += sweep(stem_path, n_plane, section_points("round", half_t * 1.3, half_t), np.ones(2))
    if rng.random() < 0.25:                                         # spikes on the outer side
        k = int(rng.integers(3, 8))
        tang = np.gradient(path, axis=0)
        mid = path.mean(0)
        for i in np.linspace(4, len(path) - 5, k).astype(int):
            o = path[i] - mid
            o -= (o @ n_plane) * n_plane
            o -= (o @ tang[i]) / (tang[i] @ tang[i]) * tang[i]
            if np.linalg.norm(o) < 1e-6:
                continue
            o /= np.linalg.norm(o)
            pieces.append(_hull(spike(rng, path[i] + o * half_t, o, half_t * 0.9, rng.uniform(0.006, 0.015))))
        p["spikes"] = k
    if where == "side" and float(body.ray.contains_points((path[len(path) // 2])[None])[0]):
        return [], {**p, "failed": "inside the body"}
    return pieces, p


# -- shapes added for self-training (seeds >= NEW_SHAPES_FROM_SEED) ----------------------------
NEW_SHAPES_FROM_SEED = 1200   # earlier seeds (the first 800 objects) are unchanged
FAMILIES = ["mug", "mug", "jhook", "knob", "lip", "crate", "crate", "classic", "classic"]


def _box(lo, hi) -> trimesh.Trimesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return _hull(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))


def make_jhook(rng, body: trimesh.Trimesh) -> tuple[list[trimesh.Trimesh], dict]:
    """A J-shaped hook on the side (a coat hook): out from the body, then curling up, open at its
    tip: nothing closes it, so no hole finder sees it; the finger goes into the bend of the J."""
    phi = rng.uniform(0, 2 * np.pi)
    ex, ez = np.array([np.cos(phi), np.sin(phi), 0.0]), np.array([0.0, 0.0, 1.0])
    top = body.bounds[1, 2]
    z = rng.uniform(0.3, 0.75) * top
    r = _surface_radius(body, z, ex)
    if r is None:
        return [], {"where": "jhook", "failed": "no surface"}
    half_t = rng.uniform(0.003, 0.008)
    arm = rng.uniform(0.01, 0.04)
    R = rng.uniform(0.012, 0.03) + half_t
    end = np.radians(rng.uniform(110, 200))
    x0 = r + arm
    seg = [np.array([r - 0.004, z]), np.array([x0, z])]
    phis = np.linspace(0, end, 18)[1:]
    curl = [np.array([x0 + R * np.sin(f), z + R - R * np.cos(f)]) for f in phis]
    P2 = np.array(seg + curl)
    path = P2[:, :1] * ex + P2[:, 1:] * ez
    n_plane = np.cross(ex, ez)
    sec_kind = str(rng.choice(SECTIONS))
    pieces = sweep(path, n_plane, section_points(sec_kind, half_t, rng.uniform(1.5, 3.0) * half_t), np.ones(len(path)))
    return pieces, {"where": "jhook", "base": "J", "wiggle": "none", "section": sec_kind,
                    "thickness": round(2 * half_t, 4), "gap": round(2 * R - 2 * half_t, 4), "span": round(R, 4),
                    "curl_deg": round(float(np.degrees(end)), 1)}


def make_knob(rng, body: trimesh.Trimesh) -> tuple[list[trimesh.Trimesh], dict]:
    """A mushroom knob on top (a lid's knob): a stem and a wider cap; a finger hooks under the cap
    beside the stem."""
    c0 = np.array([rng.uniform(-0.2, 0.2) * body.extents[0], rng.uniform(-0.2, 0.2) * body.extents[1], 0.0])
    z0 = _top_at(body, c0)
    if z0 is None:
        return [], {"where": "knob", "failed": "top too small"}
    rs, hs = rng.uniform(0.005, 0.012), rng.uniform(0.015, 0.04)
    rc, tc = rs + rng.uniform(0.012, 0.03), rng.uniform(0.008, 0.02)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)
    stem = _hull(np.vstack([c0 + rs * ring + [0, 0, z0 - 0.004], c0 + rs * ring + [0, 0, z0 + hs + 0.002]]))
    cap_shape = str(rng.choice(["disc", "dome", "ball"]))
    if cap_shape == "disc":
        cap = _hull(np.vstack([c0 + rc * ring + [0, 0, z0 + hs], c0 + rc * ring + [0, 0, z0 + hs + tc]]))
    else:
        u, v = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, 10), ang)
        sph = np.stack([np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), np.sin(u)], -1).reshape(-1, 3)
        if cap_shape == "dome":
            sph[:, 2] = np.maximum(sph[:, 2], 0.0)
        cap = _hull(c0 + sph * [rc, rc, tc] + [0, 0, z0 + hs + (0 if cap_shape == "dome" else tc)])
    return [stem, cap], {"where": "knob", "base": cap_shape, "wiggle": "none", "section": "round",
                         "thickness": round(2 * rs, 4), "gap": round(hs, 4), "span": round(rc - rs, 4)}


def make_lip(rng, body_info: dict) -> tuple[list[trimesh.Trimesh], dict]:
    """An outward lip around a container's rim (a flowerpot's, a bowl's): a finger hooks under it."""
    r = body_info["rim_radius"]
    top = body_info["height"]
    w, t = rng.uniform(0.008, 0.02), rng.uniform(0.003, 0.007)
    droop = rng.uniform(0.0, 0.01)
    n = 20
    ang = np.linspace(0, 2 * np.pi, n + 1)
    pieces = []
    for a0, a1 in itertools.pairwise(ang):
        pts = []
        for a in (a0, a1):
            d = np.array([np.cos(a), np.sin(a), 0.0])
            pts += [d * (r - 0.004) + [0, 0, top - t], d * (r - 0.004) + [0, 0, top],
                    d * (r + w) + [0, 0, top - t - droop], d * (r + w) + [0, 0, top - droop]]
        pieces.append(_hull(np.array(pts)))
    return pieces, {"where": "lip", "base": "rim", "wiggle": "none", "section": "strap",
                    "thickness": round(t, 4), "gap": round(w, 4), "span": round(droop, 4)}


def make_crate(rng) -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh], dict]:
    """An open box (a crate, a tote) with slot handholds cut through one or two walls near the
    top: body pieces (walls around the slots, bottom), handle pieces (none: the slot is the
    handle), parameters."""
    L, W, H = rng.uniform(0.12, 0.3), rng.uniform(0.1, 0.22), rng.uniform(0.08, 0.2)
    t = rng.uniform(0.004, 0.01)
    sw, sh = rng.uniform(0.05, 0.1), rng.uniform(0.02, 0.04)        # slot width, height
    below = rng.uniform(0.012, 0.035)                                  # slot top below the rim
    n_slots = int(rng.choice([1, 2]))
    pieces = [_box([-L / 2, -W / 2, 0], [L / 2, W / 2, t])]            # bottom
    pieces += [_box([-L / 2, -W / 2, 0], [L / 2, -W / 2 + t, H]),     # long walls (no slot)
               _box([-L / 2, W / 2 - t, 0], [L / 2, W / 2, H])]
    for k, side in enumerate((1, -1)):                                 # short walls, slots in 1 or 2
        x0, x1 = (L / 2 - t, L / 2) if side > 0 else (-L / 2, -L / 2 + t)
        if k < n_slots and sw < W - 2 * t - 0.02:
            zt = H - below
            zb = zt - sh
            if zb < t + 0.01:
                return [], [], {"failed": "box too low"}
            pieces += [_box([x0, -W / 2, 0], [x1, W / 2, zb]),                       # below the slot
                       _box([x0, -W / 2, zt], [x1, W / 2, H]),                        # above it
                       _box([x0, -W / 2, zb - 0.001], [x1, -sw / 2, zt + 0.001]),     # its sides
                       _box([x0, sw / 2, zb - 0.001], [x1, W / 2, zt + 0.001])]
        else:
            pieces.append(_box([x0, -W / 2, 0], [x1, W / 2, H]))
    return pieces, [], {"style": "crate", "where": "slot", "base": "slot", "wiggle": "none", "section": "strap",
                        "thickness": round(t, 4), "gap": round(sh, 4), "span": round(sw, 4), "slots": n_slots,
                        "height": round(H, 3)}


def generate_new(seed: int):
    """Objects of the families added for self-training: realistic mugs, J-hooks, knobs, lips,
    crates with slot handholds, and the classic generator. Returns (body, handle, body params,
    handle params, rng) or None."""
    rng = np.random.default_rng(seed + 888_000)
    fam = str(rng.choice(FAMILIES))
    if fam == "classic":
        return None
    if fam == "crate":
        body, handle, hp = make_crate(rng)
        return (body, handle, {"style": "crate", "family": fam}, hp, rng) if body else ("fail", fam)
    if fam in ("mug", "lip"):
        body, bp = make_container(rng, profile=str(rng.choice(["straight", "taper", "belly", "flare"])))
        bp["family"] = fam
        if fam == "mug":
            handle, hp = make_handle(rng, body=trimesh.util.concatenate(body), rim=bp["rim_radius"], where="side",
                                     base=str(rng.choice(["arc", "rect", "superellipse", "arch"])),
                                     wig=str(rng.choice(["none", "none", "none", "doodle"])), thick=(0.004, 0.007))
        else:
            handle, hp = make_lip(rng, bp)
        return (body, handle, bp, hp, rng) if handle else ("fail", fam)
    body, bp = make_body(rng)
    bp["family"] = fam
    b = trimesh.util.concatenate(body)
    handle, hp = make_jhook(rng, b) if fam == "jhook" else make_knob(rng, b)
    return (body, handle, bp, hp, rng) if handle else ("fail", fam)


# -- the whole object ---------------------------------------------------------------------------
def generate(seed: int) -> Generated | None:
    """One object from a seed; None if it came out unusable (no handle could be attached, it
    would tip over, or no spot for the QR sticker)."""
    if seed >= NEW_SHAPES_FROM_SEED:
        new = generate_new(seed)
        if isinstance(new, tuple) and new[0] == "fail":
            return None
        if new is not None:
            body_pieces, handle, bp, hp, rng = new
            hue = rng.uniform(0, 1)
            rgba = tuple(float(x) for x in (*_hsv(hue, rng.uniform(0.35, 0.8), rng.uniform(0.55, 0.95)), 1.0))
            g = Generated(f"gen_{seed:05d}", body_pieces, handle,
                          {"seed": seed, "body": bp, "handle": hp, "handles": [hp]}, rgba)
            if g.mesh.bounds[0, 2] < -1e-6 or not connected(g.pieces):
                return None
            return g
    rng = np.random.default_rng(seed)
    rim = None
    rng_c = np.random.default_rng(seed + 777_000)
    if seed >= CONTAINERS_FROM_SEED and rng_c.random() < 0.4:
        body_pieces, bp = make_container(rng_c)
        rim = bp["rim_radius"]
        rng = rng_c
    else:
        body_pieces, bp = make_body(rng)
    body = trimesh.util.concatenate(body_pieces)
    handle, hp = make_handle(rng, body, rim=rim)
    if not handle:
        return None
    hps = [hp]
    if rng.random() < 0.2:                          # a second handle, e.g. a pot's two ears
        h2, hp2 = make_handle(rng, body, phi=None, rim=rim)
        if h2:
            handle += h2
            hps.append(hp2)
    name = f"gen_{seed:05d}"
    hue = rng.uniform(0, 1)
    rgba = tuple(float(x) for x in (*_hsv(hue, rng.uniform(0.35, 0.8), rng.uniform(0.55, 0.95)), 1.0))
    g = Generated(name, body_pieces, handle, {"seed": seed, "body": bp, "handle": hp, "handles": hps}, rgba)
    lo = g.mesh.bounds[0]
    if lo[2] < -1e-6:                                         # nothing below the base
        return None
    if not connected(g.pieces):                               # every piece must hang together
        return None
    return g


def _touch(a: trimesh.Trimesh, b: trimesh.Trimesh, tol: float = 0.001) -> bool:
    """Do two convex pieces overlap or touch (within tol)? Exact enough for convex hulls: a
    vertex of one inside the other, or the closest surface points closer than tol."""
    if not (np.all(a.bounds[0] - tol <= b.bounds[1]) and np.all(b.bounds[0] - tol <= a.bounds[1])):
        return False
    if a.contains(b.vertices).any() or b.contains(a.vertices).any():
        return True
    pts = np.vstack([b.vertices, trimesh.sample.sample_surface(b, 200, seed=0)[0]])
    _, d, _ = trimesh.proximity.closest_point(a, pts)
    return bool(d.min() < tol)


def connected(pieces: list[trimesh.Trimesh]) -> bool:
    """Is the object one piece? (Each convex part touches the rest, starting from the first body
    part: a handle floating above its body, or a part of a cluster off on its own, fails.)"""
    seen, todo = {0}, [0]
    while todo:
        i = todo.pop()
        for j in range(len(pieces)):
            if j not in seen and _touch(pieces[i], pieces[j]):
                seen.add(j)
                todo.append(j)
    return len(seen) == len(pieces)


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
