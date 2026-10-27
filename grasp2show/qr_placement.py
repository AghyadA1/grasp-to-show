"""Find a flat face on a (scanned, noisy) mesh to stick a QR patch on.

Scanned meshes have no exactly planar faces, so we:
  1. fit an oriented bounding box (OBB); its six faces are candidate planes;
  2. size the QR to fit the face (capped at max_side, rejected below min_side);
  3. cast a grid of rays at the patch area along -normal and measure where the real
     surface is; the face is accepted only if every ray hits and the hit heights vary by
     less than flat_tol. A patch over a curved or recessed region fails this test;
  4. pick the largest accepted QR, preferring side faces, never the bottom face.
The QR frame has x = outward normal; y, z span the patch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .transforms import make_T


@dataclass
class QRPlacement:
    T_obj_qr: np.ndarray
    side: float          # m, including the quiet zone
    flatness: float      # m, peak-to-peak surface height under the patch

    def to_dict(self) -> dict:
        return {"T_obj_qr": np.round(self.T_obj_qr, 6).tolist(), "side": round(self.side, 5),
                "flatness": round(self.flatness, 5)}

    @classmethod
    def from_dict(cls, d: dict) -> QRPlacement:
        return cls(np.asarray(d["T_obj_qr"]), float(d["side"]), float(d["flatness"]))


def find_qr_face(mesh: trimesh.Trimesh, min_side: float = 0.03, max_side: float = 0.045,
                 flat_tol: float = 0.002, grid: int = 7, up=(0, 0, 1)) -> QRPlacement | None:
    obb = mesh.bounding_box_oriented
    T, ext = obb.primitive.transform, obb.primitive.extents
    R, c0 = T[:3, :3], T[:3, 3]
    up = np.asarray(up, float)
    best, best_key = None, None
    for k in range(3):
        for sign in (1.0, -1.0):
            n = sign * R[:, k]
            if n @ up < -0.7:                     # bottom face: the object rests on it
                continue
            i, j = [a for a in range(3) if a != k]
            u, v = R[:, i], R[:, j]
            side = min(max_side, 0.8 * min(ext[i], ext[j]))
            if side < min_side:
                continue
            centre = c0 + n * ext[k] / 2
            a = np.linspace(-0.55, 0.55, grid) * side      # cover the patch + a small margin
            A, B = np.meshgrid(a, a)
            pts = centre + A.reshape(-1, 1) * u + B.reshape(-1, 1) * v
            origins = pts + 0.03 * n
            dirs = np.repeat(-n[None], len(origins), axis=0)
            locs, ray_idx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
            if len(np.unique(ray_idx)) < len(origins):
                continue                           # some rays miss: patch hangs over an edge
            h = (locs - centre) @ n                 # surface height above the OBB face
            flat = float(h.max() - h.min())
            if flat > flat_tol:
                continue
            # Patch frame: x = normal, y/z in the face; sit on the highest point of the surface.
            Rq = np.column_stack([n, u, np.cross(n, u)])
            Tq = make_T(Rq, centre + n * h.max())
            is_side = abs(n @ up) < 0.3
            key = (round(side, 3), is_side, -flat)
            if best_key is None or key > best_key:
                best, best_key = QRPlacement(Tq, side, flat), key
    return best


def find_qr_patch(mesh: trimesh.Trimesh, min_side: float = 0.03, max_side: float = 0.045,
                  flat_tol: float = 0.004, grid: int = 7, up=(0, 0, 1), n_candidates: int = 80,
                  seed: int = 0) -> QRPlacement | None:
    """A QR sticker spot on any shape (not only on the faces of the bounding box).

    Candidate centres are sampled on the surface; at each, the patch plane is the surface's
    tangent plane, oriented with one edge horizontal. The ray test is the same as in
    find_qr_face, with a looser flatness (a 3 cm sticker on a mug of 4 cm radius bulges by
    2.9 mm): every ray must hit, and the heights under the patch must vary by less than
    flat_tol. Bottom-facing spots are skipped; the largest, then side-facing, then flattest
    spot wins. The sticker sits on the highest point under it (its edges may float a few mm
    off a curved surface). The smallest size is tested first, larger ones only where it fits.
    """
    up = np.asarray(up, float)
    pts, fid = trimesh.sample.sample_surface(mesh, n_candidates, seed=seed)
    normals = mesh.face_normals[fid]
    best, best_key = None, None

    def test(c, n, u, v, side):
        a = np.linspace(-0.55, 0.55, grid) * side
        A, B = np.meshgrid(a, a)
        origins = c + A.reshape(-1, 1) * u + B.reshape(-1, 1) * v + 0.03 * n
        dirs = np.repeat(-n[None], len(origins), axis=0)
        locs, ray_idx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        if len(np.unique(ray_idx)) < len(origins):
            return None
        h = (locs - c) @ n
        flat = float(h.max() - h.min())
        return (flat, h.max()) if flat <= flat_tol and h.max() <= 0.02 else None

    for c, n in zip(pts, normals):
        if n @ up < -0.7:
            continue
        u = np.cross(up, n)
        if np.linalg.norm(u) < 1e-3:                      # top-facing: any horizontal edge
            u = np.cross([1.0, 0.0, 0.0], n)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        fit = None
        for side in (min_side, 0.5 * (min_side + max_side), max_side):
            r = test(c, n, u, v, side)
            if r is None:
                break
            fit = (side, *r)
        if fit is None:
            continue
        side, flat, top = fit
        is_side = abs(n @ up) < 0.3
        key = (round(side, 3), is_side, -flat)
        if best_key is None or key > best_key:
            Rq = np.column_stack([n, u, np.cross(n, u)])
            best, best_key = QRPlacement(make_T(Rq, c + n * top), side, flat), key
            if side >= max_side and is_side and flat < 0.5 * flat_tol:
                break                                    # cannot do meaningfully better
    return best
