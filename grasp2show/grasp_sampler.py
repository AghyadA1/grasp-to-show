"""Antipodal grasp sampling on a mesh (the candidate generator, as in Dex-Net / ACRONYM).

A two-finger grasp can hold an object by friction alone when the line between the two
contacts lies inside both friction cones ("antipodal"). With Coulomb friction mu, the
cone half-angle is atan(mu); we sample with a stricter angle so that candidates are
not all marginal. Steps:
  1. sample surface points p1 with outward normals n1;
  2. shoot a ray from p1 into the object along -n1 (perturbed inside the cone) and take
     the exit point p2 with normal n2;
  3. keep the pair if the ray direction is within cone_deg of both -n1 and +n2 and the
     width |p2 - p1| fits the gripper;
  4. for every kept pair, rotate the approach direction around the finger axis
     (n_approach angles) and shift the TCP along the approach by a random depth (negative =
     shallow, only the fingertips catch the object) -> full 6-DoF TCP poses.
The sampling cone (30 deg) is deliberately wider than the true friction cone
(atan(0.5) = 26.6 deg): like any real sampler, it produces some candidates that fail,
and the simulator labels them.
Collision with the object is filtered later in the simulator (GripperSim.in_collision).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class Candidates:
    T_obj_tcp: np.ndarray   # (N, 4, 4)
    width: np.ndarray       # (N,)
    p1: np.ndarray          # (N, 3) contact points
    p2: np.ndarray

    def __len__(self) -> int:
        return len(self.width)

    def subset(self, mask) -> Candidates:
        return Candidates(self.T_obj_tcp[mask], self.width[mask], self.p1[mask], self.p2[mask])


def _orthonormal_basis(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to each row of a (N,3)."""
    helper = np.where(np.abs(a[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    e1 = np.cross(a, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(a, e1)


def _outward_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Face normals, flipped if the scan's winding makes most of them point inwards."""
    n = mesh.face_normals
    out = mesh.triangles_center - mesh.centroid
    return -n if np.mean(np.sum(n * out, axis=1) > 0) < 0.5 else n


def sample_antipodal(mesh: trimesh.Trimesh, n_pairs: int, rng: np.random.Generator,
                     cone_deg: float = 30.0, min_width: float = 0.008, max_width: float = 0.075,
                     oversample: int = 8) -> tuple[np.ndarray, np.ndarray]:
    normals = _outward_normals(mesh)
    pts, fid = trimesh.sample.sample_surface(mesh, n_pairs * oversample, seed=int(rng.integers(1 << 31)))
    n1 = normals[fid]
    # Perturb the inward direction inside the cone.
    e1, e2 = _orthonormal_basis(-n1)
    theta = np.radians(cone_deg) * np.sqrt(rng.random(len(pts)))
    phi = 2 * np.pi * rng.random(len(pts))
    d = (-n1 * np.cos(theta)[:, None]
         + (e1 * np.cos(phi)[:, None] + e2 * np.sin(phi)[:, None]) * np.sin(theta)[:, None])
    origins = pts + d * 1e-4
    locs, ray_idx, tri = mesh.ray.intersects_location(origins, d, multiple_hits=False)
    p1, dd, p2, n2 = pts[ray_idx], d[ray_idx], locs, normals[tri]
    width = np.linalg.norm(p2 - p1, axis=1)
    cos_c = np.cos(np.radians(cone_deg)) - 1e-6
    ok = ((width > min_width) & (width < max_width)
          & (np.sum(dd * -n1[ray_idx], axis=1) >= cos_c) & (np.sum(dd * n2, axis=1) >= cos_c))
    p1, p2 = p1[ok][:n_pairs], p2[ok][:n_pairs]
    return p1, p2


def candidates_from_pairs(p1: np.ndarray, p2: np.ndarray, n_approach: int,
                          rng: np.random.Generator,
                          depth_range: tuple[float, float] = (-0.015, 0.01)) -> Candidates:
    axis = p2 - p1
    width = np.linalg.norm(axis, axis=1)
    axis /= width[:, None]
    e1, e2 = _orthonormal_basis(axis)
    centre = 0.5 * (p1 + p2)
    Ts, ws, a1, a2 = [], [], [], []
    for k in range(n_approach):
        ang = 2 * np.pi * k / n_approach + rng.random(len(axis)) * 2 * np.pi / n_approach
        z = e1 * np.cos(ang)[:, None] + e2 * np.sin(ang)[:, None]     # approach direction
        x = np.cross(axis, z)
        T = np.tile(np.eye(4), (len(axis), 1, 1))
        depth = rng.uniform(*depth_range, size=len(axis))[:, None]
        T[:, :3, 0], T[:, :3, 1], T[:, :3, 2], T[:, :3, 3] = x, axis, z, centre + depth * z
        Ts.append(T), ws.append(width), a1.append(p1), a2.append(p2)
    return Candidates(np.concatenate(Ts), np.concatenate(ws), np.concatenate(a1), np.concatenate(a2))
