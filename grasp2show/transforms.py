"""Small rigid-body helpers. Poses are 4x4 homogeneous matrices T_a_b (frame b in frame a)."""
from __future__ import annotations

import numpy as np


def make_T(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ p)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def frame_from_axes(x=None, y=None, z=None) -> np.ndarray:
    """Right-handed rotation matrix from two given axes (the third is their cross product).

    The second given axis is re-orthogonalised against the first, so the first is exact.
    """
    given = {k: np.asarray(v, float) for k, v in (("x", x), ("y", y), ("z", z)) if v is not None}
    if len(given) != 2:
        raise ValueError("give exactly two axes")
    (k1, a1), (k2, a2) = list(given.items())
    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 - a1 * (a1 @ a2)
    a2 = a2 / np.linalg.norm(a2)
    axes = {k1: a1, k2: a2}
    if "x" not in axes:
        axes["x"] = np.cross(axes["y"], axes["z"])
    elif "y" not in axes:
        axes["y"] = np.cross(axes["z"], axes["x"])
    else:
        axes["z"] = np.cross(axes["x"], axes["y"])
    return np.column_stack([axes["x"], axes["y"], axes["z"]])


def camera_xyaxes(pos, look_at, up=(0, 0, 1)) -> np.ndarray:
    """MuJoCo camera 'xyaxes' (right, up) for a camera at pos looking at look_at.

    A MuJoCo camera looks along its local -z axis, with +x right and +y up in the image.
    """
    f = np.asarray(look_at, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, f)
    return np.concatenate([right, cam_up])


def rotation_angle(R: np.ndarray) -> float:
    """Angle (rad) of a rotation matrix."""
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> MuJoCo quaternion (w, x, y, z)."""
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q
