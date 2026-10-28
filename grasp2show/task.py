"""Grasps, the 'show' target pose, and the scripted pick -> move -> show episode.

Frames used throughout:
  world  - z up, robot base at origin.
  obj    - object body frame (box centre, or the scanned mesh frame: z up, base at z=0).
  qr     - QR patch frame: x = outward normal, origin at the patch centre.
  tcp    - gripper tool frame: z = approach direction (out of the palm),
           y = finger closing axis, origin midway between the fingertip pads.
A grasp is the constant transform T_obj_tcp. At any object pose T_world_obj, the
gripper must be at T_world_tcp = T_world_obj @ T_obj_tcp. This one equation is why a
grasp chosen at the start pose fixes the arm configuration at the target pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import SceneConfig
from .objects import ObjectModel
from .scene import start_pose  # noqa: F401  (re-exported)
from .transforms import frame_from_axes, make_T


@dataclass
class Grasp:
    name: str
    T_obj_tcp: np.ndarray
    width: float  # object thickness between the fingers (m); a hook: the finger gap it is held at
    kind: str = "pinch"  # "pinch" (closed on the object) or "hook" (fingers through a loop, gap kept)


def flipped(grasp: Grasp) -> Grasp:
    """The same grasp rotated 180 deg about the approach axis.

    A parallel-jaw gripper is symmetric under this rotation: both fingers swap places and
    touch the same two contact points. The arm, however, ends up in a different
    configuration (wrist joint 7 differs by pi), so one twin may hit a joint limit
    where the other does not. Planners should always consider both.
    """
    Rz = np.diag([-1.0, -1.0, 1.0])
    T = grasp.T_obj_tcp.copy()
    T[:3, :3] = T[:3, :3] @ Rz
    return Grasp(grasp.name + "_flipped", T, grasp.width, grasp.kind)


def scripted_grasps(cfg: SceneConfig) -> dict[str, Grasp]:
    """Two hand-designed top-down grasps on the box, used before ACRONYM is plugged in.

    'clear_qr'  : fingers on the +-y faces -> the QR (+x face) stays free.
    'covers_qr' : fingers on the +-x faces -> a finger pad lands on the QR.
    Both are perfectly stable for lifting; a stability-only planner cannot tell them apart.
    """
    hx, hy, hz = cfg.obj.half_size
    depth = 0.035  # how far below the top face the pad centres are
    down = np.array([0, 0, -1.0])
    p = np.array([0, 0, hz - depth])
    return {
        "clear_qr": Grasp("clear_qr", make_T(frame_from_axes(y=[0, 1, 0], z=down), p), 2 * hy),
        "covers_qr": Grasp("covers_qr", make_T(frame_from_axes(y=[1, 0, 0], z=down), p), 2 * hx),
    }


def show_pose(cfg: SceneConfig, obj: ObjectModel, roll_about_view: float = 0.0) -> np.ndarray:
    """Target object pose T_world_obj: QR centre at the camera's look-at point, facing it.

    The QR normal (x of the QR frame) points from the look-at point back to the camera. The
    remaining freedom is a rotation about that viewing axis; roll_about_view = 0 keeps the
    QR frame's z axis as close to vertical as possible. Sampling roll gives the 'target set'.
    Derivation: we want T_world_qr = T_world_obj @ T_obj_qr, so T_world_obj = T_world_qr @ inv(T_obj_qr).
    """
    cam, at = np.asarray(cfg.camera.pos), np.asarray(cfg.camera.look_at)
    n = (cam - at) / np.linalg.norm(cam - at)
    R0 = frame_from_axes(x=n, z=[0, 0, 1])
    c, s = np.cos(roll_about_view), np.sin(roll_about_view)
    R_roll = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # rotation about the QR normal
    T_world_qr = make_T(R0 @ R_roll, at)
    T_obj_qr = obj.T_obj_qr
    R_world_obj = T_world_qr[:3, :3] @ T_obj_qr[:3, :3].T
    return make_T(R_world_obj, at - R_world_obj @ T_obj_qr[:3, 3])


def body_pose(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> np.ndarray:
    b = data.body(body)
    return make_T(b.xmat.reshape(3, 3), b.xpos)


def site_pose(data: mujoco.MjData, site: str) -> np.ndarray:
    s = data.site(site)
    return make_T(s.xmat.reshape(3, 3), s.xpos)


