"""Scene and task parameters, kept in one place so experiments are reproducible.

Units: metres, kilograms, seconds, newtons, radians. World frame: z up, robot base
at the origin, table top at z = 0, the robot faces +x.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObjectConfig:
    """A box with a QR patch on its +x face (the 'QR face')."""

    name: str = "box"
    half_size: tuple[float, float, float] = (0.025, 0.025, 0.06)  # 5 x 5 x 12 cm
    mass: float = 0.15
    # MuJoCo friction = (sliding mu, torsional [m], rolling [m]).
    # The torsional coefficient is an effective contact-patch radius; it is what lets
    # a two-finger grasp resist torque about the finger axis (the soft-finger model
    # used later in check 2). With condim=3 that torque could not be resisted at all.
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    condim: int = 4
    start_pos: tuple[float, float, float] = (0.50, -0.15, 0.06)  # resting on the table
    start_yaw: float = 0.0
    qr_payload: str = "G2S-0001"
    qr_size: float = 0.045  # side length of the printed QR patch (incl. quiet zone)
    qr_center_z: float = 0.02  # QR centre height on the +x face, in the object frame
    rgba: tuple[float, float, float, float] = (0.85, 0.55, 0.25, 1.0)


@dataclass
class CameraConfig:
    """The fixed 'scanner' camera the QR has to be shown to."""

    pos: tuple[float, float, float] = (0.45, 0.48, 0.52)
    look_at: tuple[float, float, float] = (0.45, 0.20, 0.44)
    fovy_deg: float = 45.0
    width: int = 640
    height: int = 480


@dataclass
class GripperConfig:
    """Force-limited position gripper, like Franka's grasp(width, speed, force).

    ctrl = commanded finger gap in metres. While the fingers are blocked by an object the
    servo saturates, so each finger pushes with exactly ``finger_force`` newtons.
    Knowing the normal force is what makes the friction checks computable.
    """

    finger_force: float = 20.0  # N per finger (Franka Hand: 70 N continuous, total)
    kp: float = 4000.0  # N/m, tendon-space stiffness (high, so the force clamp is active)
    kv: float = 100.0
    max_width: float = 0.08


@dataclass
class SceneConfig:
    obj: ObjectConfig = field(default_factory=ObjectConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    tcp_offset: float = 0.1034  # hand frame -> point between the fingertip pads (m)
    timestep: float = 0.002
    impratio: float = 10.0
    table_half_size: tuple[float, float, float] = (0.45, 0.6, 0.02)
    table_center_xy: tuple[float, float] = (0.45, 0.0)

    @property
    def show_distance(self) -> float:
        """Distance from the scanner camera to the point where the QR should be."""
        return float(np.linalg.norm(np.subtract(self.camera.look_at, self.camera.pos)))
