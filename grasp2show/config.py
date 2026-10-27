"""Scene and task parameters, kept in one place so experiments are reproducible.

Units: metres, kilograms, seconds, newtons, radians. World frame: z up, robot base
at the origin, table top at z = 0, the robot faces +x.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObjectConfig:
    """The object to pick. asset=None: the primitive box (QR on its +x face).
    asset="<Google Scanned Objects name>": a scanned mesh (see grasp2show/object_set.json);
    its QR face is found automatically and its mass comes from the mesh volume x density."""

    asset: str | None = None
    density: float = 300.0  # kg/m^3, scanned objects only
    name: str = "obj"  # MuJoCo body name
    half_size: tuple[float, float, float] = (0.025, 0.025, 0.06)  # 5 x 5 x 12 cm
    mass: float = 0.15
    # MuJoCo friction = (sliding mu, torsional [m], rolling [m]).
    # The torsional coefficient is an effective contact-patch radius; it is what lets
    # a two-finger grasp resist torque about the finger axis (the soft-finger model
    # used later in check 2). With condim=3 that torque could not be resisted at all.
    # mu = 0.5: plastic fingertips on cardboard / printed packaging.
    friction: tuple[float, float, float] = (0.5, 0.005, 0.0001)
    condim: int = 4
    start_xy: tuple[float, float] = (0.50, -0.15)  # on the table; height from the mesh
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


def random_camera(rng) -> CameraConfig:
    """A random scanner pose (part 3: the robot must show the code to a camera that moves).

    Look-at point: the default one moved by up to +-10 cm in x and y and +-8 cm in height.
    The camera looks at it from a distance of 24-36 cm, from a direction within 60 deg of the
    default one (+y, facing the robot's side of the table) horizontally and between 10 deg
    below and 35 deg above the horizontal (the default is 16 deg above).
    """
    d0 = CameraConfig()
    at = np.asarray(d0.look_at) + rng.uniform([-0.10, -0.10, -0.08], [0.10, 0.10, 0.08])
    az, el = np.radians(rng.uniform(-60, 60)), np.radians(rng.uniform(-10, 35))
    dist = rng.uniform(0.24, 0.36)
    pos = at + dist * np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    return CameraConfig(pos=tuple(np.round(pos, 4)), look_at=tuple(np.round(at, 4)))


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
    # Contact time constant (s) for fingers and the object (MuJoCo default 0.02). MuJoCo's soft
    # friction lets a loaded contact creep; the creep distance grows with the impulse carried,
    # so a held box slid 13 mm in a 2.5 s transit at only 0.12 g. A 5 ms time constant (still
    # >= 2 x timestep, as MuJoCo requires for stability) cuts the creep about 5x.
    grasp_solref: float = 0.005
    table_half_size: tuple[float, float, float] = (0.45, 0.6, 0.02)
    table_center_xy: tuple[float, float] = (0.45, 0.0)

    @property
    def show_distance(self) -> float:
        """Distance from the scanner camera to the point where the QR should be."""
        return float(np.linalg.norm(np.subtract(self.camera.look_at, self.camera.pos)))
