"""Build the MuJoCo scene programmatically with MjSpec.

Scene = Franka Panda (MuJoCo Menagerie) + table + object with a QR patch + scanner camera.
The Menagerie files are used unchanged; everything we modify (gravity compensation,
TCP site, gripper actuator) is done here in code, so the changes are visible and reviewed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np

from .config import SceneConfig
from .objects import ObjectModel, object_from_config
from .qr import make_qr_texture
from .transforms import camera_xyaxes, mat_to_quat, rot_z

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "assets"

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
PANDA_BODIES = [f"link{i}" for i in range(1, 8)] + ["hand", "left_finger", "right_finger"]
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def menagerie_panda_dir() -> Path:
    """Locate menagerie/franka_emika_panda (env var MENAGERIE_PATH or third_party/)."""
    root = Path(os.environ.get("MENAGERIE_PATH", REPO_ROOT / "third_party" / "mujoco_menagerie"))
    d = root / "franka_emika_panda"
    if not (d / "panda.xml").exists():
        raise FileNotFoundError(
            f"Panda model not found in {d}. Run python scripts/get_menagerie.py or set MENAGERIE_PATH."
        )
    return d


def set_contact_options(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """MuJoCo contacts are soft: "static" friction still lets a held object creep slowly.
    Elliptic friction cones with a high impratio (friction-to-normal impedance ratio) make the
    friction constraint much stiffer; this is the setting MuJoCo recommends for grasping.
    Without it a 0.15 kg box creeps ~1 mm/s in a 20 N grip."""
    spec.option.timestep = cfg.timestep
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = cfg.impratio


def configure_hand(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """Apply our two hand modifications. Shared by the arm scene and the floating-gripper sim,
    so grasp labels and episodes use exactly the same gripper.

    1) Tool centre point (TCP) site: midway between the fingertip pads, on the approach axis.
       Grasp poses are poses of this frame.
    2) Gripper actuator: replace Menagerie's 0..255 command with a force-limited servo on the
       finger tendon. Tendon length L = 0.5*(q1 + q2) = gap/2, so a command in metres of
       finger gap needs gain kp/2. Tendon force F gives each finger a force of 0.5*F, hence
       forcerange = 2 * finger_force. While the fingers are blocked by an object the servo
       saturates, so each finger pushes with exactly finger_force newtons.
    """
    spec.body("hand").add_site(
        name="tcp", pos=[0, 0, cfg.tcp_offset], size=[0.006, 0, 0], rgba=[1, 0, 0, 0.6], group=4
    )
    g = cfg.gripper
    act = spec.actuator("actuator8")
    act.name = "gripper"
    act.ctrlrange = [0.0, g.max_width]
    act.gainprm[:3] = [g.kp / 2.0, 0, 0]
    act.biasprm[:3] = [0.0, -g.kp, -g.kv]
    act.forcerange = [-2.0 * g.finger_force, 2.0 * g.finger_force]


def robot_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    """Panda + hand with our modifications. Also used alone as the IK model."""
    spec = mujoco.MjSpec.from_file(str(menagerie_panda_dir() / "panda.xml"))
    set_contact_options(spec, cfg)
    # Gravity compensation on the arm. The real Franka controller compensates the arm's own
    # weight; without it the position servos sag a few mm under gravity. The held object's
    # weight is NOT compensated, as on the real robot.
    for name in PANDA_BODIES:
        spec.body(name).gravcomp = 1.0
    configure_hand(spec, cfg)
    # Keyframes refer to the original actuator count and qpos size; we set state ourselves.
    for key in list(spec.keys):
        spec.delete(key)
    return spec


class Scene(NamedTuple):
    model: mujoco.MjModel
    spec: mujoco.MjSpec
    obj: ObjectModel


def build_scene(cfg: SceneConfig | None = None, obj: ObjectModel | None = None) -> Scene:
    cfg = cfg or SceneConfig()
    spec = robot_spec(cfg)
    world = spec.worldbody
    o = cfg.obj
    obj = obj or object_from_config(o)

    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.global_.offwidth = max(1280, cfg.camera.width)
    spec.visual.global_.offheight = max(720, cfg.camera.height)

    # --- floor and table -------------------------------------------------------------
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.22, 0.25, 0.3], rgb2=[0.18, 0.2, 0.24], width=300, height=300)
    spec.add_material(name="floor", textures=["", "grid"], texrepeat=[6, 6], texuniform=True)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                   pos=[0, 0, -0.75], material="floor")
    th = cfg.table_half_size
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(th),
                   pos=[*cfg.table_center_xy, -th[2]], rgba=[0.62, 0.6, 0.58, 1])
    world.add_light(name="key", pos=[0.3, 0.6, 1.6], dir=[0.1, -0.3, -1], diffuse=[0.6, 0.6, 0.6],
                    castshadow=True)
    world.add_light(name="fill", pos=[1.2, -0.4, 1.0], dir=[-0.6, 0.2, -0.5], diffuse=[0.3, 0.3, 0.3],
                    castshadow=False)

    # --- object with QR patch --------------------------------------------------------
    tex_path = make_qr_texture(o.qr_payload, ASSET_DIR / f"qr_{o.qr_payload}.png")
    spec.add_texture(name="qr", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(tex_path))
    spec.add_material(name="qr", textures=["", "qr"], specular=0.0, shininess=0.0)

    T0 = start_pose(cfg, obj)
    body = world.add_body(name=obj.name, pos=T0[:3, 3], quat=mat_to_quat(T0[:3, :3]))
    body.add_freejoint(name=f"{obj.name}_free")
    obj.add_to_body(spec, body, qr_material="qr")

    # --- cameras ---------------------------------------------------------------------
    c = cfg.camera
    world.add_camera(name="scanner", pos=list(c.pos), fovy=c.fovy_deg,
                     quat=mat_to_quat(_cam_frame(c.pos, c.look_at)))
    # A visible marker for the scanner, so it shows up in the overview video.
    world.add_geom(name="scanner_marker", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.02, 0.02],
                   pos=list(c.pos), quat=mat_to_quat(_cam_frame(c.pos, c.look_at)),
                   rgba=[0.1, 0.1, 0.12, 1], contype=0, conaffinity=0)
    world.add_geom(name="scanner_pole", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[0.008, (c.pos[2] - 0.02) / 2, 0], pos=[c.pos[0], c.pos[1] + 0.03, (c.pos[2] - 0.02) / 2],
                   rgba=[0.3, 0.3, 0.32, 1], contype=0, conaffinity=0)

    ov_pos, ov_at = (1.45, -0.85, 0.85), (0.42, 0.05, 0.25)
    world.add_camera(name="overview", pos=list(ov_pos), fovy=45,
                     quat=mat_to_quat(_cam_frame(ov_pos, ov_at)))

    model = spec.compile()
    _check_joint_layout(model)
    return Scene(model, spec, obj)


def start_pose(cfg: SceneConfig, obj: ObjectModel) -> np.ndarray:
    """Object resting on the table at the configured (x, y, yaw); +1 mm to avoid initial overlap."""
    x, y = cfg.obj.start_xy
    T = np.eye(4)
    T[:3, :3] = rot_z(cfg.obj.start_yaw)
    T[:3, 3] = [x, y, obj.rest_z + 0.001]
    return T


def _cam_frame(pos, look_at) -> np.ndarray:
    """Camera rotation: columns = (image right, image up, backwards); MuJoCo looks along -z."""
    xy = camera_xyaxes(pos, look_at)
    x, y = xy[:3], xy[3:]
    return np.column_stack([x, y, np.cross(x, y)])


def _check_joint_layout(model: mujoco.MjModel) -> None:
    """The rest of the code assumes qpos[0:7] = arm, qpos[7:9] = fingers."""
    for i, name in enumerate(ARM_JOINTS + FINGER_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert model.jnt_qposadr[jid] == i, f"unexpected qpos layout for {name}"


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData, gripper_open: bool = True) -> None:
    """Arm at the Menagerie 'home' posture, gripper open, object at its start pose."""
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = HOME_Q
    width = model.actuator("gripper").ctrlrange[1] if gripper_open else 0.0
    data.qpos[7:9] = width / 2.0
    data.ctrl[:7] = HOME_Q
    data.ctrl[7] = width
    mujoco.mj_forward(model, data)
