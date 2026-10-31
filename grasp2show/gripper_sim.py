"""Floating-gripper simulation: the ground truth for grasp stability (checks 1 and 2).

A Franka Hand without the arm, with the same force-limited finger actuator as the full
robot. The hand is a dynamic body welded to an invisible "mocap" target that we move.
(Moving the hand itself as a mocap body does not work: mocap bodies teleport with zero
velocity, so MuJoCo's velocity-level friction cannot drag the held object along, and it
slides out. The weld gives the hand a real velocity.) One free object, no table. This is the procedure ACRONYM
used to label its grasps (close, lift, shake, check the object is still held), rebuilt
in MuJoCo so labels and later evaluation share one physics engine.

The object can be given any initial orientation R_world_obj with gravity along world -z:
  - check 1 (lift): R = the start orientation (object as it rests on the table);
  - check 2 ground truth (later): R = the target orientation, where gravity acts in a
    different direction in the object frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import SceneConfig
from .motion import min_jerk
from .objects import ObjectModel
from .scene import configure_hand, menagerie_panda_dir, set_contact_options
from .transforms import inv_T, make_T, mat_to_quat, rotation_angle


@dataclass
class ShakeResult:
    success: bool
    closed_on_object: bool   # both fingers touched the object after closing
    slip_mm: float           # object motion relative to the TCP, closing -> end
    slip_deg: float


def _axis_angle(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class GripperSim:
    # timings (s) and shake amplitudes
    T_CLOSE, T_LIFT, T_SHAKE_TRANS, T_SHAKE_ROT, T_SETTLE = 0.4, 0.5, 0.5, 0.5, 0.2
    LIFT, SHAKE_AMP, SHAKE_ROT, SHAKE_HZ = 0.10, 0.02, 0.25, 2.0
    DROP_MM = 20.0   # object moved more than this relative to the TCP => it fell out / slid away

    def __init__(self, cfg: SceneConfig, obj: ObjectModel, table: bool = False):
        """table: a table top under the upright object (its base at the object's lowest point), for
        tests that start with the object standing (hanging.table_lift_test)."""
        self.cfg, self.obj = cfg, obj
        hand = mujoco.MjSpec.from_file(str(menagerie_panda_dir() / "hand.xml"))
        configure_hand(hand, cfg)
        world = mujoco.MjSpec()
        # Match the hand file's compiler/integrator settings so attaching it is conflict-free.
        world.compiler.degree = False
        world.compiler.autolimits = True
        world.option.integrator = hand.option.integrator
        set_contact_options(world, cfg)
        world.option.gravity = [0, 0, 0]
        world.worldbody.add_body(name="target", mocap=True)
        world.attach(hand, prefix="g_", frame=world.worldbody.add_frame())
        hb = world.body("g_hand")
        hb.quat = [1, 0, 0, 0]
        hb.add_freejoint(name="hand_free")
        for b in ("g_hand", "g_left_finger", "g_right_finger"):
            world.body(b).gravcomp = 1.0          # the weld moves the hand; it need not carry it
        # Weld target -> hand with identity relative pose: data = anchor(3), relpos(3),
        # relquat(4), torquescale(1), unused(1). Time constant 10 ms (>= 2 * timestep for stability).
        world.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1="target", name2="g_hand", solref=[0.01, 1.0],
                           data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        if table:
            z0 = float(obj.mesh.bounds[0, 2])
            world.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.4, 0.4, 0.02],
                                     pos=[0, 0, z0 - 0.02], friction=list(obj.friction))
        body = world.worldbody.add_body(name=obj.name)
        body.add_freejoint(name="obj_free")
        obj.add_to_body(world, body, qr_material=None, solref=cfg.grasp_solref)
        self.model = world.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.T_hand_tcp = make_T(np.eye(3), [0, 0, cfg.tcp_offset])
        self._obj_body = m.body(obj.name).id
        self._fingers = {m.body("g_left_finger").id, m.body("g_right_finger").id}
        self._gripper_bodies = self._fingers | {m.body("g_hand").id}
        self._obj_qadr = m.jnt_qposadr[m.joint("obj_free").id]
        self._hand_qadr = m.jnt_qposadr[m.joint("hand_free").id]
        self._fq = [m.jnt_qposadr[m.joint(n).id] for n in ("g_finger_joint1", "g_finger_joint2")]

    # -- helpers -------------------------------------------------------------------------
    def _reset(self, R_world_obj: np.ndarray, T_world_tcp: np.ndarray, open_width: float) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        m.opt.gravity[:] = 0.0
        d.qpos[self._obj_qadr:self._obj_qadr + 3] = 0.0
        d.qpos[self._obj_qadr + 3:self._obj_qadr + 7] = mat_to_quat(R_world_obj)
        for a in self._fq:
            d.qpos[a] = open_width / 2.0
        d.ctrl[0] = open_width
        self._set_hand(T_world_tcp)
        d.qpos[self._hand_qadr:self._hand_qadr + 3] = d.mocap_pos[0]   # start hand on its target
        d.qpos[self._hand_qadr + 3:self._hand_qadr + 7] = d.mocap_quat[0]
        mujoco.mj_forward(m, d)

    def _set_hand(self, T_world_tcp: np.ndarray) -> None:
        T = T_world_tcp @ inv_T(self.T_hand_tcp)
        self.data.mocap_pos[0] = T[:3, 3]
        self.data.mocap_quat[0] = mat_to_quat(T[:3, :3])

    def _contacts(self, min_depth: float = 0.0) -> tuple[set[int], bool]:
        """(gripper bodies touching the object, any penetration deeper than min_depth)."""
        m, d = self.model, self.data
        touching, deep = set(), False
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self._obj_body in (b1, b2):
                other = b2 if b1 == self._obj_body else b1
                if other in self._gripper_bodies:
                    touching.add(other)
                    deep |= c.dist < -min_depth
        return touching, deep

    def _obj_pose(self) -> np.ndarray:
        b = self.data.body(self._obj_body)
        return make_T(b.xmat.reshape(3, 3), b.xpos)

    def _tcp_pose(self) -> np.ndarray:
        s = self.data.site("g_tcp")
        return make_T(s.xmat.reshape(3, 3), s.xpos)

    # -- public API ----------------------------------------------------------------------
    def in_collision(self, T_obj_tcp: np.ndarray, width: float, tol: float = 0.001) -> bool:
        """Does the open gripper intersect the object at this grasp (before closing)?"""
        open_w = min(width + 0.02, self.cfg.gripper.max_width)
        self._reset(np.eye(3), T_obj_tcp, open_w)
        _, deep = self._contacts(min_depth=tol)
        return deep

    def shake_test(self, T_obj_tcp: np.ndarray, width: float,
                   R_world_obj: np.ndarray | None = None) -> ShakeResult:
        """Close (no gravity) -> gravity on -> lift -> shake along approach -> rotate about
        the finger axis -> settle. Success: both fingers still on the object and it moved
        less than DROP_MM relative to the TCP."""
        m, d, dt = self.model, self.data, self.model.opt.timestep
        R = np.eye(3) if R_world_obj is None else R_world_obj
        T_world_obj = make_T(R, np.zeros(3))
        T0 = T_world_obj @ T_obj_tcp
        self._reset(R, T0, min(width + 0.02, self.cfg.gripper.max_width))

        def steps(T):
            return max(round(T / dt), 1)

        d.ctrl[0] = 0.0
        for _ in range(steps(self.T_CLOSE)):
            mujoco.mj_step(m, d)
        touching, _ = self._contacts()
        if not touching >= self._fingers:
            return ShakeResult(False, False, np.nan, np.nan)
        T_tcp_obj_ref = inv_T(self._tcp_pose()) @ self._obj_pose()

        m.opt.gravity[:] = [0, 0, -9.81]
        approach, finger_axis = T0[:3, 2], T0[:3, 1]
        n = steps(self.T_LIFT)
        for k in range(1, n + 1):                       # lift straight up
            T = T0.copy()
            T[:3, 3] += [0, 0, self.LIFT * min_jerk(k / n)]
            self._set_hand(T)
            mujoco.mj_step(m, d)
        T_up = T0.copy()
        T_up[:3, 3] += [0, 0, self.LIFT]
        w = 2 * np.pi * self.SHAKE_HZ
        for k in range(steps(self.T_SHAKE_TRANS)):      # shake along the approach axis
            T = T_up.copy()
            T[:3, 3] += approach * self.SHAKE_AMP * np.sin(w * k * dt)
            self._set_hand(T)
            mujoco.mj_step(m, d)
        for k in range(steps(self.T_SHAKE_ROT)):        # rock about the finger axis (pivot: TCP)
            T = T_up.copy()
            T[:3, :3] = _axis_angle(finger_axis, self.SHAKE_ROT * np.sin(w * k * dt)) @ T_up[:3, :3]
            self._set_hand(T)
            mujoco.mj_step(m, d)
        self._set_hand(T_up)
        for _ in range(steps(self.T_SETTLE)):
            mujoco.mj_step(m, d)

        touching, _ = self._contacts()
        dT = inv_T(T_tcp_obj_ref) @ inv_T(self._tcp_pose()) @ self._obj_pose()
        slip_mm = float(np.linalg.norm(dT[:3, 3]) * 1e3)
        slip_deg = float(np.degrees(rotation_angle(dT[:3, :3])))
        ok = touching >= self._fingers and slip_mm < self.DROP_MM
        return ShakeResult(bool(ok), True, slip_mm, slip_deg)


@dataclass
class ContactSet:
    """Every gripper-object contact after closing, in the object frame.

    p (K, 3) points; n (K, 3) unit normals along which the gripper pushes the object; mu (K,)
    sliding and mu_t (K,) torsional friction (MuJoCo's, per contact); part (K,): 0 finger, 1 palm;
    c (K, 3): for finger contacts, the closing direction of that finger (zeros for the palm).
    """
    p: np.ndarray
    n: np.ndarray
    mu: np.ndarray
    mu_t: np.ndarray
    part: np.ndarray
    c: np.ndarray

    def __len__(self) -> int:
        return len(self.p)


def _contact_set(sim: GripperSim) -> ContactSet:
    m, d = sim.model, sim.data
    T_obj = sim._obj_pose()
    R, t = T_obj[:3, :3], T_obj[:3, 3]
    left, right = m.body("g_left_finger").id, m.body("g_right_finger").id
    y = sim._tcp_pose()[:3, 1]
    gap = d.xpos[right] - d.xpos[left]
    c_left = y if gap @ y > 0 else -y                    # the left finger closes towards the right one
    rows = []
    for con in d.contact[: d.ncon]:
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        if sim._obj_body not in (b1, b2):
            continue
        other = b2 if b1 == sim._obj_body else b1
        if other not in sim._gripper_bodies:
            continue
        nrm = con.frame[:3] if b2 == sim._obj_body else -con.frame[:3]   # force on the object
        cdir = c_left if other == left else (-c_left if other == right else np.zeros(3))
        rows.append((R.T @ (con.pos - t), R.T @ nrm, con.friction[0], con.friction[2],
                     0 if other in sim._fingers else 1, R.T @ cdir))
    if not rows:
        z = np.zeros((0, 3))
        return ContactSet(z, z, np.zeros(0), np.zeros(0), np.zeros(0, int), z)
    p, n, mu, mt, part, c = (np.array(x) for x in zip(*rows))
    return ContactSet(p, n, mu, mt, part.astype(int), c)


def close_contacts(sim: GripperSim, T_obj_tcp: np.ndarray, width: float) -> ContactSet | None:
    """Close the fingers on the object (no gravity) and return every contact, or None if the
    two fingers do not both touch it."""
    m, d = sim.model, sim.data
    sim._reset(np.eye(3), T_obj_tcp, min(width + 0.02, sim.cfg.gripper.max_width))
    d.ctrl[0] = 0.0
    for _ in range(max(round(sim.T_CLOSE / m.opt.timestep), 1)):
        mujoco.mj_step(m, d)
    touching, _ = sim._contacts()
    if not touching >= sim._fingers:
        return None
    return _contact_set(sim)


def hold_test(sim: GripperSim, T_obj_tcp: np.ndarray, width: float, R_world_obj: np.ndarray,
              duration: float = 1.0) -> tuple[bool, float, float]:
    """Ground truth for the hold check at any orientation: close (no gravity) with the object
    at R_world_obj, switch gravity on, keep the hand still for `duration`. Returns (held,
    slip_mm, slip_deg); held = both fingers still on and slip < GripperSim.DROP_MM."""
    m, d = sim.model, sim.data
    T0 = make_T(R_world_obj, np.zeros(3)) @ T_obj_tcp
    sim._reset(R_world_obj, T0, min(width + 0.02, sim.cfg.gripper.max_width))
    d.ctrl[0] = 0.0
    for _ in range(max(round(sim.T_CLOSE / m.opt.timestep), 1)):
        mujoco.mj_step(m, d)
    touching, _ = sim._contacts()
    if not touching >= sim._fingers:
        return False, np.nan, np.nan
    ref = inv_T(sim._tcp_pose()) @ sim._obj_pose()
    m.opt.gravity[:] = [0, 0, -9.81]
    for _ in range(max(round(duration / m.opt.timestep), 1)):
        mujoco.mj_step(m, d)
    touching, _ = sim._contacts()
    dT = inv_T(ref) @ inv_T(sim._tcp_pose()) @ sim._obj_pose()
    slip_mm = float(np.linalg.norm(dT[:3, 3]) * 1e3)
    return bool(touching >= sim._fingers and slip_mm < sim.DROP_MM), slip_mm, float(np.degrees(rotation_angle(dT[:3, :3])))
