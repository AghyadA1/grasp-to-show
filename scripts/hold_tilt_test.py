"""Hold test: how much weight can the simulated two-finger grasp hold, finger axis horizontal
(weight carried by friction) vs tilted up to vertical (weight partly carried by the grip force)?

Floating Franka hand, 5 x 5 x 12 cm box grasped through its centre of mass, mu = 0.3, 20 N per
finger. Ideal model: horizontal holds up to 2 mu F = 12 N (1.22 kg); vertical up to 2F = 40 N.
Usage:  python scripts/hold_tilt_test.py
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp2show.config import ObjectConfig, SceneConfig
from grasp2show.gripper_sim import GripperSim, _axis_angle
from grasp2show.objects import object_from_config
from grasp2show.transforms import inv_T, make_T, rotation_angle

mu = 0.3
def hold(mass, tilt, offset=0.0):
    cfg = SceneConfig(obj=ObjectConfig(mass=mass)); obj = object_from_config(cfg.obj); sim = GripperSim(cfg, obj)
    m, d, dt = sim.model, sim.data, sim.model.opt.timestep
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in (sim._fingers | {sim._obj_body}) and (m.geom_contype[g] | m.geom_conaffinity[g])]
    m.geom_friction[geoms, 0], m.geom_friction[geoms, 1] = mu, 0.01 * mu
    T_obj_tcp = make_T(np.column_stack([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]]), [0, 0, offset])
    R = _axis_angle([0, 1, 0], -np.radians(tilt)); T0 = make_T(R, np.zeros(3)) @ T_obj_tcp
    sim._reset(R, T0, 0.07); d.ctrl[0] = 0.0
    for _ in range(int(0.4 / dt)): mujoco.mj_step(m, d)
    ref = inv_T(sim._tcp_pose()) @ sim._obj_pose()
    m.opt.gravity[:] = [0, 0, -9.81]
    for _ in range(int(1.0 / dt)):
        sim._set_hand(T0); mujoco.mj_step(m, d)
    dT = inv_T(ref) @ inv_T(sim._tcp_pose()) @ sim._obj_pose()
    return np.linalg.norm(dT[:3, 3]) * 1e3, np.degrees(rotation_angle(dT[:3, :3]))
for tilt in (0, 70, 90):
    print(f"tilt {tilt:2d}:", " | ".join(f"{mm:.1f} kg {min(s,999):5.1f} mm" for mm in (0.8, 1.0, 1.2, 1.5, 2.0, 2.5) for s, r in [hold(mm, tilt)]))
