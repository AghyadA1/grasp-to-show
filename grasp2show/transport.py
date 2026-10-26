"""Dynamics of carrying a grasped object: the load the grasp must hold at every instant, and
the measures used to judge a transport motion (part 2 of the project).

While the object moves rigidly with the gripper, the fingers must supply the force and torque
that make it follow the motion. In the object frame (d'Alembert):
    F_ext = m (g - a_com)                      external load the contacts must cancel
    T_ext = -(I alpha + omega x I omega)        about the centre of mass
The soft-finger model (stability.utilisation_wrench) turns that load into a friction
utilisation u(t): the fraction of the grasp's capacity in use. Its anisotropy is the point of
this part: a load along the finger axis is carried by the grip force itself (the two normal
forces, up to 2F), one across it only by friction (up to 2 mu F), and a torque about the
finger axis only by torsional friction (2 mu_t F). A motion that points the load where the
grasp is strong can be faster without slipping.

Motion quality measures (scripts/transport_baseline.py reports all of them):
  u_max, u_mean      peak / mean friction utilisation during the transport
  u_end              utilisation at the end (object held still at the target: gravity only)
  slip               object motion in the hand (simulated, the ground truth)
  duration           seconds
  SPARC              spectral arc length of the TCP speed profile (Balasubramanian et al.
                     2015): 0 is a single smooth bell; more negative = less smooth
  stops              interior pauses: local speed minima below 5 % of the peak
  limit ratios       peak |qd|, |qdd|, |tau| as fractions of the Franka limits
  effort             integral of sum((tau / tau_max)^2) dt
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stability import StabilityParams, finger_contacts, utilisation_wrench

G = np.array([0.0, 0.0, -9.81])

# Franka Emika Panda joint limits (Franka Control Interface documentation).
QD_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


def friction_params(mu: float, finger_force: float = 20.0) -> StabilityParams:
    """Stability parameters for a friction coefficient mu. The torsional coefficient is the
    effective contact-patch radius times mu (5 mm at mu = 0.5, as in the base setup)."""
    return StabilityParams(mu=mu, mu_t=0.01 * mu, finger_force=finger_force)


def _derivative(x: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(x, dt, axis=0)


def angular_velocity(R: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity (N, 3) from rotation matrices (N, 3, 3): [w]x = Rdot R^T."""
    W = np.einsum("nij,nkj->nik", _derivative(R, dt), R)
    return 0.5 * np.stack([W[:, 2, 1] - W[:, 1, 2], W[:, 0, 2] - W[:, 2, 0], W[:, 1, 0] - W[:, 0, 1]], 1)


def inertial_load(T_world_obj: np.ndarray, dt: float, mass: float, com: np.ndarray, inertia: np.ndarray):
    """External force and torque the grasp must cancel (object frame), for a sampled pose
    trajectory (N, 4, 4) of a rigidly carried object. Returns F_ext (N, 3), T_ext (N, 3)."""
    R, p = T_world_obj[:, :3, :3], T_world_obj[:, :3, 3]
    c = p + np.einsum("nij,j->ni", R, com)                       # COM in the world
    a = _derivative(_derivative(c, dt), dt)
    w = angular_velocity(R, dt)
    al = _derivative(w, dt)
    Rt = np.transpose(R, (0, 2, 1))
    w_o, al_o = np.einsum("nij,nj->ni", Rt, w), np.einsum("nij,nj->ni", Rt, al)
    F = mass * np.einsum("nij,nj->ni", Rt, G - a)
    Iw = w_o @ inertia.T
    T = -(al_o @ inertia.T + np.cross(w_o, Iw))
    return F, T


def utilisation_series(obj, T_obj_tcp: np.ndarray, T_world_obj: np.ndarray, dt: float,
                       prm: StabilityParams) -> np.ndarray:
    """Friction utilisation u(t) of a grasp along a pose trajectory of the carried object."""
    c1, c2, ok = finger_contacts(obj.mesh, T_obj_tcp)
    if not ok[0]:
        return np.full(len(T_world_obj), np.inf)
    F, T = inertial_load(T_world_obj, dt, obj.mass, obj.com, obj.inertia)
    n = len(F)
    rep = lambda v: np.repeat(np.atleast_2d(v), n, axis=0)
    return utilisation_wrench(rep(c1[0]), rep(c2[0]), rep(T_obj_tcp[:3, 1]), obj.com, F, T, prm)


def sparc(speed: np.ndarray, fs: float, fc: float = 10.0, amp_th: float = 0.05, pad: int = 4) -> float:
    """Spectral arc length (Balasubramanian, Melendez-Calderon, Roby-Brami, Burdet 2015)."""
    v = np.asarray(speed, float)
    if v.max() <= 0:
        return 0.0
    nfft = int(2 ** (np.ceil(np.log2(len(v))) + pad))
    f = np.arange(nfft) * fs / nfft
    mag = np.abs(np.fft.fft(v, nfft))
    mag = mag / mag.max()
    sel = f <= fc
    f, mag = f[sel], mag[sel]
    above = np.flatnonzero(mag >= amp_th)
    f, mag = f[: above[-1] + 1], mag[: above[-1] + 1]
    df, dm = np.diff(f) / (f[-1] - f[0]), np.diff(mag)
    return float(-np.sum(np.sqrt(df**2 + dm**2)))


def count_stops(speed: np.ndarray, frac: float = 0.05) -> int:
    """Interior pauses: local minima of the speed below frac x peak (start and end excluded)."""
    v = np.asarray(speed, float)
    if len(v) < 3 or v.max() <= 0:
        return 0
    inner = v[1:-1]
    mins = (inner <= v[:-2]) & (inner < v[2:]) & (inner < frac * v.max())
    return int(mins.sum())


@dataclass
class TransportLog:
    """Samples of one transport (lift start -> presentation reached), every dt seconds."""
    dt: float
    q: np.ndarray            # (N, 7)
    qd: np.ndarray
    qdd: np.ndarray
    tau: np.ndarray          # (N, 7) joint torques the robot applies (actuators + gravity compensation)
    T_world_tcp: np.ndarray  # (N, 4, 4)
    T_world_obj: np.ndarray  # (N, 4, 4) the object as simulated (may slip)


def motion_metrics(log: TransportLog, obj, T_obj_tcp: np.ndarray, prm: StabilityParams) -> dict:
    """All measures for one transport. The utilisation uses the *commanded* rigid motion
    (TCP pose x grasp), i.e. the load the grasp is asked to hold."""
    T_rigid = log.T_world_tcp @ np.linalg.inv(T_obj_tcp)
    u = utilisation_series(obj, T_obj_tcp, T_rigid, log.dt, prm)
    speed = np.linalg.norm(_derivative(log.T_world_tcp[:, :3, 3], log.dt), axis=1)
    fs = 1.0 / log.dt
    return {
        "duration_s": round(len(log.q) * log.dt, 3),
        "u_max": float(np.max(u)), "u_mean": float(np.mean(u[np.isfinite(u)])) if np.isfinite(u).any() else np.inf,
        "u_time_above_0.8_s": float((u > 0.8).sum() * log.dt),
        "sparc": round(sparc(speed, fs), 3), "stops": count_stops(speed),
        "peak_speed_mps": round(float(speed.max()), 3),
        # Accelerations by differentiating the velocity: MuJoCo's qacc also holds solver spikes.
        "qd_ratio": float(np.max(np.abs(log.qd) / QD_MAX)),
        "qdd_ratio": float(np.max(np.abs(_derivative(log.qd, log.dt)) / QDD_MAX)),
        "u_end": float(u[-1]),
        "tau_ratio": float(np.max(np.abs(log.tau) / TAU_MAX)),
        "effort": float(np.sum((log.tau / TAU_MAX) ** 2) * log.dt),
    }
