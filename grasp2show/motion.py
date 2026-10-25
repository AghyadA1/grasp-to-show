"""Trajectory helpers: minimum-jerk interpolation in joint space and Cartesian lines."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def min_jerk(s: np.ndarray | float) -> np.ndarray | float:
    """Minimum-jerk time scaling 10s^3 - 15s^4 + 6s^5 on s in [0, 1].

    Zero velocity and acceleration at both ends: the profile that minimises the integral
    of squared jerk (Flash & Hogan, 1985). It avoids the force spikes that would shake the
    object loose at the start and end of each segment.
    """
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def joint_trajectory(q_waypoints: list[np.ndarray], durations: list[float], dt: float) -> np.ndarray:
    """Concatenate min-jerk segments between joint waypoints, sampled every dt."""
    assert len(durations) == len(q_waypoints) - 1
    out = []
    for q0, q1, T in zip(q_waypoints[:-1], q_waypoints[1:], durations):
        n = max(round(T / dt), 1)
        s = min_jerk(np.arange(1, n + 1) / n)[:, None]
        out.append(q0 + s * (q1 - q0))
    return np.vstack(out)


def path_trajectory(q_path: list[np.ndarray], duration: float, dt: float) -> np.ndarray:
    """Time-parametrise a dense joint path with ONE min-jerk profile over its whole length.

    Unlike joint_trajectory (which stops at every waypoint), the arm moves smoothly through
    the intermediate points: progress s(t) = min_jerk(t/T), and q(s) is linear between
    waypoints, which are evenly spaced in Cartesian space.
    """
    q_path = np.asarray(q_path)
    u = np.linspace(0.0, 1.0, len(q_path))
    n = max(round(duration / dt), 1)
    s = min_jerk(np.arange(1, n + 1) / n)
    return np.column_stack([np.interp(s, u, q_path[:, j]) for j in range(q_path.shape[1])])


def cartesian_line(T0: np.ndarray, T1: np.ndarray, n: int) -> list[np.ndarray]:
    """n poses from T0 to T1 (excluding T0): linear in position, slerp in rotation."""
    slerp = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))
    poses = []
    for s in np.linspace(0, 1, n + 1)[1:]:
        T = np.eye(4)
        T[:3, :3] = slerp([s]).as_matrix()[0]
        T[:3, 3] = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
        poses.append(T)
    return poses
