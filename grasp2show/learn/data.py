"""Load the full-simulation dataset (scripts/generate_dataset.py) into arrays for training.

Each sample is a (grasp, twin, target) tuple in one scenario (object + start pose):
  obj      index into the per-object point clouds
  R, t     the grasp T_obj_tcp (rotation, translation): object points are expressed in the gripper
           frame inside the model, so the network sees the object as the gripper sees it
  glob     32 global features (gravity in the gripper frame at the start and at the target, QR
           normal, gripper pose in the world at start and target, width, mass, centre of mass)
  y        1 if the full pick-and-show episode succeeded
  reason   failure reason code (0 = success), for an auxiliary head
The analytic checks' outputs are kept alongside (pred_ok, aware_score) for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

G = np.array([0.0, 0.0, -1.0])
N_GLOBAL = 32
LIMIT_SCALE = 0.5


@dataclass
class Split:
    obj: np.ndarray          # (N,) int
    R: np.ndarray            # (N, 3, 3) float32
    t: np.ndarray            # (N, 3)
    glob: np.ndarray         # (N, 32)
    y: np.ndarray            # (N,) float32
    reason: np.ndarray       # (N,) int
    simulated: np.ndarray    # (N,) bool
    scenario: np.ndarray     # (N,) int: scenario id (object x yaw)
    pred_ok: np.ndarray      # (N,) bool: analytic checks say feasible
    aware_score: np.ndarray  # (N,) analytic leximin first element (smallest normalised margin)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, m) -> Split:
        return Split(*(getattr(self, f)[m] for f in self.__dataclass_fields__))


@dataclass
class Clouds:
    names: list[str]
    feats: np.ndarray        # (O, P, 7): xyz, normal, on_qr (object frame)
    mass: np.ndarray         # (O,)
    com: np.ndarray          # (O, 3)
    qr_normal: np.ndarray    # (O, 3)


def load_clouds(path: Path) -> Clouds:
    z = np.load(path)
    names = [str(n) for n in z["names"]]
    feats = np.stack([np.concatenate([z[f"points__{n}"], z[f"normals__{n}"],
                                      z[f"on_qr__{n}"][:, None].astype(np.float32)], 1) for n in names])
    return Clouds(names, feats.astype(np.float32), np.array([float(z[f"mass__{n}"]) for n in names]),
                  np.stack([z[f"com__{n}"] for n in names]), np.stack([z[f"T_obj_qr__{n}"][:3, 0] for n in names]))


def _rot6d(R: np.ndarray) -> np.ndarray:
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def global_features(T_obj_tcp, T_world_obj_start, T_world_obj_target, width, mass, com, qr_normal) -> np.ndarray:
    """(N, 32) features; all inputs batched over N except the start pose (4, 4) of the scenario."""
    R = T_obj_tcp[:, :3, :3]
    Rt = np.transpose(R, (0, 2, 1))
    g_start = Rt @ (T_world_obj_start[:3, :3].T @ G)                               # (N, 3)
    g_target = np.einsum("nij,nkj,k->ni", Rt, T_world_obj_target[:, :3, :3], G)    # R^T R_wt^T g
    qr = np.einsum("nij,j->ni", Rt, qr_normal)
    W_start = T_world_obj_start @ T_obj_tcp
    W_target = T_world_obj_target @ T_obj_tcp
    com_tcp = np.einsum("nij,nj->ni", Rt, com - T_obj_tcp[:, :3, 3])
    f = np.concatenate([g_start, g_target, qr, W_start[:, :3, 3], _rot6d(W_start[:, :3, :3]),
                        W_target[:, :3, 3], _rot6d(W_target[:, :3, :3]), 10 * width[:, None],
                        mass[:, None], 10 * com_tcp], 1)
    assert f.shape[1] == N_GLOBAL
    return f.astype(np.float32)


def analytic_score(z) -> np.ndarray:
    """The aware policy's ranking key (its first, smallest element); -inf if a check fails."""
    parts = np.stack([z["lift_margin"], z["start_limit"] / LIMIT_SCALE, z["target_limit"] / LIMIT_SCALE,
                      z["stab_margin"], (z["vis_fraction"] - 0.9) / 0.1], 1)
    s = np.nanmin(np.where(np.isnan(parts), -np.inf, parts), axis=1)
    return np.where(z["pred_ok"], s, -np.inf)


def load_dataset(root: Path) -> tuple[Clouds, Split]:
    clouds = load_clouds(root / "clouds.npz")
    idx = {n: i for i, n in enumerate(clouds.names)}
    parts, sid = [], 0
    for f in sorted(root.glob("*__yaw*.npz")):
        name = f.name.split("__yaw")[0]
        if name not in idx:
            continue
        z = np.load(f)
        n = len(z["success"])
        if n == 0:
            continue
        o = idx[name]
        T = z["T_obj_tcp"].astype(np.float64)
        glob = global_features(T, z["T_world_obj_start"], z["T_target"], z["width"],
                               np.full(n, clouds.mass[o]), clouds.com[o], clouds.qr_normal[o])
        parts.append(Split(np.full(n, o), T[:, :3, :3].astype(np.float32), T[:, :3, 3].astype(np.float32),
                           glob, z["success"].astype(np.float32), z["reason"].astype(int),
                           z["simulated"].astype(bool), np.full(n, sid), z["pred_ok"].astype(bool),
                           analytic_score(z)))
        sid += 1
    cat = {f: np.concatenate([getattr(p, f) for p in parts]) for f in Split.__dataclass_fields__}
    return clouds, Split(**cat)
