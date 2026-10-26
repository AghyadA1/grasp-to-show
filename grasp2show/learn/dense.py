"""Way B: a dense, goal-conditioned grasp heatmap network.

Input: the object's point cloud (object frame) and the task: where the object starts and how it
must be presented (target pose). Output, for every point: the score of the best grasp touching
it for this task, plus that grasp (approach direction, closing direction, TCP offset from the
point, opening width), in the style of Contact-GraspNet, but conditioned on the target.

Labels come from dense candidates scored by the analytic checks (scripts/generate_dense_labels.py):
a point's label is the best-scoring candidate whose finger contact lies within `radius` of it
(the way-A heatmap); points no candidate touches are unlabelled and ignored by the loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

G = np.array([0.0, 0.0, -1.0])
N_COND = 25
N_OUT = 12   # feasible logit, score, approach (3), closing (3), offset (3), width


def condition(T_start: np.ndarray, T_target: np.ndarray, cam_pos, mass: float, com: np.ndarray) -> np.ndarray:
    """(25,) task description in the object frame: gravity, world x axis and robot base position
    at the start and at the target, camera direction at the target, mass, centre of mass."""
    out = []
    for T in (T_start, T_target):
        R, t = T[:3, :3], T[:3, 3]
        out += [R.T @ G, R.T @ np.array([1.0, 0, 0]), R.T @ (-t)]
    cam = T_target[:3, :3].T @ (np.asarray(cam_pos) - T_target[:3, 3])
    out += [cam / np.linalg.norm(cam), [mass], np.asarray(com) * 10]
    return np.concatenate(out).astype(np.float32)


def point_labels(points: np.ndarray, d: dict, radius: float = 0.008):
    """Per point and target: best score and that grasp's parameters.

    points (P, 3); d: one dense-label file (T_obj_tcp, width, c1, c2, S (N, R)).
    Returns score (R, P) with NaN = no candidate touches; params (R, P, 10): approach, closing,
    offset (TCP - point), width (zeros where unlabelled); stab (P,): the target-blind map (best
    lift margin of the touching candidates), for comparison.
    """
    T, S = d["T_obj_tcp"].astype(float), d["S"].astype(float)
    tree = cKDTree(points)
    pi, gi = [], []
    for c in (d["c1"], d["c2"]):
        ok = ~np.isnan(c).any(1)
        for g, nb in zip(np.flatnonzero(ok), tree.query_ball_point(c[ok], radius)):
            pi += nb
            gi += [g] * len(nb)
    pi, gi = np.asarray(pi, int), np.asarray(gi, int)
    R, P = S.shape[1], len(points)
    score = np.full((R, P), np.nan, np.float32)
    params = np.zeros((R, P, 10), np.float32)
    stab = np.full(P, np.nan, np.float32)                   # target-blind map: best lift margin
    if len(pi) == 0:
        return score, params, stab
    np.fmax.at(stab, pi, np.clip(d["lift_margin"][gi], 0, 1).astype(np.float32))
    for r in range(R):
        s = S[gi, r]
        order = np.lexsort((s, pi))                         # by point, then by score
        last = np.r_[np.flatnonzero(np.diff(pi[order])), len(order) - 1]
        best_p, best_g = pi[order][last], gi[order][last]
        score[r, best_p] = S[best_g, r]
        Tg = T[best_g]
        params[r, best_p] = np.concatenate([Tg[:, :3, 2], Tg[:, :3, 1], Tg[:, :3, 3] - points[best_p],
                                            d["width"][best_g, None]], 1)
    return score, params, stab


@dataclass
class DenseSet:
    obj: np.ndarray       # (M,) object index into the clouds
    cond: np.ndarray      # (M, 25)
    score: np.ndarray     # (M, P) NaN = unlabelled
    params: np.ndarray    # (M, P, 10)
    stab: np.ndarray      # (M, P) target-blind map, for comparison only
    yaw: np.ndarray       # (M,)
    roll_idx: np.ndarray  # (M,)

    def __len__(self) -> int:
        return len(self.obj)

    def subset(self, m) -> DenseSet:
        return DenseSet(*(getattr(self, f)[m] for f in self.__dataclass_fields__))


def load_dense(root: Path, clouds, cam_pos, radius: float = 0.008) -> DenseSet:
    """All (object, start yaw, target) samples with per-point labels on the clouds' points."""
    rows = {k: [] for k in DenseSet.__dataclass_fields__}
    for f in sorted(root.glob("*__yaw*.npz")):
        name = f.name.split("__yaw")[0]
        if name not in clouds.names:
            continue
        o = clouds.names.index(name)
        d = dict(np.load(f))
        score, params, stab = point_labels(clouds.feats[o, :, :3], d, radius)
        for r in range(score.shape[0]):
            rows["obj"].append(o)
            rows["cond"].append(condition(d["T_start"], d["T_target"][r], cam_pos, clouds.mass[o], clouds.com[o]))
            rows["score"].append(score[r])
            rows["params"].append(params[r])
            rows["stab"].append(stab)
            rows["yaw"].append(float(d["yaw"]))
            rows["roll_idx"].append(r)
    return DenseSet(*(np.asarray(rows[k]) for k in DenseSet.__dataclass_fields__))


class EdgeConv(nn.Module):
    """DGCNN edge convolution: features from each point's k nearest neighbours (local shape)."""

    def __init__(self, c_in: int, c_out: int, k: int = 16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(nn.Conv2d(2 * c_in, c_out, 1), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2))

    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:   # x (B, C, P), xyz (B, P, 3)
        idx = torch.cdist(xyz, xyz).topk(self.k, largest=False).indices          # (B, P, k)
        b = torch.arange(x.shape[0], device=x.device)[:, None, None]
        nb = x.transpose(1, 2)[b, idx].permute(0, 3, 1, 2)                       # (B, C, P, k)
        ctr = x.unsqueeze(3).expand(-1, -1, -1, self.k)
        return self.mlp(torch.cat([ctr, nb - ctr], 1)).max(dim=3).values


class HeatmapNet(nn.Module):
    """Per-point grasp quality and grasp parameters, conditioned on the task."""

    def __init__(self, clouds: torch.Tensor, k: int = 16):
        super().__init__()
        self.register_buffer("clouds", clouds)                                    # (O, P, 7)
        self.edge = EdgeConv(7, 64, k)
        self.cond_mlp = nn.Sequential(nn.Linear(N_COND, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.point_mlp = nn.Sequential(nn.Conv1d(64 + 64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                       nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.head = nn.Sequential(nn.Conv1d(64 + 256 + 256 + 64, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Conv1d(256, 128, 1), nn.ReLU(), nn.Conv1d(128, N_OUT, 1))

    def inputs(self, obj, sel=None):
        pc = self.clouds[obj]                                                     # (B, P, 7)
        if sel is not None:
            pc = pc[:, sel]
        xyz = pc[..., :3]
        return torch.cat([xyz * 10.0, pc[..., 3:]], -1).transpose(1, 2), xyz

    def forward(self, obj, cond, sel=None) -> torch.Tensor:
        x, xyz = self.inputs(obj, sel)
        P = x.shape[2]
        e = self.edge(x, xyz)                                                     # (B, 64, P)
        c = self.cond_mlp(cond)[:, :, None].expand(-1, -1, P)                     # (B, 64, P)
        f = self.point_mlp(torch.cat([e, c], 1))                                  # (B, 256, P)
        g = f.max(dim=2, keepdim=True).values.expand(-1, -1, P)
        return self.head(torch.cat([e, f, g, c], 1))                              # (B, 12, P)


def split_outputs(out: torch.Tensor) -> dict:
    """Raw network outputs (B, 12, P) -> named, normalised per-point predictions."""
    o = out.transpose(1, 2)
    nrm = nn.functional.normalize
    return {"logit": o[..., 0], "score": torch.sigmoid(o[..., 1]), "approach": nrm(o[..., 2:5], dim=-1),
            "closing": nrm(o[..., 5:8], dim=-1), "offset": o[..., 8:11] / 10.0, "width": o[..., 11] / 10.0}


def grasp_from_prediction(point: np.ndarray, approach: np.ndarray, closing: np.ndarray, offset: np.ndarray):
    """T_obj_tcp from a point's predicted grasp: z = approach, y = closing made orthogonal to z."""
    z = approach / np.linalg.norm(approach)
    y = closing - (closing @ z) * z
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(y, z), y, z], 1)
    T[:3, 3] = point + offset
    return T
