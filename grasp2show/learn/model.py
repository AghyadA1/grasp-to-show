"""Learned grasp scorer: P(pick-and-show succeeds | object, grasp, start pose, target pose).

PointNet encoder (Qi et al., CVPR 2017) over the object point cloud expressed in the gripper
frame (so the network sees the object as the gripper does), concatenated with 32 global
features, then an MLP with two heads: success logit, and failure-reason logits (auxiliary).
"""
from __future__ import annotations

import torch
from torch import nn

from .data import N_GLOBAL

N_REASONS = 8


class Scorer(nn.Module):
    def __init__(self, clouds: torch.Tensor, width: int = 256, use_qr_flag: bool = True):
        """clouds: (O, P, 7) per-object points (xyz, normal, on_qr) in the object frame."""
        super().__init__()
        self.register_buffer("clouds", clouds)
        self.use_qr_flag = use_qr_flag
        self.point_mlp = nn.Sequential(
            nn.Conv1d(7, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, width, 1), nn.BatchNorm1d(width), nn.ReLU(),
        )
        self.glob_mlp = nn.Sequential(nn.Linear(N_GLOBAL, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(width + 64, 256), nn.ReLU(), nn.Dropout(0.2),
                                   nn.Linear(256, 128), nn.ReLU())
        self.success_head = nn.Linear(128, 1)
        self.reason_head = nn.Linear(128, N_REASONS)

    def point_features(self, obj, R, t, n_points: int | None = None) -> torch.Tensor:
        """(B, 7, P') gripper-frame point features for a batch of grasps."""
        pc = self.clouds[obj]                                          # (B, P, 7)
        if n_points is not None and n_points < pc.shape[1]:
            sel = torch.randperm(pc.shape[1], device=pc.device)[:n_points]
            pc = pc[:, sel]
        xyz = torch.bmm(pc[..., :3] - t[:, None, :], R) * 10.0         # (p - t) R, decimetres
        nrm = torch.bmm(pc[..., 3:6], R)
        flag = pc[..., 6:7] if self.use_qr_flag else torch.zeros_like(pc[..., 6:7])
        return torch.cat([xyz, nrm, flag], -1).transpose(1, 2)

    def forward(self, obj, R, t, glob, n_points: int | None = None):
        f = self.point_mlp(self.point_features(obj, R, t, n_points)).max(dim=2).values
        h = self.trunk(torch.cat([f, self.glob_mlp(glob)], 1))
        return self.success_head(h).squeeze(1), self.reason_head(h)
