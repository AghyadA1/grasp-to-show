"""The transport network: predicts the optimiser's motion in one forward pass.

Inputs (what the motion depends on), all relative to the robot base or the gripper:
  q0, q1, q1 - q0     start and end joint configurations (7 each): where the object starts and
                      where it must be presented, as the robot sees it
  mu                  finger-object friction
  mass, COM and inertia of the object in the gripper frame (the leverage the grasp must resist)
  grasp width
Output: the trajectory coefficients c (K x 7, same basis as grasp2show/transport_opt.py) and
log T. The prediction is checked with the exact constraints and, if needed, slowed down to the
smallest feasible duration (or refined by a few optimiser steps), as DJ-GOMP does with GOMP.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_IN = 7 * 3 + 1 + 1 + 3 + 6 + 1


def features(q0, q1, mu, T_obj_tcp, width, mass, com, inertia) -> np.ndarray:
    """(N_IN,) input vector for one problem."""
    R, p = T_obj_tcp[:3, :3], T_obj_tcp[:3, 3]
    com_t = R.T @ (np.asarray(com) - p)
    I_t = R.T @ np.asarray(inertia) @ R
    iu = I_t[np.triu_indices(3)]
    return np.concatenate([q0, q1, np.asarray(q1) - q0, [mu], [mass], com_t * 10, iu * 1e3, [width * 10]]).astype(np.float32)


class TransportNet(nn.Module):
    def __init__(self, K: int = 4, width: int = 256):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(nn.Linear(N_IN, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, K * 7 + 1))
        self.register_buffer("x_mean", torch.zeros(N_IN))
        self.register_buffer("x_std", torch.ones(N_IN))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp((x - self.x_mean) / self.x_std)
        return out[:, :-1].reshape(-1, self.K, 7), out[:, -1]      # c, log T
