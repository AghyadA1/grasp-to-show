"""The carry network (part 3): from what the robot knows about a grasped object and where it must
go (carry.features), M candidate carries and the score each is expected to get.

Several answers instead of one: where two different motions are about equally good (round the
left or the right of an obstacle, turning the wrist one way or the other), a network trained
towards both learns their average, which can be worse than either (part 2: the imitation
network). Here each of the M heads is trained only towards the best carry found from its own
proposals (scripts/train_carry.py), so each keeps to one kind of motion, and a value head
predicts each one's score: the robot runs the head with the best predicted score.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..carry import N_IN, N_THETA, K

M_HEADS = 6


def head_offsets(M: int = M_HEADS) -> np.ndarray:
    """Different starting motions for the heads: straight; the base swinging one way or the
    other on the way; the shoulder raised; the elbow bent; straight and faster."""
    off = np.zeros((M, N_THETA), np.float32)
    c = lambda m, k, j, v: off.__setitem__((m, k * 7 + j), v)
    if M > 1:
        c(1, 0, 0, 0.4)
    if M > 2:
        c(2, 0, 0, -0.4)
    if M > 3:
        c(3, 0, 1, -0.3)
    if M > 4:
        c(4, 0, 3, 0.3)
    if M > 5:
        off[5, K * 7] = -0.3
    return off


class CarryNet(nn.Module):
    def __init__(self, M: int = M_HEADS, width: int = 256):
        super().__init__()
        self.M = M
        self.trunk = nn.Sequential(nn.Linear(N_IN, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU())
        self.theta = nn.Linear(width, M * N_THETA)
        self.value = nn.Linear(width, M)
        with torch.no_grad():
            self.theta.weight.mul_(0.01)
            self.theta.bias.copy_(torch.as_tensor(head_offsets(M)).reshape(-1))
            self.value.weight.mul_(0.01)
            self.value.bias.zero_()
        self.register_buffer("x_mean", torch.zeros(N_IN))
        self.register_buffer("x_std", torch.ones(N_IN))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk((x - self.x_mean) / self.x_std)
        return self.theta(h).reshape(-1, self.M, N_THETA), self.value(h)

    @torch.no_grad()
    def best(self, x: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        """For one problem's features: (theta of the head with the best predicted score, its
        index, all predicted scores)."""
        th, v = self(torch.as_tensor(x[None], dtype=torch.float32))
        i = int(v[0].argmax())
        return th[0, i].numpy(), i, v[0].numpy()


def load_carry_net(path) -> CarryNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = CarryNet(ck.get("M", M_HEADS), ck.get("width", 256))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net
