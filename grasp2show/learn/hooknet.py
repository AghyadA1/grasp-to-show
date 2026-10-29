"""Hook network (part 3): where on an object's point cloud a fingertip can hook it, and how.

The analytic hook pipeline (grasp2show/hooks.py, hanging.py) finds a handle's hole by ray
casting, fits the fingertips' cross-section into it, nudges the pose clear of the handle and
lift-tests it in simulation: about 18 s for a generated object, minutes for a detailed scan, and
it needs the full mesh. This network does the first steps in one forward pass on a point cloud,
in the style of Contact-GraspNet: for every point, the probability that a hook grasp is anchored
there, and that grasp (the hand's approach axis = through the hole, its closing axis, the TCP's
offset from the point, and whether one finger or both go through). Only the lift test stays
(a few tenths of a second per candidate), plus the scene-dependent checks that come after.

Labels come from the analytic pipeline on generated objects (scripts/generate_hookable.py): a
point is positive if the anchor of a hook that lifts the object lies within LABEL_RADIUS of it,
and its targets are that nearest hook's parameters; every other point is negative (objects
without any hook are all negative: the network must also learn when there is nothing to hook).
The anchor is the fingertip that is in the hole: the TCP when both fingers go through, the
finger at TCP + (w/2) y with the gripper open at w when one does (the TCP is then outside, 4 cm
off). So the closing axis is signed for one-finger hooks (it points from the anchor's finger to
the other one, back across the TCP), and sign-free for two-finger hooks (the twins). The approach
axis is sign-free too: a hole can be threaded from either side, and neighbouring points are
labelled with hooks from both, so a signed target would average to nothing; both signs are
decoded (refine_hooks then fits the fingertips analytically around the predicted hole).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from torch import nn

from .dense import EdgeConv

N_POINTS = 2048
LABEL_RADIUS = 0.02     # m: points this close to a hook's TCP are labelled with it
N_OUT = 11              # logit, approach (3), closing (3), offset (3), one-finger logit
OPEN_WIDTH = 0.08       # finger gap of a one-finger hook (the other finger stays outside)


def sample_cloud(mesh: trimesh.Trimesh, n: int = N_POINTS, seed: int = 0) -> np.ndarray:
    """(n, 6) surface points and their normals, object frame."""
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.hstack([pts, mesh.face_normals[fid]]).astype(np.float32)


@dataclass
class HookSet:
    """Objects with their clouds and hooks (hooks padded to H per object; mask marks real ones)."""
    names: list[str]
    clouds: np.ndarray        # (O, P, 6)
    T: np.ndarray             # (O, H, 4, 4) hook poses T_obj_tcp
    one: np.ndarray           # (O, H) one finger through (else both)
    mask: np.ndarray          # (O, H)

    def subset(self, idx) -> HookSet:
        idx = list(idx)
        return HookSet([self.names[i] for i in idx], self.clouds[idx], self.T[idx], self.one[idx], self.mask[idx])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def anchor(T: np.ndarray, one) -> np.ndarray:
    """The fingertip in the hole (see the module docstring). T (..., 4, 4), one (...,)."""
    w = np.where(np.asarray(one, bool), OPEN_WIDTH, 0.0)[..., None]
    return T[..., :3, 3] + T[..., :3, 1] * w / 2


def labels(cloud: np.ndarray, T: np.ndarray, one: np.ndarray, mask: np.ndarray, radius: float = LABEL_RADIUS):
    """Per point: positive?, and the nearest hook's (approach, closing, anchor - point, one-finger).
    cloud (P, 6), T (H, 4, 4)."""
    P = len(cloud)
    pos = np.zeros(P, bool)
    tgt = np.zeros((P, 10), np.float32)
    Th = T[mask]
    if len(Th) == 0:
        return pos, tgt
    tcp = anchor(Th, one[mask])
    d, j = cKDTree(tcp).query(cloud[:, :3])
    pos = d < radius
    Tj = Th[j[pos]]
    tgt[pos] = np.concatenate([Tj[:, :3, 2], Tj[:, :3, 1], tcp[j[pos]] - cloud[pos, :3],
                               one[mask][j[pos], None].astype(np.float32)], 1)
    return pos, tgt


def augment(cloud: np.ndarray, T: np.ndarray, rng: np.random.Generator):
    """A random turn about the vertical (the object stands on the table at any yaw) and a little
    noise on the points (a camera's cloud is not exact)."""
    R = rot_z(rng.uniform(0, 2 * np.pi))
    c = cloud.copy()
    c[:, :3] = c[:, :3] @ R.T + rng.normal(scale=0.001, size=(len(c), 3))
    c[:, 3:] = c[:, 3:] @ R.T
    Ta = T.copy()
    Ta[..., :3, :3] = R @ T[..., :3, :3]
    Ta[..., :3, 3] = T[..., :3, 3] @ R.T
    return c.astype(np.float32), Ta


class HookNet(nn.Module):
    """Per-point hook probability and hook parameters from the object's point cloud alone."""

    def __init__(self, k: int = 16):
        super().__init__()
        self.edge1 = EdgeConv(6, 64, k)
        self.edge2 = EdgeConv(64, 64, k)
        self.point_mlp = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                       nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.head = nn.Sequential(nn.Conv1d(128 + 256 + 256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Conv1d(256, 128, 1), nn.ReLU(), nn.Conv1d(128, N_OUT, 1))

    def forward(self, cloud: torch.Tensor) -> torch.Tensor:          # cloud (B, P, 6) -> (B, 11, P)
        xyz = cloud[..., :3]
        x = torch.cat([xyz * 10.0, cloud[..., 3:]], -1).transpose(1, 2)
        e1 = self.edge1(x, xyz)
        e2 = self.edge2(e1, xyz)                                       # two hops: a handle's hole is wider than k points
        f = self.point_mlp(torch.cat([e1, e2], 1))
        g = f.max(dim=2, keepdim=True).values.expand(-1, -1, f.shape[2])
        return self.head(torch.cat([e1, e2, f, g], 1))


def split(out: torch.Tensor) -> dict:
    o = out.transpose(1, 2)
    nrm = nn.functional.normalize
    return {"logit": o[..., 0], "approach": nrm(o[..., 1:4], dim=-1), "closing": nrm(o[..., 4:7], dim=-1),
            "offset": o[..., 7:10] / 10.0, "one_logit": o[..., 10]}


def loss_fn(o: dict, pos: torch.Tensor, tgt: torch.Tensor, pos_weight: torch.Tensor) -> tuple[torch.Tensor, dict]:
    bce = nn.functional.binary_cross_entropy_with_logits(o["logit"], pos.float(), pos_weight=pos_weight)
    parts = {"bce": bce.item()}
    if not pos.any():
        return bce, parts
    a, c, off, one = tgt[..., 0:3], tgt[..., 3:6], tgt[..., 6:9], tgt[..., 9]
    la = (1 - (o["approach"] * a).sum(-1).abs())[pos].mean()         # through the hole from either side
    dot = (o["closing"] * c).sum(-1)
    lc = (1 - torch.where(one > 0.5, dot, dot.abs()))[pos].mean()   # signed for one finger, else sign-free
    lo = nn.functional.smooth_l1_loss(o["offset"][pos] * 100, off[pos] * 100)    # cm
    lk = nn.functional.binary_cross_entropy_with_logits(o["one_logit"][pos], one[pos])
    parts.update(approach=la.item(), closing=lc.item(), offset_cm=lo.item(), kind=lk.item())
    return bce + la + lc + lo + 0.5 * lk, parts


def hook_pose(point: np.ndarray, approach: np.ndarray, closing: np.ndarray, offset: np.ndarray,
              width: float = 0.0) -> np.ndarray:
    """T_obj_tcp: z = approach, y = closing made orthogonal to z, anchor = point + offset, TCP =
    anchor - (width/2) y."""
    z = approach / np.linalg.norm(approach)
    y = closing - (closing @ z) * z
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(y, z), y, z], 1)
    T[:3, 3] = point + offset - y * width / 2
    return T


@torch.no_grad()
def predict_hooks(net: HookNet, cloud: np.ndarray, k: int = 10, min_prob: float = 0.5, sep: float = 0.01,
                  device="cpu") -> list[dict]:
    """Hook candidates for one cloud (object frame): the k most probable points, at least `sep`
    apart, each decoded with both approach signs (2k poses). Each: dict T, width, kind, prob,
    anchor (the predicted fingertip in the hole), axis (the predicted hole axis)."""
    net.eval()
    o = split(net(torch.as_tensor(cloud[None], device=device)))
    p = torch.sigmoid(o["logit"][0]).cpu().numpy()
    out, tcps = [], []
    for i in np.argsort(-p):
        if p[i] < min_prob or len(tcps) == k:
            break
        one = float(torch.sigmoid(o["one_logit"][0, i])) > 0.5
        w = OPEN_WIDTH if one else 0.0
        T = hook_pose(cloud[i, :3], o["approach"][0, i].cpu().numpy(), o["closing"][0, i].cpu().numpy(),
                      o["offset"][0, i].cpu().numpy(), w)
        a = cloud[i, :3] + o["offset"][0, i].cpu().numpy()
        if any(np.linalg.norm(a - t) < sep for t in tcps):
            continue
        flip = np.diag([-1.0, 1.0, -1.0, 1.0])                         # 180 deg about y: through from the other side
        T2 = hook_pose(cloud[i, :3], -o["approach"][0, i].cpu().numpy(), o["closing"][0, i].cpu().numpy(),
                       o["offset"][0, i].cpu().numpy(), w)
        assert np.allclose(T2[:3, :3], T[:3, :3] @ flip[:3, :3], atol=1e-6)
        for Tk in (T, T2):
            out.append({"T": Tk, "width": w, "kind": "one" if one else "two", "prob": float(p[i]), "anchor": a,
                        "axis": T[:3, 2].copy()})
        tcps.append(a)
    return out


def refine_hooks(obj, preds: list[dict], merge: float = 0.015, turns: int = 4, max_holes: int = 2) -> list[dict]:
    """The analytic fingertip fit (hooks.fit_hook_poses) around the holes the network found: its
    anchors, merged within `merge`, are the holes' centres and its axes their axes. The global
    search for holes (ray casting the whole object) is what the network replaces; the fit itself
    needs millimetres, which a regression does not give, and is cheap for one or two holes."""
    from ..hooks import fit_hook_poses

    holes = []
    for h in sorted(preds, key=lambda h: -h["prob"]):                 # the most probable holes first
        if all(np.linalg.norm(h["anchor"] - c) > merge for c, _ in holes):
            holes.append((h["anchor"], h["axis"]))
        if len(holes) == max_holes:
            break
    loops = [(c, d, 16) for c, d in holes]
    return fit_hook_poses(obj, loops, turns=turns, max_holes=len(loops)) if loops else []
