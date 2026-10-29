"""Hook network labels and decoding (grasp2show/learn/hooknet.py)."""
import numpy as np
import torch

from grasp2show.learn.hooknet import OPEN_WIDTH, HookNet, anchor, augment, hook_pose, labels, predict_hooks


def _pose(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = q, rng.uniform(-0.05, 0.05, 3)
    return T


def test_labels_decode_back_to_the_hook_and_turn_with_the_object():
    rng = np.random.default_rng(0)
    cloud = np.hstack([rng.uniform(-0.06, 0.06, (500, 3)), np.tile([0, 0, 1.0], (500, 1))]).astype(np.float32)
    T = np.stack([_pose(rng), _pose(rng)])
    one = np.array([True, False])
    mask = np.array([True, True])
    pos, tgt = labels(cloud, T, one, mask, radius=0.03)
    assert pos.any()
    for i in np.flatnonzero(pos)[:20]:
        w = OPEN_WIDTH if tgt[i, 9] > 0.5 else 0.0
        Tr = hook_pose(cloud[i, :3], tgt[i, 0:3], tgt[i, 3:6], tgt[i, 6:9], w)
        assert min(np.abs(Tr - Th).max() for Th in T) < 1e-5            # the label is a hook, exactly
    # the anchor of a one-finger hook is the finger at +w/2 along y, the TCP for both fingers
    assert np.allclose(anchor(T[0], True), T[0, :3, 3] + T[0, :3, 1] * OPEN_WIDTH / 2)
    assert np.allclose(anchor(T[1], False), T[1, :3, 3])
    # turning the object turns its labels: same positive points
    c2, T2 = augment(cloud, T, np.random.default_rng(1))
    pos2, _ = labels(c2, T2, one, mask, radius=0.03)
    assert (pos2 != pos).sum() <= 5                                       # (point noise: 1 mm)


def test_predictions_are_valid_hook_poses():
    torch.manual_seed(0)
    net = HookNet()
    cloud = np.hstack([np.random.default_rng(0).uniform(-0.05, 0.05, (256, 3)), np.tile([0, 0, 1.0], (256, 1))])
    hooks = predict_hooks(net, cloud.astype(np.float32), k=5, min_prob=0.0)
    assert 1 <= len(hooks) <= 5
    for h in hooks:
        R = h["T"][:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-5) and np.isclose(np.linalg.det(R), 1.0)
        assert h["width"] in (0.0, OPEN_WIDTH)
