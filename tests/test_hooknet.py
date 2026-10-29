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
    assert 2 <= len(hooks) <= 10 and len(hooks) % 2 == 0          # each point decoded with both approach signs
    for h in hooks:
        R = h["T"][:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-5) and np.isclose(np.linalg.det(R), 1.0)
        assert h["width"] in (0.0, OPEN_WIDTH)


def test_three_way_labels():
    from grasp2show.learn.hooknet import labels3

    x = np.linspace(-0.1, 0.1, 201)
    cloud = np.zeros((201, 6), np.float32)
    cloud[:, 0], cloud[:, 5] = x, 1.0
    T = np.tile(np.eye(4), (2, 1, 1))
    T[0, :3, 3] = [-0.08, 0, 0]                                  # a hook that lifted at x = -0.08
    one, mask = np.zeros(2, bool), np.array([True, False])
    fell = np.array([[0.05, 0, 0], [0.051, 0, 0], [0.0, 0, 0]])  # two failed tries at 0.05, one at 0
    pos, _, w = labels3(cloud, T, one, mask, fell, radius=0.01)
    at = lambda v: int(np.argmin(np.abs(x - v)))
    assert pos[at(-0.08)] and w[at(-0.08)] == 1.0
    assert not pos[at(0.05)] and w[at(0.05)] == 1.0             # a real negative: tried twice, fell
    assert w[at(0.0)] == 0.5                                      # tried once
    assert w[at(-0.03)] < 0.5 and not pos[at(-0.03)]              # untested: counts little


def test_partial_view_sees_the_near_side():
    from grasp2show.learn.hooknet import view_sample, visible

    rng = np.random.default_rng(0)
    p = rng.normal(size=(3000, 3))
    p = 0.05 * p / np.linalg.norm(p, axis=1, keepdims=True)
    idx = visible(p, np.array([0.5, 0, 0]))
    assert 0.3 < len(idx) / len(p) < 0.6 and (p[idx, 0] > -0.01).all()
    c = np.hstack([p, p / 0.05]).astype(np.float32)
    assert view_sample(c, rng, n=1024, partial=1.0).shape == (1024, 6)
