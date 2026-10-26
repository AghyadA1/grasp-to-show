"""Goal-conditioned grasp heatmaps, "way A": score every candidate grasp for a target pose, then
paint each grasp's score onto the surface where its fingers touch.

The value at a surface point is the score of the *best* grasp whose finger contact lies within
`radius` of it: "how good is it to grasp the object here, for this target?". Change the target
(here: the roll of the object about the scanner's viewing axis) and the map changes, because
the fingers must not cover the QR, the arm must reach the new pose, and gravity acts in a new
direction in the object frame. No training: this is the analytic checks, made visible.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .experiment import LIMIT_SCALE
from .feasibility import GraspReport


def aware_scores(reports: list[GraspReport], lift_margin: np.ndarray, n_targets: int) -> np.ndarray:
    """(N, n_targets) score of each grasp for each target, best over its two twins.

    Same rule as the target-aware policy: 0 if any hard check fails, else the smallest
    normalised margin (lift and target stability, joint limits at start and target, QR
    visibility), clipped to [0, 1].
    """
    S = np.zeros((len(reports), n_targets))
    for i, rep in enumerate(reports):
        if not lift_margin[i] > 0:
            continue
        for tw in rep.twins:
            if not tw.start_ok:
                continue
            for j, t in enumerate(tw.targets):
                if not (t.reach.ok and t.stability_margin > 0 and t.visibility is not None and t.visibility.ok()):
                    continue
                s = min(lift_margin[i], tw.start.limit_margin / LIMIT_SCALE, t.reach.limit_margin / LIMIT_SCALE,
                        t.stability_margin, (t.visibility.fraction - 0.9) / 0.1)
                S[i, j] = max(S[i, j], float(np.clip(s, 0.0, 1.0)))
    return S


def stability_scores(reports: list[GraspReport], lift_margin: np.ndarray) -> np.ndarray:
    """(N,) what a target-blind planner sees: lift margin if the grasp is reachable at the start."""
    ok = np.array([any(tw.start_ok for tw in r.twins) for r in reports])
    return np.where(ok & (lift_margin > 0), np.clip(lift_margin, 0.0, 1.0), 0.0)


def paint(points: np.ndarray, c1: np.ndarray, c2: np.ndarray, scores: np.ndarray, radius: float = 0.008) -> np.ndarray:
    """Per surface point: max score of the grasps with a finger contact within `radius`.

    points (P, 3); c1, c2 (N, 3) finger contacts; scores (N,) or (N, T).
    Returns (P,) or (P, T); NaN where no candidate touches (no information, drawn grey).
    """
    scores = np.asarray(scores, float)
    flat = scores.ndim == 1
    S = scores[:, None] if flat else scores
    out = np.full((len(points), S.shape[1]), np.nan)
    tree = cKDTree(points)
    for c in (c1, c2):
        good = ~np.isnan(c).any(1)
        for g, nbrs in zip(np.flatnonzero(good), tree.query_ball_point(c[good], radius)):
            if nbrs:
                out[nbrs] = np.fmax(out[nbrs], S[g])
    return out[:, 0] if flat else out
