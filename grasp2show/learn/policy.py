"""Use a trained scorer as a grasp-selection policy inside the episode experiments.

The analytic kinematic checks stay (they are exact and cheap: IK + collision at the start and the
target); the learned model ranks every kinematically feasible (grasp, twin, target roll) of the
scenario by its predicted probability of success, replacing the analytic margins.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..experiment import Choice, Scenario
from ..feasibility import twin_of
from ..task import show_pose
from .data import LIMIT_SCALE, global_features, load_clouds
from .model import Scorer


def load_scorer(path: Path, clouds_path: Path, device: str = "cpu") -> tuple[Scorer, list[str], bool]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    clouds = load_clouds(clouds_path)
    if clouds.names != ckpt["names"]:
        raise ValueError("point clouds do not match the ones the model was trained with")
    with_checks = bool(cfg.get("with_checks"))
    model = Scorer(torch.as_tensor(clouds.feats), use_qr_flag=not cfg.get("no_qr_flag", False),
                   n_extra=9 if with_checks else 0)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    model.n_points_eval = int(cfg.get("points", 512))
    model.clouds_meta = clouds
    return model, clouds.names, with_checks


@torch.no_grad()
def learned_choices(sc: Scenario, model: Scorer, names: list[str], with_checks: bool, k: int = 5) -> list[Choice]:
    o = names.index(sc.asset)
    meta = model.clouds_meta
    tuples, T, Tt, chk = [], [], [], []
    for i, rep in sc.reports.items():
        for tw in rep.twins:
            if not tw.start_ok:
                continue
            Tg = sc.T[i] if tw.twin == 0 else twin_of(sc.T[i])
            for j, t in enumerate(tw.targets):
                if not t.reach.ok:
                    continue
                tuples.append((i, tw.twin, j, rep.ok))
                T.append(Tg)
                Tt.append(show_pose(sc.cfg, sc.obj, float(sc.rolls[j])))
                if with_checks:
                    vis = t.visibility.fraction if t.visibility else np.nan
                    chk.append([sc.lift_margin[i], tw.start.limit_margin / LIMIT_SCALE,
                                t.reach.limit_margin / LIMIT_SCALE, t.stability_margin, vis,
                                1.0, 1.0, 1.0, float(t.ok)])
    if not tuples:
        return []
    T, Tt = np.stack(T), np.stack(Tt)
    n = len(T)
    W = np.array([sc.W[i] for i, *_ in tuples])
    glob = global_features(T, sc.fc.T_start, Tt, W, np.full(n, meta.mass[o]), meta.com[o], meta.qr_normal[o])
    if with_checks:
        c = np.nan_to_num(np.clip(np.array(chk, float), -3, 3), nan=-1.0)
        glob = np.concatenate([glob, c.astype(np.float32)], 1)
    dev = next(model.parameters()).device
    f = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    p = []
    for s in range(0, n, 1024):
        sl = slice(s, s + 1024)
        logit, _ = model(f(np.full(len(T[sl]), o), torch.long), f(T[sl, :3, :3]), f(T[sl, :3, 3]),
                         f(glob[sl]), n_points=model.n_points_eval)
        p.append(torch.sigmoid(logit).cpu().numpy())
    p = np.concatenate(p)
    order = np.argsort(-p, kind="stable")[:k]
    return [Choice(*tuples[m][:3], tuples[m][3]) for m in order]
