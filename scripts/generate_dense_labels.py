"""Dense labels for the goal-conditioned heatmap network (way B).

For every object: sample dense antipodal candidates once (seeded per object) and drop those where
the open gripper intersects the object; then for every start rotation (yaw) run the analytic
checks on every candidate, both twins, for the 12 target poses (rotations of the presented
object about the scanner's viewing axis). Each candidate gets a score per target: 0 if a check
fails, else the smallest normalised margin, as in the target-aware policy (grasp2show/heatmap.py).

The per-point labels (best grasp touching each surface point) are derived from these files at
training time, so the painting radius can change without recomputing anything.

Usage:  python scripts/generate_dense_labels.py                  # 40 objects x 4 yaws
        python scripts/generate_dense_labels.py --objects 2 --yaws 0 --pairs 100   (quick check)
Outputs: data/dense/<object>__yaw<deg>.npz, data/dense/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp2show.config import ObjectConfig, SceneConfig
from grasp2show.experiment import lift_margins
from grasp2show.feasibility import FeasibilityChecker
from grasp2show.grasp_sampler import candidates_from_pairs, sample_antipodal
from grasp2show.gripper_sim import GripperSim
from grasp2show.heatmap import aware_scores
from grasp2show.objects import load_object_set, object_from_config
from grasp2show.scene import build_scene
from grasp2show.stability import finger_contacts
from grasp2show.task import show_pose

OUT = ROOT / "data" / "dense"


def process(job) -> list[dict]:
    asset, yaws, n_pairs, n_rolls, seed = job
    t0 = time.perf_counter()
    cfg0 = SceneConfig(obj=ObjectConfig(asset=asset))
    obj = object_from_config(cfg0.obj)
    rng = np.random.default_rng(seed)
    p1, p2 = sample_antipodal(obj.mesh, n_pairs, rng)
    cand = candidates_from_pairs(p1, p2, 8, rng)
    sim = GripperSim(cfg0, obj)
    cand = cand.subset(np.array([not sim.in_collision(T, w) for T, w in zip(cand.T_obj_tcp, cand.width)], bool))
    c1, c2, _ = finger_contacts(obj.mesh, cand.T_obj_tcp)
    rolls = np.linspace(0, 2 * np.pi, n_rolls, endpoint=False)
    rows = []
    for yaw in yaws:
        cfg = SceneConfig(obj=ObjectConfig(asset=asset, start_yaw=np.radians(yaw)))
        fc = FeasibilityChecker(build_scene(cfg, obj), cfg)
        lm = lift_margins(obj, cand.T_obj_tcp, fc.stab)
        reports = [fc.evaluate(T, w, rolls) for T, w in zip(cand.T_obj_tcp, cand.width)]
        S = aware_scores(reports, lm, n_rolls)
        np.savez_compressed(
            OUT / f"{asset}__yaw{int(yaw)}.npz", T_obj_tcp=cand.T_obj_tcp.astype(np.float32),
            width=cand.width.astype(np.float32), c1=c1.astype(np.float32), c2=c2.astype(np.float32),
            lift_margin=lm.astype(np.float32), S=S.astype(np.float32), rolls=rolls, yaw=yaw,
            T_start=fc.T_start, T_target=np.stack([show_pose(cfg, obj, r) for r in rolls]))
        rows.append({"object": asset, "yaw": yaw, "candidates": len(cand),
                     "feasible_any_target": round(float((S.max(1) > 0).mean()), 3),
                     "feasible_per_target": round(float((S > 0).mean()), 3)})
    rows[-1]["seconds"] = round(time.perf_counter() - t0, 1)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", type=int, default=None, help="only the first N objects")
    p.add_argument("--yaws", type=float, nargs="+", default=[0, 90, 180, 270])
    p.add_argument("--pairs", type=int, default=400, help="antipodal pairs per object (x 8 approach angles)")
    p.add_argument("--rolls", type=int, default=12)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    names = load_object_set()["objects"][: args.objects]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(n, args.yaws, args.pairs, args.rolls, 1000 + i) for i, n in enumerate(names)]
    print(f"{len(jobs)} objects x {len(args.yaws)} start poses x {args.rolls} targets, {args.workers} workers")
    t0, rows = time.perf_counter(), []
    with Pool(args.workers) as pool:
        for k, rs in enumerate(pool.imap_unordered(process, jobs), 1):
            rows += rs
            print(f"[{k}/{len(jobs)}] {rs[0]['object'][:45]:45s} {rs[0]['candidates']:5d} candidates, "
                  f"feasible for some target {np.mean([r['feasible_any_target'] for r in rs]):.0%} "
                  f"({rs[-1]['seconds']} s)", flush=True)
    summary = {"config": vars(args), "wall_seconds": round(time.perf_counter() - t0, 1), "scenarios": rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {summary['wall_seconds']} s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
