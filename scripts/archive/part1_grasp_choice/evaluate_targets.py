"""Run checks 2-5 (target stability, reachability, QR visibility) on the labelled grasps.

Input:  data/grasps/<object>.npz from generate_grasps.py (candidates + check-1 lift labels).
Output: data/feasibility/<object>.npz with, per grasp,
          lift_ok            check 1 (simulated lift)                     (N,)
          start_ok[2]        check 3 per twin                              (N, 2)
          target_ok[2,R]     checks 2+4+5 per twin and target roll         (N, 2, R)
          reach/stab/vis     the three parts of the target check           (N, 2, R)
          vis_frac           visible fraction of the QR                    (N, 2, R)
          feasible           lift_ok and some twin passes start and some roll  (N,)
        data/feasibility/summary.json   per-object rates and the failure breakdown
Objects run in parallel. Usage:
  python scripts/evaluate_targets.py               # all objects with labels
  python scripts/evaluate_targets.py --objects 3
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

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
from grasp2show.config import ObjectConfig, SceneConfig
from grasp2show.feasibility import FeasibilityChecker
from grasp2show.objects import load_object_set
from grasp2show.scene import build_scene

IN, OUT = ROOT / "data" / "grasps", ROOT / "data" / "feasibility"


def process(job):
    name, n_rolls = job
    t0 = time.perf_counter()
    cfg = SceneConfig(obj=ObjectConfig(asset=name))
    scene = build_scene(cfg)
    fc = FeasibilityChecker(scene, cfg)
    z = np.load(IN / f"{name}.npz")
    T, W, lift = z["T_obj_tcp"].astype(float), z["width"].astype(float), z["success"]
    rolls = np.linspace(0, 2 * np.pi, n_rolls, endpoint=False)
    N, R = len(T), n_rolls
    start_ok = np.zeros((N, 2), bool)
    reach, stab, vis = (np.zeros((N, 2, R), bool) for _ in range(3))
    vis_frac = np.full((N, 2, R), np.nan, np.float32)
    stab_margin = np.full((N, 2, R), np.nan, np.float32)
    for i in range(N):
        rep = fc.evaluate(T[i], W[i], rolls)
        for k, tw in enumerate(rep.twins):
            start_ok[i, k] = tw.start_ok
            for j, t in enumerate(tw.targets):
                reach[i, k, j] = t.reach.ok
                stab[i, k, j] = t.stability_margin > 0
                stab_margin[i, k, j] = t.stability_margin
                if t.visibility is not None:
                    vis[i, k, j] = t.visibility.ok()
                    vis_frac[i, k, j] = t.visibility.fraction
    target_ok = reach & stab & vis
    twin_ok = start_ok & target_ok.any(-1)
    feasible = lift & twin_ok.any(-1)
    np.savez_compressed(OUT / f"{name}.npz", lift_ok=lift, start_ok=start_ok, target_ok=target_ok,
                        reach=reach, stab=stab, vis=vis, vis_frac=vis_frac, stab_margin=stab_margin,
                        feasible=feasible, rolls=rolls)
    # Failure breakdown, in pipeline order: the first check that eliminates the grasp.
    has_start = start_ok.any(-1)
    s_ok = start_ok[..., None]
    reach_any = (s_ok & reach).any((1, 2))
    stab_any = (s_ok & reach & stab).any((1, 2))
    first_fail = {
        "lift (check 1)": int((~lift).sum()),
        "start reach (check 3)": int((lift & ~has_start).sum()),
        "target reach (check 4)": int((lift & has_start & ~reach_any).sum()),
        "target stability (check 2)": int((lift & reach_any & ~stab_any).sum()),
        "QR visibility (check 5)": int((lift & stab_any & ~feasible).sum()),
    }
    candidates = lift & has_start          # what a stability-only planner would consider
    single = lift & (start_ok & target_ok[..., 0]).any(-1)   # target set = roll 0 only
    return {"object": name, "grasps": N, "lift_ok": int(lift.sum()), "candidates": int(candidates.sum()),
            "feasible": int(feasible.sum()), "feasible_single_target": int(single.sum()),
            "feasible_rate": round(float(feasible.mean()), 3),
            "lift_ok_but_infeasible": round(float((lift & ~feasible).sum() / max(lift.sum(), 1)), 3),
            "first_failure": first_fail, "seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", type=int, default=None)
    p.add_argument("--rolls", type=int, default=12, help="target poses: rotations about the viewing axis")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    names = [n for n in load_object_set()["objects"] if (IN / f"{n}.npz").exists()][: args.objects]
    if not names:
        print("no labelled grasps found: run scripts/generate_grasps.py first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(names)} objects, {args.rolls} target rolls, {args.workers} workers")
    t0, rows = time.perf_counter(), []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(process, [(n, args.rolls) for n in names]):
            rows.append(r)
            print(f"[{len(rows)}/{len(names)}] {r['object'][:45]:45s} lift ok {r['lift_ok']:3d}/{r['grasps']:3d}"
                  f"  feasible {r['feasible']:3d}  ({r['seconds']} s)")
    rows.sort(key=lambda r: r["object"])
    tot = {k: sum(r[k] for r in rows) for k in ("grasps", "lift_ok", "candidates", "feasible", "feasible_single_target")}
    ff = {k: sum(r["first_failure"][k] for r in rows) for k in rows[0]["first_failure"]}
    summary = {"rolls": args.rolls, "objects": rows,
               "totals": {**tot, "feasible_rate": round(tot["feasible"] / max(tot["grasps"], 1), 3),
                          "candidates_failing_task": round(1 - tot["feasible"] / max(tot["candidates"], 1), 3),
                          "candidates_failing_task_single_target": round(
                              1 - tot["feasible_single_target"] / max(tot["candidates"], 1), 3),
                          "first_failure": ff, "wall_seconds": round(time.perf_counter() - t0, 1)}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    t = summary["totals"]
    print(f"done in {t['wall_seconds']} s: {t['feasible']}/{t['grasps']} grasps feasible. Of the "
          f"{t['candidates']} grasps a stability-only planner would accept (lift ok + reachable), "
          f"{t['candidates_failing_task']:.0%} fail the task ({t['candidates_failing_task_single_target']:.0%} "
          f"with a single target pose). First failure: {ff}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
