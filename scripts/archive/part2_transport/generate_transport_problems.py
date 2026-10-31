"""More transport problems for self-supervised training: geometry only, no optimiser, no simulation.

Same problem distribution as scripts/generate_transport_data.py (a random training object at a
random place and rotation, way B's grasp and presentation pose, planned lift), but only the
problem is stored: object, grasp, start and end joint configurations, and a friction (training
draws a fresh friction each time anyway). About a second or two per problem instead of ~40 s, so
thousands are cheap. The 5 validation objects of the training scripts are skipped, so
validation stays on unseen objects.

Usage:  python scripts/generate_transport_problems.py --problems 3000 [--workers 4] [--seed 1]
Output: data/transport_problems/<id>.npz
"""
from __future__ import annotations

import argparse
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
from demo_comparison import plan
from eval_way_b import kinematic_choices, way_b_grasps

from grasp2show.experiment import Scenario
from grasp2show.objects import load_object_set

OUT = ROOT / "data" / "transport_problems"


def run(job):
    pid, asset, yaw, xy, mu, model, clouds = job
    out = OUT / f"{pid:05d}.npz"
    if out.exists():
        return "exists"
    T, W, R, _ = way_b_grasps(asset, yaw, model, clouds, 10, start_xy=xy)
    if len(T) == 0:
        return "no grasp"
    sc = Scenario(asset, yaw, {"T_obj_tcp": T, "width": W, "success": np.ones(len(T), bool)}, start_xy=xy)
    try:
        grasp, paths = plan(sc, "way_b", kinematic_choices(sc, list(zip(range(len(T)), R.tolist()))))
    except SystemExit:
        return "no plan"
    np.savez_compressed(out, object=asset, yaw=yaw, start_xy=np.asarray(xy), mu=mu, T_obj_tcp=grasp.T_obj_tcp,
                        width=grasp.width, q0=paths["lift"][-1], q1=paths["present"][-1], T_start=sc.fc.T_start)
    return "ok"


def held_out_objects(k: int = 5, seed: int = 0) -> set[str]:
    """The validation objects of train_transport_net.py / train_transport_selfsup.py."""
    d = ROOT / "data" / "transport_dataset"
    names = sorted({str(np.load(f, allow_pickle=True)["object"]) for f in d.glob("*.npz")
                    if bool(np.load(f, allow_pickle=True)["feasible"])})
    return set(np.random.default_rng(seed).choice(names, min(k, len(names) - 1), replace=False))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problems", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    skip = held_out_objects()
    objs = [o for o in load_object_set()["train"] if o not in skip]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(i, str(rng.choice(objs)), float(rng.uniform(0, 360)),
             (float(rng.uniform(0.40, 0.60)), float(rng.uniform(-0.30, 0.0))), float(rng.uniform(0.15, 0.6)),
             args.model, args.clouds) for i in range(args.problems)]
    print(f"{len(jobs)} problems on {len(objs)} training objects (validation objects skipped), {args.workers} workers")
    t0, counts = time.perf_counter(), {}
    with Pool(args.workers, maxtasksperchild=20) as pool:
        for k, st in enumerate(pool.imap_unordered(run, jobs), 1):
            counts[st] = counts.get(st, 0) + 1
            if k % 50 == 0 or k == len(jobs):
                el = time.perf_counter() - t0
                print(f"[{k}/{len(jobs)}] {counts}  ({el / 60:.1f} min, ~{el / k * (len(jobs) - k) / 60:.0f} min left)",
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
