"""Training data for the transport network: optimised motions for many random problems.

Each problem: a random training object (never a test object) at a random place and rotation on
the table, a random finger-object friction mu in [0.15, 0.6], way B's grasp and presentation
pose, planned lift. The optimiser (grasp2show/transport_opt.py, default limits: friction use
<= 0.7 robust to +-5 mm COM error, <= 60 % of the Franka limits) then finds the shaped and timed
motion; it is the label the network learns to predict (as DJ-GOMP learns GOMP's solutions).
Each optimised motion is also executed in full simulation, so the data says whether it worked.

Stored per problem (data/transport_dataset/<id>.npz): inputs (object name, grasp T_obj_tcp and
width, start and end joint configurations q0 and q1, mu, start pose), labels (coefficients c
(K, 7) and duration T of the shaped motion; duration of the retimed straight motion), and the
simulated outcome.
Usage:  python scripts/generate_transport_data.py --problems 400 [--workers 4]
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
from demo_comparison import plan
from eval_way_b import kinematic_choices, way_b_grasps

from grasp2show.experiment import Scenario
from grasp2show.objects import load_object_set
from grasp2show.transport import friction_params, motion_metrics
from grasp2show.transport_opt import TransportProblem

OUT = ROOT / "data" / "transport_dataset"


def run(job):
    pid, asset, yaw, xy, mu, model, clouds, seed, out = job
    t0 = time.perf_counter()
    T, W, R, _ = way_b_grasps(asset, yaw, model, clouds, 10, start_xy=xy)
    if len(T) == 0:
        return {"id": pid, "object": asset, "status": "no grasp"}
    sc = Scenario(asset, yaw, {"T_obj_tcp": T, "width": W, "success": np.ones(len(T), bool)}, start_xy=xy)
    try:
        grasp, paths = plan(sc, "way_b", kinematic_choices(sc, list(zip(range(len(T)), R.tolist()))))
    except SystemExit:
        return {"id": pid, "object": asset, "status": "no plan"}
    prm = friction_params(mu)
    q0, q1 = paths["lift"][-1], paths["present"][-1]
    pr = TransportProblem(sc.obj, grasp.T_obj_tcp, grasp.width, q0, q1, prm, checker=sc.fc)
    a = pr.retime_only()
    b = pr.shape_and_time(a, seed=seed)
    p = dict(paths)
    p["transport"] = pr.sample(b.c, b.T, sc.ep.model.opt.timestep)
    r = sc.ep.execute(grasp, p, mu=mu, log_every=5)
    m = motion_metrics(r.transport, sc.obj, grasp.T_obj_tcp, prm)
    np.savez_compressed(Path(out) / f"{pid:05d}.npz", object=asset, yaw=yaw, start_xy=np.asarray(xy), mu=mu,
                        T_obj_tcp=grasp.T_obj_tcp, width=grasp.width, q0=q0, q1=q1, T_start=sc.fc.T_start,
                        c=b.c, T=b.T, T_retime=a.T, feasible=b.feasible, retime_feasible=a.feasible,
                        u_static=pr.u_static, u_lim=pr.u_lim, success=r.success, slip_mm=r.slip_mm,
                        slip_deg=r.slip_deg)
    return {"id": pid, "object": asset, "status": "ok", "yaw": round(yaw, 1), "mu": round(mu, 3),
            "T": round(b.T, 3), "T_retime": round(a.T, 3), "feasible": b.feasible, "u_static": round(pr.u_static, 3),
            "success": bool(r.success), "slip_mm": round(r.slip_mm, 2), "u_max_sim": round(m["u_max"], 3),
            "seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problems", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["train", "test"], default="train",
                   help="objects to use; test problems go to data/transport_dataset_test/")
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    global OUT
    train = load_object_set()[args.split]
    if args.split == "test":
        OUT = ROOT / "data" / "transport_dataset_test"
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(i, str(rng.choice(train)), float(rng.uniform(0, 360)),
             (float(rng.uniform(0.40, 0.60)), float(rng.uniform(-0.30, 0.0))), float(rng.uniform(0.15, 0.6)),
             args.model, args.clouds, int(rng.integers(1 << 30)), str(OUT)) for i in range(args.problems)]
    print(f"{len(jobs)} problems on {len(train)} {args.split} objects, {args.workers} workers")
    rows = []
    with Pool(args.workers, maxtasksperchild=8) as pool:
        for k, row in enumerate(pool.imap_unordered(run, jobs), 1):
            rows.append(row)
            if row["status"] == "ok":
                print(f"[{k}/{len(jobs)}] {row['object'][:30]:30s} mu {row['mu']:.2f}  T {row['T']:.2f} s "
                      f"(retimed {row['T_retime']:.2f})  sim {'ok' if row['success'] else 'FAIL'}  ({row['seconds']} s)",
                      flush=True)
            else:
                print(f"[{k}/{len(jobs)}] {row['object'][:30]:30s} {row['status']}", flush=True)
    (OUT / "problems.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["id"])))
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(ok)} problems solved; optimiser feasible {np.mean([r['feasible'] for r in ok]):.0%}; "
          f"simulated success {np.mean([r['success'] for r in ok]):.0%}; median T {np.median([r['T'] for r in ok]):.2f} s "
          f"vs retimed {np.median([r['T_retime'] for r in ok]):.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
