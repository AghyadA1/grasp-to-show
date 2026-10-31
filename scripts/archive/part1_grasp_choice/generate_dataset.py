"""Dataset for the learned scorer: full pick-and-show outcomes for (grasp, twin, target) tuples.

For each object and start rotation (yaw), every lift-stable grasp is checked (checks 2-5, both
twins, 12 target rolls). Samples:
  simulated  : --sim tuples drawn uniformly among the kinematically feasible ones (reachable at the
               start and at the target), executed in the full simulation -> success / reason;
  cheap      : --cheap negatives whose outcome is known without simulation: grasps that dropped
               in the lift test (check 1), and tuples that are unreachable at the start or target.
Every sample also stores what the analytic checks predict, so the learned scorer and the checks
can be compared on exactly the same cases.

Outputs (data/dataset/):
  <object>__yaw<deg>.npz   one file per scenario (see FIELDS below)
  clouds.npz               per object: 1024 surface points + normals + "on QR patch" flag (object
                           frame), mass, centre of mass, QR frame
  summary.json             counts and success rates
Usage:
  python scripts/generate_dataset.py                       # 40 objects x 4 yaws, ~16k episodes
  python scripts/generate_dataset.py --objects 2 --yaws 0 --sim 10   (quick check)
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
from grasp2show.config import ObjectConfig
from grasp2show.experiment import Choice, Scenario
from grasp2show.feasibility import twin_of
from grasp2show.objects import load_object_set, object_from_config
from grasp2show.task import show_pose

GRASPS, OUT = ROOT / "data" / "grasps", ROOT / "data" / "dataset"
REASONS = ["success", "qr", "slip", "dropped", "collision", "plan", "unreachable", "lift"]


def reason_code(reason: str) -> int:
    return REASONS.index(reason.split(":")[0]) if reason else 0


def object_cloud(name: str, n: int = 1024, seed: int = 0) -> dict:
    """Surface samples with normals and a flag for points on the QR patch (object frame)."""
    import trimesh

    obj = object_from_config(ObjectConfig(asset=name))
    pts, fid = trimesh.sample.sample_surface_even(obj.mesh, n, seed=seed)
    if len(pts) < n:                                         # even sampling can return fewer
        extra, fid2 = trimesh.sample.sample_surface(obj.mesh, n - len(pts), seed=seed)
        pts, fid = np.vstack([pts, extra]), np.concatenate([fid, fid2])
    normals = obj.mesh.face_normals[fid]
    T = obj.T_obj_qr
    local = (pts - T[:3, 3]) @ T[:3, :3]                     # points in the QR frame
    half = obj.qr.side / 2
    on_qr = (np.abs(local[:, 0]) < 0.004) & (np.abs(local[:, 1]) <= half) & (np.abs(local[:, 2]) <= half)
    return {"points": pts.astype(np.float32), "normals": normals.astype(np.float32),
            "on_qr": on_qr, "mass": obj.mass, "com": obj.com.astype(np.float32),
            "T_obj_qr": T.astype(np.float32), "qr_side": obj.qr.side}


def run_job(job) -> dict:
    asset, yaw, n_sim, n_cheap, seed = job
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    z = np.load(GRASPS / f"{asset}.npz")
    sc = Scenario(asset, yaw, {k: z[k] for k in ("T_obj_tcp", "width", "success")})
    R = len(sc.rolls)

    feasible_kin, unreachable = [], []
    for i, rep in sc.reports.items():
        for tw in rep.twins:
            for j in range(R):
                t = tw.targets[j] if tw.targets else None
                if tw.start_ok and t is not None and t.reach.ok:
                    feasible_kin.append((i, tw.twin, j))
                else:
                    unreachable.append((i, tw.twin, j))
    lift_failed = [(int(i), int(rng.integers(2)), int(rng.integers(R))) for i in np.nonzero(~sc.lift_ok)[0]]

    rows = []

    def add(i, twin, j, success, reason, simulated, rec=None):
        rep = sc.reports.get(i)
        tw = rep.twins[twin] if rep else None
        t = tw.targets[j] if tw is not None and tw.targets else None
        T_obj_target = show_pose(sc.cfg, sc.obj, float(sc.rolls[j]))
        rows.append({
            "grasp": i, "twin": twin, "roll": float(sc.rolls[j]),
            "T_obj_tcp": sc.T[i] if twin == 0 else twin_of(sc.T[i]), "width": float(sc.W[i]),
            "R_target": T_obj_target[:3, :3], "T_target": T_obj_target,
            "success": success, "reason": reason_code(reason),
            "simulated": simulated,
            # analytic checks for the same tuple
            "lift_ok": bool(sc.lift_ok[i]), "lift_margin": float(sc.lift_margin[i]),
            "start_ok": bool(tw.start_ok) if tw else False,
            "reach_ok": bool(t.reach.ok) if t else False,
            "stab_margin": float(t.stability_margin) if t else np.nan,
            "vis_fraction": float(t.visibility.fraction) if t and t.visibility else np.nan,
            "pred_ok": bool(t.ok) if t else False,
            "start_limit": float(tw.start.limit_margin) if tw else np.nan,
            "target_limit": float(t.reach.limit_margin) if t else np.nan,
            "slip_mm": rec.slip_mm if rec else np.nan, "slip_deg": rec.slip_deg if rec else np.nan,
        })

    for k in rng.permutation(len(feasible_kin))[:n_sim]:
        i, twin, j = feasible_kin[k]
        rec = sc.run("dataset", [Choice(i, twin, j, sc.reports[i].ok)])
        add(i, twin, j, bool(rec.success), rec.reason, True, rec)
    cheap = [(c, "unreachable") for c in unreachable] + [(c, "lift") for c in lift_failed]
    for k in rng.permutation(len(cheap))[:n_cheap]:
        (i, twin, j), why = cheap[k]
        add(i, twin, j, False, why, False)

    keys = rows[0].keys() if rows else []
    arrays = {k: np.array([r[k] for r in rows]) for k in keys}
    arrays.update(yaw_deg=yaw, T_world_obj_start=sc.fc.T_start)
    np.savez_compressed(OUT / f"{asset}__yaw{int(yaw)}.npz", **arrays)
    sim = [r for r in rows if r["simulated"]]
    return {"object": asset, "yaw": yaw, "samples": len(rows), "simulated": len(sim),
            "sim_success": int(sum(r["success"] for r in sim)),
            "pred_ok_success": int(sum(r["success"] for r in sim if r["pred_ok"])),
            "pred_ok": int(sum(r["pred_ok"] for r in sim)),
            "seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", type=int, default=None)
    p.add_argument("--yaws", type=float, nargs="+", default=[0, 90, 180, 270])
    p.add_argument("--sim", type=int, default=100, help="simulated episodes per scenario")
    p.add_argument("--cheap", type=int, default=40, help="known-negative samples per scenario")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    oset = load_object_set()
    names = [n for n in oset["objects"] if (GRASPS / f"{n}.npz").exists()][: args.objects]
    if not names:
        print("no labelled grasps: run scripts/generate_grasps.py first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    clouds = {n: object_cloud(n) for n in names}
    np.savez_compressed(OUT / "clouds.npz", names=np.array(names),
                        **{f"{k}__{n}": v for n, c in clouds.items() for k, v in c.items()})
    jobs = [(n, y, args.sim, args.cheap, 1000 * k + int(y)) for k, n in enumerate(names) for y in args.yaws]
    print(f"{len(jobs)} scenarios, up to {args.sim} simulated episodes each, {args.workers} workers")
    t0, rows = time.perf_counter(), []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            rows.append(r)
            print(f"[{len(rows)}/{len(jobs)}] {r['object'][:38]:38s} yaw {r['yaw']:3.0f}: "
                  f"{r['sim_success']}/{r['simulated']} succeed  ({r['seconds']} s)")
    sim = sum(r["simulated"] for r in rows)
    succ = sum(r["sim_success"] for r in rows)
    pok = sum(r["pred_ok"] for r in rows)
    summary = {"config": vars(args), "scenarios": rows, "reasons": REASONS, "totals": {
        "samples": sum(r["samples"] for r in rows), "simulated": sim, "sim_success_rate": round(succ / max(sim, 1), 3),
        "success_if_checks_say_feasible": round(sum(r["pred_ok_success"] for r in rows) / max(pok, 1), 3),
        "wall_seconds": round(time.perf_counter() - t0, 1)}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["totals"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
