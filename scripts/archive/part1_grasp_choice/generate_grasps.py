"""Generate grasp candidates and their lift-stability labels (check 1) for the object set.

For each object: sample antipodal contact pairs -> 6-DoF candidates -> drop those where the
open gripper intersects the object -> run the MuJoCo shake test on the rest (object upright,
as it rests on the table). Objects run in parallel, one per CPU core.

Usage:
  python scripts/generate_grasps.py                    # all 40 objects
  python scripts/generate_grasps.py --objects 4        # first 4 (quick check)
  python scripts/generate_grasps.py --split test
Outputs (data/grasps/):
  <object>.npz      T_obj_tcp, width, p1, p2, success, closed, slip_mm, slip_deg
  summary.json      per-object counts, success rates, timing  (small: read this, not the npz)
  plots/<object>.png  contact pairs coloured by label
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
from grasp2show.grasp_sampler import candidates_from_pairs, sample_antipodal
from grasp2show.gripper_sim import GripperSim
from grasp2show.objects import load_object_set, object_from_config, object_set_file

OUT = ROOT / "data" / ("grasps" if os.environ.get("G2S_OBJECT_SET", "v1") == "v1" else
                       f"grasps_{os.environ['G2S_OBJECT_SET']}")


def plot_labels(obj, cand, success, path: Path, title: str, max_lines: int = 150) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = obj.mesh.sample(6000)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=25, azim=35)
    ax.scatter(*pts.T, s=1.5, c="0.45", alpha=0.5, linewidths=0)
    idx = np.random.default_rng(0).permutation(len(success))[:max_lines]
    for i in idx:
        seg = np.stack([cand.p1[i], cand.p2[i]])
        ax.plot(*seg.T, c="tab:green" if success[i] else "tab:red", lw=1.4, alpha=0.9)
        ax.scatter(*seg.T, s=6, c="tab:green" if success[i] else "tab:red")
    ax.set_box_aspect(np.ptp(pts, axis=0))
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def process(job: tuple[str, int, int, int, int, bool]) -> dict:
    name, n_pairs, n_approach, max_tests, seed, resume, min_width = job
    t0 = time.perf_counter()
    if resume and (OUT / f"{name}.npz").exists():           # done in an earlier, interrupted run
        z = np.load(OUT / f"{name}.npz")
        s = z["success"]
        return {"object": name, "mass_kg": round(float(z["mass"]), 3), "sampled": None, "collision_free": None,
                "tested": len(s), "success": int(s.sum()), "success_rate": round(float(s.mean()) if len(s) else 0.0, 3),
                "seconds": 0.0, "resumed": True}
    cfg = SceneConfig(obj=ObjectConfig(asset=name))
    obj = object_from_config(cfg.obj)
    sim = GripperSim(cfg, obj)
    rng = np.random.default_rng(seed)
    p1, p2 = sample_antipodal(obj.mesh, n_pairs, rng, min_width=min_width)
    cand = candidates_from_pairs(p1, p2, n_approach, rng)
    n_sampled = len(cand)
    free = np.array([not sim.in_collision(T, w) for T, w in zip(cand.T_obj_tcp, cand.width)], bool)
    cand = cand.subset(free)
    cand = cand.subset(rng.permutation(len(cand))[:max_tests])
    res = [sim.shake_test(T, w) for T, w in zip(cand.T_obj_tcp, cand.width)]
    success = np.array([r.success for r in res], bool)
    np.savez_compressed(
        OUT / f"{name}.npz", T_obj_tcp=cand.T_obj_tcp.astype(np.float32), width=cand.width.astype(np.float32),
        p1=cand.p1.astype(np.float32), p2=cand.p2.astype(np.float32), success=success,
        closed=np.array([r.closed_on_object for r in res], bool),
        slip_mm=np.array([r.slip_mm for r in res], np.float32),
        slip_deg=np.array([r.slip_deg for r in res], np.float32), mass=obj.mass)
    rate = float(success.mean()) if len(success) else 0.0
    plot_labels(obj, cand, success, OUT / "plots" / f"{name}.png",
                f"{name[:40]}\nm={obj.mass:.2f} kg, lift success {rate:.0%} of {len(success)}")
    return {"object": name, "mass_kg": round(obj.mass, 3), "sampled": n_sampled,
            "collision_free": int(free.sum()), "tested": len(success), "success": int(success.sum()),
            "success_rate": round(rate, 3), "seconds": round(time.perf_counter() - t0, 1)}


def prune(bad: list[str]) -> None:
    """Drop objects that no tested grasp could lift from the active object set file."""
    f = object_set_file()
    s = json.loads(f.read_text())
    bad = set(bad)
    for k in ("objects", "train", "test", "handle_candidates"):
        if k in s:
            s[k] = [n for n in s[k] if n not in bad]
    for k in ("stats", "qr"):
        s[k] = {n: v for n, v in s[k].items() if n not in bad}
    if "classes" in s:
        s["classes"] = {c: [n for n in v if n not in bad] for c, v in s["classes"].items()}
    s["pruned_not_pickable"] = sorted(set(s.get("pruned_not_pickable", [])) | bad)
    f.write_text(json.dumps(s, indent=1))
    print(f"pruned {len(bad)} objects with no lifting grasp; {len(s['objects'])} remain in {f.name}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["all", "train", "test"], default="all")
    p.add_argument("--objects", type=int, default=None, help="only the first N objects")
    p.add_argument("--pairs", type=int, default=60, help="antipodal contact pairs per object")
    p.add_argument("--approaches", type=int, default=8, help="approach angles per pair")
    p.add_argument("--max-tests", type=int, default=300, help="shake tests per object")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-width", type=float, default=0.008,
                   help="smallest distance between the two contacts (0.002 lets thin walls in: a mug's rim)")
    p.add_argument("--resume", action="store_true", help="reuse objects already labelled (after an interruption)")
    p.add_argument("--prune", action="store_true",
                   help="remove objects with no grasp that lifts from the active object set (pickability)")
    args = p.parse_args()

    oset = load_object_set()
    names = oset["objects"] if args.split == "all" else oset[args.split]
    names = names[: args.objects]
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    jobs = [(n, args.pairs, args.approaches, args.max_tests, args.seed + i, args.resume, args.min_width)
            for i, n in enumerate(names)]
    print(f"{len(jobs)} objects on {args.workers} workers")
    t0 = time.perf_counter()
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(process, jobs):
            rows.append(r)
            print(f"[{len(rows)}/{len(jobs)}] {r['object'][:45]:45s} m={r['mass_kg']:.2f} kg  "
                  f"lift success {r['success']}/{r['tested']}  ({r['seconds']} s)")
    rows.sort(key=lambda r: r["object"])
    tested = sum(r["tested"] for r in rows)
    succ = sum(r["success"] for r in rows)
    summary = {
        "config": vars(args), "objects": rows,
        "totals": {"objects": len(rows), "tested": tested, "success": succ,
                   "success_rate": round(succ / max(tested, 1), 3),
                   "wall_seconds": round(time.perf_counter() - t0, 1)},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    if args.prune:
        prune([r["object"] for r in rows if r["success"] == 0])
    print(f"done: {succ}/{tested} successful grasps ({summary['totals']['success_rate']:.0%}) "
          f"in {summary['totals']['wall_seconds']} s -> {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
