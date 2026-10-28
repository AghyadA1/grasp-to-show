"""Dense labels for the goal-conditioned heatmap network (way B).

For every object: sample dense antipodal candidates once (seeded per object) and drop those where
the open gripper intersects the object; then for every start rotation (yaw) run the analytic
checks on every candidate, both twins, for the 12 target poses (rotations of the presented
object about the scanner's viewing axis). Each candidate gets a score per target: 0 if a check
fails, else the smallest normalised margin, as in the target-aware policy (grasp2show/heatmap.py).

The per-point labels (best grasp touching each surface point) are derived from these files at
training time, so the painting radius can change without recomputing anything.

With --sim-lift (part 3), check 1 (does the grasp lift the object) comes from the simulated
shake test instead of the analytic soft-finger margin, which was validated on boxes but rejects
every grasp of a mug (the fingers press a curved rim along a line, not at a point, and hold much
more torque than the point-contact model allows). Grasps that lift in simulation get the
analytic margin clipped to [0.1, 1]; the others are rejected. --max-candidates caps the shake
tests per object.

With --grip-model all (part 3), the stability check at each presentation target uses the grip
model with every contact (grip_model.py; validated AUC 0.96 vs 0.92) instead of the two-pad
point model: the contacts come from closing the simulated gripper on each candidate once.

With --random-scenes K (part 3), each object gets K scenes instead of the fixed yaws: a random
start rotation and place on the table and a random scanner pose (config.random_camera); the
scanner pose is stored with the labels, since the targets depend on it.

Usage:  python scripts/generate_dense_labels.py                  # 40 objects x 4 yaws
        python scripts/generate_dense_labels.py --objects 2 --yaws 0 --pairs 100   (quick check)
        G2S_OBJECT_SET=v2 python scripts/generate_dense_labels.py --random-scenes 8   (part 3)
Outputs: data/dense[_v2]/<object>__yaw<deg>.npz or <object>__s<k>.npz, summary.json
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
from grasp2show.config import CameraConfig, ObjectConfig, SceneConfig, random_camera
from grasp2show.experiment import lift_margins
from grasp2show.feasibility import FeasibilityChecker
from grasp2show.grasp_sampler import candidates_from_pairs, sample_antipodal
from grasp2show.grip_model import apply_to_reports
from grasp2show.gripper_sim import GripperSim, close_contacts
from grasp2show.heatmap import aware_scores
from grasp2show.objects import load_object_set, object_from_config
from grasp2show.scene import build_scene
from grasp2show.stability import finger_contacts
from grasp2show.task import show_pose

OUT = ROOT / "data" / ("dense" if os.environ.get("G2S_OBJECT_SET", "v1") == "v1" else
                       f"dense_{os.environ['G2S_OBJECT_SET']}")


def process(job) -> list[dict]:
    asset, scenes, n_pairs, n_rolls, seed, sim_lift, max_cand, grip_model, out, u_ok, min_width = job
    t0 = time.perf_counter()
    cfg0 = SceneConfig(obj=ObjectConfig(asset=asset))
    obj = object_from_config(cfg0.obj)
    rng = np.random.default_rng(seed)
    p1, p2 = sample_antipodal(obj.mesh, n_pairs, rng, min_width=min_width)
    cand = candidates_from_pairs(p1, p2, 8, rng)
    sim = GripperSim(cfg0, obj)
    cand = cand.subset(np.array([not sim.in_collision(T, w) for T, w in zip(cand.T_obj_tcp, cand.width)], bool))
    if max_cand:
        cand = cand.subset(rng.permutation(len(cand))[:max_cand])
    lift_sim = np.array([sim.shake_test(T, w).success for T, w in zip(cand.T_obj_tcp, cand.width)], bool) \
        if sim_lift else None
    contacts = [close_contacts(sim, T, w) for T, w in zip(cand.T_obj_tcp, cand.width)] \
        if grip_model == "all" else None
    c1, c2, _ = finger_contacts(obj.mesh, cand.T_obj_tcp)
    rolls = np.linspace(0, 2 * np.pi, n_rolls, endpoint=False)
    rows = []
    for tag, yaw, xy, cam in scenes:
        extra = {} if xy is None else {"start_xy": tuple(xy)}
        cfg = SceneConfig(obj=ObjectConfig(asset=asset, start_yaw=np.radians(yaw), **extra),
                          camera=cam if cam is not None else CameraConfig())
        fc = FeasibilityChecker(build_scene(cfg, obj), cfg)
        lm = lift_margins(obj, cand.T_obj_tcp, fc.stab)
        if lift_sim is not None:
            lm = np.where(lift_sim, np.clip(lm, 0.1, 1.0), -1.0)
        reports = [fc.evaluate(T, w, rolls) for T, w in zip(cand.T_obj_tcp, cand.width)]
        if contacts is not None:
            apply_to_reports(reports, contacts, obj, np.stack([show_pose(cfg, obj, r) for r in rolls]), u_ok=u_ok)
        S = aware_scores(reports, lm, n_rolls)
        np.savez_compressed(
            Path(out) / f"{asset}__{tag}.npz", T_obj_tcp=cand.T_obj_tcp.astype(np.float32),
            width=cand.width.astype(np.float32), c1=c1.astype(np.float32), c2=c2.astype(np.float32),
            lift_margin=lm.astype(np.float32), S=S.astype(np.float32), rolls=rolls, yaw=yaw,
            T_start=fc.T_start, T_target=np.stack([show_pose(cfg, obj, r) for r in rolls]),
            cam_pos=np.asarray(cfg.camera.pos), cam_look_at=np.asarray(cfg.camera.look_at),
            start_xy=np.asarray(cfg.obj.start_xy),
            **({} if lift_sim is None else {"lift_sim": lift_sim}))
        rows.append({"object": asset, "scene": tag, "yaw": round(yaw, 1), "candidates": len(cand),
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
    p.add_argument("--random-scenes", type=int, default=0,
                   help="K random scenes per object (start pose + scanner pose) instead of --yaws")
    p.add_argument("--split", choices=["all", "train", "test"], default="all")
    p.add_argument("--sim-lift", action="store_true", help="check 1 from the simulated shake test (part 3)")
    p.add_argument("--max-candidates", type=int, default=0, help="cap per object (0: no cap)")
    p.add_argument("--min-width", type=float, default=0.008,
                   help="smallest distance between the two contacts (0.002 lets thin walls in: a mug's rim)")
    p.add_argument("--u-ok", type=float, default=0.8, help="every-contact model: holds if u below this")
    p.add_argument("--out", type=Path, default=None, help="output folder (default data/dense[_<set>])")
    p.add_argument("--grip-model", choices=["point", "all"], default="point",
                   help="stability at the targets: two-pad point model or every contact (part 3)")
    p.add_argument("--prune", action="store_true",
                   help="drop objects with no feasible grasp in any scene (not pickable from the table)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    global OUT
    if args.out is not None:
        OUT = args.out
    oset = load_object_set()
    names = (oset["objects"] if args.split == "all" else oset[args.split])[: args.objects]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, n in enumerate(names):
        if args.random_scenes:
            r = np.random.default_rng(5000 + oset["objects"].index(n))
            scenes = [(f"s{k}", float(r.uniform(0, 360)), (float(r.uniform(0.40, 0.60)), float(r.uniform(-0.30, 0.0))),
                       random_camera(r)) for k in range(args.random_scenes)]
        else:
            scenes = [(f"yaw{int(y)}", float(y), None, None) for y in args.yaws]
        jobs.append((n, scenes, args.pairs, args.rolls, 1000 + oset["objects"].index(n), args.sim_lift,
                     args.max_candidates, args.grip_model, str(OUT), args.u_ok, args.min_width))
    n_sc = args.random_scenes or len(args.yaws)
    print(f"{len(jobs)} objects x {n_sc} scenes x {args.rolls} targets, {args.workers} workers -> {OUT}")
    t0, rows = time.perf_counter(), []
    with Pool(args.workers) as pool:
        for k, rs in enumerate(pool.imap_unordered(process, jobs), 1):
            rows += rs
            print(f"[{k}/{len(jobs)}] {rs[0]['object'][:45]:45s} {rs[0]['candidates']:5d} candidates, "
                  f"feasible for some target {np.mean([r['feasible_any_target'] for r in rs]):.0%} "
                  f"({rs[-1]['seconds']} s)", flush=True)
    summary = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "wall_seconds": round(time.perf_counter() - t0, 1), "scenarios": rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    if args.prune:
        sys.path.insert(0, str(ROOT / "scripts"))
        from generate_grasps import prune

        best = {}
        for r in rows:
            best[r["object"]] = max(best.get(r["object"], 0.0), r["feasible_any_target"])
        bad = [n for n, b in best.items() if b == 0]
        for n in bad:                                  # their label files would only teach "nothing works"
            for f in OUT.glob(f"{n}__*.npz"):
                f.unlink()
        prune(bad)
    print(f"done in {summary['wall_seconds']} s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
