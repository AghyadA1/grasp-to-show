"""Dense labels for way B v5: more objects, and every scene with its own friction and weight.

Way B v4 (scripts/archive/part1_grasp_choice/generate_dense_labels.py, --random-scenes 8
--sim-lift --min-width 0.002 --max-candidates 600) learned from 81 scans, each always at its own
weight and tested at friction 1 (the fingers' default: MuJoCo takes the larger of the two
coefficients). Here:

  objects   the v2 scans (train and test), the extra scans of object_set_more.json
            (scripts/select_more_objects.py) and the generated objects' training split
            (object_set_gen.json; its test split is the evaluation's);
  scenes    --scenes per object, each with a random start pose and scanner pose (as v4) and a
            friction and weight drawn as in the evaluation (chooser.sample_conditions: mu
            0.15-1 log-uniform, mass x 0.5-8 up to 3 kg), except scene 0: friction 1, the
            object's own weight (v4's conditions, where the most grasps work);
  labels    as v4 (heatmap.aware_scores: 0 if any check fails, else the smallest normalised
            margin) with every check at this scene's friction and weight: the stability at
            each presentation target (point model) and the lift, which is the simulated shake
            test (gripper_sim) run at this friction and weight. A shake test is only run where
            it matters (a candidate passing every other check for some target), and skipped
            where the outcome follows from one already run: a grasp that held at friction mu and
            mass m also holds with more friction and less mass, and one that failed fails with
            less friction and more mass.

One file per scene: data/dense_v5/<object>__s<k>.npz (the v4 fields plus mu, mass_scale, mass).
Resumable: objects with all their scene files are skipped. Real scans first, then generated.

Usage:  python scripts/generate_wayb_labels.py --workers 11
        python scripts/generate_wayb_labels.py --smoke         (3 objects, 2 scenes, 60 candidates)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("G2S_OBJECT_SET", "v2")
from grasp2show.parallel import Workers

OUT = ROOT / "data" / "dense_v5"


def label_object(job) -> dict:
    try:
        return _label_object(job)
    except Exception as e:  # noqa: BLE001 (an object that does not load: left out, say why)
        return {"object": job[0], "error": f"{type(e).__name__}: {e}"}


def _label_object(job) -> dict:
    name, n_scenes, n_pairs, max_cand, seed, out = job
    from grasp2show.chooser import sample_conditions
    from grasp2show.config import ObjectConfig, SceneConfig, random_camera
    from grasp2show.experiment import lift_margins
    from grasp2show.feasibility import FeasibilityChecker
    from grasp2show.grasp_sampler import candidates_from_pairs, sample_antipodal
    from grasp2show.gripper_sim import GripperSim
    from grasp2show.heatmap import aware_scores
    from grasp2show.objects import object_from_config
    from grasp2show.scene import build_scene
    from grasp2show.stability import finger_contacts
    from grasp2show.task import show_pose

    t0 = time.time()
    out = Path(out)
    obj0 = object_from_config(ObjectConfig(asset=name))
    rng = np.random.default_rng(seed)
    p1, p2 = sample_antipodal(obj0.mesh, n_pairs, rng, min_width=0.002)
    cand = candidates_from_pairs(p1, p2, 8, rng)
    sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj0)
    cand = cand.subset(np.array([not sim.in_collision(T, w) for T, w in zip(cand.T_obj_tcp, cand.width)], bool))
    cand = cand.subset(rng.permutation(len(cand))[:max_cand])
    T_all, W = cand.T_obj_tcp, cand.width
    c1, c2, _ = finger_contacts(obj0.mesh, T_all)
    rolls = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    held: list[list] = [[] for _ in range(len(T_all))]       # (mu, mass, ok) shake tests run so far
    closes = np.ones(len(T_all), bool)                        # both fingers touch after closing
    n_shake = n_inferred = 0
    rows = []
    for k in range(n_scenes):
        f = out / f"{name}__s{k}.npz"
        r = np.random.default_rng([seed, k])
        yaw, xy, cam = float(r.uniform(0, 360)), (float(r.uniform(0.40, 0.60)), float(r.uniform(-0.30, 0.0))), \
            random_camera(r)
        mu, ms = sample_conditions(r, obj0.mass) if k > 0 else (1.0, 1.0)    # scene 0: as v4 (mu 1, own mass)
        if f.exists():
            continue
        cfg = SceneConfig(obj=ObjectConfig(asset=name, start_yaw=np.radians(yaw), start_xy=xy,
                                           friction=(mu, 0.01 * mu, 0.0001)), camera=cam)
        obj = replace(obj0, mass=obj0.mass * ms, inertia=obj0.inertia * ms)
        fc = FeasibilityChecker(build_scene(cfg, obj), cfg)
        reports = [fc.evaluate(T, w, rolls) for T, w in zip(T_all, W)]
        S_other = aware_scores(reports, np.ones(len(T_all)), len(rolls))      # every check but the lift
        lm = lift_margins(obj, T_all, fc.stab)
        lift = np.zeros(len(T_all), bool)
        sim.set_conditions(mu, ms)
        for i in np.flatnonzero(S_other.max(1) > 0):
            if not closes[i]:
                continue
            known = [ok for (mu2, m2, ok) in held[i] if (ok and mu2 <= mu and m2 >= obj.mass)
                     or (not ok and mu2 >= mu and m2 <= obj.mass)]
            if known:
                lift[i] = known[0]
                n_inferred += 1
                continue
            res = sim.shake_test(T_all[i], W[i])
            n_shake += 1
            closes[i] = res.closed_on_object
            lift[i] = res.success
            held[i].append((mu, obj.mass, bool(res.success)))
        sim.set_conditions(None, 1.0)
        lm_s = np.where(lift, np.clip(lm, 0.1, 1.0), -1.0)
        S = np.where(lift[:, None], np.minimum(S_other, lm_s[:, None]), 0.0)
        np.savez_compressed(
            f, T_obj_tcp=T_all.astype(np.float32), width=W.astype(np.float32), c1=c1.astype(np.float32),
            c2=c2.astype(np.float32), lift_margin=lm_s.astype(np.float32), S=S.astype(np.float32), rolls=rolls,
            yaw=yaw, T_start=fc.T_start, T_target=np.stack([show_pose(cfg, obj, a) for a in rolls]),
            cam_pos=np.asarray(cam.pos), cam_look_at=np.asarray(cam.look_at), start_xy=np.asarray(xy),
            lift_sim=lift, mu=mu, mass_scale=ms, mass=obj.mass, com=np.asarray(obj.com))
        rows.append({"scene": k, "mu": mu, "mass": round(obj.mass, 3), "feasible_any_target": float((S.max(1) > 0).mean()),
                     "pass_but_lift": float((S_other.max(1) > 0).mean()), "lift": float(lift.mean())})
    return {"object": name, "candidates": len(T_all), "scenes": rows, "shake_tests": n_shake,
            "inferred": n_inferred, "seconds": round(time.time() - t0, 1)}


def object_lists() -> dict[str, list[str]]:
    """name lists by source, each with its split: train (and test, for the training's metrics)."""
    v2 = json.loads((ROOT / "grasp2show" / "object_set_v2.json").read_text())
    more = json.loads((ROOT / "grasp2show" / "object_set_more.json").read_text())
    gen = json.loads((ROOT / "grasp2show" / "object_set_gen.json").read_text())
    return {"v2": v2["train"] + v2["test"], "more": more["train"] + more["test"], "gen": gen["train"]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", type=int, default=6)
    p.add_argument("--pairs", type=int, default=400, help="antipodal pairs per object (x 8 approach angles)")
    p.add_argument("--max-candidates", type=int, default=400)
    p.add_argument("--sources", nargs="*", default=["v2", "more", "gen"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    lists = object_lists()
    names = [n for s in args.sources for n in lists[s]]
    if args.smoke:
        names = ["Cole_Hardware_Saucer_Glazed_6", "ACE_Coffee_Mug_Kristen_16_oz_cup", "gen_00962"]
        args.scenes, args.max_candidates, args.out = 2, 60, args.out.with_name(args.out.name + "_smoke")
    from grasp2show.objects import asset_dir

    missing = [n for n in names if not (asset_dir(n) / "model.xml").exists()]
    if missing:
        print(f"{len(missing)} objects are not on disk (left out; scans: python scripts/get_objects.py --set "
              f"object_set_more.json, generated: python scripts/generate_hookable.py): {missing[:5]} ...")
    names = [n for n in names if n not in missing]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, args.scenes, args.pairs, args.max_candidates, 1000 + i, str(args.out)) for i, n in enumerate(names)
            if not all((args.out / f"{n}__s{k}.npz").exists() for k in range(args.scenes))]
    print(f"{len(jobs)} objects to label ({len(names) - len(jobs)} done), {args.scenes} scenes each, "
          f"{args.workers} workers -> {args.out}", flush=True)
    log = open(args.out / "log.jsonl", "a", encoding="utf-8")  # noqa: SIM115 (open for the whole run)
    t0 = time.time()
    pool = Workers(args.workers, stall=3600)
    try:
        for i, (job, r) in enumerate(pool.run(label_object, jobs), 1):
            if r is None:
                print(f"  [{i}/{len(jobs)}] {job[0]}: lost (crashed or hung)", flush=True)
                continue
            log.write(json.dumps(r) + "\n")
            log.flush()
            if "error" in r:
                print(f"  [{i}/{len(jobs)}] {r['object']}: {r['error']}", flush=True)
                continue
            sc = r["scenes"]
            el = time.time() - t0
            print(f"  [{i}/{len(jobs)}] {r['object'][:36]:36s} {r['candidates']:4d} candidates; workable for some "
                  f"target {np.mean([s['feasible_any_target'] for s in sc]) if sc else 0:.0%} (all but the lift "
                  f"{np.mean([s['pass_but_lift'] for s in sc]) if sc else 0:.0%}); {r['shake_tests']} shake tests, "
                  f"{r['inferred']} inferred; {r['seconds']:.0f} s; ~{el / i * (len(jobs) - i) / 3600:.1f} h left",
                  flush=True)
    finally:
        pool.close()
        log.close()
    print(f"done in {(time.time() - t0) / 3600:.1f} h -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
