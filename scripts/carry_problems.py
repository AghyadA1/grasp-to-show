"""Carry problems for learning the carry (grasp2show/carry.py): grasped objects to bring from the
table to the scanner, on the training objects only.

Per object, --scenes scenes, each drawn like the evaluation's (pipeline.sample_scene: start pose,
scanner pose, friction, weight, and the pose error the robot does not know) but from other seeds.
In each, what the robot would run:
  pinch  way B's grasps (data/models/heatmap_net_v5.pt), checked, then shake-tested at the scene's
         friction and weight (pipeline.Checked.wayb_verified): the first two that can be planned;
  hook   the hook network's hooks (as evaluate.py's prepare stage; --hook-objects generated
         objects get them computed here), ranked robust: the first one that can be planned.
The planner's paths (approach, descend, lift; and its transit and present, as the baseline) are
kept. Objects: the v2 and extra scans' and the generated objects' training splits; 10 % of them
(by name) are validation objects (their problems are never trained on).

Usage:  python scripts/carry_problems.py --workers 11            -> data/carry/problems/<object>.pkl
        python scripts/carry_problems.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("G2S_OBJECT_SET", "v2")
from grasp2show.parallel import Workers

OUT = ROOT / "data" / "carry" / "problems"
_WAYB = None


def split_of(name: str) -> str:
    from grasp2show.pipeline import crc

    return "val" if crc("carry" + name) % 10 == 0 else "train"


def scene_of(name: str, k: int, scale: float):
    """Like pipeline.sample_scene, other seeds."""
    from grasp2show.chooser import sample_conditions
    from grasp2show.config import ObjectConfig, random_camera
    from grasp2show.objects import object_from_config
    from grasp2show.pipeline import POSE_ERR_DEG, POSE_ERR_MM, Scene, crc

    r = np.random.default_rng(crc("carry" + name) * 100 + k)
    yaw, xy, cam = float(r.uniform(0, 360)), (float(r.uniform(0.40, 0.60)), float(r.uniform(-0.30, 0.0))), \
        random_camera(r)
    cam = replace(cam, width=int(cam.width * scale), height=int(cam.height * scale))
    mu, f = sample_conditions(r, object_from_config(ObjectConfig(asset=name)).mass)
    e_xy = np.clip(r.normal(0, POSE_ERR_MM / 1000, 2), -2.5 * POSE_ERR_MM / 1000, 2.5 * POSE_ERR_MM / 1000)
    e_yaw = float(np.clip(r.normal(0, np.radians(POSE_ERR_DEG)), -np.radians(2.5 * POSE_ERR_DEG),
                          np.radians(2.5 * POSE_ERR_DEG)))
    return Scene(name, yaw, xy, cam, mu, f), (float(e_xy[0]), float(e_xy[1]), e_yaw)


def object_problems(job) -> dict:
    try:
        return _object_problems(job)
    except Exception as e:  # noqa: BLE001 (an object that does not load: left out, say why)
        return {"object": job[0], "error": f"{type(e).__name__}: {e}"}


def _object_problems(job) -> dict:
    global _WAYB
    name, n_scenes, scale, hooks_dir, out = job
    from grasp2show.carry import CarryProblem
    from grasp2show.chooser import rank_hooks
    from grasp2show.experiment import Scenario
    from grasp2show.feasibility import twin_of
    from grasp2show.gripper_sim import GripperSim
    from grasp2show.hanging import hook_options, load_hook_file, plan_hook
    from grasp2show.pipeline import Checked, WayB, add_wayb
    from grasp2show.task import show_pose
    from grasp2show.transforms import inv_T

    t0 = time.time()
    if _WAYB is None:
        _WAYB = WayB()
    f_h = Path(hooks_dir) / f"{name}.npz"
    all_hooks = load_hook_file(f_h) if f_h.exists() else []
    split = split_of(name)
    probs = []
    for k in range(n_scenes):
        scene, off = scene_of(name, k, scale)
        common = {"asset": name, "yaw_deg": scene.yaw_deg, "start_xy": tuple(scene.start_xy), "camera": scene.camera,
                  "mu": scene.mu, "mass_scale": scene.mass_scale, "offset": off, "split": split}
        ck = add_wayb(Checked(scene, None, None, [], []), _WAYB)
        n_pinch = 0
        S = ck.wb
        for c in (ck.wayb_verified() if S is not None else [])[:20]:
            tw = S.reports[c.grasp].twins[c.twin]
            t = tw.targets[c.roll_idx]
            T = S.T[c.grasp] if c.twin == 0 else twin_of(S.T[c.grasp])
            w = float(S.W[c.grasp])
            paths, _ = S.ep.plan_analytic(T, show_pose(S.cfg, S.obj, float(S.rolls[c.roll_idx])), tw.start.q,
                                          t.reach.q, checker=S.fc, width=w)
            if paths is None:
                continue
            probs.append(CarryProblem(**common, kind="pinch", T_obj_tcp=T, width=w,
                                      open_width=min(w + 0.02, S.cfg.gripper.max_width),
                                      paths={key: [np.asarray(q) for q in paths[key]] for key in ("approach", "descend", "lift")},
                                      q_show=np.asarray(paths["present"][-1]), T_tcp_obj=inv_T(T),
                                      baseline={"transit": [np.asarray(q) for q in paths["transit"]],
                                                "present": [np.asarray(q) for q in paths["present"]]}))
            n_pinch += 1
            if n_pinch == 2:
                break
        hooks = [h for h in all_hooks if h.get("mu_min", 0.0) <= scene.mu + 1e-9]
        if hooks:
            dummy = {"T_obj_tcp": np.eye(4)[None], "width": np.array([0.04]), "success": np.array([False])}
            sc = Scenario(name, scene.yaw_deg, dummy, start_xy=scene.start_xy, camera=scene.camera,
                          mass_scale=scene.mass_scale, mu=scene.mu)
            opts = hook_options(sc.fc, GripperSim(sc.cfg, sc.obj), hooks, T_start=sc.ep.resting_pose())
            for o in rank_hooks(opts, hooks)[:10]:
                paths, _ = plan_hook(sc.ep, sc.fc, o)
                if paths is None:
                    continue
                probs.append(CarryProblem(**common, kind="hook", T_obj_tcp=o.T_obj_tcp, width=o.width,
                                          open_width=o.width,
                                          paths={key: [np.asarray(q) for q in paths[key]] for key in ("approach", "descend", "lift")},
                                          q_show=np.asarray(paths["present"][-1]), T_tcp_obj=o.hang.T_tcp_obj,
                                          baseline={"transit": [np.asarray(q) for q in paths["transit"]],
                                                    "present": [np.asarray(q) for q in paths["present"]]}))
                break
    with open(Path(out) / f"{name}.pkl", "wb") as fh:
        pickle.dump(probs, fh)
    return {"object": name, "pinch": sum(p.kind == "pinch" for p in probs), "hook": sum(p.kind == "hook" for p in probs),
            "seconds": round(time.time() - t0, 1)}


def hook_test_job(args):
    name, cands, hooks_dir = args
    from grasp2show.pipeline import test_hooks

    try:
        test_hooks(name, cands, Path(hooks_dir))
        return name, ""
    except Exception as e:  # noqa: BLE001
        return name, f"{type(e).__name__}: {e}"


def main() -> int:
    from grasp2show.pipeline import SCANNER_SCALE, hook_dir

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", type=int, default=2)
    p.add_argument("--hook-objects", type=int, default=400, help="generated training objects given hooks here")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    v2 = json.loads((ROOT / "grasp2show" / "object_set_v2.json").read_text())
    more = json.loads((ROOT / "grasp2show" / "object_set_more.json").read_text())
    gen = json.loads((ROOT / "grasp2show" / "object_set_gen.json").read_text())
    from grasp2show.objects import asset_dir

    names = [n for n in v2["train"] + more["train"] + gen["train"] if (asset_dir(n) / "model.xml").exists()]
    out = OUT if not args.smoke else OUT.with_name("problems_smoke")
    if args.smoke:
        names = [n for n in ["3D_Dollhouse_Sink", "gen_00962", "ACE_Coffee_Mug_Kristen_16_oz_cup"]
                 if (asset_dir(n) / "model.xml").exists()]
        args.scenes, args.hook_objects = 2, 0
    out.mkdir(parents=True, exist_ok=True)
    hooks_dir = hook_dir()
    pool = Workers(args.workers, stall=3600)
    try:
        gen_train = [n for n in names if n.startswith("gen_")][: args.hook_objects]
        todo = [n for n in gen_train if not (hooks_dir / f"{n}.npz").exists()]
        if todo:
            import torch

            from grasp2show.learn.hooknet import load_hook_net
            from grasp2show.pipeline import propose_hooks

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            net = load_hook_net(device=dev)
            jobs = [(n, propose_hooks(net, n, device=dev), str(hooks_dir)) for n in todo]
            print(f"hooks for {len(jobs)} generated training objects", flush=True)
            for i, (job, r) in enumerate(pool.run(hook_test_job, jobs), 1):
                print(f"  [{i}/{len(jobs)}] {job[0]}: " + ("lost" if r is None else r[1] or "ok"), flush=True)
        jobs = [(n, args.scenes, SCANNER_SCALE, str(hooks_dir), str(out)) for n in names if not (out / f"{n}.pkl").exists()]
        print(f"problems: {len(jobs)} objects ({len(names) - len(jobs)} done) -> {out}", flush=True)
        t0, tot = time.time(), {"pinch": 0, "hook": 0}
        for i, (job, r) in enumerate(pool.run(object_problems, jobs), 1):
            if r is None or "error" in r:
                print(f"  [{i}/{len(jobs)}] {job[0]}: " + ("lost" if r is None else r["error"]), flush=True)
                continue
            tot["pinch"] += r["pinch"]
            tot["hook"] += r["hook"]
            el = time.time() - t0
            print(f"  [{i}/{len(jobs)}] {r['object'][:36]:36s} pinch {r['pinch']}, hook {r['hook']} ({r['seconds']:.0f} s); "
                  f"so far {tot['pinch']} pinch, {tot['hook']} hook; ~{el / i * (len(jobs) - i) / 60:.0f} min left", flush=True)
    finally:
        pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
