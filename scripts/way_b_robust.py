"""Way B with robustness: does preferring grasps that survive small errors help the network too?

Way B (frozen: data/models/heatmap_net_v4.pt, the goal-conditioned heatmap network of parts 1
and 3) scores every point of the object's cloud for every presentation target and proposes a
grasp there. Its grasps are ranked by that score alone. For the analytic pinch, adding each
candidate's robustness (the share of its 3 mm / 6 deg shifts that still hold) raised the success
of the grasp the robot runs from 68 % to 75 % where the rankings disagreed. Three rankings of
way B's grasps, on the same scenes:

  score    the network's predicted success, highest first (as way B has always run);
  cluster  that score averaged over the points within 1.5 cm (same target): a high point among
           high points before an isolated peak (the "cluster" idea; free: no simulation);
  robust   the score's best 10 grasps, each shifted 8 ways and lift-and-shake tested in the
           floating gripper (chooser.pinch_robustness): ranked by score x robustness.

Scenes: the 31 unseen objects of set v2 x --scenes (start pose and scanner pose from the seeds of
eval_way_b_v2.py, 1280 x 960 scanner), each with an error in the object's pose the robot does
not know (2 mm, 1.5 deg; the same for every ranking): the error is what robustness is for.
Every ranking runs its first grasp whose motion can be planned, in full simulation.

Usage:  python scripts/way_b_robust.py --workers 4                (data/runs/<today>_way_b_robust/)
        python scripts/way_b_robust.py --smoke
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
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):
    sys.path.insert(1, str(_d))
os.environ.setdefault("G2S_OBJECT_SET", "v2")
from grasp2show.parallel import Workers

RANKINGS = ("score", "cluster", "robust")
MODEL = ROOT / "data" / "models" / "heatmap_net_v4.pt"
CLOUDS = ROOT / "data" / "dataset_v2" / "clouds.npz"
K, MIN_SEP, CLUSTER_R = 10, 0.01, 0.015


def picks(q: np.ndarray, pts: np.ndarray, n: int) -> list[tuple[int, int]]:
    """(target, point) pairs by decreasing q (12, P), at least MIN_SEP apart per target."""
    out = []
    for flat in np.argsort(-q, axis=None):
        r, i = divmod(int(flat), q.shape[1])
        if any(r == r2 and np.linalg.norm(pts[i] - pts[i2]) < MIN_SEP for r2, i2 in out):
            continue
        out.append((r, i))
        if len(out) == n:
            break
    return out


def scene(args) -> dict:
    asset, k, yaw, xy, cam, off = args
    import torch
    from eval_way_b import _load, kinematic_choices
    from scipy.spatial import cKDTree

    from grasp2show.chooser import pinch_robustness
    from grasp2show.config import ObjectConfig, SceneConfig
    from grasp2show.experiment import Scenario
    from grasp2show.gripper_sim import GripperSim
    from grasp2show.learn.dense import condition, grasp_from_prediction, split_outputs
    from grasp2show.objects import object_from_config
    from grasp2show.scene import start_pose
    from grasp2show.task import show_pose

    t0 = time.time()
    net, clouds = _load(MODEL, CLOUDS)
    o = clouds.names.index(asset)
    cfg = SceneConfig(obj=ObjectConfig(asset=asset, start_yaw=np.radians(yaw), start_xy=tuple(xy)), camera=cam)
    obj = object_from_config(cfg.obj)
    rolls = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    T0 = start_pose(cfg, obj)
    cond = np.stack([condition(T0, show_pose(cfg, obj, r), cfg.camera.pos, clouds.mass[o], clouds.com[o])
                     for r in rolls])
    with torch.no_grad():
        pr = {kk: v.numpy() for kk, v in split_outputs(net(torch.full((12,), o), torch.as_tensor(cond))).items()}
    q = 1 / (1 + np.exp(-np.clip(pr["logit"], -50, 50))) * pr["score"]          # (12, P)
    pts = clouds.feats[o, :, :3]
    nb = cKDTree(pts).query_ball_point(pts, CLUSTER_R)
    q_cluster = np.stack([[q[r, idx].mean() for idx in nb] for r in range(12)])
    sim = GripperSim(cfg, obj)

    def grasps(pp):
        out = []
        for r, i in pp:
            T = grasp_from_prediction(pts[i], pr["approach"][r, i], pr["closing"][r, i], pr["offset"][r, i])
            w = float(np.clip(pr["width"][r, i], 0.0, 0.08))
            if not sim.in_collision(T, w):                 # the open gripper must not hit the object
                out.append(((r, i), T, w))
            if len(out) == K:
                break
        return out

    by = {"score": grasps(picks(q, pts, 3 * K)), "cluster": grasps(picks(q_cluster, pts, 3 * K))}
    robust = {}
    for key, T, w in by["score"]:
        robust[key] = pinch_robustness(sim, T, w)
    by["robust"] = sorted(by["score"], key=lambda g: -q[g[0]] * robust[g[0]])
    union, index = [], {}
    for g in by["score"] + by["cluster"]:
        if g[0] not in index:
            index[g[0]] = len(union)
            union.append(g)
    rec = {"object": asset, "scene": k, "yaw": round(yaw, 1), "pose_error": [round(off[0] * 1000, 2),
           round(off[1] * 1000, 2), round(float(np.degrees(off[2])), 2)]}
    if not union:
        for rk in RANKINGS:
            rec.update({f"{rk}_success": False, f"{rk}_reason": "no grasp"})
        return rec
    sc = Scenario(asset, yaw, {"T_obj_tcp": np.stack([g[1] for g in union]), "width": np.array([g[2] for g in union]),
                               "success": np.ones(len(union), bool)}, start_xy=xy, camera=cam)
    done = {}
    for rk in RANKINGS:
        order = [(index[g[0]], g[0][0]) for g in by[rk]]
        ok, why, first = False, "no grasp", None
        for c in kinematic_choices(sc, order):
            key = (c.grasp, c.twin, c.roll_idx)
            if key not in done:
                done[key] = sc._execute(c, obj_offset=off)
            e = done[key]
            if e.reason.startswith("plan"):
                why = e.reason
                continue
            ok, why, first = bool(e.success), e.reason, union[c.grasp][0]
            break
        rec.update({f"{rk}_success": ok, f"{rk}_reason": why,
                    f"{rk}_q": float(q[first]) if first else None,
                    f"{rk}_robust": robust.get(first) if first else None})
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def summary(recs: list[dict], classes: dict) -> dict:
    from scipy.stats import binomtest

    cls = {n: c for c, ns in classes.items() for n in ns}
    out = {"scenes": len(recs), "objects": len({r["object"] for r in recs})}
    for rk in RANKINGS:
        s = np.array([r[f"{rk}_success"] for r in recs])
        out[rk] = {"success": float(s.mean()),
                   "by_class": {c: float(np.mean([r[f"{rk}_success"] for r in recs if cls.get(r["object"]) == c]))
                                for c in classes if any(cls.get(r["object"]) == c for r in recs)}}
        if rk != "score":
            w = sum(r[f"{rk}_success"] and not r["score_success"] for r in recs)
            lo = sum(r["score_success"] and not r[f"{rk}_success"] for r in recs)
            out[rk]["vs_score"] = {"won": int(w), "lost": int(lo),
                                   "p": float(binomtest(w, w + lo).pvalue) if w + lo else 1.0}
    return out


def main() -> int:
    from grasp2show.config import random_camera
    from grasp2show.objects import load_object_set
    from grasp2show.pipeline import POSE_ERR_DEG, POSE_ERR_MM, SCANNER_SCALE

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", type=int, default=8)
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    p.add_argument("--run", type=Path, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    oset = load_object_set()
    test = oset["test"][:3] if args.smoke else oset["test"]
    scenes = 1 if args.smoke else args.scenes
    out = args.run or ROOT / "data" / "runs" / f"{time.strftime('%Y-%m-%d')}_way_b_robust{'_smoke' if args.smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    rec_f = out / "records.jsonl"
    done = {(d["object"], d["scene"]) for d in map(json.loads, rec_f.read_text().splitlines())} if rec_f.exists() else set()
    jobs = []
    for n in test:
        r = np.random.default_rng(9000 + oset["objects"].index(n))    # eval_way_b_v2.py's scenes
        e = np.random.default_rng(7000 + oset["objects"].index(n))    # the unknown pose errors
        for k in range(scenes):
            yaw, xy, cam = float(r.uniform(0, 360)), (float(r.uniform(0.40, 0.60)), float(r.uniform(-0.30, 0.0))), \
                random_camera(r)
            cam = replace(cam, width=int(cam.width * SCANNER_SCALE), height=int(cam.height * SCANNER_SCALE))
            exy = np.clip(e.normal(0, POSE_ERR_MM / 1000, 2), -2.5 * POSE_ERR_MM / 1000, 2.5 * POSE_ERR_MM / 1000)
            eyaw = float(np.clip(e.normal(0, np.radians(POSE_ERR_DEG)), -np.radians(2.5 * POSE_ERR_DEG),
                                 np.radians(2.5 * POSE_ERR_DEG)))
            if (n, k) not in done:
                jobs.append((n, k, yaw, xy, cam, (float(exy[0]), float(exy[1]), eyaw)))
    print(f"{len(jobs)} scenes to run ({len(done)} done before) -> {out}", flush=True)
    pool = Workers(args.workers)
    try:
        with open(rec_f, "a", encoding="utf-8") as fh:
            for i, (job, r) in enumerate(pool.run(scene, jobs), 1):
                if r is None:
                    print(f"  {job[0]} scene {job[1]}: lost", flush=True)
                    continue
                fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()
                print(f"  [{i}/{len(jobs)}] {r['object'][:32]:32s} " + "  ".join(
                    f"{rk} {'ok' if r[f'{rk}_success'] else r[f'{rk}_reason'][:16]}" for rk in RANKINGS)
                      + f" ({r.get('seconds', 0):.0f} s)", flush=True)
    finally:
        pool.close()
    recs = [json.loads(x) for x in rec_f.read_text().splitlines()]
    s = summary(recs, oset["classes"])
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(f"{s['scenes']} scenes, {s['objects']} objects:")
    for rk in RANKINGS:
        v = s[rk]
        extra = (f"; vs score: won {v['vs_score']['won']}, lost {v['vs_score']['lost']} (p = {v['vs_score']['p']:.3f})"
                 if "vs_score" in v else "")
        print(f"  {rk:8s} {v['success']:.0%}  (" + ", ".join(f"{c} {x:.0%}" for c, x in v["by_class"].items())
              + ")" + extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
