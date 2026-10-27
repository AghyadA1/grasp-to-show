"""Part 3 test: way B on unseen shapes shown to unseen scanner poses, in full episodes.

Object set v2 (boxes, rounded and irregular shapes); the test objects were never seen in
training, and every scene draws a new start pose and a new scanner pose (config.random_camera,
seeds disjoint from the training scenes). Policies, all executed in full simulation:
  checks       the analytic checks on the object's sampled grasps (lift from the simulated test,
               reach, QR visibility, stability at the target), best margin first
  way_b_v1     the network trained on the 40 boxes with the fixed scanner (part 1)
  way_b_v2     the network retrained on set v2 with random scenes (part 3)
  way_b_v3     (with --v3) the same, with labels from the grip model with every contact
With --grip-model all, the checks policy uses the grip model with every contact for the
stability at the targets (as the v3 labels do).
Usage:  $env:G2S_OBJECT_SET="v2"; python scripts/eval_way_b_v2.py [--scenes 4] [--workers 4]
        [--v3 data/models/heatmap_net_v3.pt --grip-model all --tag way_b_v3]
Outputs: data/episodes_way_b_v2/records.jsonl, summary.json, media/way_b_v2_policy.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_way_b import kinematic_choices, way_b_grasps
from run_experiments import wilson

from grasp2show.config import random_camera
from grasp2show.experiment import Scenario, choose
from grasp2show.objects import load_object_set

OUT = ROOT / "data" / "episodes_way_b_v2"
GRASPS = ROOT / "data" / "grasps_v2"
POLICIES = ["checks", "way_b_v1", "way_b_v2"]


def run(job):
    asset, k, yaw, xy, cam, models, clouds, top, grip_model = job
    base = {"object": asset, "scene": k, "yaw_deg": round(yaw, 1), "cam_pos": list(cam.pos)}
    recs = []
    z = np.load(GRASPS / f"{asset}.npz")
    sc = Scenario(asset, yaw, {kk: z[kk] for kk in ("T_obj_tcp", "width", "success")}, start_xy=xy, camera=cam)
    # check 1 from the simulated lift test, as in the part-3 labels (the analytic margin is box-validated)
    sc.lift_margin = np.where(sc.lift_ok, np.clip(sc.lift_margin, 0.1, 1.0), -1.0)
    if grip_model == "all":
        from grasp2show.grip_model import apply_to_reports
        from grasp2show.gripper_sim import GripperSim, close_contacts
        from grasp2show.task import show_pose

        gs = GripperSim(sc.cfg, sc.obj)
        keys = list(sc.reports)
        apply_to_reports([sc.reports[i] for i in keys], [close_contacts(gs, sc.T[i], sc.W[i]) for i in keys], sc.obj,
                         np.stack([show_pose(sc.cfg, sc.obj, r) for r in sc.rolls]))
    recs.append({**asdict(sc.run("checks", choose("aware", sc.reports, sc.lift_margin))), **base, "policy": "checks"})
    for pol, model in models.items():
        T, W, R, ms = way_b_grasps(asset, yaw, model, clouds, top, start_xy=xy, camera=cam)
        if len(T) == 0:
            recs.append({**base, "policy": pol, "success": False, "reason": "no grasp"})
            continue
        sb = Scenario(asset, yaw, {"T_obj_tcp": T, "width": W, "success": np.ones(len(T), bool)}, start_xy=xy,
                      camera=cam)
        r = asdict(sb.run(pol, kinematic_choices(sb, list(zip(range(len(T)), R.tolist())))))
        recs.append({**r, **base, "policy": pol, "net_ms": round(ms, 1)})
    return recs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", type=int, default=4, help="random scenes per test object")
    p.add_argument("--v1", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--v2", type=Path, default=ROOT / "data" / "models" / "heatmap_net_v2.pt")
    p.add_argument("--tag", default="way_b_v2", help="outputs: data/episodes_<tag>/, media/<tag>_policy.png")
    p.add_argument("--v3", type=Path, default=None, help="way B retrained on every-contact labels")
    p.add_argument("--grip-model", choices=["point", "all"], default="point")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset_v2" / "clouds.npz")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--objects", type=int, default=None)
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    args = p.parse_args()
    oset = load_object_set()
    test = [n for n in oset["test"] if (GRASPS / f"{n}.npz").exists()][: args.objects]
    models = {"way_b_v1": args.v1, "way_b_v2": args.v2}
    if args.v3:
        models["way_b_v3"] = args.v3
        POLICIES.append("way_b_v3")
    best = POLICIES[-1]
    jobs = []
    for n in test:
        r = np.random.default_rng(9000 + oset["objects"].index(n))     # disjoint from the training scenes
        for k in range(args.scenes):
            jobs.append((n, k, float(r.uniform(0, 360)), (float(r.uniform(0.40, 0.60)), float(r.uniform(-0.30, 0.0))),
                         random_camera(r), models, args.clouds, args.top, args.grip_model))
    out = OUT.with_name(f"episodes_{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} scenes ({len(test)} unseen objects x {args.scenes} random start + scanner poses), "
          f"{args.workers} workers")
    recs = []
    with Pool(args.workers) as pool:
        for kk, rs in enumerate(pool.imap_unordered(run, jobs), 1):
            recs += rs
            print(f"[{kk}/{len(jobs)}] {rs[0]['object'][:34]:34s} " + "  ".join(
                f"{r['policy']}: {'ok' if r['success'] else (r.get('reason') or 'fail').split(':')[0]}" for r in rs),
                flush=True)
    (out / "records.jsonl").write_text("".join(json.dumps(r, default=float) + "\n" for r in recs))
    cls = {n: c for c, names in oset["classes"].items() for n in names}
    summary = {"scenes": len(jobs), "objects": len(test), "grip_model": args.grip_model, "policies": {}}
    by = {}
    for r in recs:
        by.setdefault((r["object"], r["scene"]), {})[r["policy"]] = r["success"]
    for pol in POLICIES:
        rs = [r for r in recs if r["policy"] == pol]
        k = sum(r["success"] for r in rs)
        lo, hi = wilson(k, len(rs))
        fails = {}
        for r in rs:
            if not r["success"]:
                key = (r.get("reason") or "fail").split(":")[0]
                fails[key] = fails.get(key, 0) + 1
        row = {"success": round(k / len(rs), 3), "ci95": [round(lo, 3), round(hi, 3)], "failures": fails,
               "by_class": {c: round(float(np.mean([r["success"] for r in rs if cls.get(r["object"]) == c])), 3)
                            for c in oset["classes"] if any(cls.get(r["object"]) == c for r in rs)}}
        if pol != best:
            row[f"{best}_wins_losses"] = [sum(v.get(best, False) and not v.get(pol, False) for v in by.values()),
                                          sum(v.get(pol, False) and not v.get(best, False) for v in by.values())]
        summary["policies"][pol] = row
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    plot(summary, ROOT / "media" / f"{args.tag}_policy.png")
    return 0


def plot(s, path):
    """Left: success of each policy. Right: by shape class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"checks": "analytic checks", "way_b_v1": "way B, trained on\nboxes, fixed scanner",
              "way_b_v2": "way B, retrained on\nmany shapes, random scanner",
              "way_b_v3": "way B, retrained on\nevery-contact labels"}
    cols = {"checks": "#9aa5b1", "way_b_v1": "#e07a2f", "way_b_v2": "#1d4e6e", "way_b_v3": "#3f8f6b"}
    pols = [p for p in POLICIES if p in s["policies"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})
    for i, pol in enumerate(pols):
        d = s["policies"][pol]
        ax[0].bar(i, d["success"], 0.6, color=cols[pol])
        ax[0].errorbar(i, d["success"], yerr=[[d["success"] - d["ci95"][0]], [d["ci95"][1] - d["success"]]], c="k",
                       capsize=3, lw=1)
        ax[0].text(i, d["ci95"][1] + 0.02, f"{d['success']:.0%}", ha="center", fontsize=9)
    ax[0].set_xticks(range(len(pols)), [labels[p] for p in pols], fontsize=8)
    ax[0].set_ylim(0, 1.1)
    ax[0].set_ylabel("task success (95 % CI)")
    ax[0].set_title(f"{s['scenes']} scenes: {s['objects']} unseen objects, unseen scanner poses", fontsize=9)
    classes = list(s["policies"]["way_b_v2"]["by_class"])
    w = 0.8 / len(pols)
    for i, pol in enumerate(pols):
        v = [s["policies"][pol]["by_class"].get(c, 0) for c in classes]
        ax[1].bar(np.arange(len(classes)) + (i - (len(pols) - 1) / 2) * w, v, w, color=cols[pol], label=labels[pol].replace("\n", " "))
        for x, y in zip(np.arange(len(classes)) + (i - (len(pols) - 1) / 2) * w, v):
            ax[1].text(x, y + 0.02, f"{y:.0%}", ha="center", fontsize=7)
    ax[1].set_xticks(range(len(classes)), classes, fontsize=9)
    ax[1].set_ylim(0, 1.1)
    ax[1].legend(frameon=False, fontsize=7, loc="upper right")
    ax[1].set_title("by shape class", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
