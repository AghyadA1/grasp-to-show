"""The learned scorer as a grasp-selection policy, in full pick-and-show episodes on UNSEEN objects.

Scenarios: the 10 test objects (never in training) x 8 start yaws (4 of them never in the
dataset either). Policies, all with the exact analytic kinematic checks, differing in how they
rank the kinematically feasible (grasp, twin, target) tuples:
  stability_only   baseline (most stable grasp)
  aware            analytic checks 1-5
  learned          trained scorer, geometry only          (data/models/scorer.pt)
  learned_checks   trained scorer + the checks' outputs   (data/models/scorer_checks.pt)
Usage:  python scripts/eval_learned_policy.py [--yaws ...] [--workers N]
Outputs: data/episodes_learned/records.jsonl, summary.json, media/learned_policy.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_experiments import wilson

from grasp2show.experiment import Scenario, choose
from grasp2show.objects import load_object_set

GRASPS, OUT = ROOT / "data" / "grasps", ROOT / "data" / "episodes_learned"
MODELS = {"learned": "scorer.pt", "learned_checks": "scorer_checks.pt"}
_models = {}


def _get_models(clouds_path: Path):
    import torch

    from grasp2show.learn.policy import load_scorer

    torch.set_num_threads(1)
    if not _models:
        for name, f in MODELS.items():
            p = ROOT / "data" / "models" / f
            if p.exists():
                _models[name] = load_scorer(p, clouds_path)
    return _models


def run(job):
    asset, yaw, clouds_path = job
    from grasp2show.learn.policy import learned_choices

    z = np.load(GRASPS / f"{asset}.npz")
    sc = Scenario(asset, yaw, {k: z[k] for k in ("T_obj_tcp", "width", "success")})
    recs = []
    for pol in ("stability_only", "aware"):
        recs.append(asdict(sc.run(pol, choose(pol, sc.reports, sc.lift_margin))))
    for name, (model, names, with_checks) in _get_models(clouds_path).items():
        t = time.perf_counter()
        ch = learned_choices(sc, model, names, with_checks)
        r = asdict(sc.run(name, ch))
        r["select_ms"] = round((time.perf_counter() - t) * 1e3, 1)
        recs.append(r)
    return recs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaws", type=float, nargs="+", default=[0, 45, 90, 135, 180, 225, 270, 315])
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    test = [n for n in load_object_set()["test"] if (GRASPS / f"{n}.npz").exists()]
    jobs = [(n, y, args.clouds) for n in test for y in args.yaws]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} scenarios on unseen objects, {args.workers} workers")
    recs = []
    with Pool(args.workers) as pool:
        for k, rs in enumerate(pool.imap_unordered(run, jobs), 1):
            recs += rs
            print(f"[{k}/{len(jobs)}] {rs[0]['object'][:34]:34s} yaw {rs[0]['yaw_deg']:3.0f}  " +
                  "  ".join(f"{r['policy']}: {'ok' if r['success'] else (r['reason'] or 'fail').split(':')[0]}" for r in rs))
    (OUT / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    pols = list(dict.fromkeys(r["policy"] for r in recs))
    summary = {"scenarios": len(jobs), "policies": {}}
    for pol in pols:
        rs = [r for r in recs if r["policy"] == pol]
        k = sum(r["success"] for r in rs)
        lo, hi = wilson(k, len(rs))
        fails = {}
        for r in rs:
            if not r["success"]:
                key = (r["reason"] or "fail").split(":")[0]
                fails[key] = fails.get(key, 0) + 1
        summary["policies"][pol] = {"success_rate": round(k / len(rs), 3), "ci95": [round(lo, 3), round(hi, 3)],
                                    "failures": fails}
        if "select_ms" in rs[0]:
            summary["policies"][pol]["select_ms_median"] = float(np.median([r["select_ms"] for r in rs]))
    by = {}
    for r in recs:
        by.setdefault((r["object"], r["yaw_deg"]), {})[r["policy"]] = r["success"]
    for pol in pols:
        if pol == "aware":
            continue
        w = sum(v.get(pol, False) and not v["aware"] for v in by.values())
        lo_ = sum(v["aware"] and not v.get(pol, False) for v in by.values())
        summary["policies"][pol]["vs_aware_wins_losses"] = [w, lo_]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    plot(summary, ROOT / "media" / "learned_policy.png")
    return 0


def plot(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = {"stability_only": "stability only\n(baseline)", "aware": "analytic checks", "learned": "learned\n(geometry)",
           "learned_checks": "learned\n+ checks"}
    ps = list(summary["policies"])
    r = [summary["policies"][p]["success_rate"] for p in ps]
    err = np.array([[ri - summary["policies"][p]["ci95"][0] for ri, p in zip(r, ps)],
                    [summary["policies"][p]["ci95"][1] - ri for ri, p in zip(r, ps)]])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(range(len(ps)), r, 0.6, yerr=err, capsize=4, color=["#9aa5b1", "#c1121f", "#2a6f97", "#1d4e6e"][:len(ps)],
           error_kw={"lw": 1, "ecolor": "0.3"})
    for i, (ri, p) in enumerate(zip(r, ps)):
        ax.text(i, summary["policies"][p]["ci95"][1] + 0.02, f"{ri:.0%}", ha="center", fontsize=9)
    ax.set_xticks(range(len(ps)), [lab.get(p, p) for p in ps], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("task success")
    ax.set_title(f"Unseen objects: {summary['scenarios']} scenarios (10 objects x 8 start poses), 95 % CI",
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
