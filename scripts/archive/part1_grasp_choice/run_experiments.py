"""Day-5 experiments: full simulated episodes for (1) label validation and (2) policy comparison.

(1) Validation: in the yaw-0 scenario of each object, up to --validate grasps that a
    stability-only planner would accept (lift ok + reachable at the start) are executed:
    with the checks' best (twin, target) when predicted feasible, otherwise with the target a
    blind planner would pick. Predicted label vs episode outcome -> confusion matrix.
(2) Comparison: every object at --yaws start rotations; each policy (experiment.POLICIES)
    chooses one grasp per scenario; success rate per policy, on train and unseen test objects.

Usage:
  python scripts/run_experiments.py                     # 40 objects x 4 yaws, validate 30/object
  python scripts/run_experiments.py --objects 3 --yaws 0 --validate 5     (quick check)
Outputs: data/episodes/records.jsonl, data/episodes/summary.json, media/policy_comparison.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from math import comb
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
from grasp2show.experiment import POLICIES, Scenario, choose
from grasp2show.objects import load_object_set

GRASPS, OUT = ROOT / "data" / "grasps", ROOT / "data" / "episodes"


def run_scenario(job) -> list[dict]:
    asset, yaw, n_validate, seed = job
    z = np.load(GRASPS / f"{asset}.npz")
    sc = Scenario(asset, yaw, {k: z[k] for k in ("T_obj_tcp", "width", "success")})
    recs = []
    for policy in POLICIES:
        r = sc.run(policy, choose(policy, sc.reports, sc.lift_margin))
        recs.append({**asdict(r), "kind": "policy"})
    if n_validate:
        cand = [i for i, rep in sc.reports.items() if any(tw.start_ok for tw in rep.twins)]
        rng = np.random.default_rng(seed)
        for i in rng.permutation(cand)[:n_validate]:
            one = {int(i): sc.reports[int(i)]}
            c = choose("aware", one, sc.lift_margin) or choose("stability_only", one, sc.lift_margin)
            r = sc.run("validate", c[:1])
            recs.append({**asdict(r), "kind": "validate"})
    return recs


def rate(rs):
    return round(float(np.mean([r["success"] for r in rs])), 3) if rs else None


def summarise(records: list[dict], oset: dict) -> dict:
    split = {n: "test" for n in oset["test"]} | {n: "train" for n in oset["train"]}
    pol = [r for r in records if r["kind"] == "policy"]
    out = {"policies": {}}
    for p in POLICIES:
        rs = [r for r in pol if r["policy"] == p]
        reasons = {}
        for r in rs:
            if not r["success"]:
                reasons[r["reason"].split(":")[0]] = reasons.get(r["reason"].split(":")[0], 0) + 1
        out["policies"][p] = {
            "scenarios": len(rs), "success_rate": rate(rs),
            "train": rate([r for r in rs if split[r["object"]] == "train"]),
            "test": rate([r for r in rs if split[r["object"]] == "test"]),
            "failures": reasons,
        }
    # Paired comparison baseline vs aware, scenario by scenario.
    by = {}
    for r in pol:
        by.setdefault((r["object"], r["yaw_deg"]), {})[r["policy"]] = r["success"]
    pairs = [(v["stability_only"], v["aware"]) for v in by.values() if "aware" in v and "stability_only" in v]
    out["paired"] = {"aware_wins": sum(a and not b for b, a in pairs), "baseline_wins": sum(b and not a for b, a in pairs),
                     "both_succeed": sum(a and b for b, a in pairs), "both_fail": sum(not a and not b for b, a in pairs)}
    w, lo = out["paired"]["aware_wins"], out["paired"]["baseline_wins"]
    n = w + lo
    out["paired"]["sign_test_p_two_sided"] = float(min(1.0, 2 * sum(comb(n, k) for k in range(max(w, lo), n + 1)) / 2**n)) if n else None

    val = [r for r in records if r["kind"] == "validate" and r["predicted_ok"] is not None]
    if val:
        # Planning failures (no collision-free path found by our simple planner although both end
        # configurations are feasible) are not something the checks predict: reported separately.
        ex = [r for r in val if not r["reason"].startswith("plan")]
        pred = np.array([r["predicted_ok"] for r in ex])
        act = np.array([r["success"] for r in ex])
        tp, tn = int(np.sum(pred & act)), int(np.sum(~pred & ~act))
        fp, fn = int(np.sum(pred & ~act)), int(np.sum(~pred & act))
        vis_pred = np.array([r["vis_fraction"] >= 0.9 for r in ex])
        held = np.array([r["held"] and not r["reason"].startswith("collision") for r in ex])
        qr = np.array([r["qr_ok"] for r in ex])
        sm = np.array([r["stab_margin"] for r in ex])
        calib = []
        for a, b in ((0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.01)):
            s = pred & (sm >= a) & (sm < b)
            calib.append({"target_stability_margin": [a, min(b, 1.0)], "n": int(s.sum()),
                          "success_rate": round(float(act[s].mean()), 3) if s.any() else None})
        out["validation"] = {
            "episodes": len(val), "planning_failures": len(val) - len(ex), "executed": len(ex),
            "success_rate_if_predicted_feasible": round(float(act[pred].mean()), 3) if pred.any() else None,
            "success_rate_if_predicted_infeasible": round(float(act[~pred].mean()), 3) if (~pred).any() else None,
            "agreement": round((tp + tn) / max(len(ex), 1), 3),
            "precision_feasible": round(tp / max(tp + fp, 1), 3), "recall_feasible": round(tp / max(tp + fn, 1), 3),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "visibility_vs_decode_agreement_held": round(float(np.mean(vis_pred[held] == qr[held])), 3)
            if held.any() else None,
            "false_positive_reasons": _count([r["reason"].split(":")[0] for r, p, a in zip(ex, pred, act) if p and not a]),
            "calibration": calib,
        }
    return out


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a success rate k / n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def plot(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"stability_only": "stability only\n(baseline)", "aware": "target-aware\n(ours)",
              "aware_no_vis": "ours without\nvisibility check", "aware_no_stab": "ours without\ntarget stability",
              "aware_single": "ours with one\ntarget pose", "aware_leximin": "ours, leximin\ntie-break"}
    ps = list(POLICIES)
    n = [summary["policies"][p]["scenarios"] for p in ps]
    r = [summary["policies"][p]["success_rate"] or 0 for p in ps]
    ci = [wilson(round(ri * ni), ni) for ri, ni in zip(r, n)]
    err = np.array([[ri - lo for ri, (lo, _) in zip(r, ci)], [hi - ri for ri, (_, hi) in zip(r, ci)]])
    colors = ["#9aa5b1", "#2a6f97"] + ["#8fb8d6"] * (len(ps) - 2)
    x = np.arange(len(ps))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x, r, 0.6, color=colors, yerr=err, capsize=4, error_kw={"lw": 1, "ecolor": "0.3"})
    for xi, ri, (_, hi) in zip(x, r, ci):
        ax.text(xi, hi + 0.02, f"{ri:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x, [labels[p] for p in ps], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("task success\n(QR decoded, held, no gross slip)", fontsize=9)
    ax.set_title(f"Full pick-and-show episodes, {n[0]} scenarios (40 objects x 4 start poses); 95 % CI",
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def resummarise() -> None:
    """Recompute summary.json and the plot from records.jsonl (no simulation)."""
    records = [json.loads(x) for x in (OUT / "records.jsonl").read_text().splitlines()]
    old = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    summary = summarise(records, load_object_set())
    summary.update({k: old[k] for k in ("wall_seconds", "config") if k in old})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    plot(summary, ROOT / "media" / "policy_comparison.png")
    print(json.dumps(summary, indent=1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", type=int, default=None)
    p.add_argument("--yaws", type=float, nargs="+", default=[0, 90, 180, 270])
    p.add_argument("--validate", type=int, default=30, help="validation episodes per object (yaw 0)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--resummarise", action="store_true", help="only recompute the summary from records")
    args = p.parse_args()
    if args.resummarise:
        resummarise()
        return 0
    oset = load_object_set()
    names = [n for n in oset["objects"] if (GRASPS / f"{n}.npz").exists()][: args.objects]
    if not names:
        print("no labelled grasps: run scripts/generate_grasps.py first")
        return 1
    jobs = [(n, y, args.validate if y == args.yaws[0] else 0, k) for k, n in enumerate(names) for y in args.yaws]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} scenarios ({len(names)} objects x {len(args.yaws)} yaws), {args.workers} workers")
    t0, records = time.perf_counter(), []
    with Pool(args.workers) as pool:
        for k, recs in enumerate(pool.imap_unordered(run_scenario, jobs), 1):
            records += recs
            pr = {r["policy"]: ("ok" if r["success"] else r["reason"] or "fail") for r in recs if r["kind"] == "policy"}
            print(f"[{k}/{len(jobs)}] {recs[0]['object'][:34]:34s} yaw {recs[0]['yaw_deg']:3.0f}  "
                  f"baseline: {pr['stability_only']:<18s} aware: {pr['aware']}")
    with open(OUT / "records.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    summary = summarise(records, oset)
    summary["wall_seconds"] = round(time.perf_counter() - t0, 1)
    summary["config"] = vars(args)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    plot(summary, ROOT / "media" / "policy_comparison.png")
    print(json.dumps({k: summary[k] for k in ("policies", "paired") if k in summary}, indent=1))
    if "validation" in summary:
        print("validation:", json.dumps(summary["validation"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
