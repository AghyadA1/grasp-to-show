"""Robustness to pose-estimation error: the object is not exactly where the robot thinks.

Each policy chooses and plans its grasp for the nominal object pose (as before); the episode is
then run with the object actually displaced on the table by a random offset the robot does not
know about: dx, dy ~ N(0, sigma_xy), dyaw ~ N(0, sigma_yaw). Same offsets for every policy
(paired). Reported: success rate per policy and noise level, on the 10 unseen test objects x 8
start rotations.

Usage:  python scripts/robustness.py                     # 3 noise levels x 3 samples
        python scripts/robustness.py --levels 0.005:5 --samples 2 --yaws 0 90
Outputs: data/robustness/records.jsonl, summary.json, media/robustness.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_experiments import wilson

from grasp2show.experiment import Choice, Scenario, choose
from grasp2show.objects import load_object_set

GRASPS, OUT = ROOT / "data" / "grasps", ROOT / "data" / "robustness"
_models = {}


def _learned(clouds: Path):
    """The learned+checks scorer, if PyTorch and the model are available (else skipped)."""
    if "m" not in _models:
        try:
            import torch

            from grasp2show.learn.policy import load_scorer

            torch.set_num_threads(1)
            _models["m"] = load_scorer(ROOT / "data" / "models" / "scorer_checks.pt", clouds)
        except (ImportError, FileNotFoundError) as e:
            print(f"learned policy skipped: {e}")
            _models["m"] = None
    return _models["m"]


def run(job):
    asset, yaw, levels, n_samples, clouds, seed = job
    z = np.load(GRASPS / f"{asset}.npz")
    sc = Scenario(asset, yaw, {k: z[k] for k in ("T_obj_tcp", "width", "success")})
    choices = {p: choose(p, sc.reports, sc.lift_margin) for p in ("stability_only", "aware")}
    lm = _learned(clouds)
    if lm is not None:
        from grasp2show.learn.policy import learned_choices

        choices["learned_checks"] = learned_choices(sc, *lm)
    rng = np.random.default_rng(seed)
    offsets = [(0.0, 0.0, 0.0, 0.0)] + [
        (sxy, *rng.normal(0, sxy, 2), np.radians(rng.normal(0, syaw)))
        for sxy, syaw in levels for _ in range(n_samples)]
    recs = []
    for pol, ch in choices.items():
        nominal = sc.run(pol, ch)                      # the choice this policy actually executes
        for sxy, dx, dy, dyaw in offsets:
            base = {"object": asset, "yaw_deg": yaw, "policy": pol, "sigma_xy_mm": round(sxy * 1e3, 1),
                    "dx_mm": round(dx * 1e3, 2), "dy_mm": round(dy * 1e3, 2), "dyaw_deg": round(np.degrees(dyaw), 2)}
            if nominal.grasp is None or nominal.reason.startswith("plan"):
                recs.append({**base, "success": False, "reason": nominal.reason or "no grasp"})
                continue
            if sxy == 0.0:
                r = nominal
            else:
                c = Choice(nominal.grasp, nominal.twin, round(nominal.roll_deg / 360 * len(sc.rolls)) % len(sc.rolls),
                           bool(nominal.predicted_ok))
                r = sc._execute(c, obj_offset=(dx, dy, dyaw))
            recs.append({**base, "success": bool(r.success), "reason": r.reason, "slip_mm": r.slip_mm})
    return recs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", nargs="+", default=["0.002:2", "0.005:5", "0.01:10"],
                   help="noise levels sigma_xy[m]:sigma_yaw[deg]")
    p.add_argument("--samples", type=int, default=3, help="random offsets per level and scenario")
    p.add_argument("--yaws", type=float, nargs="+", default=[0, 45, 90, 135, 180, 225, 270, 315])
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    levels = [tuple(float(v) for v in s.split(":")) for s in args.levels]
    test = [n for n in load_object_set()["test"] if (GRASPS / f"{n}.npz").exists()]
    jobs = [(n, y, levels, args.samples, args.clouds, 1000 * i + int(y)) for i, n in enumerate(test) for y in args.yaws]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} scenarios x {1 + len(levels) * args.samples} object poses, {args.workers} workers")
    recs = []
    with Pool(args.workers) as pool:
        for k, rs in enumerate(pool.imap_unordered(run, jobs), 1):
            recs += rs
            ok = {}
            for r in rs:
                ok.setdefault(r["policy"], []).append(r["success"])
            print(f"[{k}/{len(jobs)}] {rs[0]['object'][:34]:34s} yaw {rs[0]['yaw_deg']:3.0f}  " +
                  "  ".join(f"{pol}: {sum(v)}/{len(v)}" for pol, v in ok.items()), flush=True)
    (OUT / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    summary = {"scenarios": len(jobs), "levels": [{"sigma_xy_mm": a * 1e3, "sigma_yaw_deg": b} for a, b in levels],
               "samples": args.samples, "success": {}}
    for pol in dict.fromkeys(r["policy"] for r in recs):
        row = {}
        for s in sorted({r["sigma_xy_mm"] for r in recs}):
            rs = [r for r in recs if r["policy"] == pol and r["sigma_xy_mm"] == s]
            k = sum(r["success"] for r in rs)
            lo, hi = wilson(k, len(rs))
            fails = {}
            for r in rs:
                if not r["success"]:
                    key = (r["reason"] or "fail").split(":")[0]
                    fails[key] = fails.get(key, 0) + 1
            row[f"{s:g} mm"] = {"rate": round(k / len(rs), 3), "ci95": [round(lo, 3), round(hi, 3)], "n": len(rs),
                                "failures": fails}
        summary["success"][pol] = row
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["success"], indent=1))
    plot(summary, ROOT / "media" / "robustness.png")
    return 0


def plot(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = {"stability_only": "stability only (baseline)", "aware": "analytic checks", "learned_checks": "learned + checks"}
    col = {"stability_only": "#9aa5b1", "aware": "#c1121f", "learned_checks": "#1d4e6e"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    pols = list(summary["success"])
    for n, (pol, row) in enumerate(summary["success"].items()):
        x0 = [float(k.split()[0]) for k in row]
        xs = [x + 0.12 * (n - (len(pols) - 1) / 2) for x in x0]       # dodge: error bars apart
        ys = [v["rate"] for v in row.values()]
        lo = [v["rate"] - v["ci95"][0] for v in row.values()]
        hi = [v["ci95"][1] - v["rate"] for v in row.values()]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=6, lw=2, capsize=3, color=col.get(pol), label=lab.get(pol, pol))
        ax.annotate(f"{ys[-1]:.0%}", (xs[-1], ys[-1]), xytext=(14, 0), textcoords="offset points", va="center",
                    fontsize=8, color="0.25")
    ax.set_xticks(x0, [f"{x:g} mm / {lv:g}°" for x, lv in zip(x0, [0] + [lv["sigma_yaw_deg"] for lv in summary["levels"]])])
    ax.set_xlabel("object pose error (std. dev., position / yaw)")
    ax.set_ylabel("task success")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(f"Unseen objects, {summary['scenarios']} scenarios, 95 % CI", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
