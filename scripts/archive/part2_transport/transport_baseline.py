"""Part 2, step 1: how does today's planner carry the object, and does the dynamic grasp model
predict slip during the motion?

For each scenario (10 unseen objects x 8 start rotations) the grasp is way B's (the heatmap
network), planned once with the analytic planner (straight Cartesian lines, one minimum-jerk
profile per segment: lift, pause, transit, present). It is then executed at several speeds
and finger-object friction coefficients, recording the whole transport. Per episode:
outcome and slip (simulated), and the motion measures of grasp2show/transport.py, among them
the friction utilisation u(t) the motion demands from the grasp (soft-finger model with
d'Alembert loads).

Reported:
  - does u_max predict slip during the motion? (AUC; slip event = slip > 5 mm or 5 deg, or lost)
  - speed vs outcome curves for each friction coefficient: today's planner, the baseline
  - how close today's planner comes to the Franka limits
Usage:  python scripts/transport_baseline.py [--workers 4]
Outputs: data/transport_baseline/records.jsonl, summary.json, media/transport_baseline.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
from demo_comparison import plan
from eval_way_b import kinematic_choices, way_b_grasps
from train_scorer import auc

from grasp2show.experiment import Scenario
from grasp2show.objects import load_object_set
from grasp2show.transport import friction_params, motion_metrics

OUT = ROOT / "data" / "transport_baseline"


def slip_event(r: dict, mm: float = 5.0, deg: float = 5.0) -> bool:
    return (not r["held"]) or r["slip_mm"] > mm or r["slip_deg"] > deg


def run(job):
    asset, yaw, speeds, mus, model, clouds = job
    T, W, R, _ = way_b_grasps(asset, yaw, model, clouds, 10)
    if len(T) == 0:
        return []
    sc = Scenario(asset, yaw, {"T_obj_tcp": T, "width": W, "success": np.ones(len(T), bool)})
    try:
        grasp, paths = plan(sc, "way_b", kinematic_choices(sc, list(zip(range(len(T)), R.tolist()))))
    except SystemExit:
        return []
    recs = []
    for mu in mus:
        for sp in speeds:
            r = sc.ep.execute(grasp, paths, speed=sp, mu=mu, log_every=5)
            m = motion_metrics(r.transport, sc.obj, grasp.T_obj_tcp, friction_params(mu))
            recs.append({"object": asset, "yaw_deg": yaw, "mu": mu, "speed": sp, "success": bool(r.success),
                         "held": bool(r.held), "qr_ok": bool(r.qr_ok), "collisions": len(r.collisions),
                         "slip_mm": round(r.slip_mm, 2), "slip_deg": round(r.slip_deg, 2), "mass": sc.obj.mass,
                         **{k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v) for k, v in m.items()}})
    return recs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaws", type=float, nargs="+", default=[0, 45, 90, 135, 180, 225, 270, 315])
    p.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--mus", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    p.add_argument("--objects", type=int, default=None)
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)),
                   help="each worker loads PyTorch (way B); more than ~4 can exhaust Windows' paging file")
    args = p.parse_args()
    test = load_object_set()["test"][: args.objects]
    jobs = [(n, y, args.speeds, args.mus, args.model, args.clouds) for n in test for y in args.yaws]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} scenarios x {len(args.speeds)} speeds x {len(args.mus)} frictions, {args.workers} workers")
    recs = []
    with Pool(args.workers, maxtasksperchild=4) as pool:
        for k, rs in enumerate(pool.imap_unordered(run, jobs), 1):
            recs += rs
            ok = sum(r["success"] for r in rs)
            print(f"[{k}/{len(jobs)}] {(rs[0]['object'][:32] if rs else '(no grasp)'):32s} "
                  f"{ok}/{len(rs)} succeeded", flush=True)
    (OUT / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    summary = summarise(recs)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "curves"}, indent=1))
    plot(recs, summary, ROOT / "media" / "transport_baseline.png")
    return 0


def summarise(recs: list[dict]) -> dict:
    ev = np.array([slip_event(r) for r in recs])
    umax = np.array([r["u_max"] if np.isfinite(r["u_max"]) else 1e9 for r in recs])
    uend = np.array([r["u_end"] if np.isfinite(r["u_end"]) else 1e9 for r in recs])
    s = {"episodes": len(recs), "slip_events": int(ev.sum()),
         "auc_u_max_for_slip": round(auc(umax, ev), 3), "auc_u_end_for_slip": round(auc(uend, ev), 3),
         "slip_rate_u_max_le_1": round(float(ev[umax <= 1].mean()), 3) if (umax <= 1).any() else None,
         "slip_rate_u_max_gt_1": round(float(ev[umax > 1].mean()), 3) if (umax > 1).any() else None,
         "curves": {}}
    for mu in sorted({r["mu"] for r in recs}):
        row = {}
        for sp in sorted({r["speed"] for r in recs}):
            rs = [r for r in recs if r["mu"] == mu and r["speed"] == sp]
            row[str(sp)] = {"n": len(rs), "success": round(np.mean([r["success"] for r in rs]), 3),
                            "slip_event": round(np.mean([slip_event(r) for r in rs]), 3),
                            "duration_s": round(float(np.median([r["duration_s"] for r in rs])), 2),
                            "u_max_median": round(float(np.median([min(r["u_max"], 9) for r in rs])), 3),
                            "qdd_ratio_median": round(float(np.median([r["qdd_ratio"] for r in rs])), 2),
                            "qd_ratio_median": round(float(np.median([r["qd_ratio"] for r in rs])), 2),
                            "tau_ratio_median": round(float(np.median([r["tau_ratio"] for r in rs])), 2),
                            "stops_median": float(np.median([r["stops"] for r in rs])),
                            "sparc_median": round(float(np.median([r["sparc"] for r in rs])), 2)}
        s["curves"][str(mu)] = row
    return s


def plot(recs, summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {0.2: "#c1121f", 0.3: "#e07a2f", 0.5: "#1d4e6e"}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for mu, row in summary["curves"].items():
        c = cols.get(float(mu), None)
        d = [v["duration_s"] for v in row.values()]
        ax[0].plot(d, [v["success"] for v in row.values()], "o-", c=c, label=f"μ = {mu}")
        ax[1].plot(d, [v["qdd_ratio_median"] for v in row.values()], "o-", c=c)
    ax[0].set_xlabel("transport duration (s)")
    ax[0].set_ylabel("task success")
    ax[0].set_ylim(0, 1)
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Today's planner: faster = less reliable", fontsize=9)
    ax[1].axhline(1, c="0.5", ls=":")
    ax[1].set_xlabel("transport duration (s)")
    ax[1].set_ylabel("peak joint acceleration / Franka limit")
    ax[1].set_title("Today's planner vs the robot's limits", fontsize=9)
    ev = np.array([slip_event(r) for r in recs])
    u = np.array([min(r["u_max"], 5) for r in recs])
    jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(u))
    ax[2].scatter(u, ev + jit, s=6, alpha=0.4, c=np.where(ev, "#c1121f", "#1d4e6e"))
    ax[2].axvline(1, c="0.5", ls=":")
    ax[2].set_yticks([0, 1], ["held", "slipped"])
    ax[2].set_xlabel("peak friction utilisation u_max (model)")
    ax[2].set_title(f"Does the model predict slip? AUC {summary['auc_u_max_for_slip']:.2f}", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
