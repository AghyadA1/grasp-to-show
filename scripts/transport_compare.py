"""Part 2, step 2: today's planner vs slowing down vs moving differently, in full episodes.

Same grasp (way B's), same start and same presentation pose for all methods:
  today        the analytic planner (Cartesian segments, pauses), at 1x and 2x speed
  retime_only  one smooth joint-space motion, the shortest duration that keeps the grasp safe
               and the robot within 60 % of its limits (the "just slow down" answer)
  shape_time   the path shape and the duration optimised together under the same constraints
               (the "carry the load where the grasp is strong" answer)
The optimised motions are also run at 1.25x their planned speed, to see how much margin the
constraints leave (as in Pham & Pham 2019).
Constraints: friction utilisation <= 0.7, robust to +-5 mm COM error (leverage), load along the
finger axis <= 14 N (measured in scripts/hold_tilt_test.py), <= 60 % of the Franka velocity and
acceleration limits, no collision.

Usage:  python scripts/transport_compare.py [--workers 4]
Outputs: data/transport_compare/records.jsonl, summary.json, media/transport_compare.png
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
sys.path.insert(0, str(ROOT / "scripts"))
from demo_comparison import plan
from eval_way_b import kinematic_choices, way_b_grasps
from run_experiments import wilson
from transport_baseline import slip_event

from grasp2show.experiment import Scenario
from grasp2show.objects import load_object_set
from grasp2show.transport import friction_params, motion_metrics
from grasp2show.transport_opt import TransportProblem

OUT = ROOT / "data" / "transport_compare"


def run(job):
    asset, yaw, mus, model, clouds = job
    T, W, R, _ = way_b_grasps(asset, yaw, model, clouds, 10)
    if len(T) == 0:
        return []
    sc = Scenario(asset, yaw, {"T_obj_tcp": T, "width": W, "success": np.ones(len(T), bool)})
    try:
        grasp, paths = plan(sc, "way_b", kinematic_choices(sc, list(zip(range(len(T)), R.tolist()))))
    except SystemExit:
        return []
    dt = sc.ep.model.opt.timestep
    recs = []
    for mu in mus:
        prm = friction_params(mu)
        base = {"object": asset, "yaw_deg": yaw, "mu": mu, "mass": sc.obj.mass}

        def episode(method, p, speed=1.0, extra=None, mu=mu, prm=prm, base=base):
            r = sc.ep.execute(grasp, p, speed=speed, mu=mu, log_every=5)
            m = motion_metrics(r.transport, sc.obj, grasp.T_obj_tcp, prm)
            recs.append({**base, "method": method, "speed": speed, "success": bool(r.success), "held": bool(r.held),
                         "qr_ok": bool(r.qr_ok), "collisions": len(r.collisions), "slip_mm": round(r.slip_mm, 2),
                         "slip_deg": round(r.slip_deg, 2),
                         **{k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v) for k, v in m.items()},
                         **(extra or {})})

        episode("today", paths, 1.0)
        episode("today", paths, 2.0)
        pr = TransportProblem(sc.obj, grasp.T_obj_tcp, grasp.width, paths["lift"][-1], paths["present"][-1], prm,
                              checker=sc.fc)
        # u_static: the robust model's need at rest at the two ends. If it is above the 0.7 limit, the
        # limit is relaxed to u_static + 0.05 for this problem (the motion must not add load); such
        # problems are kept and flagged, since the robust model is conservative for some grasps.
        info = {"u_static": round(pr.u_static, 3), "static_ok": pr.static_ok, "u_lim": round(pr.u_lim, 3)}
        t0 = time.perf_counter()
        a = pr.retime_only()
        ta = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = pr.shape_and_time(a)
        tb = time.perf_counter() - t0
        for name, s, secs in (("retime_only", a, ta), ("shape_time", b, tb)):
            p = dict(paths)
            p["transport"] = pr.sample(s.c, s.T, dt)
            extra = {**info, "planned_T": round(s.T, 3), "planned_feasible": s.feasible,
                     "planned_u_max": round(s.info["u_max"], 3), "solve_s": round(secs, 1)}
            episode(name, p, 1.0, extra)
            episode(name, p, 1.25, extra)
    return recs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaws", type=float, nargs="+", default=[0, 45, 90, 135, 180, 225, 270, 315])
    p.add_argument("--mus", type=float, nargs="+", default=[0.2, 0.3])
    p.add_argument("--objects", type=int, default=None)
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    args = p.parse_args()
    test = load_object_set()["test"][: args.objects]
    jobs = [(n, y, args.mus, args.model, args.clouds) for n in test for y in args.yaws]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} scenarios x {len(args.mus)} frictions, {args.workers} workers")
    recs = []
    with Pool(args.workers, maxtasksperchild=4) as pool:
        for k, rs in enumerate(pool.imap_unordered(run, jobs), 1):
            recs += rs
            line = "  ".join(f"{m}: {sum(r['success'] for r in rs if r['method'] == m and r['speed'] == 1.0)}"
                             for m in ("today", "retime_only", "shape_time"))
            print(f"[{k}/{len(jobs)}] {(rs[0]['object'][:30] if rs else '(no grasp)'):30s} {line}", flush=True)
    (OUT / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    s = summarise(recs)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    plot(recs, s, ROOT / "media" / "transport_compare.png")
    return 0


def summarise(recs):
    out = {"problems": len({(r["object"], r["yaw_deg"], r["mu"]) for r in recs}), "methods": {}}
    for meth, sp in (("today", 1.0), ("today", 2.0), ("retime_only", 1.0), ("retime_only", 1.25),
                     ("shape_time", 1.0), ("shape_time", 1.25)):
        rs = [r for r in recs if r["method"] == meth and r["speed"] == sp]
        if not rs:
            continue
        ok = [r["success"] for r in rs]
        lo, hi = wilson(sum(ok), len(rs))
        ran = [r for r in rs if "duration_s" in r]
        med = lambda k, ran=ran: round(float(np.median([r[k] for r in ran])), 3) if ran else None
        out["methods"][f"{meth} x{sp}"] = {
            "n": len(rs), "success": round(float(np.mean(ok)), 3), "ci95": [round(lo, 3), round(hi, 3)],
            "slip_event": round(float(np.mean([slip_event(r) for r in ran])), 3) if ran else None,
            "duration_s": med("duration_s"), "u_max": med("u_max"), "sparc": med("sparc"), "stops": med("stops"),
            "qd_ratio": med("qd_ratio"), "qdd_ratio": med("qdd_ratio"), "tau_ratio": med("tau_ratio")}
    by = {}
    for r in recs:
        by.setdefault((r["object"], r["yaw_deg"], r["mu"]), {})[(r["method"], r["speed"])] = r
    pairs = {}
    for a, b in ((("shape_time", 1.0), ("retime_only", 1.0)), (("shape_time", 1.0), ("today", 1.0)),
                 (("retime_only", 1.0), ("today", 1.0))):
        both = [v for v in by.values() if a in v and b in v and "duration_s" in v[a] and "duration_s" in v[b]]
        if both:
            pairs[f"{a[0]} vs {b[0]}"] = {
                "wins": sum(v[a]["success"] and not v[b]["success"] for v in both),
                "losses": sum(v[b]["success"] and not v[a]["success"] for v in both),
                "median_time_ratio": round(float(np.median([v[a]["duration_s"] / v[b]["duration_s"] for v in both])), 3)}
    out["paired"] = pairs
    return out


def plot(recs, s, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"today": "#9aa5b1", "retime_only": "#e07a2f", "shape_time": "#1d4e6e"}
    lab = {"today": "today's planner", "retime_only": "slow down (retime only)", "shape_time": "move differently (shape + time)"}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for meth, c in col.items():
        pts = []
        for sp in (1.0, 1.25, 2.0):
            rs = [r for r in recs if r["method"] == meth and r["speed"] == sp and "duration_s" in r]
            if rs:
                pts.append((np.median([r["duration_s"] for r in rs]), np.mean([r["success"] for r in rs]), sp))
        if pts:
            xs, ys, _ = zip(*pts)
            ax[0].plot(xs, ys, "o-", c=c, label=lab[meth])
            for x, y, sp in pts:
                ax[0].annotate(f"x{sp:g}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color="0.3")
    ax[0].set_xlabel("median transport duration (s)")
    ax[0].set_ylabel("task success")
    ax[0].set_ylim(0, 1.02)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[0].set_title(f"{s['problems']} problems (unseen objects, μ 0.2-0.3)", fontsize=9)
    keys = ["today x1.0", "retime_only x1.0", "shape_time x1.0"]
    labels = ["today", "retime only", "shape + time"]
    metrics = [("stops", "stops"), ("qdd_ratio", "peak joint acc. / limit"), ("u_max", "peak friction use")]
    w = 0.25
    for i, (k, nm) in enumerate(metrics):
        vals = [s["methods"].get(kk, {}).get(k) or 0 for kk in keys]
        ax[1].bar(np.arange(3) + (i - 1) * w, vals, w, label=nm, color=["#cfd8dc", "#9aa5b1", "#1d4e6e"][i])
    ax[1].set_xticks(range(3), labels, fontsize=8)
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("Motion quality (medians)", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
