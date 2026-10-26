"""Part 2, step 4: the transport network in full episodes on unseen objects.

The test problems of scripts/generate_transport_data.py --split test (unseen objects, random
place, rotation and friction) are rebuilt and each is carried with four planners, all under the
same constraints (friction use <= 0.7 robust to +-5 mm COM error, <= 14 N along the finger axis,
<= 60 % of the Franka limits, no collision):

  retime       the straight joint-space motion, slowed to the shortest safe duration
  optimiser    the full shape + time search (60 iterations, as in the training labels)
  net          the network's motion, slowed down if needed to satisfy the constraints;
               falls back to `retime` if no duration works (e.g. the path turns the object where
               its weight alone is too much for the grip) or it collides
  net+refine   the network's motion as the starting point of a short search (10 iterations,
               a sixth of the optimiser's), then the same fallback (DJ-GOMP's warm start)

Every motion is executed in simulation. Reported: success, transport duration, planning time,
how often the fallback was needed.
Usage:  python scripts/transport_net_episodes.py [--workers 4] [--refine-iters 10]
Outputs: data/transport_net_episodes/records.jsonl, summary.json, media/transport_net_episodes.png
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
from train_transport_net import slow_to_feasible

from grasp2show.experiment import Scenario
from grasp2show.transport import friction_params, motion_metrics
from grasp2show.transport_opt import Solution, TransportProblem

OUT = ROOT / "data" / "transport_net_episodes"
METHODS = ("retime", "optimiser", "net", "net+refine")
_NET = None


def net_predict(path: Path, x: np.ndarray) -> tuple[np.ndarray, float]:
    global _NET
    import torch

    from grasp2show.learn.transport_net import TransportNet

    torch.set_num_threads(1)
    if _NET is None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        _NET = TransportNet()
        _NET.load_state_dict(ck["state_dict"])
        _NET.eval()
    with torch.no_grad():
        c, lt = _NET(torch.as_tensor(x[None]))
    return c[0].numpy().astype(float), float(np.exp(lt[0]))


def run(job):
    f, model, clouds, net_path, refine_iters = job
    from grasp2show.learn.transport_net import features

    z = dict(np.load(f, allow_pickle=True))
    asset, yaw, xy, mu = str(z["object"]), float(z["yaw"]), tuple(z["start_xy"]), float(z["mu"])
    T, W, R, _ = way_b_grasps(asset, yaw, model, clouds, 10, start_xy=xy)
    if len(T) == 0:
        return []
    sc = Scenario(asset, yaw, {"T_obj_tcp": T, "width": W, "success": np.ones(len(T), bool)}, start_xy=xy)
    try:
        grasp, paths = plan(sc, "way_b", kinematic_choices(sc, list(zip(range(len(T)), R.tolist()))))
    except SystemExit:
        return []
    if not np.allclose(grasp.T_obj_tcp, z["T_obj_tcp"], atol=1e-6):
        return [{"file": Path(f).name, "object": asset, "status": "grasp differs from the dataset"}]
    prm = friction_params(mu)
    q0, q1 = paths["lift"][-1], paths["present"][-1]
    pr = TransportProblem(sc.obj, grasp.T_obj_tcp, grasp.width, q0, q1, prm, checker=sc.fc)
    dt = sc.ep.model.opt.timestep
    base = {"file": Path(f).name, "object": asset, "mu": round(mu, 3), "mass": sc.obj.mass,
            "u_static": round(pr.u_static, 3), "status": "ok"}

    sols = {}
    t0 = time.perf_counter()
    ret = pr.retime_only()
    sols["retime"] = (ret, time.perf_counter() - t0, {})
    t0 = time.perf_counter()
    sols["optimiser"] = (pr.shape_and_time(ret, seed=0), time.perf_counter() - t0, {})

    def with_fallback(s: Solution, extra: dict, t_start: float):
        r = pr.retime_only()                         # cheap (about 20 checks); counted in the planning time
        use_ret = not s.feasible or (r.feasible and r.T <= s.T)
        return (r if use_ret else s), time.perf_counter() - t_start, \
            {**extra, "fallback": bool(not s.feasible), "retime_shorter": bool(s.feasible and use_ret)}

    t0 = time.perf_counter()
    x = features(q0, q1, mu, grasp.T_obj_tcp, grasp.width, sc.obj.mass, sc.obj.com, sc.obj.inertia)
    c, T_pred = net_predict(net_path, x)
    as_pred = pr.evaluate(c, T_pred)["feasible"]
    T_fix = slow_to_feasible(pr, c, T_pred)
    net = Solution(T_fix, c, False, {})
    if np.isfinite(T_fix):
        e = pr.evaluate(c, T_fix)
        net = Solution(T_fix, c, e["feasible"], e)
    sols["net"] = with_fallback(net, {"as_predicted": bool(as_pred), "slowed_by": round(T_fix / T_pred, 3)
                                       if np.isfinite(T_fix) else None}, t0)
    t0 = time.perf_counter()
    start = net if net.feasible else Solution(T_pred, c, False, {})
    ref = pr.shape_and_time(start, iters=refine_iters, seed=1)
    sols["net+refine"] = with_fallback(ref, {}, t0)

    recs = []
    for name in METHODS:
        s, secs, extra = sols[name]
        p = dict(paths)
        p["transport"] = pr.sample(s.c, s.T, dt)
        r = sc.ep.execute(grasp, p, mu=mu, log_every=5)
        m = motion_metrics(r.transport, sc.obj, grasp.T_obj_tcp, prm)
        recs.append({**base, "method": name, "planned_T": round(s.T, 3), "planned_feasible": bool(s.feasible),
                     "plan_s": round(secs, 4), "success": bool(r.success), "held": bool(r.held),
                     "collisions": len(r.collisions), "slip_mm": round(r.slip_mm, 2),
                     "duration_s": round(m["duration_s"], 3), "u_max": round(m["u_max"], 3), **extra})
    return recs


def summarise(recs):
    ok = [r for r in recs if r.get("status") == "ok"]
    out = {"problems": len({r["file"] for r in ok}), "methods": {}}
    for m in METHODS:
        rs = [r for r in ok if r["method"] == m]
        if not rs:
            continue
        s = [r["success"] for r in rs]
        lo, hi = wilson(sum(s), len(rs))
        d = {"n": len(rs), "success": round(float(np.mean(s)), 3), "ci95": [round(lo, 3), round(hi, 3)],
             "planned_feasible": round(float(np.mean([r["planned_feasible"] for r in rs])), 3),
             "duration_median_s": round(float(np.median([r["planned_T"] for r in rs])), 3),
             "plan_s_median": round(float(np.median([r["plan_s"] for r in rs])), 4)}
        if m.startswith("net"):
            d["fallback_to_retime"] = round(float(np.mean([r["fallback"] for r in rs])), 3)
            d["retime_was_shorter"] = round(float(np.mean([r["retime_shorter"] for r in rs])), 3)
        if m == "net":
            d["feasible_as_predicted"] = round(float(np.mean([r["as_predicted"] for r in rs])), 3)
        out["methods"][m] = d
    by = {}
    for r in ok:
        by.setdefault(r["file"], {})[r["method"]] = r
    ratios = {}
    for m in ("optimiser", "net", "net+refine"):
        both = [v for v in by.values() if m in v and "retime" in v and v["retime"]["planned_feasible"]]
        ratios[f"{m} / retime"] = round(float(np.median([v[m]["planned_T"] / v["retime"]["planned_T"] for v in both])), 3)
        if m != "optimiser":
            both = [v for v in by.values() if m in v and "optimiser" in v]
            ratios[f"{m} / optimiser"] = round(float(np.median([v[m]["planned_T"] / v["optimiser"]["planned_T"]
                                                                 for v in both])), 3)
    out["duration_ratio_median"] = ratios
    # problems the straight motion cannot do at all (retime infeasible): only shaping helps there
    hard = [v for v in by.values() if "retime" in v and not v["retime"]["planned_feasible"]]
    out["retime_infeasible"] = {"n": len(hard), **{m: int(sum(v[m]["planned_feasible"] for v in hard if m in v))
                                                   for m in ("optimiser", "net", "net+refine")}}
    return out


def plot(recs, s, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"retime": "#e07a2f", "optimiser": "#1d4e6e", "net": "#9aa5b1", "net+refine": "#3f8f6b"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ms = [m for m in METHODS if m in s["methods"]]
    for m in ms:
        d = s["methods"][m]
        ax[0].scatter(d["plan_s_median"] * 1e3, d["duration_median_s"], s=60, c=col[m], label=m, zorder=3)
        ax[0].annotate(f"{d['success']:.0%} success", (d["plan_s_median"] * 1e3, d["duration_median_s"]),
                       xytext=(6, 4), textcoords="offset points", fontsize=8, color="0.3")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("planning time per motion, median (ms, log)")
    ax[0].set_ylabel("transport duration, median (s)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(f"{s['problems']} problems on unseen objects", fontsize=9)
    ok = [r for r in recs if r.get("status") == "ok"]
    by = {}
    for r in ok:
        by.setdefault(r["file"], {})[r["method"]] = r
    for i, m in enumerate(("net", "net+refine")):
        v = [b[m]["planned_T"] / b["optimiser"]["planned_T"] for b in by.values() if m in b and "optimiser" in b]
        ax[1].scatter(i + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, s=10, c=col[m], alpha=0.6)
        ax[1].hlines(np.median(v), i - 0.25, i + 0.25, colors="k", lw=2)
        ax[1].text(i + 0.28, np.median(v), f"median {np.median(v):.2f}", va="center", fontsize=8)
    ax[1].axhline(1, ls=":", c="0.5")
    ax[1].set_xticks([0, 1], ["network (+ slow down)", "network + short refine"], fontsize=8)
    ax[1].set_ylabel("duration / optimiser's duration")
    ax[1].set_yscale("log")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "transport_dataset_test")
    p.add_argument("--net", type=Path, default=ROOT / "data" / "models" / "transport_net.pt")
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--refine-iters", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    args = p.parse_args()
    files = sorted(args.data.glob("*.npz"))[: args.limit]
    jobs = [(str(f), args.model, args.clouds, args.net, args.refine_iters) for f in files]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} problems, {args.workers} workers")
    recs = []
    with Pool(args.workers, maxtasksperchild=4) as pool:
        for k, rs in enumerate(pool.imap_unordered(run, jobs), 1):
            recs += rs
            ok = [r for r in rs if r.get("status") == "ok"]
            line = "  ".join(f"{r['method']}: {r['planned_T']:.2f} s {'ok' if r['success'] else 'FAIL'}" for r in ok)
            print(f"[{k}/{len(jobs)}] {(rs[0]['object'][:24] if rs else '(none)'):24s} {line or (rs[0]['status'] if rs else '')}",
                  flush=True)
    (OUT / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    s = summarise(recs)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    plot(recs, s, ROOT / "media" / "transport_net_episodes.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
