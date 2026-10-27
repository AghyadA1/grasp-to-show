"""Choose the grasp and the transport motion together.

Way B ranks grasps by how likely they are to show the QR code successfully; until now the robot
took the first one it could plan and only then worked out how to carry it. But two grasps that
are nearly as good for the task can differ a lot for the carry: where the fingers hold relative
to the centre of mass (leverage), and how the finger axis is oriented along the way decide how
fast the object can be moved without slipping. The transport network plans a motion in about a
millisecond, so every candidate grasp can be tried (with the optimiser this would be ~25 s per
grasp).

Policies, on the same problems (the unseen-object test problems: random place, rotation,
friction), all carried with the raw network motion:
  way_b      way B's first plannable grasp (as before)
  joint      among the plannable grasps whose way-B score is within --delta of the best one
             (way B's top scores are close together, typically 0.86-0.89; default 0.02),
             the one with the shortest network motion that passes the safety model and the
             collision check (none passes: way B's first)
  time_only  the shortest feasible motion over all plannable grasps, ignoring way B's score
             (to show what the score protects)
Every choice is executed in simulation.

Usage:  python scripts/joint_grasp_motion.py [--workers 4] [--delta 0.02] [--mu 0.2 --tag mu02]
Outputs: data/joint_grasp_motion/records.jsonl, summary.json, media/joint_grasp_motion.png
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
from eval_way_b import kinematic_choices, way_b_grasps
from run_experiments import wilson
from transport_net_episodes import net_predict

from grasp2show.experiment import Scenario
from grasp2show.feasibility import twin_of
from grasp2show.task import Grasp, show_pose
from grasp2show.transport import friction_params, motion_metrics
from grasp2show.transport_opt import TransportProblem

OUT = ROOT / "data" / "joint_grasp_motion"
POLICIES = ("way_b", "joint", "time_only")


def candidates(sc: Scenario, choices):
    """Every plannable (grasp, target) pair, in way B's order: (choice, Grasp, paths)."""
    out = []
    for c in choices:
        tw = sc.reports[c.grasp].twins[c.twin]
        t = tw.targets[c.roll_idx]
        T = sc.T[c.grasp] if c.twin == 0 else twin_of(sc.T[c.grasp])
        paths, _ = sc.ep.plan_analytic(T, show_pose(sc.cfg, sc.obj, sc.rolls[c.roll_idx]), tw.start.q,
                                       t.reach.q, checker=sc.fc, width=float(sc.W[c.grasp]))
        if paths is not None:
            out.append((c, Grasp("way_b", T, float(sc.W[c.grasp])), paths))
    return out


def run(job):
    f, model, clouds, net_path, delta, mu_override = job
    from grasp2show.learn.transport_net import features

    z = dict(np.load(f, allow_pickle=True))
    asset, yaw, xy = str(z["object"]), float(z["yaw"]), tuple(z["start_xy"])
    mu = float(mu_override if mu_override is not None else z["mu"])
    t0 = time.perf_counter()
    T, W, R, _, Q = way_b_grasps(asset, yaw, model, clouds, 10, start_xy=xy, with_scores=True)
    if len(T) == 0:
        return []
    sc = Scenario(asset, yaw, {"T_obj_tcp": T, "width": W, "success": np.ones(len(T), bool)}, start_xy=xy)
    cands = candidates(sc, kinematic_choices(sc, list(zip(range(len(T)), R.tolist()))))
    t_plan = time.perf_counter() - t0
    if not cands:
        return []
    prm = friction_params(mu)
    t0 = time.perf_counter()
    info = []
    for c, g, paths in cands:
        q0, q1 = paths["lift"][-1], paths["present"][-1]
        pr = TransportProblem(sc.obj, g.T_obj_tcp, g.width, q0, q1, prm, checker=sc.fc)
        x = features(q0, q1, mu, g.T_obj_tcp, g.width, sc.obj.mass, sc.obj.com, sc.obj.inertia)
        coef, T_pred = net_predict(net_path, x)
        e = pr.evaluate(coef, T_pred)
        info.append({"score": float(Q[c.grasp]), "T": T_pred, "feasible": bool(e["feasible"]), "c": coef, "pr": pr,
                     "u_static": float(pr.u_static)})
    t_choose = time.perf_counter() - t0
    best_q = info[0]["score"]

    def shortest(idx):
        ok = [i for i in idx if info[i]["feasible"]]
        return min(ok, key=lambda i: info[i]["T"]) if ok else 0

    pick = {"way_b": 0,
            "joint": shortest([i for i in range(len(info)) if info[i]["score"] >= best_q - delta]),
            "time_only": shortest(range(len(info)))}
    base = {"file": Path(f).name, "object": asset, "yaw": round(yaw, 1), "mu": round(mu, 3), "mass": sc.obj.mass,
            "candidates": len(cands), "within_delta": int(sum(i["score"] >= best_q - delta for i in info)),
            "plan_all_s": round(t_plan, 2), "choose_ms": round(t_choose * 1e3, 1), "status": "ok"}
    done, recs = {}, []
    for name in POLICIES:
        i = pick[name]
        if i not in done:
            c, g, paths = cands[i]
            p = dict(paths)
            p["transport"] = info[i]["pr"].sample(info[i]["c"], info[i]["T"], sc.ep.model.opt.timestep)
            r = sc.ep.execute(g, p, mu=mu, log_every=5)
            m = motion_metrics(r.transport, sc.obj, g.T_obj_tcp, prm)
            done[i] = {"success": bool(r.success), "held": bool(r.held), "qr_ok": bool(r.qr_ok),
                       "collisions": len(r.collisions), "slip_mm": round(r.slip_mm, 2),
                       "duration_s": round(m["duration_s"], 3), "u_max": round(m["u_max"], 3)}
        recs.append({**base, "policy": name, "rank": i, "score": round(info[i]["score"], 3),
                     "planned_T": round(info[i]["T"], 3), "planned_feasible": info[i]["feasible"],
                     "u_static": round(info[i]["u_static"], 3), **done[i]})
    return recs


def summarise(recs):
    ok = [r for r in recs if r.get("status") == "ok"]
    out = {"problems": len({r["file"] for r in ok}), "policies": {}}
    for m in POLICIES:
        rs = [r for r in ok if r["policy"] == m]
        if not rs:
            continue
        s = [r["success"] for r in rs]
        lo, hi = wilson(sum(s), len(rs))
        out["policies"][m] = {"n": len(rs), "success": round(float(np.mean(s)), 3), "ci95": [round(lo, 3), round(hi, 3)],
                              "qr_read": round(float(np.mean([r["qr_ok"] for r in rs])), 3),
                              "held": round(float(np.mean([r["held"] for r in rs])), 3),
                              "planned_feasible": round(float(np.mean([r["planned_feasible"] for r in rs])), 3),
                              "transport_median_s": round(float(np.median([r["planned_T"] for r in rs])), 3),
                              "transport_mean_s": round(float(np.mean([r["planned_T"] for r in rs])), 3),
                              "way_b_score_mean": round(float(np.mean([r["score"] for r in rs])), 3),
                              "not_first_choice": round(float(np.mean([r["rank"] > 0 for r in rs])), 3)}
    by = {}
    for r in ok:
        by.setdefault(r["file"], {})[r["policy"]] = r
    for m in ("joint", "time_only"):
        v = [b for b in by.values() if m in b and "way_b" in b]
        diff = [b for b in v if b[m]["rank"] != b["way_b"]["rank"]]
        out[f"{m} vs way_b"] = {
            "different_grasp": len(diff),
            "time_ratio_median_all": round(float(np.median([b[m]["planned_T"] / b["way_b"]["planned_T"] for b in v])), 3),
            "time_ratio_median_changed": round(float(np.median([b[m]["planned_T"] / b["way_b"]["planned_T"]
                                                                 for b in diff])), 3) if diff else None,
            "wins": sum(b[m]["success"] and not b["way_b"]["success"] for b in v),
            "losses": sum(b["way_b"]["success"] and not b[m]["success"] for b in v)}
    out["median_candidates"] = float(np.median([b["way_b"]["candidates"] for b in by.values() if "way_b" in b]))
    out["choose_ms_median"] = float(np.median([b["way_b"]["choose_ms"] for b in by.values() if "way_b" in b]))
    return out


def plot(recs, s, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"way_b": "#9aa5b1", "joint": "#1d4e6e", "time_only": "#e07a2f"}
    by = {}
    for r in recs:
        if r.get("status") == "ok":
            by.setdefault(r["file"], {})[r["policy"]] = r
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for m in ("joint", "time_only"):
        v = [b for b in by.values() if m in b and "way_b" in b]
        x = [b["way_b"]["planned_T"] for b in v]
        y = [b[m]["planned_T"] for b in v]
        ax[0].scatter(x, y, s=12, c=col[m], alpha=0.7, label=m.replace("_", " "))
    lim = ax[0].get_xlim()[1]
    ax[0].plot([0, lim], [0, lim], ls=":", c="0.5")
    ax[0].set_xlabel("transport duration, way B's first grasp (s)")
    ax[0].set_ylabel("transport duration, chosen grasp (s)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(f"{s['problems']} problems on unseen objects", fontsize=9)
    ms = [m for m in POLICIES if m in s["policies"]]
    for i, m in enumerate(ms):
        d = s["policies"][m]
        ax[1].bar(i, d["success"], color=col[m], width=0.6)
        ax[1].errorbar(i, d["success"], yerr=[[d["success"] - d["ci95"][0]], [d["ci95"][1] - d["success"]]],
                       c="k", capsize=3, lw=1)
        ax[1].text(i, d["ci95"][1] + 0.01, f"{d['success']:.0%}\n{d['transport_median_s']:.2f} s", ha="center",
                   fontsize=8)
    ax[1].set_xticks(range(len(ms)), [m.replace("_", " ") for m in ms], fontsize=8)
    ax[1].set_ylim(0.5, 1.12)
    ax[1].set_ylabel("task success (95 % CI); median transport")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "transport_dataset_test")
    p.add_argument("--net", type=Path, default=ROOT / "data" / "models" / "transport_net_ss_mass.pt")
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--delta", type=float, default=0.02, help="how much way-B score the joint choice may give up")
    p.add_argument("--mu", type=float, default=None, help="one friction for every problem (default: as recorded)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    args = p.parse_args()
    files = sorted(args.data.glob("*.npz"))[: args.limit]
    jobs = [(str(f), args.model, args.clouds, args.net, args.delta, args.mu) for f in files]
    out = OUT if not args.tag else OUT.with_name(f"{OUT.name}_{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} problems, delta {args.delta}, mu {args.mu or 'as recorded'}, {args.workers} workers")
    recs = []
    with Pool(args.workers, maxtasksperchild=4) as pool:
        for k, rs in enumerate(pool.imap_unordered(run, jobs), 1):
            recs += rs
            line = "  ".join(f"{r['policy']}: #{r['rank']} {r['planned_T']:.1f}s {'ok' if r['success'] else 'FAIL'}"
                             for r in rs)
            print(f"[{k}/{len(jobs)}] {(rs[0]['object'][:22] if rs else '(none)'):22s} "
                  f"{(str(rs[0]['candidates']) + ' grasps') if rs else ''}  {line}", flush=True)
    (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    s = summarise(recs)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    plot(recs, s, ROOT / "media" / f"joint_grasp_motion{'_' + args.tag if args.tag else ''}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
