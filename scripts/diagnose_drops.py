"""Where in the motion are objects lost, and how fast is each way of picking a pinch?

Replays the scenes of an evaluation run (scripts/evaluate.py records.jsonl: same scene, same
unknown pose error) for each strategy's first candidate that can be planned, the one the robot
runs: the analytic pinch (robust ranking), way B (checked ranking; data/models/heatmap_net_v5.pt)
and the hook (robust ranking). For each: success, why it failed, and where the object was lost
(episode.EpisodeResult.lost: the phase - close, lift, settle, transit, present, hold - and how far
into it, at the first moment the object had moved 10 mm / 10 deg in the hand, or a hooked object
had touched no finger for 0.2 s), the slip at the end of every phase, and the load on a pinch.
Also the planning time of each pinch: the analytic checks of all ~300 candidates vs way B (its
network plus the checks of its 10 grasps).

Usage:  python scripts/diagnose_drops.py --run data/runs/2026-10-31_chooser_table --workers 4
Outputs: <run>/drops.jsonl, drops_summary.json (printed)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("G2S_OBJECT_SET", "v2")
from grasp2show.parallel import Workers

STRATEGIES = ("pinch", "wayb", "hook")
_WAYB = None


def replay(args) -> dict:
    global _WAYB
    name, k, scale, hooks_dir = args
    from grasp2show.feasibility import twin_of
    from grasp2show.hanging import HOOK_SPEED, plan_hook
    from grasp2show.pipeline import Assets, WayB, add_wayb, check, sample_scene
    from grasp2show.task import Grasp, show_pose

    if _WAYB is None:
        _WAYB = WayB()
    scene, off = sample_scene(name, k, scale)
    assets = Assets.load(name, Path(hooks_dir))
    ck = check(scene, assets)
    add_wayb(ck, _WAYB, ck.sc.cfg, ck.sc.obj)
    rec = {"object": name, "scene": k, "mu": scene.mu, "mass_kg": round(ck.sc.obj.mass, 3),
           "load": ck.features["load"], "t_pinch_checks": round(ck.seconds["pinch"], 2),
           "t_wayb": round(ck.seconds["wayb"], 2), "t_hook_checks": round(ck.seconds["hook"], 2)}

    def result(r, hook=False):
        return {"success": bool(r.success), "reason": r.reason(hook=hook), "lost": r.lost, "phases": r.phase_slip}

    for strat in STRATEGIES:
        out = None
        if strat == "hook":
            for o in ck.hook_ranked(True)[:20]:
                paths, _ = plan_hook(ck.sc.ep, ck.sc.fc, o)
                if paths is None:
                    continue
                r = ck.sc.ep.execute(Grasp("hook", o.T_obj_tcp, o.width, "hook"), paths, speed=HOOK_SPEED,
                                     mu=scene.mu, obj_offset=off)
                out = result(r, hook=True)
                break
        else:
            S = ck.sc if strat == "pinch" else ck.wb
            ranked = ck.pinch_ranked(True) if strat == "pinch" else ck.wayb_ranked(True)
            for c in ranked[:20]:
                tw = S.reports[c.grasp].twins[c.twin]
                t = tw.targets[c.roll_idx]
                T = S.T[c.grasp] if c.twin == 0 else twin_of(S.T[c.grasp])
                paths, _ = S.ep.plan_analytic(T, show_pose(S.cfg, S.obj, float(S.rolls[c.roll_idx])), tw.start.q,
                                              t.reach.q, checker=S.fc, width=float(S.W[c.grasp]))
                if paths is None:
                    continue
                r = S.ep.execute(Grasp("g", T, float(S.W[c.grasp])), paths, obj_offset=off, mu=scene.mu)
                out = {**result(r), "lift_margin": round(float(S.lift_margin[c.grasp]), 3),
                       "stab_margin": round(float(t.stability_margin), 3)}
                break
        rec[strat] = out
    return rec


def summary(recs: list[dict]) -> dict:
    out = {"scenes": len(recs), "planning_s": {"pinch_checks_median": float(np.median([r["t_pinch_checks"] for r in recs])),
                                               "wayb_median": float(np.median([r["t_wayb"] for r in recs]))}}
    for s in STRATEGIES:
        ran = [r[s] for r in recs if r.get(s)]
        lost = [x["lost"] for x in ran if x["lost"]]
        if s != "hook":     # did the checks expect it to hold? (lift margin > 0 at this friction and weight)
            for tag, sel in (("predicted_to_hold", [x for x in ran if x["lift_margin"] > 0]),
                             ("predicted_to_fail", [x for x in ran if x["lift_margin"] <= 0])):
                out.setdefault(f"{s}_{tag}", {"ran": len(sel), "success": float(np.mean([x["success"] for x in sel]))
                                              if sel else float("nan"),
                                              "lost_in_phase": dict(Counter(x["lost"][0] for x in sel if x["lost"]))})
        failed = [x for x in ran if not x["success"]]
        out[s] = {"ran": len(ran), "success": float(np.mean([x["success"] for x in ran])) if ran else float("nan"),
                  "reasons": dict(Counter((x["reason"] or "ok").split(":")[0] for x in ran).most_common()),
                  "lost_in_phase": dict(Counter(x[0] for x in lost).most_common()),
                  "failures_lost_in_phase": dict(Counter(x["lost"][0] if x["lost"] else "never lost"
                                                         for x in failed).most_common())}
        by = {}
        for x in lost:
            by.setdefault(x[0], []).append(x[1])
        out[s]["seconds_into_phase_median"] = {p: float(np.median(v)) for p, v in by.items()}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    recs = [json.loads(x) for x in (args.run / "records.jsonl").read_text().splitlines()]
    recs = [r for r in recs if r["pinch_tries_robust"] or r["hook_tries_robust"]][: args.limit]
    out_f = args.run / "drops.jsonl"
    done = {(d["object"], d["scene"]) for d in map(json.loads, out_f.read_text().splitlines())} if out_f.exists() else set()
    jobs = [(r["object"], r["scene"], float(cfg.get("scanner_scale", 2.0)), cfg["hooks_dir"]) for r in recs
            if (r["object"], r["scene"]) not in done]
    print(f"{len(jobs)} scenes to replay ({len(done)} done) -> {out_f}", flush=True)
    pool = Workers(args.workers)
    try:
        with open(out_f, "a", encoding="utf-8") as fh:
            for i, (job, r) in enumerate(pool.run(replay, jobs), 1):
                if r is None:
                    print(f"  {job[0]} scene {job[1]}: lost (crashed or hung)", flush=True)
                    continue
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                f = lambda x: "-" if not x else "ok" if x["success"] else (x["reason"][:10] + (
                    f" in {x['lost'][0]} +{x['lost'][1]}s" if x["lost"] else ""))
                print(f"  [{i}/{len(jobs)}] {r['object'][:28]:28s} load {r['load']:.2f}: pinch {f(r['pinch'])}; "
                      f"way B {f(r['wayb'])}; hook {f(r['hook'])}  (checks {r['t_pinch_checks']:.1f} s, "
                      f"way B {r['t_wayb']:.1f} s)", flush=True)
    finally:
        pool.close()
    allr = [json.loads(x) for x in out_f.read_text().splitlines()]
    s = summary(allr)
    (args.run / "drops_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
