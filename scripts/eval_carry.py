"""The carry network against the planner, in full episodes on the evaluation's scenes.

The scenes of an evaluation run (scripts/evaluate.py records.jsonl; held-out objects, the same
unknown pose errors): for way B's pinch (verified ranking) and the hook (robust ranking), the
first candidate whose motion can be planned is run twice, with the planner's carry and with the
carry network's (data/models/carry_net.pt), from the same grasp: QR read, held, no collision.
Way B's training objects are left out.

Usage:  python scripts/eval_carry.py --run data/runs/2026-10-31_chooser_table --workers 4
Outputs: <run>/carry_eval.jsonl, carry_eval_summary.json (printed)
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

_CACHE: dict = {}


def scene_job(args) -> dict:
    name, k, scale, hooks_dir, net_path = args
    from grasp2show.learn.carry_net import load_carry_net
    from grasp2show.pipeline import Assets, WayB, add_wayb, check, execute_hook, execute_wayb, sample_scene

    if "wayb" not in _CACHE:
        _CACHE["wayb"] = WayB()
    if _CACHE.get("net_path") != net_path:
        _CACHE["net"], _CACHE["net_path"] = load_carry_net(net_path), net_path
    net = _CACHE["net"]
    scene, off = sample_scene(name, k, scale)
    ck = check(scene, Assets.load(name, Path(hooks_dir)))
    add_wayb(ck, _CACHE["wayb"], ck.sc.cfg, ck.sc.obj)
    rec = {"object": name, "scene": k, "mu": scene.mu, "load": ck.features["load"]}
    for strat, ranked, run in (("pinch", ck.wayb_verified() if ck.wb is not None else [], execute_wayb),
                               ("hook", ck.hook_ranked(True), execute_hook)):
        for c in ranked[:20]:
            ok, why = run(ck, c, off)
            if ok is None:
                continue
            ok2, why2 = run(ck, c, off, carry_net=net)
            rec[strat] = {"planner": bool(ok), "planner_why": why, "net": bool(ok2), "net_why": why2}
            break
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--net", type=Path, default=ROOT / "data" / "models" / "carry_net.pt")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    train = set(json.loads((ROOT / "grasp2show" / "object_set_v2.json").read_text())["train"])
    recs = [json.loads(x) for x in (args.run / "records.jsonl").read_text().splitlines()]
    recs = [r for r in recs if r["object"] not in train][: args.limit]
    out_f = args.run / "carry_eval.jsonl"
    done = {(d["object"], d["scene"]) for d in map(json.loads, out_f.read_text().splitlines())} if out_f.exists() else set()
    jobs = [(r["object"], r["scene"], float(cfg.get("scanner_scale", 2.0)), cfg["hooks_dir"], str(args.net))
            for r in recs if (r["object"], r["scene"]) not in done]
    print(f"{len(jobs)} scenes ({len(done)} done) -> {out_f}", flush=True)
    pool = Workers(args.workers)
    try:
        with open(out_f, "a", encoding="utf-8") as fh:
            for i, (job, r) in enumerate(pool.run(scene_job, jobs), 1):
                if r is None:
                    print(f"  {job[0]} scene {job[1]}: lost", flush=True)
                    continue
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                f = lambda x: "-" if not x else (f"planner {'ok' if x['planner'] else x['planner_why'][:12]}, "
                                                  f"net {'ok' if x['net'] else x['net_why'][:12]}")
                print(f"  [{i}/{len(jobs)}] {r['object'][:28]:28s} pinch: {f(r.get('pinch'))}; hook: {f(r.get('hook'))}",
                      flush=True)
    finally:
        pool.close()
    from scipy.stats import binomtest

    allr = [json.loads(x) for x in out_f.read_text().splitlines()]
    summ = {}
    for s in ("pinch", "hook"):
        xs = [r[s] for r in allr if r.get(s)]
        if not xs:
            continue
        w = sum(x["net"] and not x["planner"] for x in xs)
        lo = sum(x["planner"] and not x["net"] for x in xs)
        summ[s] = {"n": len(xs), "planner": float(np.mean([x["planner"] for x in xs])),
                   "net": float(np.mean([x["net"] for x in xs])), "won": w, "lost": lo,
                   "p": float(binomtest(w, w + lo).pvalue) if w + lo else 1.0,
                   "planner_failures": dict(Counter((x["planner_why"] or "ok").split(":")[0] for x in xs if not x["planner"])),
                   "net_failures": dict(Counter((x["net_why"] or "ok").split(":")[0] for x in xs if not x["net"]))}
    (args.run / "carry_eval_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
