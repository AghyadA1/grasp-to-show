"""Hook vs pinch in full episodes, as the object gets heavier (part 3).

Objects with a handle (object_set_v2 "handle_candidates" that have grasp data). For each: random
scenes (start pose and scanner pose, seeds 9000 + object index as in eval_way_b_v2.py) and mass
factors (the object x1, x2, x4, x8; the checks and the simulator both know the mass). Policies:

- pinch_way_b (with --way-b MODEL): way B's top grasps for the scene (the network was trained at
  the objects' own mass and is not told the weight), first one that plans;
- pinch: the analytic checks (lift, kinematics, collisions, stability at the target, QR
  visibility) on the object's antipodal grasps (data/grasps_v2, rims included when generated with
  --min-width 0.002), best ranked choice that plans;
- hook: grasp2show/hanging.py: hook grasps that fit the handle and lift it (cached in
  data/hooks/), where the object hangs, the hand turned so the QR faces the scanner, reach /
  collision / visibility checks, then a carry that keeps the fingers from pointing down.

Both are executed in the simulator and judged by the same ground truth: QR decoded by the scanner,
object held, no unexpected contact; a pinch must also not slip (< 20 mm / 15 deg after the lift),
while a hooked object hangs and may swing (its motion relative to the hand is reported, not
judged). --scanner-scale 2: a 1280 x 960 scanner instead of 640 x 480 (for both policies).

Usage:  $env:G2S_OBJECT_SET="v2"; python scripts/hook_episodes.py --workers 4 --way-b data/models/heatmap_net_v4.pt
Outputs: data/hook_episodes/records.json, summary.json; media/hook_episodes.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from grasp2show.config import ObjectConfig, SceneConfig, random_camera
from grasp2show.experiment import Scenario, choose
from grasp2show.gripper_sim import GripperSim
from grasp2show.hanging import HOOK_SPEED, hook_options, plan_hook, usable_hooks
from grasp2show.objects import load_object_set, object_from_config
from grasp2show.task import Grasp

OUT = ROOT / "data" / "hook_episodes"
HOOKS = ROOT / "data" / "hooks"
GRASPS = ROOT / "data" / "grasps_v2"


def hooks_for(name: str) -> list[dict]:
    """Usable hooks of an object, computed once (about 2 min) and cached."""
    f = HOOKS / f"{name}.npz"
    if f.exists():
        z = np.load(f)
        return [{"T": T, "width": float(w), "kind": str(k)} for T, w, k in zip(z["T"], z["width"], z["kind"])]
    obj = object_from_config(ObjectConfig(asset=name))
    hooks = usable_hooks(obj, GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj))
    HOOKS.mkdir(parents=True, exist_ok=True)
    np.savez(f, T=np.array([h["T"] for h in hooks]).reshape(-1, 4, 4), width=np.array([h["width"] for h in hooks]),
             kind=np.array([h["kind"] for h in hooks]))
    return hooks


def run_hook(sc: Scenario, hooks: list[dict], tries: int = 5) -> dict:
    if not hooks:
        return {"success": False, "reason": "no hook fits", "n_feasible": 0}
    sim = GripperSim(sc.cfg, sc.obj)                    # the object at this scene's mass
    opts = hook_options(sc.fc, sim, hooks, T_start=sc.ep.resting_pose())
    ok = [o for o in opts if o.ok]
    rec = {"n_feasible": len(ok)}
    if not ok:
        whys = [o.why or "qr not visible" for o in opts]
        return {**rec, "success": False, "reason": "no hook: " + max(set(whys), key=whys.count)}
    why = ""
    for o in ok[:tries]:
        paths, why = plan_hook(sc.ep, sc.fc, o)
        if paths is None:
            continue
        r = sc.ep.execute(Grasp(f"hook{o.hook}", o.T_obj_tcp, o.width, "hook"), paths, speed=HOOK_SPEED)
        reason = "" if r.success else ("collision" if r.collisions else "dropped" if not r.held else "qr")
        return {**rec, "success": bool(r.success), "reason": reason, "kind": hooks[o.hook]["kind"],
                "pred_angle_deg": round(o.vis.angle_deg, 1), "pred_pixels": round(o.vis.pixels, 1),
                "hang_tilt_deg": round(o.hang.tilt_deg, 1), "qr_ok": bool(r.qr_ok), "held": bool(r.held),
                "slip_mm": round(r.slip_mm, 1), "slip_deg": round(r.slip_deg, 1), "collisions": r.collisions}
    return {**rec, "success": False, "reason": f"plan: {why}"}


def run(job) -> list[dict]:
    name, k, yaw, xy, cam, mass, way_b, clouds, scale = job
    if scale != 1:
        cam = replace(cam, width=int(cam.width * scale), height=int(cam.height * scale))
    t0 = time.time()
    hooks = hooks_for(name)
    z = np.load(GRASPS / f"{name}.npz")
    sc = Scenario(name, yaw, {kk: z[kk] for kk in ("T_obj_tcp", "width", "success")}, start_xy=xy, camera=cam,
                  mass_scale=mass)
    sc.lift_margin = np.where(sc.lift_ok, np.clip(sc.lift_margin, 0.1, 1.0), -1.0)   # as eval_way_b_v2
    base = {"object": name, "scene": k, "mass_scale": mass, "mass_kg": round(sc.obj.mass, 3)}
    pinch = asdict(sc.run("pinch", choose("aware", sc.reports, sc.lift_margin)))
    hook = run_hook(sc, hooks)
    out = [{**base, **pinch, "policy": "pinch"}, {**base, **hook, "policy": "hook"}]
    if way_b is not None:
        from eval_way_b import kinematic_choices, way_b_grasps

        T, W, R, _ = way_b_grasps(name, yaw, way_b, clouds, 10, start_xy=xy, camera=cam)
        if len(T):
            sb = Scenario(name, yaw, {"T_obj_tcp": T, "width": W, "success": np.ones(len(T), bool)}, start_xy=xy,
                          camera=cam, mass_scale=mass)
            out.append({**base, **asdict(sb.run("pinch_way_b", kinematic_choices(sb, list(zip(range(len(T)),
                                                                                              R.tolist()))))),
                        "policy": "pinch_way_b"})
        else:
            out.append({**base, "policy": "pinch_way_b", "success": False, "reason": "no grasp"})
    print(f"{name[:28]:28s} scene {k} x{mass}: " + "  ".join(
        f"{r['policy']} {'ok' if r['success'] else r['reason']}" for r in out) + f"  ({time.time() - t0:.0f} s)",
        flush=True)
    return out


def plot(recs, masses, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for pol, c in (("pinch", "#4C78A8"), ("pinch_way_b", "#72B7B2"), ("hook", "#E45756")):
        if not any(r["policy"] == pol for r in recs):
            continue
        y = [np.mean([r["success"] for r in recs if r["policy"] == pol and r["mass_scale"] == m]) for m in masses]
        ax.plot(masses, np.array(y) * 100, "o-", color=c, label=pol, lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(masses, [f"x{m}" for m in masses])
    ax.set_xlabel("object mass (x the scanned mug's)")
    ax.set_ylabel("episodes succeeded (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Handle objects: hook vs pinch, full pick-and-show episodes")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", type=int, default=12)
    p.add_argument("--masses", type=float, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--objects", nargs="*", default=None, help="default: handle candidates with grasp data")
    p.add_argument("--way-b", type=Path, default=None, help="also way B's pinch choice with this network")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset_v2" / "clouds.npz")
    p.add_argument("--scanner-scale", type=float, default=1.0, help="scanner resolution x this (both policies)")
    p.add_argument("--tag", default="", help="outputs: data/hook_episodes<tag>/, media/hook_episodes<tag>.png")
    p.add_argument("--update", action="store_true", help="replace only these masses in an existing run (--tag)")
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    args = p.parse_args()
    oset = load_object_set()
    names = args.objects or [n for n in oset["handle_candidates"] if (GRASPS / f"{n}.npz").exists()]
    masses = [int(m) if float(m).is_integer() else m for m in args.masses]
    with Pool(args.workers) as pool:                       # hooks first (cached), one object per worker
        for n, h in zip(names, pool.map(hooks_for, names)):
            print(f"{n}: {len(h)} usable hooks", flush=True)
    jobs = []
    for n in names:
        r = np.random.default_rng(9000 + oset["objects"].index(n))
        for k in range(args.scenes):
            yaw, xy, cam = float(r.uniform(0, 360)), (float(r.uniform(0.40, 0.60)), float(r.uniform(-0.30, 0.0))), \
                random_camera(r)
            jobs += [(n, k, yaw, xy, cam, m, args.way_b, args.clouds, args.scanner_scale) for m in masses]
    print(f"{len(jobs)} scenes ({len(names)} objects x {args.scenes} scenes x {len(masses)} masses), "
          f"{args.workers} workers", flush=True)
    recs = []
    with Pool(args.workers) as pool:
        for rr in pool.imap_unordered(run, jobs):
            recs += rr
    out = OUT.with_name(OUT.name + args.tag)
    out.mkdir(parents=True, exist_ok=True)
    if args.update and (out / "records.json").exists():     # re-run some masses, keep the others
        old = json.loads((out / "records.json").read_text())
        recs = [r for r in old if r["mass_scale"] not in masses] + recs
        masses = sorted({r["mass_scale"] for r in recs})
    (out / "records.json").write_text(json.dumps(recs, indent=1, default=float))
    summary = {"objects": names, "scenes": args.scenes, "masses": masses, "scanner_scale": args.scanner_scale,
               "policies": {}}
    pols = [p for p in ("pinch", "pinch_way_b", "hook") if any(r["policy"] == p for r in recs)]
    for pol in pols:
        summary["policies"][pol] = {
            str(m): {"success": float(np.mean([r["success"] for r in recs if r["policy"] == pol and r["mass_scale"] == m])),
                     "failures": {f: sum(1 for r in recs if r["policy"] == pol and r["mass_scale"] == m and not r["success"]
                                        and r["reason"].split(":")[0] == f)
                                  for f in sorted({r["reason"].split(":")[0] for r in recs if not r["success"]})}}
            for m in masses}
        summary["policies"][pol]["by_object"] = {
            n: [float(np.mean([r["success"] for r in recs if r["policy"] == pol and r["object"] == n
                               and r["mass_scale"] == m])) for m in masses] for n in names}
    summary["hook_or_pinch"] = {      # what a chooser that tries the hook when one is feasible could reach
        str(m): float(np.mean([any(r["success"] for r in recs if r["object"] == o and r["scene"] == s
                                   and r["mass_scale"] == m) for o in names for s in range(args.scenes)]))
        for m in masses}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    plot(recs, masses, ROOT / "media" / f"hook_episodes{args.tag}.png")
    for pol in pols:
        print(pol, {m: round(v["success"], 2) for m, v in summary["policies"][pol].items() if m != "by_object"})
    print(f"wrote {out / 'summary.json'} and media/hook_episodes{args.tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
