"""Generate objects with handles and check which ones the robot can really hook (part 3).

1. Objects from seeds (grasp2show/shapegen.py): random bodies and handles, density 150-1200
   kg/m^3, kept if they stand on their base and have a spot for the QR sticker. Written to
   data/generated_objects/models/gen_<seed>/.
2. Each one goes through the hook pipeline used for the scans (grasp2show/hanging.py):
   fingertip placements fitted to its holes, nudged clear of the handle, kept if they lift the
   object in the floating-gripper test; where it hangs on each; cached in data/hooks/.
3. Table check: with the object standing on the table, the hand at the hook must stay above the
   table top (the porcelain mug's lesson). An object is "hookable" if at least one hook passes.

Usage:  python scripts/generate_hookable.py --count 500 --workers 4
        python scripts/generate_hookable.py --count 20 --figure     (and a picture of them)
Outputs: grasp2show/object_set_gen.json (names, hookable, per-object hook counts, train/test
         split), media/generated_objects.png with --figure. Use the set with G2S_OBJECT_SET=gen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp2show.config import ObjectConfig, SceneConfig
from grasp2show.feasibility import twin_of
from grasp2show.gripper_sim import GripperSim
from grasp2show.hanging import hand_min_z, hook_library
from grasp2show.objects import GENERATED_DIR, object_from_config
from grasp2show.shapegen import generate, write

SET_FILE = ROOT / "grasp2show" / "object_set_gen.json"


def make(seed: int) -> dict | None:
    g = generate(seed)
    if g is None:
        return None
    density = float(np.random.default_rng(seed + 10_000).uniform(150, 1200))
    props = write(g, GENERATED_DIR, density)
    if props is None:
        return None
    return {"name": g.name, "seed": seed, "mass": round(props["mass"], 3), "handle": g.params["handle"],
            "body": g.params["body"]}


def check(name: str) -> dict:
    t0 = time.time()
    hooks = hook_library(name)
    obj = object_from_config(ObjectConfig(asset=name))
    sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj)
    table_ok = 0
    for h in hooks:
        for T in (h["T"], twin_of(h["T"])):
            sim._reset(np.eye(3), T, h["width"])
            table_ok += hand_min_z(sim) > 0.002            # object upright, base on z = 0
    return {"name": name, "hooks": len(hooks), "table_ok": int(table_ok), "hookable": table_ok > 0,
            "kinds": dict(Counter(h["kind"] for h in hooks)), "seconds": round(time.time() - t0, 1)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=500, help="objects to keep")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    p.add_argument("--figure", action="store_true", help="also draw media/generated_objects.png (first 20)")
    args = p.parse_args()
    made, seed = [], args.start_seed
    while len(made) < args.count:
        m = make(seed)
        seed += 1
        if m is not None:
            made.append(m)
    print(f"{len(made)} objects from seeds {args.start_seed}-{seed - 1}; checking hooks with {args.workers} workers",
          flush=True)
    res = {}
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(check, [m["name"] for m in made]), 1):
            res[r["name"]] = r
            print(f"[{i}/{len(made)}] {r['name']}: {r['hooks']} hooks, {r['table_ok']} clear of the table "
                  f"({r['seconds']} s)", flush=True)
    objects = [{**m, **res[m["name"]]} for m in made]
    rng = np.random.default_rng(0)
    test = set(rng.choice([o["name"] for o in objects], size=max(1, len(objects) // 5), replace=False).tolist())
    kinds = Counter(o["handle"]["where"] for o in objects)
    hookable = Counter(o["handle"]["where"] for o in objects if o["hookable"])
    out = {"source": "grasp2show/shapegen.py via scripts/generate_hookable.py", "seeds": [args.start_seed, seed - 1],
           "objects": [o["name"] for o in objects], "train": [o["name"] for o in objects if o["name"] not in test],
           "test": sorted(test), "hookable": [o["name"] for o in objects if o["hookable"]],
           "by_handle": {k: {"objects": kinds[k], "hookable": hookable[k]} for k in kinds},
           "details": objects}
    SET_FILE.write_text(json.dumps(out, indent=1))
    n_h = len(out["hookable"])
    print(f"{n_h}/{len(objects)} hookable from the table; by handle: "
          + ", ".join(f"{k} {v['hookable']}/{v['objects']}" for k, v in out["by_handle"].items()))
    print(f"wrote {SET_FILE}")
    if args.figure:
        figure(objects[:20])
    return 0


def figure(objects: list[dict]) -> None:
    """Each object standing on a grey table, with the hand at a hook that clears the table (or at
    one that does not, or no hand: no hook at all)."""
    import cv2
    import mujoco

    S = 300
    tiles = []
    for o in objects:
        name = o["name"]
        obj = object_from_config(ObjectConfig(asset=name))
        sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj)
        hooks = hook_library(name)
        pose = None
        for h in hooks:
            for T in (h["T"], twin_of(h["T"])):
                sim._reset(np.eye(3), T, h["width"])
                if hand_min_z(sim) > 0.002 or pose is None:
                    pose = (T, h["width"], hand_min_z(sim) > 0.002)
            if pose and pose[2]:
                break
        m, d = sim.model, sim.data
        m.vis.headlight.ambient[:] = 0.5
        if pose:
            sim._reset(np.eye(3), pose[0], pose[1])
        else:                                                        # hand out of the way
            sim._reset(np.eye(3), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2.0], [0, 0, 0, 1.0]]), 0.08)
        r = mujoco.Renderer(m, S, S)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        c = obj.mesh.bounds.mean(0)
        cam.lookat[:] = c if not pose else 0.5 * (c + pose[0][:3, 3])
        cam.distance = 0.32 + 1.2 * float(obj.mesh.extents.max())
        cam.azimuth = 45.0 if not pose else float(np.degrees(np.arctan2(*(pose[0][:2, 3] - c[:2])[::-1]))) + 60
        cam.elevation = -22.0
        r.update_scene(d, camera=cam)
        scn = r.scene
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([0.3, 0.3, 0.0008]),
                            np.array([c[0], c[1], -0.0008]), np.eye(3).ravel(), np.array([0.55, 0.55, 0.55, 0.5], np.float32))
        scn.ngeom += 1
        img = r.render().copy()
        r.close()
        hp = o["handle"]
        verdict = (f"hookable: {o['table_ok']} hooks" if o["hookable"] else
                   f"{o['hooks']} hooks, all hit the table" if o["hooks"] else "no hook fits")
        col = (60, 170, 60) if o["hookable"] else (200, 60, 60)
        cv2.rectangle(img, (0, 0), (S, 40), (245, 245, 245), -1)
        cv2.putText(img, f"{name}  {hp['where']} {hp['base']} {hp['wiggle']} {hp['section']}  {o['mass']:.2f} kg",
                    (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(img, f"{verdict}  (gap {hp['gap'] * 100:.1f} x {hp['span'] * 100:.1f} cm)", (6, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 5:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.concatenate([np.concatenate(tiles[i:i + 5], 1) for i in range(0, len(tiles), 5)], 0)
    out = ROOT / "media" / "generated_objects.png"
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
