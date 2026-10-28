"""Hook vs pinch: how heavy can the object get before the grasp lets go? (part 3)

For each object with a handle or rim: the hook grasps that fit (hooks.fit_hook_poses) and lift
the object as it is, and pinch grasps (antipodal, as everywhere else in the project) that lift it,
all with the same simulated lift (hooks.lift_test: gravity on, lift 10 cm, hold 0.8 s). Then the
object is made heavier (mass and inertia x 2 ... x 16, the Franka's payload is 3 kg) and every
grasp that worked is tried again. The point of hooking: the weight presses the handle onto the
fingers and goes into the hand's structure, so the grip force (2 x 20 N) and friction stop being
the limit.
Also renders one hook per object (media/hook_<object>.gif).

Usage:  $env:G2S_OBJECT_SET="v2"; python scripts/hook_strength.py
Outputs: data/hook_strength/summary.json, media/hook_strength.png, media/hook_<object>.gif
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp2show.config import ObjectConfig, SceneConfig
from grasp2show.grasp_sampler import candidates_from_pairs, sample_antipodal
from grasp2show.gripper_sim import GripperSim
from grasp2show.hooks import find_loops, fit_hook_poses, lift_test
from grasp2show.objects import load_object_set, object_from_config

OUT = ROOT / "data" / "hook_strength"
SCALES = [1, 2, 4, 8, 12, 16]


def run(name: str) -> dict:
    rng = np.random.default_rng(0)
    obj = object_from_config(ObjectConfig(asset=name))
    sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj)
    hooks = [h for h in fit_hook_poses(obj, find_loops(obj.mesh, rng))
             if lift_test(sim, h["T"], h["width"], close=False)["success"]][:8]
    p1, p2 = sample_antipodal(obj.mesh, 60, rng)
    cand = candidates_from_pairs(p1, p2, 4, rng)
    cand = cand.subset(rng.permutation(len(cand)))
    pinches = []
    for T, w in zip(cand.T_obj_tcp, cand.width):
        if len(pinches) >= 8:
            break
        if not sim.in_collision(T, w) and lift_test(sim, T, w, close=True)["success"]:
            pinches.append({"T": T, "width": float(w)})
    curve = {"hook": [], "pinch": []}
    for k in SCALES:
        curve["hook"].append(int(sum(lift_test(sim, h["T"], h["width"], close=False, mass_scale=k)["success"]
                                     for h in hooks)))
        curve["pinch"].append(int(sum(lift_test(sim, p["T"], p["width"], close=True, mass_scale=k)["success"]
                                      for p in pinches)))
    if hooks:
        render(sim, obj, hooks[0], ROOT / "media" / f"hook_{name[:24]}.gif")
    return {"object": name, "mass_kg": round(obj.mass, 3), "hooks": len(hooks), "pinches": len(pinches),
            "hook_kinds": [h["kind"] for h in hooks], "held": curve}


def render(sim, obj, hook, path):
    """GIF of one hook lift, seen from the side of the fingers (along the hand's x axis, so the
    palm does not hide the handle), plus a still of the start and the end next to each other."""
    import imageio.v2 as imageio

    m = sim.model
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.diffuse[:] = 0.75
    r = mujoco.Renderer(m, 400, 400)
    T = hook["T"]
    view = T[:3, 0] if T[2, 0] < 0.5 else -T[:3, 0]       # the side of the fingers, not from below
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = T[:3, 3] - 0.02 * T[:3, 2] + [0, 0, 0.05]
    cam.distance = 0.42
    cam.azimuth = float(np.degrees(np.arctan2(view[1], view[0])))
    cam.elevation = -12.0
    frames = []
    lift_test(sim, T, hook["width"], close=False, frames=frames, renderer=r, camera=cam)
    bg = np.array([238, 240, 243], np.uint8)
    frames = [np.where(f.sum(-1, keepdims=True) == 0, bg, f) for f in frames]
    imageio.mimsave(path, frames[::2], duration=0.1, loop=0)
    imageio.imwrite(path.with_suffix(".png"), np.hstack([frames[1], frames[-1]]))
    r.close()


def main() -> int:
    oset = load_object_set()
    names = sys.argv[1:] or oset.get("handle_candidates", [])
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(max(1, min(len(names), (os.cpu_count() or 2) - 1))) as pool:
        rows = pool.map(run, names)
    for r in rows:
        print(f"{r['object'][:34]:34s} {r['mass_kg']:.2f} kg  hooks {r['hooks']} ({''.join(k[0] for k in r['hook_kinds'])})"
              f"  pinches {r['pinches']}  held at x{SCALES}: hook {r['held']['hook']}  pinch {r['held']['pinch']}")
    (OUT / "summary.json").write_text(json.dumps({"scales": SCALES, "objects": rows}, indent=1))
    plot(rows, ROOT / "media" / "hook_strength.png")
    return 0


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["hooks"] and r["pinches"]]
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(3.4 * len(rows), 3.2), squeeze=False)
    for ax, r in zip(axes[0], rows):
        kg = np.array(SCALES) * r["mass_kg"]
        ax.plot(kg, np.array(r["held"]["hook"]) / r["hooks"], "o-", c="#1d4e6e", label=f"hook ({r['hooks']})")
        ax.plot(kg, np.array(r["held"]["pinch"]) / r["pinches"], "o-", c="#e07a2f", label=f"pinch ({r['pinches']})")
        ax.axvline(3.0, ls=":", c="0.5")
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        ax.minorticks_off()
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("object mass (kg); dotted: Franka payload")
        ax.set_title(r["object"].replace("_", " ")[:30], fontsize=8)
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("share of grasps that still lift it")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
