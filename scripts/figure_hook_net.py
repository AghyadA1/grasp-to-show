"""Pictures of what the self-trained hook network chooses (part 3).

For each object: the network looks at the object's point cloud (no mesh, no rules), proposes its
best points (both approach directions), and each proposal is nudged clear and lift-tested in the
floating-gripper simulator. Each tile shows the first proposal that lifts: left, the hand placed
at it with the object on the table (dots: the network's top points, yellow = most probable);
right, the object hanging after the lift. Red title: none of the proposals lifted (the best one
is shown).

Usage:  python scripts/figure_hook_net.py                     (held-out objects, a few of each kind)
        python scripts/figure_hook_net.py --objects A B C --out media/x.png
Outputs: media/hook_net_choices_real.png, media/hook_net_choices_generated.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("G2S_OBJECT_SET", "v2")

REAL = ["ACE_Coffee_Mug_Kristen_16_oz_cup", "Threshold_Porcelain_Teapot_White", "Threshold_Porcelain_Pitcher_White",
        "RJ_Rabbit_Easter_Basket_Blue", "Room_Essentials_Mug_White_Yellow", "Diamond_Visions_Scissors_Red",
        "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White", "Spritz_Easter_Basket_Plastic_Teal",
        "Sterilite_Caddy_Blue_Sky_17_58_x_12_58_x_9_14", "Cole_Hardware_School_Bell_Solid_Brass_38",
        "Aroma_Stainless_Steel_Milk_Frother_2_Cup", "Target_Basket_Medium", "Ocedar_Snap_On_Dust_Pan_And_Brush_1_ct",
        "STACKING_RING", "Embark_Lunch_Cooler_Blue", "Vtech_Stack_Sing_Rings_636_Months"]


def generated_pick(per_kind: int = 2, seed: int = 0) -> list[str]:
    """A few held-out generated objects of every kind of handle (including the new families)."""
    g = json.loads((ROOT / "grasp2show" / "object_set_gen.json").read_text())
    test = set(g["test"])
    kind = lambda o: ("container-" if o["body"].get("style") == "container" else "") + o["handle"]["where"]
    by = defaultdict(list)
    for o in g["details"]:
        if o["name"] in test:
            by[kind(o)].append(o["name"])
    rng = np.random.default_rng(seed)
    out = []
    for k in sorted(by):
        out += list(rng.choice(by[k], min(per_kind, len(by[k])), replace=False))
    return out


def ensure_generated(name: str) -> None:
    from grasp2show.objects import GENERATED_DIR
    from grasp2show.shapegen import generate, write

    if (GENERATED_DIR / name / "model.xml").exists():
        return
    seed = int(name.split("_")[1])
    g = generate(seed)
    write(g, GENERATED_DIR, float(np.random.default_rng(seed + 10_000).uniform(150, 1200)))


def cloud_of(name: str) -> np.ndarray:
    import trimesh

    from grasp2show.learn.hooknet import N_POINTS, sample_cloud
    from grasp2show.objects import asset_dir

    d = asset_dir(name)
    mesh = trimesh.load(d / "model.obj", force="mesh", process=True)
    pieces = ([trimesh.load(p, force="mesh") for p in sorted(d.glob("model_collision_*.obj"))]
              if name.startswith("gen_") else None)
    return sample_cloud(mesh, N_POINTS, seed=1, pieces=pieces)


def tile(name: str, net, S: int = 300) -> np.ndarray:
    import cv2
    import mujoco

    from grasp2show.config import ObjectConfig, SceneConfig
    from grasp2show.gripper_sim import GripperSim
    from grasp2show.hanging import nudge
    from grasp2show.hooks import lift_test
    from grasp2show.learn.hooknet import predict_hooks
    from grasp2show.objects import object_from_config

    obj = object_from_config(ObjectConfig(asset=name))
    sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj)
    cloud = cloud_of(name)
    preds = predict_hooks(net, cloud, k=6, min_prob=0.0)
    chosen, lifted = None, False
    for h in preds:
        T = nudge(sim, h["T"], h["width"], step=0.003, reach=3, depths=(-0.006, 0.0, 0.006))
        if T is None:
            continue
        if chosen is None:
            chosen = (T, h)
        if lift_test(sim, T, h["width"], close=False)["success"]:
            chosen, lifted = (T, h), True
            break
    m, d = sim.model, sim.data
    m.vis.headlight.ambient[:] = 0.5
    hand = {m.body(b).id for b in ("g_hand", "g_left_finger", "g_right_finger")}
    for g in range(m.ngeom):                                   # see-through hand: the finger in the handle shows
        if m.geom_bodyid[g] in hand:
            m.geom_matid[g] = -1
            m.geom_rgba[g] = (0.85, 0.87, 0.92, 0.35)
    r = mujoco.Renderer(m, S, S)
    c = obj.mesh.bounds.mean(0)
    ext = float(obj.mesh.extents.max())

    def camera(T, up=0.0):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (0.5 * (c + T[:3, 3]) if T is not None else c) + [0, 0, up]
        cam.distance = 0.38 + 1.5 * ext
        a = T[:3, 2] if T is not None else np.array([1.0, 0, 0])
        cam.azimuth = float(np.degrees(np.arctan2(a[1], a[0]))) + 70.0       # from the side of the approach
        cam.elevation = -20.0
        return cam

    def dots(scn, alpha=1.0):
        anchors = np.array([h["anchor"] for h in preds[::2]])
        probs = np.array([h["prob"] for h in preds[::2]])
        for a, p in zip(anchors, probs):
            if scn.ngeom >= scn.maxgeom:
                break
            q = p / max(probs.max(), 1e-6)
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.005, 0, 0]), a,
                                np.eye(3).ravel(), np.array([0.9, 0.2 + 0.7 * q, 0.1, alpha], np.float32))
            scn.ngeom += 1

    def floor(scn, z=0.0):
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([0.3, 0.3, 0.0008]),
                            np.array([c[0], c[1], z - 0.0008]), np.eye(3).ravel(),
                            np.array([0.55, 0.55, 0.55, 0.5], np.float32))
        scn.ngeom += 1

    T = chosen[0] if chosen else None
    if T is not None:
        sim._reset(np.eye(3), T, chosen[1]["width"])
    else:
        sim._reset(np.eye(3), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2.0], [0, 0, 0, 1.0]]), 0.08)
    cam = camera(T)
    r.update_scene(d, camera=cam)
    floor(r.scene)
    dots(r.scene)
    left = r.render().copy()
    if T is not None:
        frames = []
        lift_test(sim, T, chosen[1]["width"], close=False, frames=frames, renderer=r, camera=camera(T, up=0.05))
        right = frames[-1] if frames else np.full_like(left, 255)
    else:
        right = np.full_like(left, 255)
    r.close()
    img = np.concatenate([left, right], 1)
    cv2.rectangle(img, (0, 0), (2 * S, 40), (245, 245, 245), -1)
    cv2.putText(img, name[:52], (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    if chosen:
        h = chosen[1]
        txt = (f"{'lifts' if lifted else 'no proposal lifts'}: {'one finger' if h['kind'] == 'one' else 'both fingers'}"
               f" through, p = {h['prob']:.2f}")
    else:
        txt = "no proposal could be placed"
    cv2.putText(img, txt, (6, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 140, 40) if lifted else (200, 50, 50), 1,
                cv2.LINE_AA)
    cv2.line(img, (S, 40), (S, S), (255, 255, 255), 2)
    return img, lifted


def grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    return np.concatenate([np.pad(r, ((0, 6), (0, 0), (0, 0)), constant_values=255) for r in rows], 0)


def main() -> int:
    import cv2

    from grasp2show.learn.hooknet import load_hook_net

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "hook_net_self.pt")
    p.add_argument("--objects", nargs="*", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cols", type=int, default=3)
    args = p.parse_args()
    net = load_hook_net(args.model)
    sets = ({"custom": args.objects} if args.objects else
            {"real": REAL, "generated": generated_pick()})
    for key, names in sets.items():
        tiles, n_lift = [], 0
        for n in names:
            if n.startswith("gen_"):
                ensure_generated(n)
            try:
                img, ok = tile(n, net)
            except Exception as e:  # noqa: BLE001 (a scan that does not load: skip it, say why)
                print(f"{n}: skipped ({type(e).__name__}: {e})", flush=True)
                continue
            tiles.append(img)
            n_lift += ok
            print(f"{n}: {'lifts' if ok else 'no lift'}", flush=True)
        out = args.out or ROOT / "media" / f"hook_net_choices_{key}.png"
        cv2.imwrite(str(out), cv2.cvtColor(grid(tiles, args.cols), cv2.COLOR_RGB2BGR))
        print(f"wrote {out}: {n_lift}/{len(tiles)} objects lifted by the network's choice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
