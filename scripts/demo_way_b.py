"""Video: way B picking and showing objects it has never seen (full simulation, side by side).

For each object: the heatmap network proposes grasps for all target poses, the exact reach checks
and the planner pick the first executable one (as in scripts/eval_way_b.py), and the episode is
filmed with the scanner view as an inset.
Usage:  python scripts/demo_way_b.py [--objects A B] [--yaw 90]
Writes media/way_b_demo.mp4 and media/way_b_demo.gif
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from demo_comparison import film
from demo_pick_and_show import write_comparison_gif
from eval_way_b import kinematic_choices, way_b_grasps

from grasp2show.experiment import Scenario
from grasp2show.objects import load_object_set


def clip(job):
    n, yaw, model, clouds, fps = job
    T, W, R, _ = way_b_grasps(n, yaw, model, clouds, 10)
    sc = Scenario(n, yaw, {"T_obj_tcp": T, "width": W, "success": np.ones(len(T), bool)})
    ch = kinematic_choices(sc, list(zip(range(len(T)), R.tolist())))
    return film(sc, "way_b", fps, ch, label=f"way B, unseen: {' '.join(n.split('_')[:3])}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", nargs=2, default=["Nestle_Skinny_Cow_Dreamy_Clusters_Candy_Dark_Chocolate_6_pack_1_oz_pouches",
                                                  "Office_Depot_Canon_PGI5BK_Remanufactured_Ink_Cartridge_Black"])
    p.add_argument("--yaw", type=float, default=90.0)
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()
    test = set(load_object_set()["test"])
    assert all(n in test for n in args.objects), "show unseen (test) objects only"
    # One fresh process per clip: filming a second scene in the same process gave a noise image
    # with the OSMesa backend (the scanner check in PickAndShow.execute now refuses such frames).
    with Pool(1, maxtasksperchild=1) as pool:
        clips = pool.map(clip, [(n, args.yaw, args.model, args.clouds, args.fps) for n in args.objects], chunksize=1)
    m = max(len(c) for c in clips)
    a, b = (c + [c[-1]] * (m - len(c)) for c in clips)
    both = [np.hstack([x, y]) for x, y in zip(a, b)]
    both = [cv2.resize(f, (f.shape[1] // 2 * 2, f.shape[0] // 2 * 2)) for f in both]
    imageio.mimwrite(ROOT / "media" / "way_b_demo.mp4", both, fps=args.fps, quality=7, macro_block_size=8)
    write_comparison_gif(a, b, args.fps, ROOT / "media" / "way_b_demo.gif")
    print("wrote media/way_b_demo.mp4 and .gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
