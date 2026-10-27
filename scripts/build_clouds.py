"""Point clouds of the active object set for the networks (1024 surface points, normals, QR flag,
mass, centre of mass), without running the dataset episodes of generate_dataset.py.

Usage:  G2S_OBJECT_SET=v2 python scripts/build_clouds.py      -> data/dataset_v2/clouds.npz
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from generate_dataset import object_cloud

from grasp2show.objects import load_object_set


def main() -> int:
    v = os.environ.get("G2S_OBJECT_SET", "v1")
    out = ROOT / "data" / ("dataset" if v == "v1" else f"dataset_{v}")
    out.mkdir(parents=True, exist_ok=True)
    names = load_object_set()["objects"]
    with Pool(max(1, min(4, (os.cpu_count() or 2) - 1))) as pool:
        clouds = dict(zip(names, pool.map(object_cloud, names)))
    np.savez_compressed(out / "clouds.npz", names=np.array(names),
                        **{f"{k}__{n}": v for n, c in clouds.items() for k, v in c.items()})
    print(f"{len(names)} clouds -> {out / 'clouds.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
