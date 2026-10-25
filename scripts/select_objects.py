"""Select the project's object set from all 1030 Google Scanned Objects (run once).

Criteria (written down so the selection is reproducible and not cherry-picked):
  - graspable by the Franka Hand: smallest OBB side in [2, 7] cm (hand opens to 8 cm);
  - portable: largest OBB side <= 25 cm;
  - box-like package: convex-hull volume / OBB volume >= 0.85 (labels go on flat faces);
  - a QR patch of at least 3 cm fits on a flat, non-bottom face (qr_placement.find_qr_face).
The passing objects are split 75/25 into train/test (seeded) for the unseen-object test.
Writes grasp2show/object_set.json, including the QR placements, so users download only the
selected objects (scripts/get_objects.py).

Needs all meshes locally:  SCANNED_OBJECTS_PATH=<dir containing models/>  (model.obj files).
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp2show.objects import OBJECT_SET_FILE, scanned_objects_dir
from grasp2show.qr_placement import find_qr_face

CRITERIA = {"min_side_range_m": [0.02, 0.07], "max_side_max_m": 0.25, "boxiness_min": 0.85,
            "qr_min_side_m": 0.03, "qr_flat_tol_m": 0.002, "test_fraction": 0.25, "max_objects": 40,
            "seed": 0}


def evaluate(d: Path):
    try:
        mesh = trimesh.load(d / "model.obj", force="mesh", process=True)
        ext = np.sort(mesh.bounding_box_oriented.primitive.extents)
        lo, hi = CRITERIA["min_side_range_m"]
        if not (lo <= ext[0] <= hi and ext[2] <= CRITERIA["max_side_max_m"]):
            return None
        box = mesh.convex_hull.volume / np.prod(ext)
        if box < CRITERIA["boxiness_min"]:
            return None
        qr = find_qr_face(mesh, min_side=CRITERIA["qr_min_side_m"], flat_tol=CRITERIA["qr_flat_tol_m"])
        if qr is None:
            return None
        return d.name, {"extents_cm": np.round(ext * 100, 1).tolist(), "boxiness": round(float(box), 3)}, qr
    except (ValueError, IndexError) as e:  # a few scans are malformed; skip them
        print(f"skip {d.name}: {e}")
        return None


def main():
    dirs = sorted(p for p in scanned_objects_dir().iterdir() if (p / "model.obj").exists())
    print(f"evaluating {len(dirs)} objects")
    with Pool() as pool:
        results = [r for r in pool.map(evaluate, dirs, chunksize=8) if r is not None]
    print(f"{len(results)} pass the criteria")
    rng = np.random.default_rng(CRITERIA["seed"])
    idx = rng.permutation(len(results))[: CRITERIA["max_objects"]]
    chosen = sorted((results[i] for i in idx), key=lambda r: r[0])
    names = [r[0] for r in chosen]
    n_test = round(len(names) * CRITERIA["test_fraction"])
    test = sorted(rng.choice(names, n_test, replace=False).tolist())
    out = {
        "source": "https://github.com/kevinzakka/mujoco_scanned_objects (Google Scanned Objects, CC-BY 4.0)",
        "criteria": CRITERIA,
        "n_candidates": len(dirs), "n_passing": len(results),
        "objects": names,
        "train": [n for n in names if n not in test], "test": test,
        "stats": {r[0]: r[1] for r in chosen},
        "qr": {r[0]: r[2].to_dict() for r in chosen},
    }
    OBJECT_SET_FILE.write_text(json.dumps(out, indent=1))
    print(f"wrote {OBJECT_SET_FILE}: {len(names)} objects ({len(names) - n_test} train / {n_test} test)")


if __name__ == "__main__":
    main()
