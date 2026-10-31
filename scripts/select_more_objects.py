"""More real objects for way B's training (way B v5): every Google Scanned Object that passes the
criteria of set v2 (scripts/archive/part3_early/select_objects_v2.py: graspable by the Franka
Hand, portable, an ASCII name, room for a 3 cm QR sticker) and is in none of the sets used so far
(v2, the real handle objects, v2's objects pruned as not pickable).

Split 85 / 15 per shape class (train / test); test objects are never trained on. The evaluation
(scripts/evaluate.py) keeps its own objects, none of which are here. QR placements are cached in
the file (objects.object_from_config looks here for a scan outside the active set).

Needs all meshes: SCANNED_OBJECTS_PATH=<dir containing models/<name>/model.obj>.
Usage:  python scripts/select_more_objects.py        -> grasp2show/object_set_more.json
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts" / "archive" / "part3_early"))
from select_objects_v2 import CRITERIA, evaluate

from grasp2show.objects import scanned_objects_dir

OUT = ROOT / "grasp2show" / "object_set_more.json"
TEST_FRACTION = 0.15


def main() -> int:
    v2 = json.loads((ROOT / "grasp2show" / "object_set_v2.json").read_text())
    hr = json.loads((ROOT / "grasp2show" / "object_set_hookreal.json").read_text())
    used = set(v2["objects"]) | set(v2["pruned_not_pickable"]) | set(hr["objects"])
    dirs = sorted(p for p in scanned_objects_dir().iterdir() if (p / "model.obj").exists() and p.name not in used)
    print(f"evaluating {len(dirs)} objects ({len(used)} already used are left out)", flush=True)
    results = []
    with Pool(max(1, (os.cpu_count() or 2))) as pool:
        for k, r in enumerate(pool.imap(evaluate, dirs, chunksize=4), 1):
            if r is not None:
                results.append(r)
            if k % 50 == 0:
                print(f"  {k}/{len(dirs)} evaluated, {len(results)} pass", flush=True)
    passing = {r[0]: r for r in results}
    rng = np.random.default_rng(1)
    train, test = [], []
    for cls in CRITERIA["classes"]:
        names = sorted(n for n, r in passing.items() if r[1]["class"] == cls)
        t = set(rng.choice(names, round(len(names) * TEST_FRACTION), replace=False).tolist()) if names else set()
        train += [n for n in names if n not in t]
        test += sorted(t)
    names = sorted(passing)
    out = {"source": v2["source"], "use": "way B v5 training (scripts/generate_wayb_labels.py)",
           "criteria": CRITERIA, "n_candidates": len(dirs), "objects": names, "train": sorted(train),
           "test": sorted(test), "stats": {n: passing[n][1] for n in names},
           "qr": {n: passing[n][2].to_dict() for n in names},
           "classes": {c: sorted(n for n in names if passing[n][1]["class"] == c) for c in CRITERIA["classes"]}}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}: {len(names)} objects ({len(train)} train / {len(test)} test), "
          + ", ".join(f"{c} {len(v)}" for c, v in out["classes"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
