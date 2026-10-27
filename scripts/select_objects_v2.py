"""Object set v2: many shapes, not only boxes (run once; part 3 of the project).

From all 1030 Google Scanned Objects, written-down criteria (reproducible, not cherry-picked):
  - graspable somewhere by the Franka Hand: at least 10 antipodal contact pairs of width
    0.8-7.5 cm among 400 tries (grasp_sampler.sample_antipodal), so a mug qualifies by its
    handle or rim even though it is 9 cm across;
  - portable: largest bounding-box side <= 30 cm, smallest >= 1 cm;
  - an ASCII name (MuJoCo on Windows cannot open paths with other letters, e.g. Pokémon_...);
  - a QR sticker of at least 3 cm fits: on a flat face (qr_placement.find_qr_face, 2 mm), or
    on a gently curved spot (qr_placement.find_qr_patch, 4 mm: a mug's side);
then a stratified draw so shapes are mixed: "box" (convex hull / bounding box >= 0.85),
"rounded" (0.6-0.85: bottles, cans, jars), "irregular" (< 0.6: mugs, toys, shoes, tools).
Objects whose name says they have a handle (mug, cup, pitcher, kettle, frother, basket, pan) are
always kept when they pass: they are the candidates for hanging grasps.
The 40 objects of set v1 keep their train/test split; new objects are split 75/25 per class.
Whether each object can really be picked is decided afterwards in simulation
(scripts/generate_grasps.py --object-set v2 --prune).

Needs all meshes: SCANNED_OBJECTS_PATH=<dir containing models/<name>/model.obj>.
Writes grasp2show/object_set_v2.json.
"""
from __future__ import annotations

import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp2show.grasp_sampler import sample_antipodal
from grasp2show.objects import DEFAULT_DENSITY, OBJECT_SET_FILE, scanned_objects_dir
from grasp2show.qr_placement import find_qr_face, find_qr_patch

OUT = OBJECT_SET_FILE.with_name("object_set_v2.json")
CRITERIA = {"antipodal_pairs_min": 10, "antipodal_tries": 400, "max_side_max_m": 0.30, "min_side_min_m": 0.01,
            "qr_min_side_m": 0.03, "qr_flat_tol_face_m": 0.002, "qr_flat_tol_curved_m": 0.004,
            "classes": {"box": [0.85, 9], "rounded": [0.6, 0.85], "irregular": [0, 0.6]},
            "new_per_class": {"box": 25, "rounded": 50, "irregular": 50}, "test_fraction": 0.25, "seed": 0}
HANDLE = re.compile(r"mug|(?<!gold_)cup(?!cake|sole)|pitcher|kettle|frother|basket(?!_weave)|(?<!cake)(?<!bundt)_pan"
                    r"|teapot|bucket|jug", re.IGNORECASE)


def evaluate(d: Path):
    if not d.name.isascii():
        return None
    try:
        mesh = trimesh.load(d / "model.obj", force="mesh", process=True)
        ext = np.sort(mesh.bounding_box_oriented.primitive.extents)
        if ext[2] > CRITERIA["max_side_max_m"] or ext[0] < CRITERIA["min_side_min_m"]:
            return None
        p1, _ = sample_antipodal(mesh, 1000, np.random.default_rng(0), oversample=1)
        n_pairs = len(p1[: CRITERIA["antipodal_tries"]])
        if n_pairs < CRITERIA["antipodal_pairs_min"]:
            return None
        box = float(mesh.convex_hull.volume / np.prod(ext))
        qr = find_qr_face(mesh, min_side=CRITERIA["qr_min_side_m"], flat_tol=CRITERIA["qr_flat_tol_face_m"])
        kind = "face"
        if qr is None:
            qr = find_qr_patch(mesh, min_side=CRITERIA["qr_min_side_m"], flat_tol=CRITERIA["qr_flat_tol_curved_m"])
            kind = "curved"
        if qr is None:
            return None
        cls = next(c for c, (lo, hi) in CRITERIA["classes"].items() if lo <= box < hi)
        mass = float(np.clip(mesh.convex_hull.volume * DEFAULT_DENSITY, 0.05, 1.5))
        return d.name, {"extents_cm": np.round(ext * 100, 1).tolist(), "boxiness": round(box, 3), "class": cls,
                        "antipodal_pairs": n_pairs, "qr_kind": kind, "mass_kg_est": round(mass, 3),
                        "handle_name": bool(HANDLE.search(d.name))}, qr
    except (ValueError, IndexError, AttributeError) as e:
        print(f"skip {d.name}: {e}")
        return None


def main():
    v1 = json.loads(OBJECT_SET_FILE.read_text())
    dirs = sorted(p for p in scanned_objects_dir().iterdir() if (p / "model.obj").exists())
    print(f"evaluating {len(dirs)} objects", flush=True)
    results = []
    with Pool() as pool:
        for k, r in enumerate(pool.imap(evaluate, dirs, chunksize=4), 1):
            if r is not None:
                results.append(r)
            if k % 50 == 0:
                print(f"  {k}/{len(dirs)} evaluated, {len(results)} pass", flush=True)
    passing = {r[0]: r for r in results}
    print(f"{len(passing)} pass the criteria", flush=True)
    rng = np.random.default_rng(CRITERIA["seed"])
    chosen = {n: passing[n] for n in v1["objects"] if n in passing}
    for n in v1["objects"]:                          # keep v1 even if the new test is stricter
        if n not in chosen:
            chosen[n] = (n, v1["stats"][n] | {"class": "box"}, None)
    new_split = {}
    for cls, k in CRITERIA["new_per_class"].items():
        pool_ = sorted(n for n, r in passing.items() if r[1]["class"] == cls and n not in chosen)
        handles = [n for n in pool_ if passing[n][1]["handle_name"]]
        rest = [n for n in pool_ if n not in handles]
        pick = handles + [rest[i] for i in rng.permutation(len(rest))[: max(k - len(handles), 0)]]
        n_test = round(len(pick) * CRITERIA["test_fraction"])
        test = set(rng.choice(pick, n_test, replace=False).tolist()) if pick else set()
        for n in pick:
            chosen[n] = passing[n]
            new_split[n] = "test" if n in test else "train"
    names = sorted(chosen)
    train = sorted([n for n in v1["train"]] + [n for n, s in new_split.items() if s == "train"])
    test = sorted([n for n in v1["test"]] + [n for n, s in new_split.items() if s == "test"])
    stats = {n: chosen[n][1] for n in names}
    qr = {n: (chosen[n][2].to_dict() if chosen[n][2] is not None else v1["qr"][n]) for n in names}
    for n in v1["objects"]:
        qr[n] = v1["qr"][n]                          # identical placements for the v1 objects
    out = {"source": v1["source"], "criteria": CRITERIA, "n_candidates": len(dirs), "n_passing": len(passing),
           "objects": names, "train": train, "test": test, "stats": stats, "qr": qr,
           "classes": {c: sorted(n for n in names if stats[n].get("class") == c) for c in CRITERIA["classes"]},
           "handle_candidates": sorted(n for n in names if stats[n].get("handle_name"))}
    OUT.write_text(json.dumps(out, indent=1))
    cnt = {c: len(v) for c, v in out["classes"].items()}
    print(f"wrote {OUT}: {len(names)} objects ({len(train)} train / {len(test)} test), classes {cnt}, "
          f"{len(out['handle_candidates'])} with a handle in the name")


if __name__ == "__main__":
    main()
