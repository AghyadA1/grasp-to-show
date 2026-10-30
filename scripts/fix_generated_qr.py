"""Move generated objects' QR stickers out from under their handles (part 3).

The sticker spot was chosen on the body alone, with rays cast from 3 cm out: a handle or a bail
further out could hide it (half the time on some kinds; the camera never reads it, whether the
object is pinched or hooked). Now a spot must be clear of the whole object within 25 deg of its
normal (qr_placement.qr_clear). This rewrites props.json "qr" for every generated object whose
sticker is hidden (the meshes, hooks, grasps and clouds do not change).

Usage:  python scripts/fix_generated_qr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp2show.objects import GENERATED_DIR
from grasp2show.qr_placement import QRPlacement, find_qr_patch, qr_clear
from grasp2show.shapegen import generate


def main() -> int:
    names = json.loads((ROOT / "grasp2show" / "object_set_gen.json").read_text())["objects"]
    moved = kept = hidden = 0
    for i, n in enumerate(names, 1):
        d = GENERATED_DIR / n
        if not (d / "props.json").exists():
            continue
        props = json.loads((d / "props.json").read_text())
        mesh = trimesh.load(d / "model.obj", force="mesh", process=False)
        qr = QRPlacement.from_dict(props["qr"])
        if qr_clear(mesh, qr.T_obj_qr, qr.side) >= 0.95:
            kept += 1
            continue
        g = generate(int(n.split("_")[1]))
        body = trimesh.util.concatenate(g.body)
        new = None
        for k in (60, 200, 600):
            new = find_qr_patch(body, n_candidates=k, seed=g.params["seed"], occluders=g.mesh)
            if new is not None:
                break
        if new is None:
            props["qr_hidden"] = True                       # no clear spot: left as it was, flagged
            hidden += 1
        else:
            props["qr"] = new.to_dict()
            props.pop("qr_hidden", None)
            moved += 1
        (d / "props.json").write_text(json.dumps(props, indent=1))
        if i % 100 == 0:
            print(f"  {i}/{len(names)}", flush=True)
    print(f"{kept} stickers were clear, {moved} moved, {hidden} objects have no clear spot (flagged qr_hidden)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
