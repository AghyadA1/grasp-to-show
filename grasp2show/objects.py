"""Objects: the primitive QR box and Google Scanned Objects (GSO) meshes.

Every object exposes the same interface to the rest of the code:
  - a trimesh mesh in the object frame (for grasp sampling and ray casting),
  - mass properties (mass, centre of mass, inertia),
  - the QR patch pose T_obj_qr (x axis = outward normal of the patch) and side length,
  - rest_z: height of the object origin above the table when it rests in its start pose,
  - add_to_body(): adds visual + collision geoms to a MuJoCo body.

Scanned objects come from https://github.com/kevinzakka/mujoco_scanned_objects (assets
CC-BY 4.0, Google Research), which already provides a convex decomposition for collision.
Their meshes are scanned surfaces, not closed solids, so volume, centre of mass and inertia
are computed from the convex hull at uniform density. That is accurate for box-like
packages, which is the object set selected for this project (see scripts/select_objects.py).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from .qr_placement import QRPlacement, find_qr_face, find_qr_patch
from .transforms import make_T, mat_to_quat

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECT_SET_FILE = Path(__file__).resolve().parent / "object_set.json"
DEFAULT_DENSITY = 300.0  # kg/m^3: a packaged product is mostly cardboard and air


def scanned_objects_dir() -> Path:
    root = Path(os.environ.get("SCANNED_OBJECTS_PATH",
                               REPO_ROOT / "third_party" / "mujoco_scanned_objects"))
    return root / "models"


GENERATED_DIR = REPO_ROOT / "data" / "generated_objects" / "models"   # grasp2show/shapegen.py


def asset_dir(asset: str) -> Path:
    """Where an object's files are: a generated object (name starting with "gen_") or a scan."""
    return (GENERATED_DIR if asset.startswith("gen_") else scanned_objects_dir()) / asset


def object_set_file() -> Path:
    """The active object set: v1 (40 box-like packages, the default) or, with G2S_OBJECT_SET=v2,
    the set of many shapes (scripts/select_objects_v2.py)."""
    v = os.environ.get("G2S_OBJECT_SET", "v1")
    return OBJECT_SET_FILE if v == "v1" else OBJECT_SET_FILE.with_name(f"object_set_{v}.json")


def load_object_set() -> dict:
    """Selected object names, train/test split and cached QR placements."""
    return json.loads(object_set_file().read_text())


@dataclass
class ObjectModel:
    name: str
    mesh: trimesh.Trimesh          # object frame, metres
    mass: float
    com: np.ndarray
    inertia: np.ndarray            # 3x3 about the COM, object-frame axes
    qr: QRPlacement
    rest_z: float
    friction: tuple[float, float, float]
    condim: int
    rgba: tuple[float, float, float, float] = (0.85, 0.55, 0.25, 1.0)
    half_size: tuple[float, float, float] | None = None   # primitive box only
    asset_dir: Path | None = None                          # scanned objects only
    collision_files: list[Path] = field(default_factory=list)

    @property
    def T_obj_qr(self) -> np.ndarray:
        return self.qr.T_obj_qr

    # -- MuJoCo -------------------------------------------------------------------------
    def add_to_body(self, spec: mujoco.MjSpec, body: mujoco.MjsBody, qr_material: str | None,
                    solref: float | None = None) -> None:
        """Add inertia, visual and collision geoms, and (if qr_material) the QR plate to `body`.
        solref: contact time constant for the collision geoms (None: MuJoCo default)."""
        body.explicitinertial = True
        body.mass = self.mass
        body.ipos = self.com
        I = self.inertia
        body.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
        n = self.name
        if self.half_size is not None:
            body.add_geom(name=f"{n}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(self.half_size),
                          friction=list(self.friction), condim=self.condim, rgba=list(self.rgba))
        else:
            tex = self.asset_dir / "texture.png"
            mat = None
            if tex.exists():
                spec.add_texture(name=f"{n}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(tex))
                spec.add_material(name=f"{n}_mat", textures=["", f"{n}_tex"])
                mat = f"{n}_mat"
            spec.add_mesh(name=f"{n}_visual", file=str(self.asset_dir / "model.obj"))
            g = body.add_geom(name=f"{n}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{n}_visual",
                              contype=0, conaffinity=0, group=2)
            if mat:
                g.material = mat
            else:                                   # generated objects: plain colour
                g.rgba = list(self.rgba)
            for i, f in enumerate(self.collision_files):
                spec.add_mesh(name=f"{n}_col{i}", file=str(f))
                body.add_geom(name=f"{n}_col{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{n}_col{i}",
                              friction=list(self.friction), condim=self.condim, group=3,
                              rgba=[0.5, 0.5, 0.5, 1])
        if solref is not None:
            for g in body.geoms:
                if g.contype or g.conaffinity:
                    g.solref = [solref, 1.0]
        if qr_material is None:
            return
        # QR: visual-only thin plate, own geom group (1) so visibility rays can target it.
        q = self.qr
        p = q.T_obj_qr[:3, 3] + q.T_obj_qr[:3, 0] * 0.0005
        body.add_geom(name=f"{n}_qr", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0005, q.side / 2, q.side / 2],
                      pos=p, quat=mat_to_quat(q.T_obj_qr[:3, :3]), material=qr_material,
                      contype=0, conaffinity=0, group=1)
        body.add_site(name=f"{n}_qr_center", pos=q.T_obj_qr[:3, 3] + q.T_obj_qr[:3, 0] * 0.001,
                      size=[0.003, 0, 0], group=4)


def make_box(name: str, half_size, mass: float, friction, condim: int, qr_side: float,
             qr_center_z: float, rgba) -> ObjectModel:
    """The day 1-2 box: QR on the +x face at height qr_center_z."""
    hx, hy, hz = half_size
    mesh = trimesh.creation.box(extents=2 * np.asarray(half_size))
    I = mass / 12.0 * np.diag([(2 * hy) ** 2 + (2 * hz) ** 2, (2 * hx) ** 2 + (2 * hz) ** 2,
                               (2 * hx) ** 2 + (2 * hy) ** 2])
    qr = QRPlacement(make_T(np.eye(3), [hx, 0.0, qr_center_z]), qr_side, 0.0)
    return ObjectModel(name, mesh, mass, np.zeros(3), I, qr, hz, tuple(friction), condim, tuple(rgba),
                       half_size=tuple(half_size))


def make_scanned(asset: str, friction, condim: int, density: float = DEFAULT_DENSITY,
                 name: str = "obj", qr: QRPlacement | None = None) -> ObjectModel:
    d = asset_dir(asset)
    if not (d / "model.obj").exists():
        raise FileNotFoundError(f"{d} missing. Run: python scripts/get_objects.py "
                                "(generated objects: python scripts/generate_hookable.py)")
    mesh = trimesh.load(d / "model.obj", force="mesh", process=True)
    props = json.loads((d / "props.json").read_text()) if (d / "props.json").exists() else None
    if props is not None:                           # a generated object: exact solid, own colour and QR
        if qr is None and props.get("qr"):
            qr = QRPlacement.from_dict(props["qr"])
        xml = (d / "model.xml").read_text()
        cols = [d / f for f in sorted(set(re.findall(r'file="(model_collision_\d+\.obj)"', xml)),
                                      key=lambda s: int(re.findall(r"\d+", s)[0]))]
        if qr is None:
            raise ValueError(f"{asset}: no QR placement")
        return ObjectModel(name, mesh, float(props["mass"]), np.asarray(props["com"]), np.asarray(props["inertia"]),
                           qr, rest_z=float(-mesh.bounds[0, 2]), friction=tuple(friction), condim=condim,
                           rgba=tuple(props.get("rgba", (0.7, 0.7, 0.7, 1.0))), asset_dir=d, collision_files=cols)
    hull = mesh.convex_hull
    hull.density = density
    mass = float(np.clip(hull.mass, 0.05, 1.5))
    hull.density = density * mass / hull.mass       # keep inertia consistent with clipped mass
    xml = (d / "model.xml").read_text()
    cols = [d / f for f in sorted(set(re.findall(r'file="(model_collision_\d+\.obj)"', xml)),
                                  key=lambda s: int(re.findall(r"\d+", s)[0]))]
    if qr is None:
        qr = find_qr_face(mesh) or find_qr_patch(mesh, n_candidates=60)
        if qr is None:                  # no sticker spot (scissors, rings): fine for hook labels, the
            top = mesh.bounds[1, 2]     # QR is only used when showing; a nominal one on top, facing up
            c = mesh.bounds.mean(0)
            qr = QRPlacement(make_T(np.array([[0, 0, -1.0], [0, 1, 0], [1, 0, 0]]), [c[0], c[1], top]), 0.03, np.inf)
    return ObjectModel(name, mesh, mass, hull.center_mass, hull.moment_inertia, qr,
                       rest_z=float(-mesh.bounds[0, 2]), friction=tuple(friction), condim=condim,
                       asset_dir=d, collision_files=cols)


def object_from_config(ocfg) -> ObjectModel:
    """Build the ObjectModel described by config.ObjectConfig."""
    if ocfg.asset is None:
        return make_box(ocfg.name, ocfg.half_size, ocfg.mass, ocfg.friction, ocfg.condim, ocfg.qr_size,
                        ocfg.qr_center_z, ocfg.rgba)
    cached = load_object_set().get("qr", {}).get(ocfg.asset)
    qr = QRPlacement.from_dict(cached) if cached else None
    return make_scanned(ocfg.asset, ocfg.friction, ocfg.condim, ocfg.density, ocfg.name, qr)
