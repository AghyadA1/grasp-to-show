"""Generated objects with handles (grasp2show/shapegen.py)."""
import json

import numpy as np

from grasp2show.shapegen import generate, mass_properties, write


def test_generated_objects_are_convex_pieces_standing_on_the_table(tmp_path):
    kept, kinds = 0, set()
    for seed in range(12):
        g = generate(seed)
        if g is None:
            continue
        assert g.name == f"gen_{seed:05d}"
        for p in g.pieces:                                                 # exact collision geometry
            assert p.is_watertight and np.isclose(p.volume, p.convex_hull.volume, rtol=1e-6)
        assert g.mesh.bounds[0, 2] >= -1e-6                                 # nothing below the base
        props = write(g, tmp_path, density=600.0)
        if props is None:
            continue
        kept += 1
        kinds.add(props["params"]["handle"]["kind"])
        d = tmp_path / g.name
        files = sorted(d.glob("model_collision_*.obj"))
        assert len(files) == len(g.pieces) and (d / "model.obj").exists()
        assert json.loads((d / "props.json").read_text())["qr"]["side"] >= 0.03
        assert 0.05 - 1e-9 <= props["mass"] <= 1.5 + 1e-9
        I = np.asarray(props["inertia"])
        assert np.allclose(I, I.T) and np.all(np.linalg.eigvalsh(I) > 0)
    assert kept >= 6 and len(kinds) >= 3


def test_mass_properties_of_one_box_match_the_formula():
    import trimesh

    from grasp2show.shapegen import Generated

    box = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
    m, com, I = mass_properties(Generated("b", [box], []), density=1000.0)
    assert np.isclose(m, 6.0)
    assert np.allclose(com, 0.0)
    assert np.allclose(np.diag(I), m / 12 * np.array([0.2**2 + 0.3**2, 0.1**2 + 0.3**2, 0.1**2 + 0.2**2]))
