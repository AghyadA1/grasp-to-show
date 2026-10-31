"""The hook teacher (part 3): propose many kinds of hook candidates, let the simulator judge them.

The first hook labels came from the hole finder (hooks.find_loops): its geometric rules decided
which places were ever tried, so whatever they skip (a closed pocket with a lip, a J-shaped hook,
the underside of a rim, a knob's cap) was labelled "not hookable" without a test, and a network
trained on those labels inherits the rules' blind spots. Here the rules only rank:

  holes      the hole finder in ranking mode (strict=False): every point a ring of material
             surrounds, scored by ring closure, free line and "opens again"; the fingertip fit
             (hooks.fit_hook_poses) in the best few holes;
  overhang   a fingertip just under a downward-facing surface (a lip, a rim, a knob's cap, the
             bend of a J, the top bar of a handle) with the finger level or raised, pointing at
             the object, across it or along its surface: whatever can hang on a finger from
             above, hole or not;
  random     a fingertip anywhere near the surface, in any orientation that does not point the
             fingers down (exploration: the share of the budget no rule chooses);
  network    (self-training) the hook network's own proposals, see scripts/self_train_hooks.py.

Every candidate is nudged clear of the object (hanging.nudge) and lift-tested in the floating
gripper (hooks.lift_test). Every outcome is kept: lifted, tried and fell (a real negative), or
could not be placed without colliding (says nothing about the spot: not a negative).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .gripper_sim import GripperSim
from .hanging import max_penetration, nudge
from .hooks import find_loops, fit_hook_poses, lift_test

OPEN_W = 0.08
LIFTED, FELL, COLLIDES = 1, 0, -1
FLOAT_STORE = Path(__file__).resolve().parents[1] / "data" / "hook_teach"   # judged by the floating lift test
STORE = Path(os.environ.get("G2S_HOOK_TEACH", FLOAT_STORE))
# the judge of a hook: "lift" = the floating-gripper lift test (hooks.lift_test, object hung in the
# air); "table" = that, and the robot's lift rehearsed with the object standing on a table, fingers
# turned up 25 deg about the fingertip first, friction 0.5 (hanging.table_lift_test): the weak hooks
# the floating test passes (one finger in a narrow gap, beside a knob with no lip) fail it
JUDGE = os.environ.get("G2S_HOOK_JUDGE", "lift")
TABLE_PITCH, TABLE_MU = 25.0, 0.5


@dataclass
class Tested:
    T: np.ndarray          # (N, 4, 4) as placed (after the nudge when there was one)
    width: np.ndarray      # (N,) finger gap: 0 (both fingers through) or OPEN_W (one)
    anchor: np.ndarray     # (N, 3) the fingertip that hooks (hooknet.anchor)
    result: np.ndarray     # (N,) LIFTED, FELL or COLLIDES
    source: np.ndarray     # (N,) holes / overhang / random / net

    @staticmethod
    def empty() -> Tested:
        return Tested(np.zeros((0, 4, 4)), np.zeros(0), np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, "<U8"))

    def __len__(self) -> int:
        return len(self.result)

    def merged(self, other: Tested) -> Tested:
        return Tested(*(np.concatenate([a, b]) for a, b in zip(
            (self.T, self.width, self.anchor, self.result, self.source),
            (other.T, other.width, other.anchor, other.result, other.source))))


def anchor_of(T: np.ndarray, width: float) -> np.ndarray:
    return T[:3, 3] + T[:3, 1] * width / 2


def pose_from(anchor: np.ndarray, z: np.ndarray, y_hint: np.ndarray, width: float) -> np.ndarray | None:
    """T_obj_tcp with approach z, closing axis y (y_hint made orthogonal to z) and the hooking
    fingertip (TCP + y w/2) at `anchor`."""
    z = z / np.linalg.norm(z)
    y = y_hint - (y_hint @ z) * z
    if np.linalg.norm(y) < 1e-6:
        return None
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(y, z), y, z], 1)
    T[:3, 3] = anchor - y * width / 2
    return T


def _horizontal(v: np.ndarray) -> np.ndarray | None:
    h = np.array([v[0], v[1], 0.0])
    n = np.linalg.norm(h)
    return h / n if n > 1e-6 else None


def hole_candidates(obj, rng, n_holes: int = 4, turns: int = 6, limit: int = 40) -> list[dict]:
    loops = find_loops(obj.mesh, rng, strict=False) + find_loops(obj.mesh, rng, n_points=2000, strict=False,
                                                                  reach=0.13)          # big carrying handles
    if not loops:
        return []
    ranked = sorted(loops, key=lambda lp: -lp[3])
    fits = fit_hook_poses(obj, [(p, d, 100 * sc) for p, d, _, sc in ranked], turns=turns, max_holes=n_holes)
    fits.sort(key=lambda f: -f["clearance"])
    out = [{"T": f["T"], "width": f["width"], "source": "holes"} for f in fits[:limit - limit // 3]]
    return out + under_bar(obj, rng, ranked, limit // 3)


def under_bar(obj, rng, loops, n: int) -> list[dict]:
    """The fingertip just under the top of a hole (where the object will hang from), through it
    either way. The best fit in a big hole (a basket's bail) is its middle, deep in the basket,
    with the hand through the wall; the place to hook is under the bar."""
    mesh = obj.mesh
    centre = mesh.bounds.mean(0)
    out = []
    for i in rng.permutation(min(len(loops), 4 * n))[:n]:
        p, d = np.asarray(loops[i][0]), np.asarray(loops[i][1])
        u = np.array([0, 0, 1.0]) - d[2] * d
        if np.linalg.norm(u) < 0.3:                                     # horizontal hole: no bar above
            continue
        u /= np.linalg.norm(u)
        hit, _, _ = mesh.ray.intersects_location([p], [u], multiple_hits=False)
        if len(hit) == 0:
            continue
        h = float(np.linalg.norm(hit[0] - p))
        tip = p + u * max(h - rng.uniform(0.012, 0.02), 0.0)
        dirs = []
        for s in (1, -1):
            for yaw in (-20, 0, 20):
                c, sn = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
                z = s * d
                dirs.append(c * z + sn * np.cross(u, z))
        alts = _alternatives(rng, tip, dirs, centre)
        if alts:
            out.append({"T": alts[0][0], "width": alts[0][1], "alts": alts, "source": "holes"})
    return out


def _alternatives(rng, tip, dirs, centre, k: int = 12) -> list[tuple[np.ndarray, float]]:
    """Up to k poses with the hooking fingertip at `tip`: approach directions from `dirs` (pitched
    up 0-30 deg), several rolls, one finger or both."""
    out = []
    for _ in range(k):
        z = dirs[int(rng.integers(len(dirs)))]
        pitch = np.radians(rng.uniform(0, 30))
        z = np.cos(pitch) * z + np.sin(pitch) * np.array([0, 0, 1.0])
        width = OPEN_W if rng.random() < 0.6 else 0.0
        y_hint = rng.normal(size=3)
        if width > 0 and rng.random() < 0.7:                           # the other finger away from the object
            y_hint = -(centre - tip) + 0.3 * rng.normal(size=3)
        T = pose_from(tip, z, y_hint, width)
        if T is not None:
            out.append((T, width))
    return out


def overhang_candidates(obj, rng, n: int = 40) -> list[dict]:
    """Fingertips just under downward-facing surfaces (several orientations each: the hand is big,
    most orientations put the palm into the object; the first clear one is tested)."""
    mesh = obj.mesh
    pts, fid = trimesh.sample.sample_surface(mesh, 3000, seed=int(rng.integers(1 << 31)))
    nrm = mesh.face_normals[fid]
    down = np.flatnonzero((nrm[:, 2] < -0.3) & (pts[:, 2] > 0.012))   # not the base
    if len(down) == 0:
        return []
    centre = mesh.bounds.mean(0)
    out = []
    for i in rng.choice(down, size=min(n, len(down)), replace=False):
        p, nv = pts[i], nrm[i]
        tip = p + nv * rng.uniform(0.008, 0.016)                       # just below the surface
        to_c = _horizontal(centre - tip)
        tang = None if to_c is None else np.cross([0, 0, 1.0], to_c)
        dirs = [d for d in (to_c, -to_c if to_c is not None else None, tang,
                            -tang if tang is not None else None, _horizontal(rng.normal(size=3))) if d is not None]
        alts = _alternatives(rng, tip, dirs, centre)
        if alts:
            out.append({"T": alts[0][0], "width": alts[0][1], "alts": alts, "source": "overhang"})
    return out


def random_candidates(obj, rng, n: int = 20) -> list[dict]:
    mesh = obj.mesh
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
    out = []
    centre = mesh.bounds.mean(0)
    for p, nv in zip(pts, mesh.face_normals[fid]):
        tip = p + nv * rng.uniform(0.008, 0.03)
        dirs = []
        for _ in range(6):
            z = rng.normal(size=3)
            z[2] = abs(z[2]) * 0.5                                      # fingers not pointing down
            dirs.append(z / np.linalg.norm(z))
        alts = _alternatives(rng, tip, dirs, centre)
        if alts:
            out.append({"T": alts[0][0], "width": alts[0][1], "alts": alts, "source": "random"})
    return out


def test_candidates(sim: GripperSim, cands: list[dict], table_sim: GripperSim | None = None) -> Tested:
    """Nudge clear and lift-test every candidate; keep every outcome. table_sim: also rehearse the
    lift from the table (JUDGE "table"); a hook that fails it is a FELL."""
    if not cands:
        return Tested.empty()
    Ts, ws, anchors, res, src = [], [], [], [], []
    for c in cands:
        if "alts" in c:                                                 # the first orientation that is clear
            pens = []
            for Ta, wa in c["alts"]:
                pen = max_penetration(sim, Ta, wa)
                pens.append(pen)
                if pen == 0.0:
                    break
            k = int(np.argmin(pens))
            c = {**c, "T": c["alts"][k][0], "width": c["alts"][k][1]}
        T = (nudge(sim, c["T"], c["width"], step=0.003, reach=3, depths=(-0.006, 0.0, 0.006))
             if "alts" in c or c.get("wide") else nudge(sim, c["T"], c["width"]))   # a guess: shifts up to 9 mm
        if T is None:
            r, T = COLLIDES, c["T"]
        else:
            r = LIFTED if lift_test(sim, T, c["width"], close=False)["success"] else FELL
            if r == LIFTED and table_sim is not None and not table_ok(table_sim, T, c["width"]):
                r = FELL
        Ts.append(T)
        ws.append(c["width"])
        anchors.append(anchor_of(T, c["width"]))
        res.append(r)
        src.append(c["source"])
    return Tested(np.array(Ts), np.array(ws, float), np.array(anchors), np.array(res, int), np.array(src, "<U8"))


def table_ok(table_sim: GripperSim, T: np.ndarray, width: float) -> bool:
    from .hanging import table_lift_test

    return table_lift_test(table_sim, T, width, TABLE_PITCH, TABLE_MU)


def teach(obj, sim: GripperSim, seed: int = 0, n_holes: int = 3, budget=(40, 40, 20),
          table_sim: GripperSim | None = None) -> Tested:
    """All three proposers on one object, tested: budget = (holes, overhang, random); what the hole
    proposer does not use (no holes) goes to the other two."""
    rng = np.random.default_rng(seed)
    holes = hole_candidates(obj, rng, n_holes=n_holes, limit=budget[0])
    spare = budget[0] - len(holes)
    over = overhang_candidates(obj, rng, n=budget[1] + spare // 2)
    spare += budget[1] - len(over)
    rand = random_candidates(obj, rng, n=budget[2] + max(spare - spare // 2, 0))
    return test_candidates(sim, holes + over + rand, table_sim)


# ---- the store and the per-object jobs (run in worker processes by scripts/self_train_hooks.py)

def load_tested(name: str) -> Tested:
    f = STORE / f"{name}.npz"
    if not f.exists():
        return Tested.empty()
    z = np.load(f)
    return Tested(z["T"], z["width"], z["anchor"], z["result"], z["source"])


def save_tested(name: str, t: Tested) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    tmp = STORE / f"{name}.tmp.npz"
    np.savez_compressed(tmp, T=t.T, width=t.width, anchor=t.anchor, result=t.result, source=t.source)
    tmp.replace(STORE / f"{name}.npz")


def _load(name: str):
    from .config import ObjectConfig, SceneConfig
    from .objects import object_from_config

    obj = object_from_config(ObjectConfig(asset=name))
    return obj, GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj)


def _table_sim(name: str, obj) -> GripperSim | None:
    """The table sim when JUDGE is "table", else None."""
    if JUDGE != "table":
        return None
    from .config import ObjectConfig, SceneConfig

    return GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj, table=True)


def relabel_job(name: str) -> tuple[str, int, int, float, str]:
    """Worker: an object's floating-test tests (FLOAT_STORE) judged again by the table rehearsal:
    each hook that lifted keeps LIFTED only if its lift from the table holds; saved to STORE.
    -> name, lifted before, lifted now, seconds, error."""
    import time

    t0 = time.time()
    try:
        z = np.load(FLOAT_STORE / f"{name}.npz")
        t = Tested(z["T"], z["width"], z["anchor"], z["result"].copy(), z["source"])
        from .config import ObjectConfig, SceneConfig

        obj, _ = _load(name)
        tsim = GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj, table=True)
        before = int((t.result == LIFTED).sum())
        for i in np.flatnonzero(t.result == LIFTED):
            if not table_ok(tsim, t.T[i], float(t.width[i])):
                t.result[i] = FELL
    except Exception as e:  # noqa: BLE001 (a scan that does not load: skip it, say why)
        return name, 0, 0, time.time() - t0, f"{type(e).__name__}: {e}"
    save_tested(name, t)
    return name, before, int((t.result == LIFTED).sum()), time.time() - t0, ""


def library_hooks(name: str) -> Tested:
    """The analytic pipeline's hooks (hanging.hook_library), if cached: already lift-tested."""
    from .hanging import HOOK_CACHE

    f = HOOK_CACHE / f"{name}.npz"
    if not f.exists():
        return Tested.empty()
    z = np.load(f)
    T, w = z["T"], z["width"]
    if len(T) == 0:
        return Tested.empty()
    return Tested(T, w.astype(float), np.array([anchor_of(t, x) for t, x in zip(T, w)]),
                  np.full(len(T), LIFTED), np.full(len(T), "library", "<U8"))


def teach_job(args) -> tuple[str, int, int, float, str]:
    """Worker: the teacher on one object, stored (skipped if already stored). -> name, tested,
    lifted, seconds, error."""
    import time

    name, seed = args
    if (STORE / f"{name}.npz").exists():
        t = load_tested(name)
        return name, len(t), int((t.result == LIFTED).sum()), 0.0, ""
    t0 = time.time()
    try:
        obj, sim = _load(name)
        lib = library_hooks(name) if JUDGE == "lift" else Tested.empty()   # (the library: floating test only)
        t = lib.merged(teach(obj, sim, seed=seed, table_sim=_table_sim(name, obj)))
    except Exception as e:  # noqa: BLE001 (a scan that does not load: skip it, say why)
        return name, 0, 0, time.time() - t0, f"{type(e).__name__}: {e}"
    save_tested(name, t)
    return name, len(t), int((t.result == LIFTED).sum()), time.time() - t0, ""


def test_job(args) -> tuple[str, Tested, float]:
    """Worker: lift-test given candidates on one object (not stored here). args = (name, cands)."""
    import time

    name, cands = args
    t0 = time.time()
    try:
        obj, sim = _load(name)
        t = test_candidates(sim, cands, _table_sim(name, obj))
    except Exception:  # noqa: BLE001 (one bad object must not stop a 12-hour run)
        t = Tested.empty()
    return name, t, time.time() - t0


def random_job(args) -> list[dict]:
    """Worker helper: exploration candidates (random_candidates) for one object."""
    name, n, seed = args
    from .config import ObjectConfig
    from .objects import object_from_config

    return random_candidates(object_from_config(ObjectConfig(asset=name)), np.random.default_rng(seed), n)
