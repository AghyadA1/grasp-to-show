"""The whole system in one place: given a scene, pick the object up and show its QR to the scanner.

    from grasp2show.pipeline import Pipeline, Scene
    pipe = Pipeline()                                  # the trained chooser and hook network
    scene = Scene.sample("ACE_Coffee_Mug_Kristen_16_oz_cup", k=0)
    plan = pipe.plan(scene)                            # which strategy, which candidate, and why
    result = pipe.run(scene)                           # planned and executed in the simulator

A scene is what the robot is told: the object (its mesh), where it stands (x, y, yaw), where the
scanner is, and two physical facts, its weight and the friction between it and the fingers (part
3: a pinch holds by friction, a hook by shape; grasp2show/chooser.py). Per object, once, the
assets both strategies need are prepared and cached (prepare):

  pinch  antipodal grasps lift-tested in the floating gripper (data/grasps_choose), and the
         robustness of each (the share of 3 mm / 6 deg shifts of it that still hold);
  hook   the hook network's proposals (learn/hooknet.py, DEFAULT_HOOK_NET), each pushed as deep
         through the handle as fits and lift-tested; up to 6, each with its robustness, the lowest
         friction it holds at, where the object hangs on it, and whether the robot's lift from a
         table holds in rehearsal (data/hooks_net_<tag>).

Per scene, both strategies run their own checks (pinch: experiment.Scenario, lift / stability at
this friction and weight, reach, collisions, QR visibility; hook: hanging.hook_options, with only
the hooks that hold at this friction), each ranks its candidates by their smallest normalised
margin, robustness included (chooser.rank_pinch, rank_hooks). The chooser (chooser.Chooser,
data/models/chooser.pt) predicts each strategy's success from what the checks found and picks
one; the robot runs that strategy's first candidate whose motion can be planned (the other
strategy's if none can).

scripts/evaluate.py measures all of this on held-out scenes; scripts/run_scene.py runs one and
films it.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .chooser import (
    Chooser,
    hook_features,
    pinch_features,
    rank_hooks,
    rank_pinch,
    sample_conditions,
    scene_features,
    vector,
    wayb_features,
)
from .config import CameraConfig, ObjectConfig, SceneConfig, random_camera

ROOT = Path(__file__).resolve().parents[1]
GRASP_DIR = ROOT / "data" / "grasps_choose"
CLOUD_DIR = ROOT / "data" / "hook_clouds"
CHOOSER = ROOT / "data" / "models" / "chooser.pt"
SCANNER_SCALE = 2.0              # 1280 x 960 (part 3's scanner; config.CameraConfig is 640 x 480)
POSE_ERR_MM, POSE_ERR_DEG = 2.0, 1.5     # evaluation: where the object really is vs where the robot thinks
MAX_ROBUST = 150                 # pinch grasps whose robustness is measured per object (others: 0.5)
WAYB_MODEL = ROOT / "data" / "models" / "heatmap_net_v5.pt"     # way B v5 (scripts/train_wayb.py); v4 before
WAYB_CLOUD_DIR = ROOT / "data" / "wayb_clouds"
WAYB_K = 10
VERIFY_K = 5          # way B grasps shake-tested at the scene's friction and weight (Checked.wayb_verified)                      # way B grasps per scene (the network's best, 1 cm apart per target)


def crc(s: str) -> int:
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


def hook_dir(tag: str = "table") -> Path:
    """Where the hooks of hook network hook_net_<tag>.pt are cached ("self": data/hooks_net)."""
    return ROOT / "data" / ("hooks_net" if tag in ("", "self") else f"hooks_net_{tag}")


# ---------------------------------------------------------------- scenes

@dataclass
class Scene:
    """What the robot is told about one pick-and-show task."""
    asset: str
    yaw_deg: float
    start_xy: tuple[float, float]
    camera: CameraConfig
    mu: float                       # finger-object friction
    mass_scale: float = 1.0         # the object is this many times heavier than its scan's estimate

    @classmethod
    def sample(cls, asset: str, k: int, scale: float = SCANNER_SCALE) -> Scene:
        """Scene k of an object, as in the evaluation (deterministic): start pose on the table,
        scanner pose (config.random_camera), friction and weight (chooser.sample_conditions)."""
        return sample_scene(asset, k, scale)[0]


def sample_scene(asset: str, k: int, scale: float = SCANNER_SCALE) -> tuple[Scene, tuple[float, float, float]]:
    """Scene k of an object, and the error in the object's pose the robot does not know about
    (dx, dy [m], dyaw [rad]), drawn from the same seed: evaluate.py's scenes."""
    from .objects import object_from_config

    r = np.random.default_rng(crc(asset) * 100 + k)
    yaw, xy, cam = float(r.uniform(0, 360)), (float(r.uniform(0.40, 0.60)), float(r.uniform(-0.30, 0.0))), \
        random_camera(r)
    cam = replace(cam, width=int(cam.width * scale), height=int(cam.height * scale))
    mu, f = sample_conditions(r, object_from_config(ObjectConfig(asset=asset)).mass)
    e_xy = np.clip(r.normal(0, POSE_ERR_MM / 1000, 2), -2.5 * POSE_ERR_MM / 1000, 2.5 * POSE_ERR_MM / 1000)
    e_yaw = float(np.clip(r.normal(0, np.radians(POSE_ERR_DEG)), -np.radians(2.5 * POSE_ERR_DEG),
                          np.radians(2.5 * POSE_ERR_DEG)))
    return Scene(asset, yaw, xy, cam, mu, f), (float(e_xy[0]), float(e_xy[1]), e_yaw)


# ---------------------------------------------------------------- per-object assets (cached)

def label_grasps(asset: str, out: Path = GRASP_DIR, n_pairs: int = 300, n_approach: int = 8,
                 max_tests: int = 400, min_width: float = 0.002) -> Path:
    """Antipodal grasps lift-and-shake tested in the floating gripper (scripts/generate_grasps.py's
    test; 300 contact pairs, rims included: with 60 a mug kept 10 lifting grasps of 17)."""
    from .grasp_sampler import candidates_from_pairs, sample_antipodal
    from .gripper_sim import GripperSim
    from .objects import object_from_config

    f = out / f"{asset}.npz"
    if f.exists():
        return f
    cfg = SceneConfig(obj=ObjectConfig(asset=asset))
    obj = object_from_config(cfg.obj)
    sim = GripperSim(cfg, obj)
    rng = np.random.default_rng(crc(asset))
    p1, p2 = sample_antipodal(obj.mesh, n_pairs, rng, min_width=min_width)
    cand = candidates_from_pairs(p1, p2, n_approach, rng)
    cand = cand.subset(np.array([not sim.in_collision(T, w) for T, w in zip(cand.T_obj_tcp, cand.width)], bool))
    cand = cand.subset(rng.permutation(len(cand))[:max_tests])
    res = [sim.shake_test(T, w) for T, w in zip(cand.T_obj_tcp, cand.width)]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, T_obj_tcp=cand.T_obj_tcp.astype(np.float32), width=cand.width.astype(np.float32),
                        p1=cand.p1.astype(np.float32), p2=cand.p2.astype(np.float32),
                        success=np.array([r.success for r in res], bool),
                        closed=np.array([r.closed_on_object for r in res], bool),
                        slip_mm=np.array([r.slip_mm for r in res], np.float32),
                        slip_deg=np.array([r.slip_deg for r in res], np.float32), mass=obj.mass)
    return f


def pinch_robustness_of(asset: str, out: Path = GRASP_DIR) -> Path:
    """Robustness (chooser.pinch_robustness) of up to MAX_ROBUST lifting grasps; NaN: not tested."""
    from .chooser import pinch_robustness
    from .gripper_sim import GripperSim
    from .objects import object_from_config

    f = out / f"{asset}.robust.npy"
    if f.exists():
        return f
    z = np.load(out / f"{asset}.npz")
    obj = object_from_config(ObjectConfig(asset=asset))
    sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=asset)), obj)
    r = np.full(len(z["success"]), np.nan)
    for i in np.random.default_rng(crc(asset)).permutation(np.flatnonzero(z["success"]))[:MAX_ROBUST]:
        r[i] = pinch_robustness(sim, z["T_obj_tcp"][i].astype(float), float(z["width"][i]))
    np.save(f, r)
    return f


def hook_cloud(asset: str, n_stored: int = 4096) -> np.ndarray:
    """The hook network's input: 2048 points (and normals) of the object's outer surface, from a
    cached 4096 (generated objects: hidden faces between their convex parts left out)."""
    import trimesh

    from .learn.hooknet import N_POINTS, sample_cloud
    from .objects import asset_dir

    f = CLOUD_DIR / f"{asset}.v1.npy"
    if not f.exists():
        d = asset_dir(asset)
        mesh = trimesh.load(d / "model.obj", force="mesh", process=True)
        pieces = ([trimesh.load(p, force="mesh") for p in sorted(d.glob("model_collision_*.obj"))]
                  if asset.startswith("gen_") else None)
        CLOUD_DIR.mkdir(parents=True, exist_ok=True)
        np.save(f, sample_cloud(mesh, n_stored, seed=crc(asset), pieces=pieces))
    c = np.load(f)
    return c[np.random.default_rng(crc(asset)).choice(len(c), N_POINTS, replace=False)]


def propose_hooks(net, asset: str, k: int = 8, device="cpu") -> list[dict]:
    """The hook network's k best points on the object (both approach directions)."""
    from .learn.hooknet import predict_hooks

    return [{key: h[key] for key in ("T", "width", "kind", "prob")}
            for h in predict_hooks(net, hook_cloud(asset), k=k, min_prob=0.0, device=device)]


def test_hooks(asset: str, cands: list[dict], out: Path, keep: int = 6) -> Path:
    """The network's proposals (propose_hooks), most probable first: each also with both fingers
    through (closed) and tried that way first; nudged clear, pushed as deep through the handle as
    fits (hanging.deepen), lift-tested; up to `keep` distinct ones that lift, each with its
    robustness, the lowest friction it holds at, where the object hangs on it and whether the
    lift from a table holds in rehearsal (hanging.hang_table). Saved to out/<asset>.npz."""
    from .chooser import hook_robustness
    from .gripper_sim import GripperSim
    from .hanging import deepen, hang_table, hook_mu_min, nudge, save_hook_file
    from .hooks import lift_test
    from .learn.hooknet import anchor
    from .objects import object_from_config

    obj = object_from_config(ObjectConfig(asset=asset))
    sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=asset)), obj)
    hooks = []
    for c in sorted(cands, key=lambda c: -c["prob"]):
        variants = [c]
        if c["width"] > 0.01:                         # both fingers through cannot slide off past the other
            T2 = c["T"].copy()
            T2[:3, 3] = anchor(c["T"], True)
            variants = [{**c, "T": T2, "width": 0.0, "kind": "two"}, c]
        for v in variants:
            T = nudge(sim, v["T"], v["width"], step=0.003, reach=3, depths=(-0.006, 0.0, 0.006))
            if T is None:
                continue
            T = deepen(sim, T, v["width"])
            if not lift_test(sim, T, v["width"], close=False)["success"]:
                continue
            a = anchor(T, v["width"] > 0.01)
            if any(np.linalg.norm(a - anchor(h["T"], h["width"] > 0.01)) < 0.01 and
                   abs(T[:3, 2] @ h["T"][:3, 2]) > 0.9 for h in hooks):
                continue                              # the same hook again
            hooks.append({"T": T, "width": float(v["width"]), "kind": v["kind"], "prob": float(c["prob"])})
            break
        if len(hooks) == keep:
            break
    for h in hooks:
        h["robust"] = hook_robustness(sim, h["T"], h["width"])
        h["mu_min"] = hook_mu_min(sim, h["T"], h["width"])
    hang_table(sim, hooks, table_sim=GripperSim(SceneConfig(obj=ObjectConfig(asset=asset)), obj, table=True))
    f = out / f"{asset}.npz"
    save_hook_file(f, hooks)
    return f


# ---------------------------------------------------------------- way B: the network that proposes pinches

def wayb_cloud(asset: str, n: int = 1024) -> np.ndarray:
    """Way B's input for an object: (n, 7) surface points, normals, on-the-QR flag (as
    scripts/archive/part1_grasp_choice/generate_dataset.object_cloud), cached."""
    import trimesh

    from .objects import object_from_config

    f = WAYB_CLOUD_DIR / f"{asset}.npy"
    if f.exists():
        return np.load(f)
    obj = object_from_config(ObjectConfig(asset=asset))
    pts, fid = trimesh.sample.sample_surface_even(obj.mesh, n, seed=0)
    if len(pts) < n:
        extra, fid2 = trimesh.sample.sample_surface(obj.mesh, n - len(pts), seed=0)
        pts, fid = np.vstack([pts, extra]), np.concatenate([fid, fid2])
    T = obj.T_obj_qr
    local = (pts - T[:3, 3]) @ T[:3, :3]
    half = obj.qr.side / 2
    on_qr = (np.abs(local[:, 0]) < 0.004) & (np.abs(local[:, 1]) <= half) & (np.abs(local[:, 2]) <= half)
    feats = np.concatenate([pts, obj.mesh.face_normals[fid], on_qr[:, None]], 1).astype(np.float32)
    WAYB_CLOUD_DIR.mkdir(parents=True, exist_ok=True)
    np.save(f, feats)
    return feats


class WayB:
    """The goal-conditioned heatmap network (way B, parts 1 and 3): for a scene, a grasp score and
    a grasp at every point of the object's cloud for each of the 12 presentation targets."""

    def __init__(self, model: Path = WAYB_MODEL):
        import torch

        from .learn.dense import HeatmapNet

        torch.set_num_threads(1)
        ck = torch.load(model, map_location="cpu", weights_only=False)
        sd = {k: v for k, v in ck["state_dict"].items() if k != "clouds"}
        self.n_cond = int(sd["cond_mlp.0.weight"].shape[1])     # 25: v4; 27: v5 (friction and load too)
        self.net = HeatmapNet(torch.zeros(1, 1024, 7), n_cond=self.n_cond)
        self.net.load_state_dict(sd, strict=False)
        self.net.eval()

    def grasps(self, asset: str, cfg: SceneConfig, obj, k: int = WAYB_K, min_sep: float = 0.01):
        """Up to k (T_obj_tcp, width, target index, score) for this scene: the highest-scoring
        points, min_sep apart per target, whose open gripper does not hit the object. The network
        is told the object's mass in this scene (v4 was trained at the scans' own masses; v5 at
        many masses and frictions, and is told the friction too)."""
        import torch

        from .gripper_sim import GripperSim
        from .learn.dense import condition, grasp_from_prediction, split_outputs
        from .scene import start_pose
        from .task import show_pose

        feats = wayb_cloud(asset)
        self.net.clouds = torch.as_tensor(feats[None])
        rolls = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        T0 = start_pose(cfg, obj)
        mu = float(cfg.obj.friction[0]) if self.n_cond > 25 else None
        cond = np.stack([condition(T0, show_pose(cfg, obj, r), cfg.camera.pos, obj.mass, obj.com, mu) for r in rolls])
        with torch.no_grad():
            pr = {kk: v.numpy() for kk, v in split_outputs(self.net(torch.zeros(12, dtype=torch.long),
                                                                    torch.as_tensor(cond))).items()}
        q = 1 / (1 + np.exp(-np.clip(pr["logit"], -50, 50))) * pr["score"]
        pts = feats[:, :3]
        sim = GripperSim(cfg, obj)
        out, used = [], []
        for flat in np.argsort(-q, axis=None):
            r, i = divmod(int(flat), q.shape[1])
            if any(r == r2 and np.linalg.norm(pts[i] - pts[i2]) < min_sep for r2, i2 in used):
                continue
            used.append((r, i))
            T = grasp_from_prediction(pts[i], pr["approach"][r, i], pr["closing"][r, i], pr["offset"][r, i])
            w = float(np.clip(pr["width"][r, i], 0.0, 0.08))
            if not sim.in_collision(T, w):
                out.append((T, w, r, float(q[r, i])))
            if len(out) == k or len(used) >= 4 * k:
                break
        return out


def kinematic_choices(sc, order: list[tuple[int, int]]) -> list:
    """(grasp, target) pairs in preference order -> choices reachable at the start and at the
    target (the better twin): how way B has always been run (eval_way_b.py)."""
    from .experiment import Choice

    out = []
    for i, j in order:
        best, key = None, -np.inf
        for tw in sc.reports[i].twins:
            if tw.start_ok and j < len(tw.targets) and tw.targets[j].reach.ok:
                m = min(tw.start.limit_margin, tw.targets[j].reach.limit_margin)
                if m > key:
                    best, key = tw.twin, m
        if best is not None:
            out.append(Choice(i, best, j, sc.reports[i].ok))
    return out


@dataclass
class Assets:
    """One object's cached candidates for both strategies."""
    grasps: dict              # T_obj_tcp, width, success (pinch)
    robust: np.ndarray        # per grasp; 0.5 where not measured
    hooks: list[dict]         # hanging.load_hook_file

    @classmethod
    def load(cls, asset: str, hooks_dir: Path) -> Assets:
        from .hanging import load_hook_file

        z = np.load(GRASP_DIR / f"{asset}.npz")
        g = {k: z[k] for k in ("T_obj_tcp", "width", "success")}
        f_r = GRASP_DIR / f"{asset}.robust.npy"
        R = np.nan_to_num(np.load(f_r), nan=0.5) if f_r.exists() else np.full(len(g["width"]), 0.5)
        f_h = hooks_dir / f"{asset}.npz"
        return cls(g, R, load_hook_file(f_h) if f_h.exists() else [])


# ---------------------------------------------------------------- one scene: both strategies' checks

@dataclass
class Checked:
    """A scene with both strategies' checks done: the pinch scenario (its reports), the hooks
    that hold at this friction and their options, and the chooser's inputs."""
    scene: Scene
    sc: object                  # experiment.Scenario
    assets: Assets
    hooks: list[dict]           # those that hold at this scene's friction
    opts: list                  # hanging.HookOption for them
    features: dict = field(default_factory=dict)
    wb: object = None           # experiment.Scenario of way B's grasps for this scene (None: no way B)
    wb_q: np.ndarray | None = None      # their scores
    wb_roll: np.ndarray | None = None   # the target each was proposed for
    seconds: dict = field(default_factory=dict)   # time each strategy's checks took
    _verified: list | None = None

    def pinch_ranked(self, robust: bool = True) -> list:
        return rank_pinch(self.sc, self.assets.robust if robust else None)

    def hook_ranked(self, robust: bool = True) -> list:
        return rank_hooks(self.opts, self.hooks if robust else None)

    def wayb_ranked(self, checked: bool = False, fallback: bool = True) -> list:
        """Way B's choices: the network's order, reach checked only (as way B always ran), or
        checked: every check at this friction and weight (lift, stability, visibility), the
        network's score as one more margin (2q - 1), the largest smallest margin first; if none
        passes, the network's order (fallback)."""
        if self.wb is None:
            return []
        if checked:
            out = rank_pinch(self.wb, extra=2 * self.wb_q - 1)
            if out or not fallback:
                return out
        order = [(int(i), int(self.wb_roll[i])) for i in np.argsort(-self.wb_q)]
        return kinematic_choices(self.wb, order)

    def wayb_verified(self, k: int = VERIFY_K) -> list:
        """Way B's checked ranking with its first k grasps shake-tested in the floating-gripper
        simulator at this scene's friction and weight (gripper_sim.shake_test: close, lift,
        shake): those that hold first, in their order, then the rest. The static lift model is
        optimistic where it matters (grasps it gives a margin under 0.6 hold half the time in full
        episodes, most lost 0.3 s into the lift, as the table lets go: scripts/diagnose_drops.py).
        Adds the chooser inputs wayb_sim_any, wayb_sim_frac. Cached."""
        if self._verified is not None:
            return self._verified
        from .gripper_sim import GripperSim

        ranked = self.wayb_ranked(True)
        held, tested = {}, []
        if ranked:
            sim = GripperSim(self.wb.cfg, self.wb.obj)
            sim.set_conditions(self.scene.mu, 1.0)            # the scene's obj already has its mass
            for c in ranked:
                if c.grasp in held:
                    continue
                held[c.grasp] = bool(sim.shake_test(self.wb.T[c.grasp], float(self.wb.W[c.grasp])).success)
                tested.append(held[c.grasp])
                if len(tested) == k:
                    break
        self._verified = [c for c in ranked if held.get(c.grasp)] + [c for c in ranked if not held.get(c.grasp)]
        self.features.update(wayb_sim_any=float(any(tested)), wayb_sim_frac=float(np.mean(tested)) if tested else 0.0)
        return self._verified

    def chooser_inputs(self, robust: bool = True) -> dict:
        """features plus the robustness of each strategy's first candidate."""
        pr, hr = self.pinch_ranked(robust), self.hook_ranked(robust)
        return {**self.features, "pinch_r": float(self.assets.robust[pr[0].grasp]) if pr else 0.0,
                "hook_r": float(self.hooks[hr[0].hook].get("robust", 0.5)) if hr else 0.0}


def check(scene: Scene, assets: Assets, wayb: WayB | None = None) -> Checked:
    """The strategies' checks for one scene (no motion yet): pinch and hook, and way B's grasps
    if `wayb` is given."""
    from .experiment import Scenario
    from .gripper_sim import GripperSim
    from .hanging import hook_options

    t0 = time.time()
    sc = Scenario(scene.asset, scene.yaw_deg, assets.grasps, start_xy=scene.start_xy, camera=scene.camera,
                  mass_scale=scene.mass_scale, mu=scene.mu)
    sc.lift_margin = np.where(sc.lift_ok, sc.lift_margin, -1.0)      # this friction and weight, not clipped
    t1 = time.time()
    hooks = [h for h in assets.hooks if h.get("mu_min", 0.0) <= scene.mu + 1e-9]
    opts = hook_options(sc.fc, GripperSim(sc.cfg, sc.obj), hooks, T_start=sc.ep.resting_pose()) if hooks else []
    feats = {**scene_features(scene.mu, sc.obj.mass, sc.obj.mesh.extents), **pinch_features(sc),
             **hook_features(hooks, opts)}
    ck = Checked(scene, sc, assets, hooks, opts, feats)
    ck.seconds = {"pinch": t1 - t0, "hook": time.time() - t1}
    if wayb is not None:
        add_wayb(ck, wayb, sc.cfg, sc.obj)
    return ck


def add_wayb(ck: Checked, wayb: WayB, cfg=None, obj=None) -> Checked:
    """Way B's grasps for ck's scene, checked at its friction and weight (ck.wb, wb_q, wb_roll),
    and their chooser inputs. cfg, obj: the scene's (built here if not given)."""
    from .experiment import Scenario
    from .objects import object_from_config

    t0 = time.time()
    scene = ck.scene
    if cfg is None:
        extra = {"friction": (float(scene.mu), 0.01 * float(scene.mu), 0.0001)}
        cfg = SceneConfig(obj=ObjectConfig(asset=scene.asset, start_yaw=np.radians(scene.yaw_deg),
                                           start_xy=tuple(scene.start_xy), **extra), camera=scene.camera)
        obj = object_from_config(cfg.obj)
        if scene.mass_scale != 1.0:
            obj = replace(obj, mass=obj.mass * scene.mass_scale, inertia=obj.inertia * scene.mass_scale)
    g = wayb.grasps(scene.asset, cfg, obj)
    if g:
        ck.wb = Scenario(scene.asset, scene.yaw_deg, {"T_obj_tcp": np.stack([x[0] for x in g]),
                                                      "width": np.array([x[1] for x in g]),
                                                      "success": np.ones(len(g), bool)},
                         start_xy=scene.start_xy, camera=scene.camera, mass_scale=scene.mass_scale, mu=scene.mu)
        ck.wb.lift_margin = np.where(np.isfinite(ck.wb.lift_margin), ck.wb.lift_margin, -1.0)
        ck.wb_q, ck.wb_roll = np.array([x[3] for x in g]), np.array([x[2] for x in g])
    ck.features.update(wayb_features(ck.wb, ck.wayb_ranked(), ck.wayb_ranked(True, fallback=False), ck.wb_q))
    ck.seconds["wayb"] = time.time() - t0
    return ck


def execute_pinch(ck: Checked, choice, obj_offset=None, on_frame=None, fps: int = 30,
                  carry_net=None) -> tuple[bool | None, str]:
    """(success, why) for one pinch choice; success None: its motion could not be planned."""
    e = ck.sc._execute(choice, obj_offset=obj_offset, on_frame=on_frame, fps=fps,
                       carry=carry_fn(carry_net, "pinch", ck.scene.mu))
    if e.reason.startswith("plan"):
        return None, e.reason
    return bool(e.success), e.reason


def carry_fn(net, kind: str, mu: float):
    """A paths transformer for Scenario._execute: the carry network's carry (None: the planner's)."""
    if net is None:
        return None
    from .carry import carry_paths

    dt = SceneConfig().timestep
    return lambda paths, T, w, T_tcp_obj, obj: carry_paths(net, kind, paths, T, w, T_tcp_obj, mu, obj, dt)


def execute_wayb(ck: Checked, choice, obj_offset=None, on_frame=None, fps: int = 30,
                 carry_net=None) -> tuple[bool | None, str]:
    """(success, why) for one of way B's choices; None: its motion could not be planned.
    carry_net: carry with the carry network (learn/carry_net.py) instead of the planner."""
    e = ck.wb._execute(choice, obj_offset=obj_offset, on_frame=on_frame, fps=fps,
                       carry=carry_fn(carry_net, "pinch", ck.scene.mu))
    if e.reason.startswith("plan"):
        return None, e.reason
    return bool(e.success), e.reason


def execute_hook(ck: Checked, option, obj_offset=None, on_frame=None, fps: int = 30,
                 carry_net=None) -> tuple[bool | None, str]:
    """(success, why) for one hook option; success None: its motion could not be planned."""
    from .hanging import HOOK_SPEED, plan_hook
    from .task import Grasp

    paths, why = plan_hook(ck.sc.ep, ck.sc.fc, option)
    if paths is None:
        return None, f"plan: {why}"
    if carry_net is not None:
        from .carry import carry_paths

        paths = carry_paths(carry_net, "hook", paths, option.T_obj_tcp, option.width, option.hang.T_tcp_obj,
                            ck.scene.mu, ck.sc.obj, ck.sc.ep.model.opt.timestep)
    res = ck.sc.ep.execute(Grasp(f"hook{option.hook}", option.T_obj_tcp, option.width, "hook"), paths,
                           speed=HOOK_SPEED, mu=ck.scene.mu, obj_offset=obj_offset, on_frame=on_frame, fps=fps)
    return bool(res.success), res.reason(hook=True)


# ---------------------------------------------------------------- the pipeline

@dataclass
class Plan:
    method: str                 # the chosen strategy: "pinch", "hook", "wayb" or "wayb_checked"
    probs: dict                 # the chooser's predicted success of each strategy it knows
    ranked: dict                # each strategy's ranked candidates (experiment.Choice / HookOption)
    checked: Checked

    @property
    def candidates(self) -> list:
        return self.ranked[self.method]


@dataclass
class Result:
    method: str
    success: bool
    reason: str
    plan: Plan
    seconds: float


def ranked_all(ck: Checked) -> dict:
    """Every strategy's ranked candidates (the robust rankings for pinch and hook)."""
    out = {"pinch": ck.pinch_ranked(), "hook": ck.hook_ranked()}
    if ck.wb is not None:
        out.update(wayb=ck.wayb_ranked(), wayb_checked=ck.wayb_ranked(True), wayb_verified=ck.wayb_verified())
    return out


def execute(ck: Checked, strategy: str, c, obj_offset=None, on_frame=None, carry_net=None) -> tuple[bool | None, str]:
    if strategy == "hook":
        return execute_hook(ck, c, obj_offset, on_frame=on_frame, carry_net=carry_net)
    if strategy.startswith("wayb"):
        return execute_wayb(ck, c, obj_offset, on_frame=on_frame, carry_net=carry_net)
    return execute_pinch(ck, c, obj_offset, on_frame=on_frame, carry_net=carry_net)


class Pipeline:
    """The trained parts: the hook network (finds hooks on a new object) and the chooser."""

    def __init__(self, chooser: Path = CHOOSER, hook_net: Path | None = None, device: str = "cpu",
                 carry_net: Path | None = None):
        """carry_net: carry with the carry network (data/models/carry_net.pt, scripts/train_carry.py)
        instead of the planner's lift and transit."""
        from .learn.carry_net import load_carry_net

        self.carry_net = load_carry_net(carry_net) if carry_net is not None else None
        from .learn.hooknet import DEFAULT_HOOK_NET

        self.hook_net_path = Path(hook_net) if hook_net is not None else DEFAULT_HOOK_NET
        tag = self.hook_net_path.stem.replace("hook_net_", "")
        self.hooks_dir = hook_dir(tag)
        self.chooser = Chooser.load(chooser) if Path(chooser).exists() else None
        self.strategies = self.chooser.strategies if self.chooser else ("pinch", "hook")
        wb_model = ROOT / self.chooser.meta["wayb_model"] if self.chooser and "wayb_model" in self.chooser.meta \
            else WAYB_MODEL
        self.wayb = WayB(wb_model) if any(s.startswith("wayb") for s in self.strategies) else None
        self.device = device
        self._net = None

    def prepare(self, asset: str) -> Assets:
        """The object's grasps and hooks, computed once and cached (minutes for a new object)."""
        label_grasps(asset)
        pinch_robustness_of(asset)
        if not (self.hooks_dir / f"{asset}.npz").exists():
            if self._net is None:
                from .learn.hooknet import load_hook_net

                self._net = load_hook_net(self.hook_net_path, self.device)
            test_hooks(asset, propose_hooks(self._net, asset, device=self.device), self.hooks_dir)
        return Assets.load(asset, self.hooks_dir)

    def plan(self, scene: Scene) -> Plan:
        """The strategies checked, the chooser's pick and every strategy's ranked candidates."""
        ck = check(scene, self.prepare(scene.asset), self.wayb)
        ranked = ranked_all(ck)          # (way B verified: adds its chooser inputs)
        names = [s for s in self.strategies if s in ranked]
        if self.chooser is not None:
            x = vector(ck.chooser_inputs(), "", self.chooser.features)[None]
            probs = dict(zip(self.chooser.strategies, map(float, self.chooser.predict(x)[0])))
        else:                                          # no trained chooser: the rule (chooser.rule_choice)
            from .chooser import rule_choice

            probs = {"pinch": 0.0, "hook": 1.0} if rule_choice(ck.features) == "hook" else {"pinch": 1.0, "hook": 0.0}
        usable = [s for s in names if ranked[s]] or names
        best = max(usable, key=lambda s: probs.get(s, 0.0) - (self.chooser.margin if self.chooser and s != names[0]
                                                               else 0.0))
        return Plan(best, probs, ranked, ck)

    def run(self, scene: Scene, obj_offset=None, max_tries: int = 12, on_frame=None,
            plan: Plan | None = None) -> Result:
        """Plan (or take `plan`), then execute the chosen strategy's first candidate whose motion
        can be planned (the other strategy's if none can). obj_offset: an error in the object's
        pose the robot does not know (evaluation). on_frame(data): for videos."""
        t0 = time.time()
        pl = plan or self.plan(scene)
        why = "no candidate"
        order = [pl.method] + sorted((s for s in pl.ranked if s != pl.method), key=lambda s: -pl.probs.get(s, 0.0))
        for method in order:
            for c in pl.ranked[method][:max_tries]:
                ok, why = execute(pl.checked, method, c, obj_offset, on_frame=on_frame, carry_net=self.carry_net)
                if ok is not None:
                    return Result(method, ok, why, pl, time.time() - t0)
        return Result(pl.method, False, why, pl, time.time() - t0)
