"""Grasp-selection policies and full-episode evaluation (the experiments of day 5).

A *scenario* is an object at a start pose (x, y, yaw) on the table. For every grasp that held
in the simulated lift test (check 1), the feasibility checker computes checks 2-5 for both twins
and every target roll. A *policy* then chooses one (grasp, twin, target roll), and the choice is
executed in the full simulation (pick -> transit -> show -> hold) and judged by the QR decoder.

Policies (each returns a ranked list; the first choice whose motion can be planned is executed,
as a real system would fall back to its next grasp when motion planning fails):
  stability_only  the baseline: grasps ranked by analytic lift margin (most stable first), each
                  with a reachable target pose (largest joint margin). A standard "grasp
                  planner + motion planner" pipeline: kinematics are checked, but not whether the
                  QR will be visible or the grasp stable once the object is turned.
  aware           ours: the best grasp passing all checks, ranked by its smallest normalised
                  margin (the rule fixed before any experiment; ties keep the enumeration order:
                  grasp index, twin, target roll).
  aware_no_vis    ablation: aware without the QR-visibility check (check 5);
  aware_no_stab   ablation: aware without the target-stability check (check 2);
  aware_single    ablation: aware with one target pose (roll 0) instead of the target set;
  aware_leximin   sensitivity: ties broken by the next-smallest margin (leximin). Added after the
                  main experiment because of one demo case; it raised slip failures (static
                  margins do not capture slip during motion), so it is reported, not adopted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .config import CameraConfig, ObjectConfig, SceneConfig
from .episode import PickAndShow
from .feasibility import FeasibilityChecker, GraspReport, twin_of
from .objects import ObjectModel, object_from_config
from .scene import build_scene
from .stability import StabilityParams, finger_contacts, utilisation_shake
from .task import Grasp, show_pose

POLICIES = ("stability_only", "aware", "aware_no_vis", "aware_no_stab", "aware_single", "aware_leximin")
LIMIT_SCALE = 0.5   # rad: a joint-limit margin of 0.5 rad counts as "fully comfortable"


@dataclass
class Choice:
    grasp: int
    twin: int
    roll_idx: int
    predicted_ok: bool       # passes all checks (full aware definition)


@dataclass
class EpisodeRecord:
    object: str
    yaw_deg: float
    policy: str
    grasp: int | None
    twin: int | None
    roll_deg: float | None
    predicted_ok: bool | None
    success: bool
    reason: str              # "", or why it failed: no grasp, plan, qr, dropped, slip, collision
    qr_ok: bool = False
    held: bool = False
    slip_mm: float = float("nan")
    slip_deg: float = float("nan")
    vis_fraction: float = float("nan")
    stab_margin: float = float("nan")


def lift_margins(obj: ObjectModel, T_obj_tcp: np.ndarray, prm: StabilityParams) -> np.ndarray:
    """Analytic lift-stability margin 1 - u (shake-test loads, object upright); -inf: no contact."""
    c1, c2, ok = finger_contacts(obj.mesh, T_obj_tcp)
    m = np.full(len(T_obj_tcp), -np.inf)
    if ok.any():
        m[ok] = 1.0 - utilisation_shake(c1[ok], c2[ok], T_obj_tcp[ok], obj.com, obj.mass, obj.inertia, prm)
    return m


def _target_ok(t, use_vis: bool, use_stab: bool) -> bool:
    if not t.reach.ok:
        return False
    if use_stab and not t.stability_margin > 0:
        return False
    return not use_vis or (t.visibility is not None and t.visibility.ok())


def choose(policy: str, reports: dict[int, GraspReport], lift_margin: dict[int, float],
           k: int = 5) -> list[Choice]:
    """Apply a policy to the reports of the lift-stable grasps of one scenario: up to k ranked
    choices (distinct grasps for the baseline, distinct (grasp, twin, roll) for the others)."""
    if policy == "stability_only":
        out = []
        for i in sorted(reports, key=lambda g: -lift_margin[g]):
            best, key = None, -np.inf
            for tw in reports[i].twins:
                if not tw.start_ok:
                    continue
                for j, t in enumerate(tw.targets):
                    m = min(tw.start.limit_margin, t.reach.limit_margin)
                    if t.reach.ok and m > key:
                        best, key = (tw.twin, j), m
            if best is not None:
                out.append(Choice(i, best[0], best[1], reports[i].ok))
                if len(out) == k:
                    break
        return out

    use_vis = policy != "aware_no_vis"
    use_stab = policy != "aware_no_stab"
    single = policy == "aware_single"
    scored = []
    for i, r in reports.items():
        for tw in r.twins:
            if not tw.start_ok:
                continue
            for j, t in enumerate(tw.targets):
                if single and j != 0:
                    continue
                if not _target_ok(t, use_vis, use_stab):
                    continue
                parts = [lift_margin[i], tw.start.limit_margin / LIMIT_SCALE, t.reach.limit_margin / LIMIT_SCALE]
                if use_stab:
                    parts.append(t.stability_margin)
                if use_vis:
                    parts.append((t.visibility.fraction - 0.9) / 0.1)
                key = tuple(sorted(parts)) if policy == "aware_leximin" else (min(parts),)
                scored.append((key, Choice(i, tw.twin, j, r.ok)))
    scored.sort(key=lambda x: x[0], reverse=True)       # stable: ties keep enumeration order
    return [c for _, c in scored[:k]]


class Scenario:
    """One object at one start pose: feasibility reports and episode execution."""

    def __init__(self, asset: str, yaw_deg: float, grasps: dict, n_rolls: int = 12,
                 start_xy: tuple[float, float] | None = None, camera: CameraConfig | None = None,
                 mass_scale: float = 1.0, mu: float | None = None):
        """mass_scale: the object is that many times heavier (mass and inertia scaled after the
        loader's 50 g - 1.5 kg clip, so the checks and the simulator agree). mu: finger-object
        friction, known to the checks and set in the simulator (PickAndShow.set_friction: MuJoCo
        takes the larger of the two surfaces' coefficients, and the fingers' default is 1.0, so
        the object's own coefficient alone changes nothing); None: as before."""
        self.asset, self.yaw_deg, self.mu = asset, yaw_deg, mu
        extra = {} if start_xy is None else {"start_xy": tuple(start_xy)}
        if mu is not None:
            extra["friction"] = (float(mu), 0.01 * float(mu), 0.0001)
        cam = {} if camera is None else {"camera": camera}
        self.cfg = SceneConfig(obj=ObjectConfig(asset=asset, start_yaw=np.radians(yaw_deg), **extra), **cam)
        self.obj = object_from_config(self.cfg.obj)
        if mass_scale != 1.0:
            self.obj = replace(self.obj, mass=self.obj.mass * mass_scale, inertia=self.obj.inertia * mass_scale)
        self.scene = build_scene(self.cfg, self.obj)
        self.fc = FeasibilityChecker(self.scene, self.cfg)
        self.ep = PickAndShow(self.scene, self.cfg)
        self.rolls = np.linspace(0, 2 * np.pi, n_rolls, endpoint=False)
        self.T = grasps["T_obj_tcp"].astype(float)
        self.W = grasps["width"].astype(float)
        self.lift_ok = grasps["success"].astype(bool)
        prm = self.fc.stab
        self.lift_margin = lift_margins(self.obj, self.T, prm)
        self.reports: dict[int, GraspReport] = {
            int(i): self.fc.evaluate(self.T[i], self.W[i], self.rolls) for i in np.nonzero(self.lift_ok)[0]
        }
        self._cache: dict[tuple, EpisodeRecord] = {}

    def run(self, policy: str, choices: list[Choice]) -> EpisodeRecord:
        """Execute the first choice whose motion can be planned."""
        base = {"object": self.asset, "yaw_deg": self.yaw_deg, "policy": policy}
        if not choices:
            return EpisodeRecord(**base, grasp=None, twin=None, roll_deg=None, predicted_ok=None,
                                 success=False, reason="no grasp")
        rec = None
        for c in choices:
            key = (c.grasp, c.twin, c.roll_idx)
            if key not in self._cache:
                self._cache[key] = self._execute(c)
            rec = self._cache[key]
            if not rec.reason.startswith("plan"):
                break
        return EpisodeRecord(**{**asdict(rec), **base})

    def _execute(self, c: Choice, obj_offset: tuple[float, float, float] | None = None) -> EpisodeRecord:
        """Plan and run one choice; obj_offset: unknown error in the object's start pose (see
        PickAndShow.execute)."""
        tw = self.reports[c.grasp].twins[c.twin]
        t = tw.targets[c.roll_idx]
        T_obj_tcp = self.T[c.grasp] if c.twin == 0 else twin_of(self.T[c.grasp])
        roll = float(self.rolls[c.roll_idx])
        common = {"object": self.asset, "yaw_deg": self.yaw_deg, "policy": "", "grasp": c.grasp,
                  "twin": c.twin, "roll_deg": float(np.degrees(roll)), "predicted_ok": bool(t.ok),
                  "vis_fraction": t.visibility.fraction if t.visibility else float("nan"),
                  "stab_margin": float(t.stability_margin)}
        paths, why = self.ep.plan_analytic(T_obj_tcp, show_pose(self.cfg, self.obj, roll), tw.start.q, t.reach.q,
                                           checker=self.fc, width=float(self.W[c.grasp]))
        if paths is None:
            return EpisodeRecord(**common, success=False, reason=f"plan: {why}")
        r = self.ep.execute(Grasp(f"g{c.grasp}", T_obj_tcp, float(self.W[c.grasp])), paths, obj_offset=obj_offset,
                            mu=self.mu)
        if r.success:
            reason = ""
        elif r.collisions:
            reason = "collision: " + ", ".join(r.collisions[:2])
        elif not r.held:
            reason = "dropped"
        elif r.slip_mm >= self.ep.slip_tol_mm or r.slip_deg >= self.ep.slip_tol_deg:
            reason = "slip"
        else:
            reason = "qr"
        return EpisodeRecord(**common, success=bool(r.success), reason=reason, qr_ok=bool(r.qr_ok),
                             held=bool(r.held), slip_mm=float(r.slip_mm), slip_deg=float(r.slip_deg))
