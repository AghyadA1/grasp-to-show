"""Hook or pinch? The chooser (part 3).

Each scene is an object at a start pose, a scanner pose, and two things the robot is told about
the object: its weight and the friction between its surface and the fingers. Both strategies
are then planned with their own checks, and the chooser picks one before anything moves:

  pinch  the analytic pipeline ("aware": lift / target stability with this friction and weight,
         reach, collisions, QR visibility) on the object's antipodal grasps;
  hook   the self-trained hook network's proposals (grasp2show/learn/hooknet.py), lift-tested
         once per object, with where the object hangs on each (hanging.hang_table), then the
         hook checks (reach, collisions, QR visibility) for this scene (hanging.hook_options).

Why friction and weight. A pinch holds by friction: two fingers pressing 20 N each can carry
about 2 mu F / g (4 mu kg) when the grip is centred, less when the object's weight pulls at a
distance (torque) or when it is moved. A hook carries the object on the finger's surface: the
weight is a normal force, nearly independent of friction, up to what the arm can lift. So a wet
glass mug (mu ~ 0.2) full of water slips out of a pinch and hangs fine on a finger, while a light
rubbery box is best pinched (faster, steadier, no swing). The scenes span both:
  mu   log-uniform 0.15-1.0 (wet glazed porcelain or glass ~0.15-0.3, plastic ~0.3-0.5, rubber ~1)
  mass the object's own times a factor log-uniform 0.5-8 (empty to filled / heavier material),
       at most 3 kg (the Panda's payload).

Two rankings are compared, for both strategies: "score" (the checks' margins only) and "robust"
(the same plus the candidate's robustness r: the share of its small shifts that still hold, see
below). Each scene adds the error a real robot has: the object lies a few millimetres and a
degree or two from where the robot believes (unknown to the planner, the same for every
candidate), so a choice on a knife edge can fail.

Each strategy runs its best candidate: the pinch's (grasp, side, turn) with the largest smallest
margin (experiment.choose "aware"), the hook's (hook, side, carry pitch, turn) likewise
(hook_score); if its motion cannot be planned, the next. The episodes also run each strategy's
2nd and 3rd candidates, to measure how good the rankings are (did the best-ranked one succeed
when another would have?).

The chooser predicts each strategy's success from what the checks found (features below) with
a small network per strategy, trained on simulated episodes where both were executed, and picks
the larger (the pinch when they tie: it is faster and does not swing). A rule chooser (pinch if
all its checks pass with a positive lift margin, else the hook if one passes, else the pinch) and
the oracle (whichever succeeded) bound it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MU_RANGE = (0.15, 1.0)
MASS_FACTOR = (0.5, 8.0)
MAX_MASS = 3.0
G = 9.81
LIMIT_SCALE = 0.5


def sample_conditions(rng: np.random.Generator, mass0: float) -> tuple[float, float]:
    """(mu, mass_scale) for one scene: see the module docstring."""
    mu = float(np.exp(rng.uniform(np.log(MU_RANGE[0]), np.log(MU_RANGE[1]))))
    f = float(np.exp(rng.uniform(np.log(MASS_FACTOR[0]), np.log(MASS_FACTOR[1]))))
    f = min(f, MAX_MASS / mass0)
    return round(mu, 3), round(f, 3)


# ---------------------------------------------------------------- features

PINCH_KEYS = ("pinch_any", "pinch_n_ok", "pinch_best", "pinch_lift", "pinch_stab", "pinch_vis", "pinch_n_lift_ok")
HOOK_KEYS = ("hook_n", "hook_any", "hook_n_ok", "hook_best", "hook_pixels", "hook_angle", "hook_swing", "hook_tilt",
             "hook_pitch", "hook_one", "hook_prob")
WAYB_KEYS = ("wayb_any", "wayb_q", "wayb_n_ok", "wayb_best", "wayb_lift")
SCENE_KEYS = ("mu", "log_mass", "load", "ext_max", "ext_min")
ROBUST_KEYS = ("pinch_r", "hook_r")       # robustness of the candidate each strategy would run
FEATURES = SCENE_KEYS + PINCH_KEYS + HOOK_KEYS + ROBUST_KEYS
VERIFY_KEYS = ("wayb_sim_any", "wayb_sim_frac")   # way B's top grasps shake-tested at this friction and weight
STRATEGY_KEYS = {"pinch": PINCH_KEYS + ("pinch_r",), "hook": HOOK_KEYS + ("hook_r",), "wayb": WAYB_KEYS,
                 "wayb_checked": WAYB_KEYS, "wayb_verified": WAYB_KEYS + VERIFY_KEYS}


def features_for(strategies) -> tuple:
    """The chooser's inputs when it chooses among `strategies`: the scene's, and each strategy's
    own checks (none of the strategies it does not have)."""
    return tuple(dict.fromkeys(SCENE_KEYS + sum((STRATEGY_KEYS[s] for s in strategies), ())))


def pinch_features(sc) -> dict:
    """What the pinch checks found in this scene (experiment.Scenario, lift margins at this
    friction and weight): is there a choice passing every check, how many, the best one's
    smallest normalised margin and its parts."""
    best, n_ok, parts_best = -np.inf, 0, None
    for i, r in sc.reports.items():
        for tw in r.twins:
            if not tw.start_ok:
                continue
            for t in tw.targets:
                if not t.reach.ok:
                    continue
                vis = (t.visibility.fraction - 0.9) / 0.1 if t.visibility is not None else -1.0
                parts = [sc.lift_margin[i], tw.start.limit_margin / LIMIT_SCALE, t.reach.limit_margin / LIMIT_SCALE,
                         t.stability_margin, vis]
                ok = t.stability_margin > 0 and t.visibility is not None and t.visibility.ok() and sc.lift_margin[i] > 0
                n_ok += ok
                if min(parts) > best:
                    best, parts_best = min(parts), parts
    if parts_best is None:
        return {"pinch_any": 0.0, "pinch_n_ok": 0.0, "pinch_best": -2.0, "pinch_lift": -2.0, "pinch_stab": -2.0,
                "pinch_vis": -2.0, "pinch_n_lift_ok": float(np.log1p(np.sum(sc.lift_margin > 0)))}
    c = lambda x: float(np.clip(x, -2.0, 2.0))
    return {"pinch_any": 1.0, "pinch_n_ok": float(np.log1p(n_ok)), "pinch_best": c(best), "pinch_lift": c(parts_best[0]),
            "pinch_stab": c(parts_best[3]), "pinch_vis": c(parts_best[4]),
            "pinch_n_lift_ok": float(np.log1p(np.sum(sc.lift_margin > 0)))}


# ---------------------------------------------------------------- robustness (local support)
#
# A candidate that scores well may sit on a knife edge: the fit has 1 mm to spare, or the label
# came from one lucky test, or the scan's shape is slightly wrong there. A candidate whose small
# shifts also work has room for the errors a real robot makes (where the object is, what shape
# it has). Counting nearby candidates that worked would measure where the proposers looked, not
# robustness (the network proposes densely where it is confident), so each candidate is tested
# shifted by the same fixed set of small displacements and turns: r = the share that still holds.

PERTURB_MM, PERTURB_DEG = 3.0, 6.0


def perturbations(mm: float = PERTURB_MM, deg: float = PERTURB_DEG, axes_shift="xyz", axes_turn="xz") -> list:
    """Fixed small displacements of a grasp pose, in the TCP frame: +-mm along each of axes_shift,
    +-deg about each of axes_turn."""
    out = []
    for a in axes_shift:
        for s in (-1, 1):
            T = np.eye(4)
            T["xyz".index(a), 3] = s * mm / 1000
            out.append(T)
    for a in axes_turn:
        for s in (-1, 1):
            th = np.radians(s * deg)
            c, sn = np.cos(th), np.sin(th)
            R = {"x": [[1, 0, 0], [0, c, -sn], [0, sn, c]], "y": [[c, 0, sn], [0, 1, 0], [-sn, 0, c]],
                 "z": [[c, -sn, 0], [sn, c, 0], [0, 0, 1]]}[a]
            T = np.eye(4)
            T[:3, :3] = R
            out.append(T)
    return out


def hook_robustness(sim, T: np.ndarray, width: float) -> float:
    """Share of the perturbed hooks (shifted +-3 mm along x, y, z, turned +-6 deg about the
    approach and the across axes) that still lift the object (no nudge: the shift is the test)."""
    from .hanging import max_penetration
    from .hooks import lift_test

    ok = 0
    P = perturbations(axes_turn="xz")
    for dT in P:
        Tp = T @ dT
        ok += max_penetration(sim, Tp, width) <= 0.001 and lift_test(sim, Tp, width, close=False)["success"]
    return ok / len(P)


def pinch_robustness(sim, T: np.ndarray, width: float) -> float:
    """Share of the perturbed pinches (shifted +-3 mm along the approach and across the fingers,
    turned +-6 deg about those axes; a shift along the closing axis is taken up by the fingers)
    that are collision-free and pass the lift-and-shake test."""
    ok = 0
    P = perturbations(axes_shift="xz", axes_turn="xz")
    for dT in P:
        Tp = T @ dT
        ok += (not sim.in_collision(Tp, width)) and sim.shake_test(Tp, width).success
    return ok / len(P)


def robust_part(r: float) -> float:
    """r as a margin like the others: 0 at r = 0.5, 1 at r = 1, -1 at r = 0."""
    return 2.0 * (r - 0.5)


def rank_pinch(sc, robust: np.ndarray | None = None, k: int = 20, extra: np.ndarray | None = None) -> list:
    """The pinch's ranked choices (experiment.choose "aware": every check passes, the largest
    smallest normalised margin first), with robust (per grasp, r) added as one more margin, and
    extra (per grasp, already a margin: way B's 2q - 1) as another."""
    from .experiment import Choice

    scored = []
    for i, r in sc.reports.items():
        for tw in r.twins:
            if not tw.start_ok:
                continue
            for j, t in enumerate(tw.targets):
                if not (t.reach.ok and t.stability_margin > 0 and t.visibility is not None and t.visibility.ok()):
                    continue
                parts = [sc.lift_margin[i], tw.start.limit_margin / LIMIT_SCALE, t.reach.limit_margin / LIMIT_SCALE,
                         t.stability_margin, (t.visibility.fraction - 0.9) / 0.1]
                if robust is not None:
                    parts.append(robust_part(float(robust[i])))
                if extra is not None:
                    parts.append(float(extra[i]))
                scored.append((min(parts), Choice(i, tw.twin, j, r.ok)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def hook_score(o) -> float:
    """A feasible hook option's smallest normalised margin, as the pinch's "aware" ranking: the QR's
    size above 50 px and angle below 45 deg, and the joint-limit margins at the grasp, the
    pre-grasp and the show pose."""
    parts = [(o.vis.pixels - 50.0) / 50.0, (45.0 - o.vis.angle_deg) / 45.0]
    parts += [c.limit_margin / LIMIT_SCALE for c in (o.start, o.pre, o.show) if c is not None]
    return float(min(parts))


def rank_hooks(opts: list, hooks: list[dict] | None = None) -> list:
    """Feasible hook options, best first (hook_score); with hooks carrying "robust" (r), r is
    one more margin (robust_part); and an option whose lift from the table failed in rehearsal
    (hanging.table_lift_test) comes after every one that held (a preference, not a filter: as a
    filter it removed as many hooks that worked as hooks that would drop)."""
    def key(o):
        k = hook_score(o) if hooks is None else min(hook_score(o), robust_part(hooks[o.hook]["robust"]))
        held = True if hooks is None else hooks[o.hook].get("table", {}).get((o.twin, o.pitch_deg), True)
        return (held, k)
    return sorted([o for o in opts if o.ok], key=key, reverse=True)


def hook_features(hooks: list[dict], opts: list) -> dict:
    """What the hook checks found: how many hooks the network's proposals gave (that lifted the
    object once), how many (hook, twin) options pass reach / collision / visibility here, and the
    best option's QR size and angle, swing and tilt while hanging, carry pitch, one finger or two,
    and the network's probability for that hook."""
    ok = rank_hooks(opts)
    out = {"hook_n": float(np.log1p(len(hooks))), "hook_any": float(bool(ok)), "hook_n_ok": float(np.log1p(len(ok)))}
    if not ok:
        return {**out, "hook_best": -2.0, "hook_pixels": 0.0, "hook_angle": 90.0 / 45.0, "hook_swing": 0.0,
                "hook_tilt": 0.0, "hook_pitch": 0.0, "hook_one": 0.0, "hook_prob": 0.0}
    o = ok[0]
    return {**out, "hook_best": float(np.clip(hook_score(o), -2.0, 2.0)), "hook_pixels": float(o.vis.pixels) / 100.0, "hook_angle": float(o.vis.angle_deg) / 45.0,
            "hook_swing": float(np.nan_to_num(o.hang.swing_deg)) / 30.0,
            "hook_tilt": float(np.nan_to_num(o.hang.tilt_deg)) / 30.0, "hook_pitch": float(o.pitch_deg) / 40.0,
            "hook_one": float(o.width > 0.01), "hook_prob": float(hooks[o.hook].get("prob", 0.0))}


def scene_features(mu: float, mass: float, extents, finger_force: float = 20.0) -> dict:
    e = np.sort(np.asarray(extents, float))
    return {"mu": mu, "log_mass": float(np.log(mass)), "load": float(mass * G / (2 * mu * finger_force)),
            "ext_max": float(e[-1]) * 10, "ext_min": float(e[0]) * 10}


def vector(rec: dict, variant: str = "robust", features: tuple = FEATURES) -> np.ndarray:
    """The chooser's inputs for one scene; the robustness features are those of the candidates
    the given ranking would run."""
    get = lambda k: rec[f"{k}_{variant}"] if k in ROBUST_KEYS and variant else rec[k]
    return np.array([float(get(k)) for k in features], np.float32)


def wayb_features(wb, net_ranked: list, checked_ranked: list, q: np.ndarray) -> dict:
    """Way B's candidates in this scene: any it can reach, the network's score of its first, how
    many pass every check at this friction and weight, and the first checked one's smallest
    normalised margin and lift margin. wb: the experiment.Scenario of way B's grasps."""
    if wb is None:
        return {"wayb_any": 0.0, "wayb_q": 0.0, "wayb_n_ok": 0.0, "wayb_best": -2.0, "wayb_lift": -2.0}
    f = pinch_features(wb)
    c = lambda x: float(np.clip(x, -2.0, 2.0))
    return {"wayb_any": float(bool(net_ranked)), "wayb_q": float(q[net_ranked[0].grasp]) if net_ranked else 0.0,
            "wayb_n_ok": f["pinch_n_ok"], "wayb_best": c(f["pinch_best"]) if checked_ranked else -2.0,
            "wayb_lift": c(wb.lift_margin[checked_ranked[0].grasp]) if checked_ranked else -2.0}


# ---------------------------------------------------------------- choosers

def rule_choice(rec: dict) -> str:
    """Pinch if a pinch choice passes every check with a positive lift margin at this friction
    and weight; else the hook if one passes its checks; else the pinch (its best effort)."""
    if rec["pinch_n_ok"] > 0 and rec["pinch_lift"] > 0:
        return "pinch"
    if rec["hook_any"] > 0:
        return "hook"
    return "pinch"


@dataclass
class Chooser:
    """One small network predicting each strategy's success (sigmoid outputs, one per strategy)
    from what the checks found (`features`); picks the most likely, the first strategy on ties."""
    mean: np.ndarray
    std: np.ndarray
    state: dict
    margin: float = 0.0                      # another strategy must be this much likelier than the first
    strategies: tuple = ("pinch", "hook")
    features: tuple = FEATURES
    meta: dict = field(default_factory=dict)  # e.g. wayb_model: the way B network its inputs came from

    def net(self):
        import torch
        from torch import nn

        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(len(self.features), 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 32),
                             nn.ReLU(), nn.Linear(32, len(self.strategies)))

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, epochs: int = 400, wd: float = 1e-3, margin: float = 0.0,
            strategies: tuple = ("pinch", "hook"), features: tuple = FEATURES) -> Chooser:
        """X (N, len(features)), Y (N, len(strategies)): did each strategy succeed."""
        import torch

        mean, std = X.mean(0), X.std(0) + 1e-6
        ch = cls(mean, std, {}, margin, tuple(strategies), tuple(features))
        net = ch.net()
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=wd)
        x = torch.as_tensor((X - mean) / std, dtype=torch.float32)
        y = torch.as_tensor(np.asarray(Y, float), dtype=torch.float32)
        for _ in range(epochs):
            net.train()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        ch.state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        return ch

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(N, len(strategies)) predicted success probabilities."""
        import torch

        net = self.net()
        net.load_state_dict(self.state)
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(torch.as_tensor((X - self.mean) / self.std, dtype=torch.float32))).numpy()

    def choose(self, X: np.ndarray) -> np.ndarray:
        """(N,) the chosen strategy's name."""
        p = self.predict(X)
        p[:, 1:] -= self.margin
        return np.array(self.strategies)[p.argmax(1)]

    def save(self, path) -> None:
        import torch

        torch.save({"mean": self.mean, "std": self.std, "state": self.state, "margin": self.margin,
                    "features": self.features, "strategies": self.strategies, "meta": self.meta}, path)

    @classmethod
    def load(cls, path) -> Chooser:
        import torch

        z = torch.load(path, weights_only=False)
        return cls(z["mean"], z["std"], z["state"], z["margin"], tuple(z.get("strategies", ("pinch", "hook"))),
                   tuple(z["features"]), dict(z.get("meta", {})))
