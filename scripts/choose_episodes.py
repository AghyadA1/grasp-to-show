"""Hook or pinch: episodes with both, and the chooser trained and tested on them (part 3).

See grasp2show/chooser.py for the idea. Objects: the real scans with something to hook
(object_set_hookreal.json), the v2 scans (mostly without a handle: the chooser must learn that
the hook is no option there) and --gen held-out generated objects (every kind of handle). Each
gets --scenes scenes: start pose, scanner pose (random_camera), friction and weight
(chooser.sample_conditions). In each scene both strategies are planned with their own checks
(the chooser's inputs) and both are executed in the simulator (the chooser's targets).

Stages (all by default; each resumable):
  prepare   grasps (scripts/generate_grasps.py's lift test; 300 contact pairs, rims included, in
            data/grasps_choose: with grasps_v2's 60 a mug kept 10 lifting grasps, now about 80), and the hook network's hooks: its 8 best points (both directions),
            nudged clear and lift-tested, up to 6 that lift, with where the object hangs on each
            (data/hooks_net/<name>.npz);
  episodes  data/choose/records.jsonl (one line per scene: features and both outcomes). Each
            strategy runs its best-ranked candidate (what the robot would do) and, for the record,
            its 2nd and 3rd (how good is its ranking?);
  fit       the chooser, 5-fold cross-validated by object (no object in both training and test):
            success of always-pinch, always-hook, the rule chooser, the learned chooser, the
            oracle; overall, on real and generated objects, and over friction and weight. Then
            trained on everything: data/models/chooser.pt.

Usage:  python scripts/choose_episodes.py --workers 4
        python scripts/choose_episodes.py --smoke          (4 objects, 2 scenes: checks the pipeline)
        python scripts/choose_episodes.py --stage fit      (refit / replot from the records)
Outputs: data/choose/, data/models/chooser.pt, media/chooser.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("G2S_OBJECT_SET", "v2")
from grasp2show.chooser import FEATURES, Chooser, rule_choice, sample_conditions, vector
from grasp2show.parallel import Workers

OUT = ROOT / "data" / "choose"
HOOKS = ROOT / "data" / "hooks_net"
GRASPS = ROOT / "data" / "grasps_choose"     # denser than grasps_v2: 300 contact pairs, rims included
CLOUDS = ROOT / "data" / "hook_clouds"
TOP = 3              # candidates executed per strategy and scene (the first is the one the robot would run)


def crc(s: str) -> int:
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------- objects

def objects(n_gen: int, smoke: bool) -> dict[str, str]:
    """name -> group ("real" / "gen")."""
    real = json.loads((ROOT / "grasp2show" / "object_set_hookreal.json").read_text())["objects"]
    v2 = json.loads((ROOT / "grasp2show" / "object_set_v2.json").read_text())["objects"]
    g = json.loads((ROOT / "grasp2show" / "object_set_gen.json").read_text())
    test = set(g["test"])
    kind = lambda o: ("container-" if o["body"].get("style") == "container" else "") + o["handle"]["where"]
    by = defaultdict(list)
    for o in g["details"]:
        if o["name"] in test:
            by[kind(o)].append(o["name"])
    rng = np.random.default_rng(0)
    gen, kinds = [], sorted(by)
    for i, k in enumerate(kinds):                               # about the same number of each kind
        want = n_gen // len(kinds) + (i < n_gen % len(kinds))
        gen += list(rng.choice(by[k], min(want, len(by[k])), replace=False))
    if smoke:
        from grasp2show.objects import GENERATED_DIR

        real = ["ACE_Coffee_Mug_Kristen_16_oz_cup", "Threshold_Porcelain_Teapot_White"]
        v2 = ["Cole_Hardware_Saucer_Glazed_6"]
        gen = [n for n in gen if (GENERATED_DIR / n).exists()][:1]
    out = {n: "real" for n in list(dict.fromkeys(real + v2))}
    out.update({n: "gen" for n in gen})
    return out


def grasp_job(name: str):
    import generate_grasps

    if (GRASPS / f"{name}.npz").exists():
        return name, "", 0.0
    generate_grasps.OUT = GRASPS
    (GRASPS / "plots").mkdir(parents=True, exist_ok=True)
    try:   # 300 pairs (60 left a mug 10 lifting grasps of 17: its body is too wide, only rim and handle fit)
        r = generate_grasps.process((name, 300, 8, 400, crc(name), False, 0.002))
    except Exception as e:  # noqa: BLE001 (an object that does not load: left out, say why)
        return name, f"{type(e).__name__}: {e}", 0.0
    return name, "", r["seconds"]


def cloud_for(name: str) -> np.ndarray:
    """The hook network's input: 2048 points of the object's outer surface (the self-training cache
    if there, else sampled)."""
    from grasp2show.learn.hooknet import N_POINTS

    f = CLOUDS / f"{name}.v1.npy"
    if not f.exists():
        from self_train_hooks import cloud_job

        cloud_job(name)
    c = np.load(f)
    idx = np.random.default_rng(crc(name)).choice(len(c), N_POINTS, replace=False)
    return c[idx]


def hook_job(args):
    """Lift-test the network's proposals, keep up to 6 distinct ones that lift (most probable
    first), their robustness (chooser.hook_robustness) and where the object hangs on each;
    saved to data/hooks_net/<name>.npz."""
    name, cands = args
    from grasp2show.chooser import hook_robustness
    from grasp2show.config import ObjectConfig, SceneConfig
    from grasp2show.gripper_sim import GripperSim
    from grasp2show.hanging import hang_table, nudge, save_hook_file
    from grasp2show.hooks import lift_test
    from grasp2show.learn.hooknet import anchor
    from grasp2show.objects import object_from_config

    t0 = time.time()
    try:
        obj = object_from_config(ObjectConfig(asset=name))
        sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj)
        hooks = []
        for c in sorted(cands, key=lambda c: -c["prob"]):
            T = nudge(sim, c["T"], c["width"], step=0.003, reach=3, depths=(-0.006, 0.0, 0.006))
            if T is None or not lift_test(sim, T, c["width"], close=False)["success"]:
                continue
            a = anchor(T, c["width"] > 0.01)
            if any(np.linalg.norm(a - anchor(h["T"], h["width"] > 0.01)) < 0.01 and
                   abs(T[:3, 2] @ h["T"][:3, 2]) > 0.9 for h in hooks):
                continue                                          # the same hook again
            hooks.append({"T": T, "width": float(c["width"]), "kind": c["kind"], "prob": float(c["prob"])})
            if len(hooks) == 6:
                break
        for h in hooks:
            h["robust"] = hook_robustness(sim, h["T"], h["width"])
        hang_table(sim, hooks)
        save_hook_file(HOOKS / f"{name}.npz", hooks)
    except Exception as e:  # noqa: BLE001 (an object that does not load: left out, say why)
        return name, -1, time.time() - t0, f"{type(e).__name__}: {e}"
    return name, len(hooks), time.time() - t0, ""


MAX_ROBUST = 150     # pinch grasps tested for robustness per object (the rest count as r = 0.5)


def pinch_robust_job(name: str):
    """Robustness of every grasp that lifts (chooser.pinch_robustness), up to MAX_ROBUST of them:
    data/grasps_choose/<name>.robust.npy (NaN: not tested)."""
    from grasp2show.chooser import pinch_robustness
    from grasp2show.config import ObjectConfig, SceneConfig
    from grasp2show.gripper_sim import GripperSim
    from grasp2show.objects import object_from_config

    f = GRASPS / f"{name}.robust.npy"
    if f.exists():
        return name, "", 0.0
    t0 = time.time()
    try:
        z = np.load(GRASPS / f"{name}.npz")
        obj = object_from_config(ObjectConfig(asset=name))
        sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj)
        r = np.full(len(z["success"]), np.nan)
        idx = np.flatnonzero(z["success"])
        idx = np.random.default_rng(crc(name)).permutation(idx)[:MAX_ROBUST]
        for i in idx:
            r[i] = pinch_robustness(sim, z["T_obj_tcp"][i].astype(float), float(z["width"][i]))
    except Exception as e:  # noqa: BLE001 (an object that does not load: left out, say why)
        return name, f"{type(e).__name__}: {e}", time.time() - t0
    np.save(f, r)
    return name, "", time.time() - t0


# ---------------------------------------------------------------- episodes

POSE_ERR_MM, POSE_ERR_DEG = 2.0, 1.5     # where the object really is vs where the robot thinks (1 sigma)
VARIANTS = ("score", "robust")


def episode(args) -> dict:
    """One scene: both strategies' checks (features) and, for both rankings ("score": the
    checks' margins; "robust": also the candidate's robustness), each strategy's top TOP
    candidates executed with the same unknown error in the object's pose."""
    name, k, group, scale = args
    from grasp2show.chooser import hook_features, pinch_features, rank_hooks, rank_pinch, scene_features
    from grasp2show.config import ObjectConfig, random_camera
    from grasp2show.experiment import Scenario
    from grasp2show.gripper_sim import GripperSim
    from grasp2show.hanging import HOOK_SPEED, hook_options, load_hook_file, plan_hook
    from grasp2show.objects import object_from_config
    from grasp2show.task import Grasp

    t0 = time.time()
    r = np.random.default_rng(crc(name) * 100 + k)
    yaw, xy, cam = float(r.uniform(0, 360)), (float(r.uniform(0.40, 0.60)), float(r.uniform(-0.30, 0.0))), \
        random_camera(r)
    cam = replace(cam, width=int(cam.width * scale), height=int(cam.height * scale))
    mass0 = object_from_config(ObjectConfig(asset=name)).mass
    mu, f = sample_conditions(r, mass0)
    e_xy = np.clip(r.normal(0, POSE_ERR_MM / 1000, 2), -2.5 * POSE_ERR_MM / 1000, 2.5 * POSE_ERR_MM / 1000)
    e_yaw = float(np.clip(r.normal(0, np.radians(POSE_ERR_DEG)), -np.radians(2.5 * POSE_ERR_DEG),
                          np.radians(2.5 * POSE_ERR_DEG)))
    off = (float(e_xy[0]), float(e_xy[1]), e_yaw)
    z = np.load(GRASPS / f"{name}.npz")
    sc = Scenario(name, yaw, {kk: z[kk] for kk in ("T_obj_tcp", "width", "success")}, start_xy=xy, camera=cam,
                  mass_scale=f, mu=mu)
    sc.lift_margin = np.where(sc.lift_ok, sc.lift_margin, -1.0)   # this friction and weight, not clipped
    f_r = GRASPS / f"{name}.robust.npy"
    R = np.nan_to_num(np.load(f_r), nan=0.5) if f_r.exists() else np.full(len(sc.T), 0.5)
    rec = {"object": name, "scene": k, "group": group, "mu": mu, "mass_scale": f, "mass_kg": round(sc.obj.mass, 3),
           "pose_error": [round(off[0] * 1000, 2), round(off[1] * 1000, 2), round(float(np.degrees(off[2])), 2)],
           **scene_features(mu, sc.obj.mass, sc.obj.mesh.extents), **pinch_features(sc)}

    t1 = time.time()
    done: dict = {}
    for v in VARIANTS:
        ranked = rank_pinch(sc, R if v == "robust" else None)
        tries, why = [], "no grasp"
        for c in ranked:
            key = (c.grasp, c.twin, c.roll_idx)
            if key not in done:
                done[key] = sc._execute(c, obj_offset=off)
            e = done[key]
            if e.reason.startswith("plan"):
                why = why if tries else e.reason
                continue
            tries.append((bool(e.success), e.reason, float(R[c.grasp])))
            if len(tries) == TOP:
                break
        rec.update({f"pinch_success_{v}": bool(tries and tries[0][0]), f"pinch_reason_{v}": tries[0][1] if tries else why,
                    f"pinch_tries_{v}": [t[0] for t in tries], f"pinch_r_{v}": tries[0][2] if tries else 0.0})
    rec["pinch_s"] = round(time.time() - t1, 1)

    t1 = time.time()
    f_h = HOOKS / f"{name}.npz"
    hooks = load_hook_file(f_h) if f_h.exists() else []
    opts = hook_options(sc.fc, GripperSim(sc.cfg, sc.obj), hooks, T_start=sc.ep.resting_pose()) if hooks else []
    rec.update(hook_features(hooks, opts))
    done = {}
    for v in VARIANTS:
        ok = rank_hooks(opts, hooks if v == "robust" else None)
        tries, h_why = [], "no hook" if not hooks else "no hook passes the checks" if not ok else "plan"
        for o in ok[:12]:
            key = (o.hook, o.twin, o.pitch_deg, o.extra_deg)
            if key not in done:
                paths, why = plan_hook(sc.ep, sc.fc, o)
                if paths is None:
                    done[key] = (None, f"plan: {why}")
                else:
                    res = sc.ep.execute(Grasp(f"hook{o.hook}", o.T_obj_tcp, o.width, "hook"), paths, speed=HOOK_SPEED,
                                        mu=mu, obj_offset=off)
                    done[key] = (bool(res.success), "" if res.success else
                                 "collision" if res.collisions else "dropped" if not res.held else "qr")
            ok_, why = done[key]
            if ok_ is None:
                h_why = h_why if tries else why
                continue
            tries.append((ok_, why, float(hooks[o.hook].get("robust", 0.5))))
            if len(tries) == TOP:
                break
        rec.update({f"hook_success_{v}": bool(tries and tries[0][0]), f"hook_reason_{v}": tries[0][1] if tries else h_why,
                    f"hook_tries_{v}": [t[0] for t in tries], f"hook_r_{v}": tries[0][2] if tries else 0.0})
    rec.update(hook_s=round(time.time() - t1, 1), seconds=round(time.time() - t0, 1))
    return rec


# ---------------------------------------------------------------- fit and report

def folds(names: list[str], k: int = 5) -> dict[str, int]:
    return {n: crc("fold" + n) % k for n in names}


def evaluate(recs: list[dict], v: str = "robust", margin: float = 0.0) -> tuple[dict, np.ndarray]:
    """The chooser for ranking v, cross-validated by object, against the fixed policies."""
    X = np.stack([vector(r, v) for r in recs])
    yp = np.array([r[f"pinch_success_{v}"] for r in recs], float)
    yh = np.array([r[f"hook_success_{v}"] for r in recs], float)
    fold = folds(sorted({r["object"] for r in recs}))
    fi = np.array([fold[r["object"]] for r in recs])
    choice = np.empty(len(recs), "<U5")
    prob = np.zeros((len(recs), 2))
    for f in range(5):
        tr, te = fi != f, fi == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        ch = Chooser.fit(X[tr], yp[tr], yh[tr], margin=margin)
        prob[te] = ch.predict(X[te])
        choice[te] = ch.choose(X[te])
    rule = np.array([rule_choice(r) for r in recs])
    out = {}
    for grp in ("all", "real", "gen"):
        m = np.array([grp == "all" or r["group"] == grp for r in recs])
        if not m.any():
            continue
        succ = lambda c, m=m: float(np.mean(np.where(c[m] == "hook", yh[m], yp[m])))
        any3 = lambda key, m=m: float(np.mean([any(r.get(key, [])) for r, mm in zip(recs, m) if mm]))
        out[grp] = {"scenes": int(m.sum()), "objects": len({r["object"] for r, mm in zip(recs, m) if mm}),
                    "always_pinch": float(yp[m].mean()), "always_hook": float(yh[m].mean()),
                    "rule": succ(rule), "learned": succ(choice), "oracle": float(np.maximum(yp, yh)[m].mean()),
                    "learned_hook_share": float(np.mean(choice[m] == "hook")),
                    "pinch_any_of_top3": any3(f"pinch_tries_{v}"), "hook_any_of_top3": any3(f"hook_tries_{v}"),
                    "only_hook_works": float(np.mean((yh > yp)[m])), "only_pinch_works": float(np.mean((yp > yh)[m]))}
    return out, np.stack([prob[:, 0], prob[:, 1], choice == "hook", yp, yh], 1)


def ranking_table(recs: list[dict]) -> dict:
    """Score vs robust ranking, per strategy: success of the candidate each would run, over the
    scenes where that strategy had a candidate; and where the two picked different ones."""
    out = {}
    for s in ("pinch", "hook"):
        m = [r for r in recs if r[f"{s}_tries_score"] or r[f"{s}_tries_robust"]]
        diff = [r for r in m if r[f"{s}_r_score"] != r[f"{s}_r_robust"] or
                r[f"{s}_tries_score"] != r[f"{s}_tries_robust"]]
        out[s] = {"scenes_with_a_candidate": len(m),
                  "score": float(np.mean([r[f"{s}_success_score"] for r in m])) if m else float("nan"),
                  "robust": float(np.mean([r[f"{s}_success_robust"] for r in m])) if m else float("nan"),
                  "scenes_where_they_differ": len(diff),
                  "score_where_differ": float(np.mean([r[f"{s}_success_score"] for r in diff])) if diff else float("nan"),
                  "robust_where_differ": float(np.mean([r[f"{s}_success_robust"] for r in diff])) if diff else float("nan"),
                  "r_of_choice_score": float(np.mean([r[f"{s}_r_score"] for r in m])) if m else float("nan"),
                  "r_of_choice_robust": float(np.mean([r[f"{s}_r_robust"] for r in m])) if m else float("nan")}
    return out


def plot(recs: list[dict], res: dict, pred: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axs[0]
    pols = ("always_pinch", "always_hook", "rule", "learned", "oracle")
    cols = ("#4C78A8", "#E45756", "#B279A2", "#54A24B", "#9D9D9D")
    groups = [g for g in ("all", "real", "gen") if g in res]
    w = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        for pi, (p, c) in enumerate(zip(pols, cols)):
            v = res[g][p] * 100
            ax.bar(pi + (gi - (len(groups) - 1) / 2) * w, v, w * 0.95, color=c, alpha=1.0 - 0.3 * gi)
            ax.text(pi + (gi - (len(groups) - 1) / 2) * w, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(pols)), [p.replace("_", "\n") for p in pols])
    ax.set_ylabel("episodes succeeded (%)")
    ax.set_title("Held-out objects (5-fold by object); bars: " + " / ".join(groups), fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    ax = axs[1]
    mu = np.array([r["mu"] for r in recs])
    mass = np.array([r["mass_kg"] for r in recs])
    yp, yh = pred[:, 3].astype(bool), pred[:, 4].astype(bool)
    cat = np.where(yp & yh, 0, np.where(yp, 1, np.where(yh, 2, 3)))
    for c, (lab, col) in enumerate((("both", "#9D9D9D"), ("only pinch", "#4C78A8"), ("only hook", "#E45756"),
                                    ("neither", "#E0E0E0"))):
        m = cat == c
        ax.scatter(mu[m], mass[m], s=10 if c < 3 else 5, c=col, label=f"{lab} ({m.sum()})", alpha=0.8, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("friction mu")
    ax.set_ylabel("mass (kg)")
    ax.set_title("Which strategy succeeded, by friction and weight", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axs[2]
    load = np.array([r["load"] for r in recs])
    bins = np.quantile(load, np.linspace(0, 1, 7))
    idx = np.clip(np.digitize(load, bins[1:-1]), 0, 5)
    ch = pred[:, 2].astype(bool)
    learned = np.where(ch, yh, yp)
    for y, lab, col in ((yp, "always pinch", "#4C78A8"), (yh, "always hook", "#E45756"),
                        (learned, "learned chooser", "#54A24B"), (np.maximum(yp, yh), "oracle", "#9D9D9D")):
        ax.plot([np.median(load[idx == b]) for b in range(6)], [y[idx == b].mean() * 100 for b in range(6)], "o-",
                color=col, label=lab, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("load on a pinch: m g / (2 mu F)  (>1: friction cannot hold it, even centred)")
    ax.set_ylabel("episodes succeeded (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Success as the pinch gets harder", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# ---------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "prepare", "episodes", "fit"), default="all")
    p.add_argument("--scenes", type=int, default=6, help="scenes per object")
    p.add_argument("--gen", type=int, default=120, help="held-out generated objects")
    p.add_argument("--scanner-scale", type=float, default=2.0, help="1280 x 960 scanner (part 3's choice)")
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "hook_net_self.pt")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    out = OUT / ("smoke" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.scenes = 2
    log_f = open(out / "log.txt", "a", encoding="utf-8")  # noqa: SIM115 (open for the whole run)

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    objs = objects(args.gen, args.smoke)
    log(f"== chooser {time.strftime('%Y-%m-%d %H:%M')}: {sum(g == 'real' for g in objs.values())} real and "
        f"{sum(g == 'gen' for g in objs.values())} generated objects, {args.scenes} scenes each")
    rec_f = out / "records_v2.jsonl"       # v2: both rankings, pose error
    if args.stage in ("all", "prepare", "episodes"):
        from self_train_hooks import download_missing

        names = download_missing(sorted(objs))
        pool = Workers(args.workers)
        try:
            if args.stage in ("all", "prepare"):
                todo = [n for n in names if not (GRASPS / f"{n}.npz").exists()]
                log(f"grasps: {len(todo)} objects to label")
                for i, (n, r) in enumerate(pool.run(grasp_job, todo), 1):
                    log(f"  [{i}/{len(todo)}] {n}: " + ("lost" if r is None else r[1] or f"{r[2]:.0f} s"))
                todo = [n for n in names if (GRASPS / f"{n}.npz").exists() and not (GRASPS / f"{n}.robust.npy").exists()]
                log(f"pinch robustness: {len(todo)} objects")
                for i, (n, r) in enumerate(pool.run(pinch_robust_job, todo), 1):
                    log(f"  [{i}/{len(todo)}] {n}: " + ("lost" if r is None else r[1] or f"{r[2]:.0f} s"))
                import torch

                from grasp2show.learn.hooknet import load_hook_net, predict_hooks

                dev = "cuda" if torch.cuda.is_available() else "cpu"
                net = load_hook_net(args.model, dev)
                todo = [n for n in names if not (HOOKS / f"{n}.npz").exists()
                        or "robust" not in np.load(HOOKS / f"{n}.npz").files]
                log(f"hooks: {len(todo)} objects")
                jobs = []
                for n in todo:
                    try:
                        jobs.append((n, [{k: h[k] for k in ("T", "width", "kind", "prob")}
                                         for h in predict_hooks(net, cloud_for(n), k=8, min_prob=0.0, device=dev)]))
                    except Exception as e:  # noqa: BLE001 (no cloud: no hooks; say why)
                        log(f"  {n}: no cloud ({type(e).__name__}: {e})")
                for i, (job, r) in enumerate(pool.run(hook_job, jobs), 1):
                    log(f"  [{i}/{len(jobs)}] {job[0]}: " + ("lost" if r is None else r[3] or
                                                             f"{r[1]} hooks ({r[2]:.0f} s)"))
            if args.stage in ("all", "episodes"):
                done = set()
                if rec_f.exists():
                    done = {(d["object"], d["scene"]) for d in map(json.loads, rec_f.read_text().splitlines())}
                ready = [n for n in names if (GRASPS / f"{n}.npz").exists()]
                jobs = [(n, k, objs[n], args.scanner_scale) for n in ready for k in range(args.scenes)
                        if (n, k) not in done]
                log(f"episodes: {len(jobs)} scenes to run ({len(done)} done before)")
                t0 = time.time()
                with open(rec_f, "a", encoding="utf-8") as fh:
                    for i, (job, r) in enumerate(pool.run(episode, jobs), 1):
                        if r is None:
                            log(f"  {job[0]} scene {job[1]}: lost (worker crashed or hung)")
                            continue
                        fh.write(json.dumps(r, default=float) + "\n")
                        fh.flush()
                        o_ = lambda s, v, r=r: "ok" if r[f"{s}_success_{v}"] else r[f"{s}_reason_{v}"][:18]
                        log(f"  [{i}/{len(jobs)}] {r['object'][:30]:30s} mu {r['mu']:.2f} {r['mass_kg']:.2f} kg: pinch "
                            f"{o_('pinch', 'score')} / {o_('pinch', 'robust')}, hook {o_('hook', 'score')} / "
                            f"{o_('hook', 'robust')} ({r['seconds']:.0f} s)")
                log(f"episodes done ({(time.time() - t0) / 60:.0f} min)")
        finally:
            pool.close()
    recs = [json.loads(x) for x in rec_f.read_text().splitlines()] if rec_f.exists() else []
    if len({r["object"] for r in recs}) < 5:
        log(f"{len(recs)} scenes: too few objects to cross-validate a chooser")
        return 0
    results = {"ranking": ranking_table(recs)}
    for s_, t in results["ranking"].items():
        log(f"{s_:5s} ranking, success of the candidate it would run: score {t['score']:.0%}, robust {t['robust']:.0%} "
            f"({t['scenes_with_a_candidate']} scenes; they picked differently in {t['scenes_where_they_differ']}: "
            f"score {t['score_where_differ']:.0%} vs robust {t['robust_where_differ']:.0%}; robustness of the "
            f"chosen candidate {t['r_of_choice_score']:.2f} vs {t['r_of_choice_robust']:.2f})")
    preds = {}
    for v in VARIANTS:
        results[v], preds[v] = evaluate(recs, v)
        for g, x in results[v].items():
            log(f"[{v}] {g:4s} ({x['scenes']} scenes, {x['objects']} objects): always pinch {x['always_pinch']:.0%}, "
                f"always hook {x['always_hook']:.0%}, rule {x['rule']:.0%}, learned {x['learned']:.0%} (hook chosen "
                f"{x['learned_hook_share']:.0%}), oracle {x['oracle']:.0%}; only the hook worked in "
                f"{x['only_hook_works']:.0%}, only the pinch in {x['only_pinch_works']:.0%}; any of the top 3: "
                f"pinch {x['pinch_any_of_top3']:.0%}, hook {x['hook_any_of_top3']:.0%}")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    best = max(VARIANTS, key=lambda v: results[v]["all"]["learned"])
    for v in VARIANTS:
        plot(recs, results[v], preds[v], ROOT / "media" / f"chooser_{v}{'_smoke' if args.smoke else ''}.png")
    X = np.stack([vector(r, best) for r in recs])
    ch = Chooser.fit(X, np.array([r[f"pinch_success_{best}"] for r in recs], float),
                     np.array([r[f"hook_success_{best}"] for r in recs], float))
    if not args.smoke:
        ch.save(ROOT / "data" / "models" / "chooser.pt")
    log(f"better ranking: {best}; features: {', '.join(FEATURES)}; wrote data/models/chooser.pt, "
        f"media/chooser_score.png, media/chooser_robust.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
