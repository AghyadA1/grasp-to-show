"""Part 3: does the grip model with every contact predict holding better than the point model?

Ground truth: the simulated hold test (gripper_sim.hold_test): close the fingers on the object
held at a random orientation, switch gravity on, keep still 1 s; held = both fingers still on
and slip < 20 mm. To get both outcomes often, every test also draws a friction mu in [0.15, 0.6]
(set in the simulator, as in the transport experiments) and scales the object's mass and inertia
by a log-uniform factor in [0.5, 4].
Predictions, no fitted parameters:
  point   the two-pad point-contact model (stability.utilisation), validated on boxes in part 1
  all     every contact the closed simulated gripper makes (grip_model.utilisation_static)
Reported per shape class: AUC of u for "not held" and balanced accuracy of u <= 1.

Usage:  $env:G2S_OBJECT_SET="v2"; python scripts/validate_grip_model.py [--objects 60] [--grasps 20]
Outputs: data/grip_model_validation/records.jsonl, summary.json, media/grip_model_validation.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
from grasp2show.config import ObjectConfig, SceneConfig
from grasp2show.grasp_sampler import candidates_from_pairs, sample_antipodal
from grasp2show.grip_model import utilisation_static
from grasp2show.gripper_sim import GripperSim, close_contacts, hold_test
from grasp2show.objects import load_object_set, object_from_config
from grasp2show.stability import StabilityParams, finger_contacts, utilisation

OUT = ROOT / "data" / "grip_model_validation"


def random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def run(job):
    name, n_grasps, seed = job
    rng = np.random.default_rng(seed)
    obj = object_from_config(ObjectConfig(asset=name))
    sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj)
    m = sim.model
    body = sim._obj_body
    mass0, inertia0 = m.body_mass[body], m.body_inertia[body].copy()
    fgeoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sim._gripper_bodies | {body}]
    f0 = m.geom_friction[fgeoms].copy()
    p1, p2 = sample_antipodal(obj.mesh, 4 * n_grasps, rng)
    cand = candidates_from_pairs(p1, p2, 4, rng)
    cand = cand.subset(rng.permutation(len(cand)))
    recs = []
    for T, w in zip(cand.T_obj_tcp, cand.width):
        if len(recs) >= n_grasps:
            break
        if sim.in_collision(T, w):
            continue
        mu = float(rng.uniform(0.15, 0.6))
        k = float(np.exp(rng.uniform(np.log(0.5), np.log(4.0))))
        m.geom_friction[fgeoms] = f0
        m.geom_friction[fgeoms, 0], m.geom_friction[fgeoms, 1] = mu, 0.01 * mu
        m.body_mass[body], m.body_inertia[body] = mass0 * k, inertia0 * k
        cs = close_contacts(sim, T, w)
        if cs is None:
            continue
        R = random_rotation(rng)
        g = R.T @ np.array([0.0, 0.0, -9.81])
        u_all = utilisation_static(cs, obj.com, obj.mass * k, g)
        c1, c2, ok = finger_contacts(obj.mesh, T[None])
        u_pt = float(utilisation(c1, c2, T[None][:, :3, 1], obj.com, obj.mass * k, R,
                                 StabilityParams(mu=mu, mu_t=0.01 * mu))[0]) if ok[0] else np.inf
        held, slip, _ = hold_test(sim, T, w, R)
        recs.append({"object": name, "mu": round(mu, 3), "mass": round(obj.mass * k, 3), "contacts": len(cs),
                     "palm_contacts": int((cs.part == 1).sum()), "u_all": u_all, "u_point": u_pt, "held": held,
                     "slip_mm": slip})
    m.body_mass[body], m.body_inertia[body] = mass0, inertia0
    return recs


def auc(score, positive):
    from scipy.stats import rankdata

    s = np.where(np.isfinite(score), score, 1e9)
    pos, neg = s[positive], s[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def scores(rs):
    held = np.array([r["held"] for r in rs])
    out = {"n": len(rs), "held": round(float(held.mean()), 3)}
    for key in ("u_point", "u_all"):
        u = np.array([r[key] for r in rs], float)
        pred = u <= 1
        bacc = 0.5 * (np.mean(pred[held]) + np.mean(~pred[~held])) if held.any() and (~held).any() else float("nan")
        out[key] = {"auc_not_held": round(auc(u, ~held), 3), "balanced_accuracy": round(float(bacc), 3)}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", type=int, default=60)
    p.add_argument("--grasps", type=int, default=20)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    oset = load_object_set()
    rng = np.random.default_rng(0)
    names = []
    for c, members in oset["classes"].items():           # the same number from each class
        names += list(rng.choice(members, min(len(members), args.objects // 3), replace=False))
    jobs = [(n, args.grasps, 100 + i) for i, n in enumerate(names)]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} objects x {args.grasps} grasps, {args.workers} workers", flush=True)
    recs = []
    with Pool(args.workers) as pool:
        for k, rs in enumerate(pool.imap_unordered(run, jobs), 1):
            recs += rs
            if rs:
                print(f"[{k}/{len(jobs)}] {rs[0]['object'][:36]:36s} held {np.mean([r['held'] for r in rs]):.0%}",
                      flush=True)
    (OUT / "records.jsonl").write_text("".join(json.dumps(r, default=float) + "\n" for r in recs))
    cls = {n: c for c, v in oset["classes"].items() for n in v}
    summary = {"all": scores(recs)}
    for c in oset["classes"]:
        rs = [r for r in recs if cls.get(r["object"]) == c]
        if rs:
            summary[c] = scores(rs)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    plot(summary, ROOT / "media" / "grip_model_validation.png")
    return 0


def plot(s, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g for g in ("all", "box", "rounded", "irregular") if g in s]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.36
    for i, (key, lab, col) in enumerate((("u_point", "two-pad point model", "#9aa5b1"),
                                         ("u_all", "every contact", "#1d4e6e"))):
        v = [s[g][key]["auc_not_held"] for g in groups]
        ax.bar(np.arange(len(groups)) + (i - 0.5) * w, v, w, color=col, label=lab)
        for x, y in zip(np.arange(len(groups)) + (i - 0.5) * w, v):
            ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(groups)), [f"{g}\n({s[g]['n']} tests)" for g in groups], fontsize=8)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, ls=":", c="0.6")
    ax.set_ylabel("AUC: predicts which grasps drop")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
