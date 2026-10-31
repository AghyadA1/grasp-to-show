"""How well does the analytic friction model predict the simulated lift labels?

For every labelled grasp in data/grasps/*.npz where the simulated fingers closed on the
object, the model's friction utilisation u (u <= 1: predicted to hold) is compared with the
simulator's success label. Two versions of the model, both with NO fitted parameters (the
simulator's own mu = 0.5, mu_t = 5 mm, 20 N per finger):
  static  : the object's weight only;
  dynamic : worst case over the shake test's motion, with d'Alembert inertial loads
            (stability.utilisation_shake).
Metrics, on the 30 train and the 10 unseen test objects:
  AUC               how well u ranks holding grasps above failing ones (threshold-free);
  balanced accuracy of the prediction u <= 1.
Also a reliability table: simulated success rate per bin of u (all objects).

Writes data/grasps/stability_validation.json and media/stability_validation.png.
"""
from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
from grasp2show.config import ObjectConfig, SceneConfig
from grasp2show.objects import load_object_set, object_from_config
from grasp2show.stability import (
    StabilityParams,
    finger_contacts,
    utilisation,
    utilisation_shake,
)

DATA = ROOT / "data" / "grasps"
BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 4.0, np.inf]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney) formula; higher score = predicted success."""
    r = rankdata(score)
    n1, n0 = label.sum(), len(label) - label.sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(u: np.ndarray, label: np.ndarray) -> dict:
    pred = u <= 1.0
    tp, tn = np.sum(pred & label), np.sum(~pred & ~label)
    fp, fn = np.sum(pred & ~label), np.sum(~pred & label)
    bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    finite = np.where(np.isfinite(u), u, 1e9)
    return {"n": len(u), "auc": round(auc(-finite, label), 3),
            "balanced_accuracy": round(float(bal), 3), "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def evaluate(names: list[str], prm: StabilityParams):
    us, ud, lab, ok_all, closed_all = [], [], [], [], []
    for n in names:
        f = DATA / f"{n}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        obj = object_from_config(ObjectConfig(asset=n))
        T = z["T_obj_tcp"].astype(float)
        c1, c2, ok = finger_contacts(obj.mesh, T)
        ok_all.append(ok)
        closed_all.append(z["closed"])
        sel = ok & z["closed"]
        if not sel.any():
            continue
        us.append(utilisation(c1[sel], c2[sel], T[sel, :3, 1], obj.com, obj.mass, np.eye(3), prm))
        ud.append(utilisation_shake(c1[sel], c2[sel], T[sel], obj.com, obj.mass, obj.inertia, prm))
        lab.append(z["success"][sel])
    if not lab:
        return None
    cat = np.concatenate
    return cat(us), cat(ud), cat(lab), cat(ok_all), cat(closed_all)


def main() -> int:
    cfg = SceneConfig()
    prm = StabilityParams(mu=cfg.obj.friction[0], mu_t=cfg.obj.friction[1], finger_force=cfg.gripper.finger_force)
    oset = load_object_set()
    out = {"params": {"mu": prm.mu, "mu_t_m": prm.mu_t, "finger_force_N": prm.finger_force,
                      "fitted_parameters": 0}}
    all_u, all_l = [], []
    for split in ("train", "test"):
        res = evaluate(oset[split], prm)
        if res is None:
            continue
        us, ud, lab, ok, closed = res
        out[split] = {"static": metrics(us, lab), "dynamic": metrics(ud, lab),
                      "contacts_found_vs_sim_closed": {"agree": round(float(np.mean(ok == closed)), 3)}}
        all_u.append(ud)
        all_l.append(lab)
    if not all_u:
        print("no labelled objects in data/grasps: run scripts/generate_grasps.py first")
        return 1
    u, lab = np.concatenate(all_u), np.concatenate(all_l)
    table = []
    for a, b in pairwise(BINS):
        m = (u >= a) & (u < b)
        if m.any():
            table.append({"u_from": a, "u_to": None if np.isinf(b) else b, "n": int(m.sum()),
                          "sim_success_rate": round(float(lab[m].mean()), 3)})
    out["reliability_dynamic_all_objects"] = table
    (DATA / "stability_validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("params", "train", "test") if k in out}, indent=1))
    _plot(table)
    return 0


def _plot(table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{t['u_from']:g}–{t['u_to']:g}" if t["u_to"] else f">{t['u_from']:g}" for t in table]
    rate = [t["sim_success_rate"] for t in table]
    n = [t["n"] for t in table]
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = ["#2a6f97" if (t["u_to"] or np.inf) <= 1.0 else "#c1121f" for t in table]
    ax.bar(x, rate, color=colors, width=0.75)
    for xi, r, ni in zip(x, rate, n):
        ax.text(xi, r + 0.02, str(ni), ha="center", fontsize=7, color="0.35")
    ax.axhline(0.5, ls=":", c="0.5", lw=1)
    ax.axvline(9.5, ls="--", c="0.3", lw=1)
    ax.text(9.6, 0.93, "friction limit\nu = 1", fontsize=8, va="top")
    ax.set_xticks(x, labels, rotation=45, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("predicted friction utilisation u (dynamic model, no fitted parameters)")
    ax.set_ylabel("simulated success rate")
    ax.set_title("Analytic stability model vs MuJoCo shake test (40 objects; bar labels = grasps)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (ROOT / "media").mkdir(exist_ok=True)
    fig.savefig(ROOT / "media" / "stability_validation.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
