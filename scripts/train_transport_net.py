"""Train the transport network on optimised motions and compare it with the optimiser.

Data: data/transport_dataset/ (training objects; 5 of them held out for validation) and, if
present, data/transport_dataset_test/ (unseen objects). Labels: the optimiser's feasible
shaped motions.

Reported on held-out problems (grasp, friction, leverage and limit constraints exactly as the
optimiser's; collisions are checked later, in full episodes):
  feasible as predicted   share of predicted motions that satisfy every constraint as they are
  T after slowing down    the network's path, uniformly slowed until feasible, vs the optimiser's
                          duration and vs the retimed straight motion
  ms per motion           one forward pass, vs seconds for the optimiser
Usage:  python scripts/train_transport_net.py
Outputs: data/models/transport_net.pt, transport_net_metrics.json, media/transport_net_eval.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp2show.config import ObjectConfig
from grasp2show.learn.transport_net import TransportNet, features
from grasp2show.objects import object_from_config
from grasp2show.transport import friction_params
from grasp2show.transport_opt import TransportProblem


@cache
def obj_model(name: str):
    return object_from_config(ObjectConfig(asset=name))


def load(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.glob("*.npz")):
        z = dict(np.load(f, allow_pickle=True))
        if not bool(z["feasible"]):
            continue
        name = str(z["object"])
        o = obj_model(name)
        z["name"] = name
        z["x"] = features(z["q0"], z["q1"], float(z["mu"]), z["T_obj_tcp"], float(z["width"]), o.mass, o.com, o.inertia)
        rows.append(z)
    return rows


def problem(z) -> TransportProblem:
    return TransportProblem(obj_model(z["name"]), z["T_obj_tcp"], float(z["width"]), z["q0"], z["q1"],
                            friction_params(float(z["mu"])), checker=None)


def slow_to_feasible(pr: TransportProblem, c: np.ndarray, T: float, T_max: float = 20.0) -> float:
    """Smallest uniform duration >= T at which the path c satisfies every constraint (inf: none)."""
    if pr.evaluate(c, T, collisions=False)["feasible"]:
        return T
    if not pr.evaluate(c, T_max, collisions=False)["feasible"]:
        return np.inf
    lo, hi = T, T_max
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if pr.evaluate(c, mid, collisions=False)["feasible"] else (mid, hi)
    return hi


def evaluate(model: TransportNet, rows: list[dict]) -> dict:
    if not rows:
        return {}
    x = torch.as_tensor(np.stack([r["x"] for r in rows]))
    with torch.no_grad():
        model(x[:1])
        t0 = time.perf_counter()
        c, logT = model(x)
        ms = (time.perf_counter() - t0) / len(rows) * 1e3
    c, T = c.numpy(), np.exp(logT.numpy())
    feas, t_fix, t_opt, t_ret = [], [], [], []
    for k, r in enumerate(rows):
        pr = problem(r)
        feas.append(pr.evaluate(c[k], float(T[k]), collisions=False)["feasible"])
        t_fix.append(slow_to_feasible(pr, c[k], float(T[k])))
        t_opt.append(float(r["T"]))
        t_ret.append(float(r["T_retime"]))
    t_fix, t_opt, t_ret = map(np.asarray, (t_fix, t_opt, t_ret))
    ok = np.isfinite(t_fix)
    return {"problems": len(rows), "feasible_as_predicted": round(float(np.mean(feas)), 3),
            "feasible_after_slowing": round(float(ok.mean()), 3),
            "T_net_over_T_opt_median": round(float(np.median(t_fix[ok] / t_opt[ok])), 3),
            "T_net_over_T_retime_median": round(float(np.median(t_fix[ok] / t_ret[ok])), 3),
            "T_opt_over_T_retime_median": round(float(np.median(t_opt / t_ret)), 3),
            "ms_per_motion_cpu": round(ms, 4), "_t": (t_fix.tolist(), t_opt.tolist(), t_ret.tolist())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "transport_dataset")
    p.add_argument("--test", type=Path, default=ROOT / "data" / "transport_dataset_test")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--val-objects", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rows = load(args.data)
    names = sorted({r["name"] for r in rows})
    val_names = set(rng.choice(names, min(args.val_objects, len(names) - 1), replace=False))
    tr = [r for r in rows if r["name"] not in val_names]
    va = [r for r in rows if r["name"] in val_names]
    te = load(args.test) if args.test.exists() else []
    print(f"feasible problems: train {len(tr)}, val {len(va)} ({len(val_names)} objects), test {len(te)}")

    model = TransportNet()
    X = torch.as_tensor(np.stack([r["x"] for r in tr]))
    C = torch.as_tensor(np.stack([r["c"] for r in tr]), dtype=torch.float32)
    LT = torch.as_tensor(np.log([float(r["T"]) for r in tr]), dtype=torch.float32)
    model.x_mean.copy_(X.mean(0))
    model.x_std.copy_(torch.nan_to_num(X.std(0), nan=1.0).clamp_min(1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            c, lt = model(X[b])
            loss = nn.functional.mse_loss(c, C[b]) + nn.functional.mse_loss(lt, LT[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep % 500 == 0:
            print(f"epoch {ep}  loss {loss.item():.4f}", flush=True)
    model.eval()
    m = {"train_seconds": round(time.perf_counter() - t0, 1), "val": evaluate(model, va), "test": evaluate(model, te)}
    out = ROOT / "data" / "models"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(args) | {"data": str(args.data), "test": str(args.test)}},
               out / "transport_net.pt")
    times = {k: v.pop("_t", None) for k, v in m.items() if isinstance(v, dict)}
    (out / "transport_net_metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))
    plot(times, ROOT / "media" / "transport_net_eval.png")
    return 0


def plot(times, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for k, c in (("val", "#9aa5b1"), ("test", "#1d4e6e")):
        if not times.get(k):
            continue
        t_fix, t_opt, _ = map(np.asarray, times[k])
        ok = np.isfinite(t_fix)
        ax.scatter(t_opt[ok], t_fix[ok], s=12, c=c, alpha=0.7, label=f"{k} ({ok.sum()})")
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], ls=":", c="0.5")
    ax.set_xlabel("optimiser duration (s)")
    ax.set_ylabel("network duration, after slowing to feasible (s)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
