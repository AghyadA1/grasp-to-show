"""Train the transport network on the constraints themselves (self-supervised).

Instead of imitating the optimiser's solutions (scripts/train_transport_net.py), the network
is trained to do what the optimiser does: for each problem, make the motion as short as
possible while the grasp, leverage and robot-limit constraints hold. The loss is computed by
the PyTorch twin of the optimiser's checker (grasp2show/learn/transport_torch.py, identical to
TransportProblem.evaluate to 1e-11), so no labels are needed and the friction can be drawn
fresh for every problem in every batch:

    loss = log T + 0.05 relu(extra jerk) + 30 (max_t v + mean_t v)   for each constraint
    v    = how far the constraint is violated at sample t, with a small margin
           (u <= u_lim - 0.1, limits <= 0.9), so the raw output stays feasible on new problems
           (picked on validation: smaller margins overfit the boundary, see the README)

Judged on the RAW network output only (no slowing down, no refinement, no fallback):
  feasible        share of motions that satisfy every constraint as predicted
  T / T_opt       duration vs the optimiser's, where both are feasible (median)
  score           mean over problems of T_opt / T if feasible else 0: 1.0 = always feasible and
                  as fast as the optimiser
Validation: 5 training-set objects held out (the same as in train_transport_net.py); test: the
10 unseen objects (data/transport_dataset_test), at their recorded friction. The imitation
network is reported on the same problems for comparison.

Usage:  python scripts/train_transport_selfsup.py [--epochs 500] [--device cpu] [--init imitation|scratch|MODEL.pt] [--mass-scale 0.6 2.5]
        (uses data/transport_problems/ too when present: scripts/generate_transport_problems.py)
Outputs: data/models/transport_net_ss.pt, transport_net_ss_metrics.json, media/transport_net_ss.png
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp2show.config import ObjectConfig
from grasp2show.learn.transport_net import TransportNet, features
from grasp2show.learn.transport_torch import Constraints, pack, subset
from grasp2show.objects import object_from_config
from grasp2show.transport import friction_params
from grasp2show.transport_opt import TransportProblem

MU_COL, MASS_COL = 21, 22         # positions in the feature vector (see transport_net.features)
INERTIA_COLS = slice(26, 32)
MU_RANGE = (0.15, 0.6)
DT = torch.float64                # float32 on a GPU (laptop GPUs are slow in float64)


@cache
def obj_model(name: str):
    return object_from_config(ObjectConfig(asset=name))


def load(dirs: list[Path]) -> list[dict]:
    rows = []
    for d in dirs:
        for f in sorted(d.glob("*.npz")):
            z = dict(np.load(f, allow_pickle=True))
            o = obj_model(str(z["object"]))
            rows.append({"file": f"{d.name}/{f.name}", "name": str(z["object"]), "obj": o, "q0": z["q0"], "q1": z["q1"],
                         "T_obj_tcp": z["T_obj_tcp"], "width": float(z["width"]), "mu": float(z["mu"]),
                         "T_opt": float(z["T"]) if "T" in z else np.nan,
                         "opt_feasible": bool(z["feasible"]) if "feasible" in z else False,
                         "T_retime": float(z["T_retime"]) if "T_retime" in z else np.nan,
                         "retime_feasible": bool(z["retime_feasible"]) if "retime_feasible" in z else False,
                         "x": features(z["q0"], z["q1"], float(z["mu"]), z["T_obj_tcp"], float(z["width"]), o.mass,
                                       o.com, o.inertia)})
    return rows


def val_split(rows: list[dict], k: int, seed: int) -> set[str]:
    """The same held-out objects as train_transport_net.py (drawn from objects with feasible labels)."""
    names = sorted({r["name"] for r in rows if r["opt_feasible"]})
    return set(np.random.default_rng(seed).choice(names, min(k, len(names) - 1), replace=False))


class Data:
    """Problem tensors + features on the device; only problems the finger rays can grasp."""

    def __init__(self, rows, device):
        P = pack(rows, dtype=DT, device=device)
        keep = P.pop("index").tolist()
        self.rows = [rows[i] for i in keep]
        self.P = P
        self.X = torch.as_tensor(np.stack([r["x"] for r in self.rows]), dtype=torch.float32, device=device)
        self.n = len(self.rows)


def predict(model, X):
    c, lt = model(X)
    return c.to(DT), torch.exp(lt.to(DT).clamp(np.log(0.2), np.log(30.0)))


def loss_fn(con: Constraints, c, T, P, u_lim, w_lin: float = 10.0, m_u: float = 0.02, m_l: float = 0.02):
    out = con(c, T, P)
    vs = [torch.relu(out["u"] - (u_lim[:, None] - m_u)), torch.relu(out["lim"] - (1 - m_l)), torch.relu(out["jl"] + 0.01)]
    pen = sum(v.amax(1) + v.mean(1) for v in vs)
    loss = torch.log(T) + con.w_smooth * torch.relu(out["jerk_rel"]) + w_lin * pen
    return loss.mean(), out


@torch.no_grad()
def judge_torch(model, con, D: Data) -> dict:
    """Fast in-training judge (same as the numpy one, without collisions)."""
    c, T = predict(model, D.X)
    u_lim = torch.clamp(con.static_u(D.P) + 0.05, min=con.u_lim0)
    feas = con.feasible(con(c, T, D.P), u_lim).cpu().numpy()
    return summarise(feas, T.cpu().numpy(), D.rows)


def summarise(feas, T, rows) -> dict:
    T_opt = np.array([r["T_opt"] for r in rows])
    ok_opt = np.array([r["opt_feasible"] for r in rows])
    both = feas & ok_opt
    score = np.where(feas & ok_opt, T_opt / np.maximum(T, 1e-9), 0.0)[ok_opt]
    ret_ok = np.array([r["retime_feasible"] for r in rows])
    T_ret = np.array([r["T_retime"] for r in rows])
    hard = ~ret_ok
    return {"problems": len(rows), "feasible": round(float(np.mean(feas)), 3),
            "T_over_T_opt_median": round(float(np.median(T[both] / T_opt[both])), 3) if both.any() else None,
            "as_fast_as_opt": round(float(np.mean(feas & (T <= T_opt * 1.0 + 1e-9))), 3),
            "score": round(float(np.mean(score)), 3) if len(score) else None,
            "T_over_T_retime_median": round(float(np.median(T[feas & ret_ok] / T_ret[feas & ret_ok])), 3)
            if (feas & ret_ok).any() else None,
            "retime_impossible": {"n": int(hard.sum()), "net_feasible": int((feas & hard).sum()),
                                  "optimiser_feasible": int((ok_opt & hard).sum())}}


def judge_numpy(model, rows, device) -> tuple[dict, np.ndarray, np.ndarray]:
    """The reference judge: TransportProblem.evaluate on the raw prediction (no collisions)."""
    X = torch.as_tensor(np.stack([r["x"] for r in rows]), dtype=torch.float32, device=device)
    with torch.no_grad():
        c, T = predict(model, X)
    c, T = c.cpu().numpy(), T.cpu().numpy()
    feas = np.array([TransportProblem(r["obj"], r["T_obj_tcp"], r["width"], r["q0"], r["q1"], friction_params(r["mu"]))
                     .evaluate(c[k], float(T[k]), collisions=False)["feasible"] for k, r in enumerate(rows)])
    return summarise(feas, T, rows), feas, T


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, nargs="+", default=[ROOT / "data" / "transport_dataset", ROOT / "data" / "transport_problems"],
                   help="training problem folders (labels not needed; extra geometry-only folders welcome)")
    p.add_argument("--test", type=Path, default=ROOT / "data" / "transport_dataset_test")
    p.add_argument("--init", default="imitation",
                   help="imitation (data/models/transport_net.pt), scratch, or the path of a model to continue from")
    p.add_argument("--mass-scale", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="also train on heavier/lighter versions of the objects: mass and inertia scaled by a "
                        "log-uniform factor in [LO, HI] (half of the problems in each batch; mass capped at 1.5 kg)")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--val-objects", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--w-lin", type=float, default=30.0, help="penalty weight on constraint violations")
    p.add_argument("--margin", type=float, default=0.1, help="safety margin on u and on the limit ratios")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default=None,
                   help="default: float32 on a GPU, float64 on the CPU (the final judge is always float64)")
    p.add_argument("--out", default="transport_net_ss")
    args = p.parse_args()
    global DT
    DT = getattr(torch, args.dtype or ("float32" if args.device.startswith("cuda") else "float64"))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    dev = args.device

    args.data = [d for d in args.data if d.exists()]
    rows = load(args.data)
    val_names = val_split([r for r in rows if r["file"].startswith(args.data[0].name)], args.val_objects, args.seed)
    tr = Data([r for r in rows if r["name"] not in val_names], dev)
    va = Data([r for r in rows if r["name"] in val_names and r["file"].startswith(args.data[0].name)], dev)
    te_rows = load([args.test]) if args.test.exists() else []
    te = Data(te_rows, dev) if te_rows else None
    print(f"problems: train {tr.n}, val {va.n} ({len(val_names)} objects), test {te.n if te else 0}; device {dev}")

    model = TransportNet().to(dev)
    imit = None
    ck_path = ROOT / "data" / "models" / "transport_net.pt"
    if ck_path.exists():
        imit = TransportNet().to(dev)
        imit.load_state_dict(torch.load(ck_path, map_location=dev, weights_only=False)["state_dict"])
        imit.eval()
    if args.init not in ("imitation", "scratch"):
        model.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["state_dict"])
    elif args.init == "imitation":
        if imit is None:
            raise SystemExit("no data/models/transport_net.pt: run train_transport_net.py or use --init scratch")
        model.load_state_dict(imit.state_dict())
    else:
        model.x_mean.copy_(tr.X.mean(0))
        model.x_std.copy_(tr.X.std(0).clamp_min(1e-3))
        last = model.mlp[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        last.bias.data[-1] = float(np.log(3.0))
    lr = args.lr or (1e-3 if args.init == "scratch" else 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    con = Constraints(dtype=DT).to(dev)
    tr_check = Data([tr.rows[i] for i in np.random.default_rng(0).choice(tr.n, min(tr.n, 400), replace=False)], dev)

    hist, best, best_state = [], -1.0, None
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(tr.n, device=dev)
        tot = 0.0
        for i in range(0, tr.n, args.batch):
            b = perm[i:i + args.batch]
            P = subset(tr.P, b)
            mu = torch.as_tensor(rng.uniform(*MU_RANGE, len(b)), dtype=DT, device=dev)
            P["mu"] = mu                                        # fresh friction every time
            X = tr.X[b].clone()
            X[:, MU_COL] = mu.float()
            if args.mass_scale:                                 # heavier / lighter versions of the same objects
                lo, hi = np.log(args.mass_scale)
                k = np.where(rng.random(len(b)) < 0.5, np.exp(rng.uniform(lo, hi, len(b))), 1.0)
                k = torch.as_tensor(k, dtype=DT, device=dev)
                k = torch.minimum(k, 1.5 / P["mass"])
                P["mass"], P["inertia"] = P["mass"] * k, P["inertia"] * k[:, None, None]
                X[:, MASS_COL] *= k.float()
                X[:, INERTIA_COLS] *= k.float()[:, None]
            with torch.no_grad():
                u_lim = torch.clamp(con.static_u(P) + 0.05, min=con.u_lim0)
            c, T = predict(model, X)
            loss, _ = loss_fn(con, c, T, P, u_lim, args.w_lin, args.margin, args.margin)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        el = time.perf_counter() - t0
        print(f"epoch {ep}/{args.epochs}  loss {tot / tr.n:.4f}  ({el / ep:.1f} s/epoch, ~{el / ep * (args.epochs - ep) / 60:.0f} min left)",
              flush=True)
        if ep % args.eval_every == 0 or ep == args.epochs:
            model.eval()
            v = judge_torch(model, con, va)
            t_ = judge_torch(model, con, tr_check)
            hist.append({"epoch": ep, "loss": tot / tr.n, "train_feasible": t_["feasible"],
                         **{f"val_{k}": v[k] for k in ("feasible", "score", "T_over_T_opt_median")}})
            if v["score"] is not None and v["score"] > best:
                best, best_state = v["score"], {k: x.detach().clone() for k, x in model.state_dict().items()}
            print(f"  check: train feasible {t_['feasible']:.2f}  val feasible {v['feasible']:.2f}  T/T_opt "
                  f"{v['T_over_T_opt_median']}  score {v['score']}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    model.load_state_dict(best_state)
    model.eval()

    m = {"train_seconds": round(time.perf_counter() - t0, 1), "selected_val_score": best, "config": {
        k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v) for k, v in vars(args).items()}}
    times = {}
    for name, D in (("val", va), ("test", te)):
        if D is None:
            continue
        m[name], feas, T = judge_numpy(model, D.rows, dev)
        times[name] = (feas, T, np.array([r["T_opt"] for r in D.rows]), np.array([r["opt_feasible"] for r in D.rows]))
        if imit is not None:
            m[f"{name}_imitation"], *_ = judge_numpy(imit, D.rows, dev)
    out = ROOT / "data" / "models"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": m["config"]}, out / f"{args.out}.pt")
    (out / f"{args.out}_metrics.json").write_text(json.dumps({**m, "history": hist}, indent=1))
    print(json.dumps(m, indent=1))
    plot(hist, times, ROOT / "media" / f"{args.out}.png")
    return 0


def plot(hist, times, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    ep = [h["epoch"] for h in hist]
    ax[0].plot(ep, [h["val_feasible"] for h in hist], c="#1d4e6e", label="feasible as predicted")
    ax[0].plot(ep, [h["val_score"] for h in hist], c="#3f8f6b", label="score (1 = optimiser)")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("validation objects, raw network output", fontsize=9)
    for k, col in (("val", "#9aa5b1"), ("test", "#1d4e6e")):
        if k not in times:
            continue
        feas, T, T_opt, ok = times[k]
        s = feas & ok
        ax[1].scatter(T_opt[s], T[s], s=12, c=col, alpha=0.7, label=f"{k}: feasible {feas.mean():.0%}")
    lim = ax[1].get_xlim()[1]
    ax[1].plot([0, lim], [0, lim], ls=":", c="0.5")
    ax[1].set_xlabel("optimiser duration (s)")
    ax[1].set_ylabel("network duration, as predicted (s)")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
