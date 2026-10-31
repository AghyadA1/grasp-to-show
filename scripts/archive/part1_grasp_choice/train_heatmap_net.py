"""Train the dense goal-conditioned heatmap network (way B) and evaluate it on unseen objects.

Data: data/dense/ (scripts/generate_dense_labels.py) on the point clouds of data/dataset/clouds.npz.
Split by object as everywhere: the 10 test objects are never seen; --val-objects of the training
objects select the epoch.

Reported on the unseen objects (every start pose x target):
  point AUC      does the predicted per-point quality separate feasible from infeasible points?
  top-1          the point the network ranks first: is a feasible grasp there for this target?
                 (no candidate touching the point counts as no) vs a random labelled point, the
                 target-blind stability map, and way A (all checks on dense candidates: the oracle)
  grasp error    on feasible points: approach angle, TCP offset, width vs the best label grasp
  speed          one forward pass (all 1,024 points, one target)
Usage:  python scripts/train_heatmap_net.py            (GPU if available)
        python scripts/train_heatmap_net.py --epochs 3 (quick check)
Outputs: data/models/heatmap_net.pt, heatmap_net_metrics.json, media/heatmap_net_eval.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
from train_scorer import auc

from grasp2show.config import CameraConfig
from grasp2show.learn.data import load_clouds
from grasp2show.learn.dense import DenseSet, HeatmapNet, load_dense, split_outputs
from grasp2show.objects import load_object_set


def tensors(s: DenseSet, device):
    f = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    return f(s.obj, torch.long), f(s.cond), f(np.nan_to_num(s.score, nan=-1.0)), f(s.params)


def loss_fn(out: dict, score: torch.Tensor, params: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    known = score >= 0
    feas = score > 0
    bce = nn.functional.binary_cross_entropy_with_logits(out["logit"][known], feas[known].float(),
                                                         pos_weight=pos_weight)
    if not feas.any():
        return bce
    a, c, off, w = params[..., 0:3], params[..., 3:6], params[..., 6:9], params[..., 9]
    reg = (nn.functional.l1_loss(out["score"][feas], score[feas])
           + (1 - (out["approach"] * a).sum(-1))[feas].mean()
           + (1 - (out["closing"] * c).sum(-1).abs())[feas].mean()        # closing axis: sign-free (twins)
           + nn.functional.smooth_l1_loss(out["offset"][feas] * 100, off[feas] * 100)   # cm
           + nn.functional.l1_loss(out["width"][feas] * 100, w[feas] * 100))
    return bce + reg


@torch.no_grad()
def predict(model: HeatmapNet, s: DenseSet, device, batch: int = 16) -> dict:
    model.eval()
    obj, cond, _, _ = tensors(s, device)
    outs = [split_outputs(model(obj[i:i + batch], cond[i:i + batch])) for i in range(0, len(s), batch)]
    return {k: torch.cat([o[k] for o in outs]).cpu().numpy() for k in outs[0]}


def quality(pred: dict) -> np.ndarray:
    """Ranking value per point: P(feasible) x predicted score."""
    return 1 / (1 + np.exp(-pred["logit"])) * pred["score"]


def evaluate(pred: dict, s: DenseSet, rng) -> dict:
    q = quality(pred)
    known = ~np.isnan(s.score)
    feas = np.nan_to_num(s.score) > 0
    m = {"samples": len(s), "labelled_points": int(known.sum()), "feasible_fraction": round(float(feas[known].mean()), 3),
         "point_auc": round(auc(q[known], feas[known]), 3)}

    def top1(values):   # is there a feasible grasp at the top-ranked point? (unlabelled = no)
        i = values.argmax(1)
        return round(float(feas[np.arange(len(s)), i].mean()), 3)

    has = feas.any(1)
    m["top1"] = {"network": top1(q), "stability_map": top1(np.nan_to_num(s.stab, nan=-1.0) + 1e-6 * rng.random(q.shape)),
                 "random": top1(np.where(known, rng.random(q.shape), -1.0)), "way_A_oracle": round(float(has.mean()), 3)}
    a = s.params[..., 0:3]
    ang = np.degrees(np.arccos(np.clip((pred["approach"] * a).sum(-1), -1, 1)))[feas]
    m["grasp_error_on_feasible_points"] = {
        "approach_deg_median": round(float(np.median(ang)), 1),
        "offset_mm_median": round(float(np.median(np.linalg.norm(pred["offset"] - s.params[..., 6:9], axis=-1)[feas]) * 1e3), 1),
        "width_mm_median": round(float(np.median(np.abs(pred["width"] - s.params[..., 9])[feas]) * 1e3), 1)}
    return m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dense", type=Path, default=ROOT / "data" / "dense")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--radius", type=float, default=0.008)
    p.add_argument("--val-objects", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="heatmap_net")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    clouds = load_clouds(args.clouds)
    t0 = time.perf_counter()
    data = load_dense(args.dense, clouds, CameraConfig().pos, args.radius)
    print(f"{len(data)} samples (object x start pose x target), labels in {time.perf_counter() - t0:.0f} s")
    oset = load_object_set()
    test_ids = [i for i, n in enumerate(clouds.names) if n in oset["test"]]
    train_ids = [i for i in np.unique(data.obj) if clouds.names[i] not in oset["test"]]
    val_ids = list(rng.choice(train_ids, args.val_objects, replace=False))
    is_te, is_va = np.isin(data.obj, test_ids), np.isin(data.obj, val_ids)
    tr, va, te = data.subset(~is_te & ~is_va), data.subset(is_va), data.subset(is_te)
    print(f"device {device}; samples train {len(tr)}, val {len(va)}, test {len(te)}")

    model = HeatmapNet(torch.as_tensor(clouds.feats)).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    obj, cond, score, params = tensors(tr, device)
    fr = float(np.nanmean(np.nan_to_num(tr.score, nan=np.nan) > 0))
    pos_weight = torch.tensor((1 - fr) / max(fr, 1e-3), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, best_state, history = -1.0, None, []
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), device=device)
        tot = 0.0
        for i in range(0, len(tr), args.batch):
            b = perm[i:i + args.batch]
            loss = loss_fn(split_outputs(model(obj[b], cond[b])), score[b], params[b], pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        mv = evaluate(predict(model, va, device), va, rng)
        history.append({"epoch": ep, "loss": round(tot / len(tr), 4), "val_point_auc": mv["point_auc"],
                        "val_top1": mv["top1"]["network"]})
        print(f"epoch {ep:3d}  loss {tot / len(tr):.4f}  val point AUC {mv['point_auc']:.3f}  "
              f"val top-1 {mv['top1']['network']:.2f}", flush=True)
        key = mv["point_auc"] + mv["top1"]["network"]
        if key > best:
            best, best_state = key, {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)

    m = evaluate(predict(model, te, device), te, rng)
    m["train_seconds"] = round(train_s, 1)
    one = te.subset(np.arange(1))
    for dev in sorted({"cpu", str(device)}):
        mdl = model.to(dev)
        predict(mdl, one, dev)
        n = 20
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            predict(mdl, one, dev)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        m[f"ms_per_heatmap_{dev.split(':')[0]}"] = round((time.perf_counter() - t) / n * 1e3, 2)
    model.to(device)
    m["history"] = history
    m["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    out = ROOT / "data" / "models"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": m["config"], "names": clouds.names}, out / f"{args.tag}.pt")
    (out / f"{args.tag}_metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: v for k, v in m.items() if k not in ("history", "config")}, indent=1))
    plot(m, ROOT / "media" / f"{args.tag}_eval.png")
    return 0


def plot(m, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["random", "stability_map", "network", "way_A_oracle"]
    labels = ["random point", "stability map\n(target-blind)", "heatmap network\n(way B)", "way A\n(all checks, slow)"]
    vals = [m["top1"][k] for k in keys]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.bar(range(4), vals, 0.6, color=["#cfd8dc", "#9aa5b1", "#1d4e6e", "#c1121f"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xticks(range(4), labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("best point has a feasible grasp")
    ax.set_title(f"Unseen objects, {m['samples']} object x start x target cases; point AUC {m['point_auc']:.2f}",
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
