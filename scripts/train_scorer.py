"""Train the learned grasp scorer on full-simulation outcomes and compare it with the checks.

Split by object: the 10 test objects of grasp2show/object_set.json are never seen in training;
--val-objects of the 30 train objects are held out for model selection (best validation AUC).

Reported on the unseen test objects, on exactly the same samples for every method:
  AUC (all samples, and the simulated ones only: the hard cases, all kinematically feasible)
  top-1 success: in each scenario, pick the highest-scored simulated tuple; how often does it
                 succeed? vs the analytic checks' best, a random pick, and an oracle
  speed: scoring time per grasp-target tuple
Usage:
  python scripts/train_scorer.py                    # GPU if available
  python scripts/train_scorer.py --epochs 5         # quick check
  python scripts/train_scorer.py --no-qr-flag       # ablation: hide where the QR is
Outputs: data/models/scorer.pt, data/models/scorer_metrics.json, media/scorer_eval.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp2show.learn.data import Split, load_dataset
from grasp2show.learn.model import Scorer
from grasp2show.objects import load_object_set


def auc(score: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tensors(s: Split, device):
    f = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    return (f(s.obj, torch.long), f(s.R), f(s.t), f(s.glob), f(s.y), f(s.reason, torch.long))


@torch.no_grad()
def predict(model: Scorer, s: Split, device, batch: int = 2048) -> np.ndarray:
    model.eval()
    obj, R, t, g, _, _ = tensors(s, device)
    out = [torch.sigmoid(model(obj[i:i + batch], R[i:i + batch], t[i:i + batch], g[i:i + batch])[0])
           for i in range(0, len(s), batch)]
    return torch.cat(out).cpu().numpy()


def top1(score: np.ndarray, s: Split) -> dict:
    """Mean success of the top-scored simulated tuple per scenario (ties: average over the tied)."""
    m = s.simulated
    res = []
    for sc in np.unique(s.scenario[m]):
        k = m & (s.scenario == sc)
        sk, yk = score[k], s.y[k]
        if not np.isfinite(sk).any():
            res.append(0.0)                     # the method finds nothing it believes in
            continue
        best = sk == np.max(sk)
        res.append(float(yk[best].mean()))
    return {"scenarios": len(res), "success": round(float(np.mean(res)), 3)}


def train(model, tr: Split, va: Split, args, device):
    obj, R, t, g, y, reason = tensors(tr, device)
    pos = float(tr.y.mean())
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-3), device=device))
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, best_state, history = -1.0, None, []
    n = len(tr)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            logit, rlogit = model(obj[b], R[b], t[b], g[b], n_points=args.points)
            loss = bce(logit, y[b]) + args.aux * ce(rlogit, reason[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        p = predict(model, va, device)
        va_auc = auc(p[va.simulated], va.y[va.simulated])
        history.append({"epoch": ep, "loss": round(tot / n, 4), "val_auc_simulated": round(va_auc, 4)})
        print(f"epoch {ep:3d}  loss {tot / n:.4f}  val AUC (simulated) {va_auc:.3f}")
        if va_auc > best:
            best, best_state = va_auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best, history


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "dataset")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--points", type=int, default=512, help="points sampled per cloud while training")
    p.add_argument("--aux", type=float, default=0.3, help="weight of the failure-reason loss")
    p.add_argument("--val-objects", type=int, default=5)
    p.add_argument("--no-qr-flag", action="store_true", help="ablation: hide the QR location")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="scorer")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    clouds, data = load_dataset(args.data)
    oset = load_object_set()
    test_ids = [i for i, n in enumerate(clouds.names) if n in oset["test"]]
    train_ids = [i for i, n in enumerate(clouds.names) if n not in oset["test"]]
    val_ids = list(rng.choice(train_ids, min(args.val_objects, len(train_ids) - 1), replace=False))
    is_test, is_val = np.isin(data.obj, test_ids), np.isin(data.obj, val_ids)
    tr, va, te = data.subset(~is_test & ~is_val), data.subset(is_val), data.subset(is_test)
    print(f"device {device}; samples: train {len(tr)}, val {len(va)}, test {len(te)} "
          f"(simulated {tr.simulated.sum()}/{va.simulated.sum()}/{te.simulated.sum()}); "
          f"success rate (simulated, train) {tr.y[tr.simulated].mean():.2f}")

    model = Scorer(torch.as_tensor(clouds.feats), use_qr_flag=not args.no_qr_flag).to(device)
    t0 = time.perf_counter()
    best_val, history = train(model, tr, va, args, device)
    train_s = time.perf_counter() - t0

    # -- evaluation on unseen objects --------------------------------------------------------
    p_te = predict(model, te, device)
    sim = te.simulated
    rand = rng.random(len(te))
    m = {
        "test_objects": len(test_ids), "train_seconds": round(train_s, 1), "best_val_auc": round(best_val, 3),
        "auc_all": {"learned": round(auc(p_te, te.y), 3), "checks_score": round(auc(te.aware_score, te.y), 3),
                    "checks_binary": round(auc(te.pred_ok.astype(float), te.y), 3)},
        "auc_simulated": {"learned": round(auc(p_te[sim], te.y[sim]), 3),
                          "checks_score": round(auc(te.aware_score[sim], te.y[sim]), 3),
                          "checks_binary": round(auc(te.pred_ok[sim].astype(float), te.y[sim]), 3)},
        "top1": {"learned": top1(p_te, te), "checks": top1(te.aware_score, te),
                 "learned_within_checks": top1(np.where(te.pred_ok, p_te, -np.inf), te),
                 "random": top1(rand, te), "oracle": top1(te.y.astype(float), te)},
        "success_rate_simulated_test": round(float(te.y[sim].mean()), 3),
    }
    # Speed: score 4096 tuples (e.g. 170 grasps x 2 twins x 12 targets) at once.
    big = te.subset(np.arange(min(4096, len(te))))
    for dev in {"cpu", str(device)}:
        mdl = model.to(dev)
        predict(mdl, big, dev)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        predict(mdl, big, dev)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        m[f"us_per_tuple_{dev.split(':')[0]}"] = round((time.perf_counter() - t) / len(big) * 1e6, 2)
    model.to(device)
    m["history"] = history
    m["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    out = ROOT / "data" / "models"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": m["config"], "names": clouds.names},
               out / f"{args.tag}.pt")
    (out / f"{args.tag}_metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: v for k, v in m.items() if k not in ("history", "config")}, indent=1))
    plot(model, te, p_te, m, ROOT / "media" / f"{args.tag}_eval.png")
    return 0


def plot(model, te, p_te, m, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def roc(score, y):
        o = np.argsort(-score, kind="stable")
        y = y[o].astype(bool)
        return np.concatenate([[0], np.cumsum(~y) / max((~y).sum(), 1)]), np.concatenate([[0], np.cumsum(y) / max(y.sum(), 1)])

    sim = te.simulated
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for score, lab, c in ((p_te[sim], f"learned (AUC {m['auc_simulated']['learned']:.2f})", "#2a6f97"),
                          (np.nan_to_num(te.aware_score[sim], neginf=-1e9),
                           f"analytic checks (AUC {m['auc_simulated']['checks_score']:.2f})", "#c1121f")):
        fpr, tpr = roc(score, te.y[sim])
        ax[0].plot(fpr, tpr, label=lab, c=c)
    ax[0].plot([0, 1], [0, 1], ls=":", c="0.6")
    ax[0].set_xlabel("false positive rate")
    ax[0].set_ylabel("true positive rate")
    ax[0].set_title("Unseen objects, simulated tuples", fontsize=9)
    ax[0].legend(fontsize=8, frameon=False)
    names = ["random", "checks", "learned", "learned_within_checks", "oracle"]
    labels = ["random", "analytic\nchecks", "learned", "learned among\ncheck-feasible", "oracle"]
    vals = [m["top1"][k]["success"] for k in names]
    ax[1].bar(range(len(vals)), vals, color=["#9aa5b1", "#c1121f", "#2a6f97", "#1d4e6e", "#cfd8dc"])
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    ax[1].set_xticks(range(len(vals)), labels, fontsize=8)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Top-1 choice per scenario: success", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
