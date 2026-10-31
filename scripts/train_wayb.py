"""Train way B v5: the dense goal-conditioned heatmap network (grasp2show/learn/dense.py) on the
labels of scripts/generate_wayb_labels.py, told each scene's friction and weight.

v4 (scripts/archive/part1_grasp_choice/train_heatmap_net.py) saw 81 scans, each at its own weight
and friction 1. v5 sees the v2 and extra scans' training splits and the generated objects'
training split, every scene at its own friction and weight, and takes the friction and the load
on a pinch as two more inputs (dense.condition, 27 numbers). Same network otherwise, same loss.

Split by object: the v2 and extra scans' test objects are never trained on (the metrics below are
on them); --val-objects of the training objects choose the epoch. The evaluation of full
episodes is scripts/evaluate.py --stage wayb --wayb-model data/models/heatmap_net_v5.pt.

The labels are turned into per-point targets once and cached (data/dense_v5/targets.npz, half
precision: ~2 GB for 1,300 objects x 6 scenes x 12 targets); training reads batches from it.

Usage:  python scripts/train_wayb.py                     (GPU if available; ~20-40 epochs, hours)
        python scripts/train_wayb.py --epochs 2 --limit 20   (quick check)
Outputs: data/models/heatmap_net_v5.pt, heatmap_net_v5_metrics.json, media/heatmap_net_v5_eval.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts" / "archive" / "part1_grasp_choice"))
os.environ.setdefault("G2S_OBJECT_SET", "v2")
from train_heatmap_net import evaluate, loss_fn, plot, predict

from grasp2show.learn.dense import N_COND_MU, DenseSet, HeatmapNet, condition, point_labels, split_outputs

DENSE = ROOT / "data" / "dense_v5"


def splits() -> tuple[set, set]:
    """(test objects, objects never to train on) and nothing else: everything labelled is
    training data except the v2 and extra scans' test splits."""
    v2 = json.loads((ROOT / "grasp2show" / "object_set_v2.json").read_text())
    more = json.loads((ROOT / "grasp2show" / "object_set_more.json").read_text())
    return set(v2["test"]) | set(more["test"]), set(more["test"])


def build_targets(dense: Path, radius: float, limit: int | None) -> dict:
    """Per (scene file, target): object index, condition, per-point score and grasp (float16)."""
    from grasp2show.pipeline import wayb_cloud

    files = sorted(dense.glob("*__s*.npz"))
    names = sorted({f.name.split("__")[0] for f in files})[:limit]
    keep = set(names)
    files = [f for f in files if f.name.split("__")[0] in keep]
    t0 = time.time()
    feats = []
    for i, n in enumerate(names):
        feats.append(wayb_cloud(n))
        if i % 100 == 99:
            print(f"  clouds {i + 1}/{len(names)} ({time.time() - t0:.0f} s)", flush=True)
    feats = np.stack(feats).astype(np.float32)
    idx = {n: i for i, n in enumerate(names)}
    M = 12 * len(files)
    out = {"names": np.array(names), "feats": feats, "obj": np.zeros(M, np.int32),
           "cond": np.zeros((M, N_COND_MU), np.float32), "score": np.full((M, feats.shape[1]), np.nan, np.float16),
           "params": np.zeros((M, feats.shape[1], 10), np.float16), "stab": np.zeros((M, feats.shape[1]), np.float16),
           "yaw": np.zeros(M, np.float32), "roll_idx": np.zeros(M, np.int16), "mu": np.zeros(M, np.float32),
           "mass_scale": np.zeros(M, np.float32)}
    for j, f in enumerate(files):
        o = idx[f.name.split("__")[0]]
        d = dict(np.load(f))
        score, params, stab = point_labels(feats[o, :, :3], d, radius)
        for r in range(12):
            m = 12 * j + r
            out["obj"][m], out["score"][m], out["params"][m], out["stab"][m] = o, score[r], params[r], stab
            out["cond"][m] = condition(d["T_start"], d["T_target"][r], d["cam_pos"], float(d["mass"]), d["com"],
                                       float(d["mu"]))
            out["yaw"][m], out["roll_idx"][m] = float(d["yaw"]), r
            out["mu"][m], out["mass_scale"][m] = float(d["mu"]), float(d["mass_scale"])
        if j % 500 == 499:
            print(f"  targets {j + 1}/{len(files)} scene files ({time.time() - t0:.0f} s)", flush=True)
    return out


def as_set(t: dict, m: np.ndarray) -> DenseSet:
    return DenseSet(t["obj"][m], t["cond"][m], t["score"][m].astype(np.float32), t["params"][m].astype(np.float32),
                    t["stab"][m].astype(np.float32), t["yaw"][m], t["roll_idx"][m])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dense", type=Path, default=DENSE)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--radius", type=float, default=0.008)
    p.add_argument("--val-objects", type=int, default=40)
    p.add_argument("--limit", type=int, default=None, help="only the first N objects (quick check)")
    p.add_argument("--rebuild", action="store_true", help="rebuild the cached targets (new labels)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="heatmap_net_v5")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    cache = args.dense / ("targets.npz" if args.limit is None else f"targets_{args.limit}.npz")
    if args.rebuild or not cache.exists():
        print(f"building per-point targets from {args.dense} ...", flush=True)
        np.savez(cache, **build_targets(args.dense, args.radius, args.limit))
    t = dict(np.load(cache))
    names = [str(n) for n in t["names"]]
    test_names, _ = splits()
    is_test = np.array([n in test_names for n in names])
    train_ids = np.flatnonzero(~is_test)
    val_ids = rng.choice(train_ids, min(args.val_objects, max(1, len(train_ids) // 10)), replace=False)
    te_m, va_m = is_test[t["obj"]], np.isin(t["obj"], val_ids)
    tr_m = ~te_m & ~va_m
    te, va = as_set(t, te_m), as_set(t, va_m)
    tr = np.flatnonzero(tr_m)
    print(f"{len(names)} objects ({int(is_test.sum())} test, {len(val_ids)} val); samples train {len(tr)}, "
          f"val {len(va)}, test {len(te)}; device {device}", flush=True)

    model = HeatmapNet(torch.as_tensor(t["feats"]), n_cond=N_COND_MU).to(device)
    print(f"{sum(q.numel() for q in model.parameters()):,} parameters", flush=True)
    sc_tr = t["score"][tr].astype(np.float32)
    fr = float(np.nanmean(sc_tr > 0)) if np.isfinite(sc_tr).any() else 0.5
    del sc_tr
    pos_weight = torch.tensor((1 - fr) / max(fr, 1e-3), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, best_state, history = -1.0, None, []
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(tr)
        tot, te0 = 0.0, time.perf_counter()
        for i in range(0, len(perm), args.batch):
            b = np.sort(perm[i:i + args.batch])
            f = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
            out = split_outputs(model(f(t["obj"][b], torch.long), f(t["cond"][b])))
            loss = loss_fn(out, f(np.nan_to_num(t["score"][b].astype(np.float32), nan=-1.0)),
                           f(t["params"][b].astype(np.float32)), pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        mv = evaluate(predict(model, va, device), va, rng)
        history.append({"epoch": ep, "loss": round(tot / len(tr), 4), "val_point_auc": mv["point_auc"],
                        "val_top1": mv["top1"]["network"], "seconds": round(time.perf_counter() - te0)})
        print(f"epoch {ep:3d}  loss {tot / len(tr):.4f}  val point AUC {mv['point_auc']:.3f}  val top-1 "
              f"{mv['top1']['network']:.2f} (oracle {mv['top1']['way_A_oracle']:.2f})  "
              f"{time.perf_counter() - te0:.0f} s", flush=True)
        key = float(np.nan_to_num(mv["point_auc"])) + mv["top1"]["network"]
        if key > best or best_state is None:
            best, best_state = key, {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "config": {k: str(v) for k, v in vars(args).items()},
                        "names": names, "epoch": ep}, ROOT / "data" / "models" / f"{args.tag}.pt")
    model.load_state_dict(best_state)
    m = evaluate(predict(model, te, device), te, rng)
    hard = te.cond[:, -1] > np.log(0.5)                         # load on a pinch above 0.5
    if hard.any() and (~hard).any():
        m["test_load_below_0.5"] = evaluate(predict(model, te.subset(~hard), device), te.subset(~hard), rng)["top1"]
        m["test_load_above_0.5"] = evaluate(predict(model, te.subset(hard), device), te.subset(hard), rng)["top1"]
    m["train_seconds"] = round(time.perf_counter() - t0, 1)
    m["history"] = history
    m["config"] = {k: str(v) for k, v in vars(args).items()}
    (ROOT / "data" / "models" / f"{args.tag}_metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: v for k, v in m.items() if k not in ("history", "config")}, indent=1))
    plot(m, ROOT / "media" / f"{args.tag}_eval.png")
    print(f"wrote data/models/{args.tag}.pt (epoch {torch.load(ROOT / 'data' / 'models' / f'{args.tag}.pt', weights_only=False)['epoch']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
