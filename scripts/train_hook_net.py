"""Train the hook network (grasp2show/learn/hooknet.py) on the generated objects and test it.

Data: the generated objects and their hook libraries (scripts/generate_hookable.py writes both:
grasp2show/object_set_gen.json, data/generated_objects/, data/hooks/). Point clouds are sampled
from each object's surface (2,048 points with normals) and cached in data/hook_dataset.npz.
Split by object: the set's test objects are never seen; 10 % of the training objects pick the
epoch. The real scanned handle objects (object_set_v2 "handle_candidates") are a second test,
zero-shot: generated shapes only in training.

Reported on the unseen objects:
  point AUC     does the per-point hook probability separate anchor points from the rest?
  object AUC    does the highest point probability tell objects with a hook from those without?
  lift check    the top --verify-k predicted hooks of each object are nudged clear of the
                handle and lift-tested in simulation (the same test that labels the data):
                precision = share that lift the object; hit rate = objects with a hook where at
                least one prediction lifts; and the time this takes vs the analytic pipeline.
                raw: the network's own poses; refined: the analytic fingertip fit around the
                holes it found (hooknet.refine_hooks; the network replaces the hole search).
Usage:  python scripts/train_hook_net.py                 (GPU if available)
        python scripts/train_hook_net.py --epochs 3      (quick check)
Outputs: data/models/hook_net.pt, data/models/hook_net_metrics.json, media/hook_net_eval.png
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
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("G2S_OBJECT_SET", "v2")          # the scans' QR placements (for the lift test)
from train_scorer import auc

from grasp2show.learn.hooknet import (
    HookNet,
    HookSet,
    augment,
    labels,
    loss_fn,
    predict_hooks,
    sample_cloud,
    split,
)
from grasp2show.objects import asset_dir

DATA = ROOT / "data" / "hook_dataset.npz"
GEN_SET = ROOT / "grasp2show" / "object_set_gen.json"
V2_SET = ROOT / "grasp2show" / "object_set_v2.json"
H_MAX = 40


def build(names: list[str]) -> HookSet:
    from grasp2show.hanging import hook_library

    O = len(names)
    clouds = np.zeros((O, 2048, 6), np.float32)
    T = np.tile(np.eye(4), (O, H_MAX, 1, 1))
    one = np.zeros((O, H_MAX), bool)
    mask = np.zeros((O, H_MAX), bool)
    for i, n in enumerate(names):
        mesh = trimesh.load(asset_dir(n) / "model.obj", force="mesh", process=True)
        clouds[i] = sample_cloud(mesh, seed=i)
        hooks = hook_library(n)[:H_MAX]
        for j, h in enumerate(hooks):
            T[i, j], one[i, j], mask[i, j] = h["T"], h["kind"] == "one", True
        if (i + 1) % 50 == 0:
            print(f"  clouds and hooks: {i + 1}/{O}", flush=True)
    return HookSet(list(names), clouds, T, one, mask)


def load_data(rebuild: bool) -> tuple[HookSet, dict]:
    gen = json.loads(GEN_SET.read_text())
    scans = [n for n in json.loads(V2_SET.read_text())["handle_candidates"] if (asset_dir(n) / "model.obj").exists()]
    names = gen["objects"] + scans
    if DATA.exists() and not rebuild:
        z = np.load(DATA, allow_pickle=False)
        if list(z["names"]) == names:
            return HookSet(names, z["clouds"], z["T"], z["one"], z["mask"]), {"gen": gen, "scans": scans}
    print(f"building the dataset: {len(names)} objects", flush=True)
    s = build(names)
    np.savez_compressed(DATA, names=np.array(names), clouds=s.clouds, T=s.T, one=s.one, mask=s.mask)
    return s, {"gen": gen, "scans": scans}


def batch(s: HookSet, idx, rng, aug: bool, device):
    C, P, Tg = [], [], []
    for i in idx:
        c, T = (augment(s.clouds[i], s.T[i], rng) if aug else (s.clouds[i], s.T[i]))
        pos, tgt = labels(c, T, s.one[i], s.mask[i])
        C.append(c)
        P.append(pos)
        Tg.append(tgt)
    f = lambda a, dt=torch.float32: torch.as_tensor(np.stack(a), dtype=dt, device=device)
    return f(C), f(P, torch.bool), f(Tg)


@torch.no_grad()
def evaluate(net, s: HookSet, device) -> dict:
    net.eval()
    probs, poss, obj_p, obj_y = [], [], [], []
    for i in range(len(s.names)):
        c, pos, _ = batch(s, [i], None, False, device)
        p = torch.sigmoid(split(net(c))["logit"])[0].cpu().numpy()
        probs.append(p)
        poss.append(pos[0].cpu().numpy())
        obj_p.append(p.max())
        obj_y.append(bool(s.mask[i].any()))
    pp, yy = np.concatenate(probs), np.concatenate(poss)
    return {"point_auc": auc(pp, yy) if 0 < yy.sum() < len(yy) else float("nan"),
            "object_auc": auc(np.array(obj_p), np.array(obj_y)) if 0 < sum(obj_y) < len(obj_y) else float("nan"),
            "objects": len(s.names), "with_hooks": int(sum(obj_y))}


def lift_check(net, s: HookSet, k: int, device, refine: bool) -> dict:
    """Predicted hooks, nudged clear of the handle and lift-tested in the floating-gripper
    simulator: raw (the network's poses, both approach signs) or refined (the analytic fingertip
    fit around the predicted holes, hooknet.refine_hooks)."""
    from grasp2show.config import ObjectConfig, SceneConfig
    from grasp2show.gripper_sim import GripperSim
    from grasp2show.hanging import nudge
    from grasp2show.hooks import lift_test
    from grasp2show.learn.hooknet import refine_hooks
    from grasp2show.objects import object_from_config

    n_pred = n_lift = hit = has = 0
    t_net = t_sim = 0.0
    per = {}
    for i, name in enumerate(s.names):
        t0 = time.perf_counter()
        hooks = predict_hooks(net, s.clouds[i], k=k, device=device)
        obj = object_from_config(ObjectConfig(asset=name))
        if refine:
            hooks = refine_hooks(obj, hooks)[: 4 * k]
        t1 = time.perf_counter()
        sim = GripperSim(SceneConfig(obj=ObjectConfig(asset=name)), obj)
        ok = 0
        for h in hooks:
            T = nudge(sim, h["T"], h["width"])
            ok += T is not None and lift_test(sim, T, h["width"], close=False)["success"]
        t_net += t1 - t0
        t_sim += time.perf_counter() - t1
        n_pred += len(hooks)
        n_lift += ok
        has += bool(s.mask[i].any())
        hit += bool(s.mask[i].any()) and ok > 0
        per[name] = {"candidates": len(hooks), "lift": int(ok), "analytic_hooks": int(s.mask[i].sum())}
    return {"precision": n_lift / max(n_pred, 1), "hit_rate": hit / max(has, 1), "candidates": n_pred,
            "objects_with_hooks": has, "found_s_per_object": t_net / len(s.names),
            "check_s_per_object": t_sim / len(s.names), "per_object": per}


def figure(net, s: HookSet, idx, path, device):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3.2 * len(idx), 6.4))
    for col, i in enumerate(idx):
        c = s.clouds[i]
        with torch.no_grad():
            p = torch.sigmoid(split(net(torch.as_tensor(c[None], device=device)))["logit"])[0].cpu().numpy()
        pos, _ = labels(c, s.T[i], s.one[i], s.mask[i])
        for row, (val, title) in enumerate(((pos.astype(float), "label"), (p, "prediction"))):
            ax = fig.add_subplot(2, len(idx), row * len(idx) + col + 1, projection="3d")
            ax.scatter(*c[:, :3].T, c=val, cmap="viridis", vmin=0, vmax=1, s=2)
            ax.set_title(f"{s.names[i][:22]}\n{title}", fontsize=8)
            ax.set_axis_off()
            ax.set_box_aspect(np.ptp(c[:, :3], axis=0))
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--verify-k", type=int, default=5, help="predicted hooks lift-tested per test object")
    p.add_argument("--rebuild", action="store_true", help="rebuild data/hook_dataset.npz")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="hook_net")
    args = p.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data, meta = load_data(args.rebuild)
    names = data.names
    test_gen = set(meta["gen"]["test"])
    scans = set(meta["scans"])
    train_all = [i for i, n in enumerate(names) if n not in test_gen and n not in scans]
    rng.shuffle(train_all)
    n_val = max(1, len(train_all) // 10)
    val, train = train_all[:n_val], train_all[n_val:]
    test = [i for i, n in enumerate(names) if n in test_gen]
    real = [i for i, n in enumerate(names) if n in scans]
    tr = data.subset(train)
    n_pos = sum(labels(tr.clouds[i], tr.T[i], tr.one[i], tr.mask[i])[0].sum() for i in range(len(train)))
    n_all = len(train) * tr.clouds.shape[1]
    pos_weight = torch.tensor(min(max((n_all - n_pos) / max(n_pos, 1), 1.0), 50.0), device=args.device)
    print(f"{len(train)} train / {len(val)} val / {len(test)} test generated objects, {len(real)} real scans; "
          f"{n_pos / n_all:.1%} of training points are hook anchors (pos_weight {float(pos_weight):.1f}); "
          f"device {args.device}", flush=True)
    net = HookNet().to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    va = data.subset(val)
    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(len(train))
        tot, parts_sum, n = 0.0, {}, 0
        for b in range(0, len(order), args.batch):
            c, pos, tgt = batch(tr, order[b:b + args.batch], rng, True, args.device)
            loss, parts = loss_fn(split(net(c)), pos, tgt, pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            n += 1
            for k_, v in parts.items():
                parts_sum[k_] = parts_sum.get(k_, 0.0) + v
        sched.step()
        m = evaluate(net, va, args.device)
        score = m["point_auc"] if np.isfinite(m["point_auc"]) else 0.0
        if score > best:
            best, best_state = score, {k_: v.detach().clone() for k_, v in net.state_dict().items()}
        print(f"epoch {ep:3d}  loss {tot / n:.3f} ({', '.join(f'{k_} {v / n:.3f}' for k_, v in parts_sum.items())})"
              f"  val point AUC {m['point_auc']:.3f} object AUC {m['object_auc']:.3f}", flush=True)
    net.load_state_dict(best_state)
    out_dir = ROOT / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "names_train": [names[i] for i in train]}, out_dir / f"{args.tag}.pt")
    metrics = {"train": len(train), "val": len(val), "best_val_point_auc": best}
    for key, idx in (("test", test), ("real_scans", real)):
        sub = data.subset(idx)
        metrics[key] = {**evaluate(net, sub, args.device),
                        "raw": lift_check(net, sub, args.verify_k, args.device, refine=False),
                        "refined": lift_check(net, sub, args.verify_k, args.device, refine=True)}
    gen_details = {d["name"]: d for d in meta["gen"].get("details", [])}
    t_an = [gen_details[names[i]]["seconds"] for i in test if names[i] in gen_details]
    metrics["test"]["analytic_s_per_object"] = float(np.mean(t_an)) if t_an else float("nan")
    (out_dir / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=1))
    for k_ in ("test", "real_scans"):
        mm = metrics[k_]
        print(f"{k_}: point AUC {mm['point_auc']:.3f}, object AUC {mm['object_auc']:.3f}", flush=True)
        for mode in ("raw", "refined"):
            r = mm[mode]
            print(f"  {mode:8s} lift precision {r['precision']:.0%} of {r['candidates']} candidates, hit rate "
                  f"{r['hit_rate']:.0%} of {r['objects_with_hooks']} objects with hooks; finding "
                  f"{r['found_s_per_object']:.2f} s + lift checks {r['check_s_per_object']:.1f} s per object", flush=True)
    show = [i for i in test if data.mask[i].any()][:3] + [i for i in test if not data.mask[i].any()][:1] + real[:2]
    figure(net, data, show, ROOT / "media" / f"{args.tag}_eval.png", args.device)
    print(f"wrote {out_dir / (args.tag + '.pt')}, metrics, media/{args.tag}_eval.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
