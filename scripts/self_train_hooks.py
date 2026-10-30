"""Self-training of the hook network (part 3): the simulator is the teacher, not the rules.

The first hook network learned from the analytic hole finder's hooks, so it could only learn what
those rules find (holes that open on both sides, small enough for a 6 cm reach). Here every label
is a lift test in the floating-gripper simulator (hooks.lift_test), and the candidates come from
several proposers so that no rule decides alone what is ever tried:

  teach    (once per object) grasp2show/hookteach.py: the hole finder in ranking mode (also big
           carrying handles, and under the top bar of a hole), fingertips under any overhang
           (lips, rims, knob caps, J-hooks, closed pockets), random placements; ~100 tests per
           object, every outcome kept (data/hook_teach/<name>.npz): lifted, tried and fell (a real
           negative), or could not be placed (not a label).
  rounds   train the network on all tests so far (labels: learn/hooknet.labels3; random yaw,
           1 mm noise, 30 % single-camera partial views), then let it propose on the training
           objects (its best points, points sampled by its probability with jittered poses, and
           random exploration), test those too and add them: its false positives become hard
           negatives, the hooks it finds that the teacher missed become positives. Repeat.

Objects: the generated set (grasp2show/object_set_gen.json, its own train/test split; new shape
families from seed 1200), the real scans with something to hook (object_set_hookreal.json:
26 train, 8 held out) and the scanned objects of the v2 set (mostly without a handle: the network
must also learn where there is nothing to hook; its train/test split). Held-out objects are taught
too (their tests are the reference for the evaluation) but never trained on or proposed on.

Each round reports on the held-out generated objects and the held-out real scans: point AUC
against the teacher's labels; and the real check: the network's top --eval-k points (both
approach directions) nudged clear and lift-tested: precision, hit rate (objects with a hook where
at least one of them lifts), coverage (share of the teacher's distinct hooks, 2.5 cm apart, that
the network also found) and discoveries (lifting hooks of the network far from any of the
teacher's).

Usage (resumable: rerun the same command after an interruption):
  python scripts/self_train_hooks.py --rounds 8 --workers 4
  python scripts/self_train_hooks.py --smoke          (a few objects, minutes: checks the pipeline)
Outputs: data/models/hook_net_self.pt (best round) and hook_net_self_r<k>.pt, data/self_train/
         (state, per-round metrics), media/self_train_hooks.png
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("G2S_OBJECT_SET", "v2")
if "--smoke" in sys.argv:                                   # keep the smoke test's tests apart
    os.environ["G2S_HOOK_TEACH"] = str(ROOT / "data" / "self_train" / "smoke" / "hook_teach")
from grasp2show.hookteach import FELL, LIFTED, STORE, load_tested, save_tested, teach_job, test_candidates

OUT = ROOT / "data" / "self_train"
CLOUDS = ROOT / "data" / "hook_clouds"
MODELS = ROOT / "data" / "models"
N_STORED = 4096
CLOUD_TAG = "v1"
H_POS, H_NEG = 96, 256


def seed_of(name: str) -> int:
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------- workers

class Workers:
    """A process pool that survives a worker dying. With multiprocessing.Pool, a worker that
    crashes (on Windows, silently) is replaced but its job never returns, and imap waits forever
    (the first run stopped at 1337/1340 teacher jobs this way). Here results come back as they
    finish; if nothing finishes for `stall` seconds, the jobs still out are given up (returned
    with result None) and the pool is restarted."""

    def __init__(self, n: int, stall: float = 600.0):
        self.n, self.stall = n, stall
        self.pool = Pool(n)

    def run(self, fn, jobs: list):
        """Yields (job, result or None), in the order they finish."""
        pending = {i: self.pool.apply_async(fn, (j,)) for i, j in enumerate(jobs)}
        last = time.time()
        while pending:
            done = [i for i, a in pending.items() if a.ready()]
            for i in done:
                a = pending.pop(i)
                try:
                    res = a.get()
                except Exception:  # noqa: BLE001 (an error inside the job: give it up, keep going)
                    res = None
                yield jobs[i], res
                last = time.time()
            if not done:
                if time.time() - last > self.stall:
                    for i in sorted(pending):
                        yield jobs[i], None
                    self.restart()
                    return
                time.sleep(0.2)

    def restart(self) -> None:
        self.pool.terminate()
        self.pool.join()
        self.pool = Pool(self.n)

    def close(self) -> None:
        self.pool.terminate()
        self.pool.join()


# ---------------------------------------------------------------- objects

def object_pools(smoke: bool) -> dict:
    gen = json.loads((ROOT / "grasp2show" / "object_set_gen.json").read_text())
    real = json.loads((ROOT / "grasp2show" / "object_set_hookreal.json").read_text())
    v2 = json.loads((ROOT / "grasp2show" / "object_set_v2.json").read_text())
    gen_test = set(gen["test"])
    pools = {"train": [n for n in gen["objects"] if n not in gen_test] + real["train"] + list(v2["train"]),
             "test_gen": sorted(gen_test), "test_real_handles": real["test"], "test_scans_other": list(v2["test"])}
    if smoke:
        from grasp2show.objects import GENERATED_DIR

        here = lambda ns: [n for n in ns if (GENERATED_DIR / n).exists()]
        pools["train"], pools["test_gen"] = here(pools["train"]), here(pools["test_gen"])
        pools = {"train": pools["train"][:6] + real["train"][:2], "test_gen": pools["test_gen"][:3],
                 "test_real_handles": real["test"][:2], "test_scans_other": list(v2["test"])[:1]}
    return pools


def download_missing(names: list[str]) -> list[str]:
    from get_objects import download_object

    from grasp2show.objects import asset_dir

    ok = []
    for n in names:
        if n.startswith("gen_") or (asset_dir(n) / "model.xml").exists():
            ok.append(n)
            continue
        try:
            print("  downloading", download_object(n), flush=True)
            ok.append(n)
        except OSError as e:
            print(f"  could not download {n}: {e}", flush=True)
    return ok


def cloud_job(name: str) -> tuple[str, str]:
    import trimesh

    from grasp2show.learn.hooknet import sample_cloud
    from grasp2show.objects import asset_dir

    f = CLOUDS / f"{name}.{CLOUD_TAG}.npy"
    if f.exists():
        return name, ""
    try:
        d = asset_dir(name)
        mesh = trimesh.load(d / "model.obj", force="mesh", process=True)
        pieces = ([trimesh.load(p, force="mesh") for p in sorted(d.glob("model_collision_*.obj"))]
                  if name.startswith("gen_") else None)                 # generated: outer surface only
        c = sample_cloud(mesh, N_STORED, seed=seed_of(name), pieces=pieces)
    except Exception as e:  # noqa: BLE001 (skip an object that does not load, say why)
        return name, f"{type(e).__name__}: {e}"
    CLOUDS.mkdir(parents=True, exist_ok=True)
    np.save(f, c)
    return name, ""


# ---------------------------------------------------------------- data

class Data:
    """Clouds (N_STORED points) and the tests of each object, as training targets."""

    def __init__(self, names: list[str]):
        self.names, self.clouds, self.T, self.one, self.mask, self.fell = [], [], [], [], [], []
        rng = np.random.default_rng(0)
        for n in names:
            f = CLOUDS / f"{n}.{CLOUD_TAG}.npy"
            if not f.exists():
                continue
            t = load_tested(n)
            lifted = np.flatnonzero(t.result == LIFTED)
            if len(lifted) > H_POS:
                lifted = rng.choice(lifted, H_POS, replace=False)
            fell = np.flatnonzero(t.result == FELL)
            if len(fell) > H_NEG:
                fell = rng.choice(fell, H_NEG, replace=False)
            T = np.tile(np.eye(4), (H_POS, 1, 1))
            one, mask = np.zeros(H_POS, bool), np.zeros(H_POS, bool)
            T[:len(lifted)], one[:len(lifted)], mask[:len(lifted)] = t.T[lifted], t.width[lifted] > 0.01, True
            self.names.append(n)
            self.clouds.append(np.load(f))
            self.T.append(T)
            self.one.append(one)
            self.mask.append(mask)
            self.fell.append(t.anchor[fell])

    def __len__(self) -> int:
        return len(self.names)

    def sample(self, i: int, rng, aug: bool, partial: float):
        """(cloud 2048, pos, tgt, weight) for object i, turned about z by a random yaw if aug."""
        from grasp2show.learn.hooknet import labels3, rot_z, view_sample

        c, T, fell = self.clouds[i], self.T[i], self.fell[i]
        if aug:
            R = rot_z(rng.uniform(0, 2 * np.pi))
            c = c.copy()
            c[:, :3] = c[:, :3] @ R.T + rng.normal(scale=0.001, size=(len(c), 3))
            c[:, 3:] = c[:, 3:] @ R.T
            T = T.copy()
            T[:, :3, :3] = R @ T[:, :3, :3]
            T[:, :3, 3] = T[:, :3, 3] @ R.T
            fell = fell @ R.T
        c = view_sample(c, rng, partial=partial if aug else 0.0).astype(np.float32)
        pos, tgt, w = labels3(c, T, self.one[i], self.mask[i], fell)
        return c, pos, tgt, w


def fixed_cloud(data: Data, i: int) -> np.ndarray:
    """The evaluation / proposal cloud of object i: a fixed subset of 2048 stored points."""
    from grasp2show.learn.hooknet import N_POINTS

    idx = np.random.default_rng(seed_of(data.names[i])).choice(len(data.clouds[i]), N_POINTS, replace=False)
    return data.clouds[i][idx]


# ---------------------------------------------------------------- training

def train(data: Data, val_idx, epochs: int, lr: float, batch: int, device: str, init: dict | None,
          partial: float, log) -> tuple[object, float]:
    import torch
    from train_scorer import auc

    from grasp2show.learn.hooknet import HookNet, loss_fn, split

    torch.manual_seed(0)
    rng = np.random.default_rng(int(time.time()))
    net = HookNet(hops=3).to(device)
    if init is not None:
        net.load_state_dict(init)
    tr = [i for i in range(len(data)) if i not in set(val_idx)]
    n_pos = n_all = 0
    for i in tr[::5]:
        _, pos, _, w = data.sample(i, rng, False, 0.0)
        n_pos += pos.sum()
        n_all += w.sum()
    pos_weight = torch.tensor(float(np.clip((n_all - n_pos) / max(n_pos, 1), 1.0, 30.0)), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    f = lambda a, dt=torch.float32: torch.as_tensor(np.stack(a), dtype=dt, device=device)
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        net.train()
        order = rng.permutation(tr)
        tot, n = 0.0, 0
        t0 = time.time()
        for b in range(0, len(order), batch):
            s = [data.sample(i, rng, True, partial) for i in order[b:b + batch]]
            if len(s) < 2:
                continue
            o = split(net(f([x[0] for x in s])))
            loss, _ = loss_fn(o, f([x[1] for x in s], torch.bool), f([x[2] for x in s]), pos_weight,
                              f([x[3] for x in s]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        sched.step()
        net.eval()
        pp, yy = [], []
        with torch.no_grad():
            for i in val_idx:
                c, pos, _, _ = data.sample(i, np.random.default_rng(i), False, 0.0)
                pp.append(torch.sigmoid(split(net(f([c])))["logit"])[0].cpu().numpy())
                yy.append(pos)
        yy_ = np.concatenate(yy)
        score = auc(np.concatenate(pp), yy_) if 0 < yy_.sum() < len(yy_) else 0.0
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in net.state_dict().items()}
        log(f"    epoch {ep:3d}  loss {tot / max(n, 1):.3f}  val point AUC {score:.3f}  ({time.time() - t0:.0f} s)")
    net.load_state_dict(best_state)
    return net, best


# ---------------------------------------------------------------- proposing and checking

def jitter(T: np.ndarray, rng, deg: float = 25.0, shift: float = 0.004) -> np.ndarray:
    """T turned by up to `deg` about a random axis through its fingertip area, shifted a little."""
    from scipy.spatial.transform import Rotation

    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    R = Rotation.from_rotvec(ax * np.radians(rng.uniform(0, deg))).as_matrix()
    out = T.copy()
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = T[:3, 3] + rng.normal(scale=shift, size=3)
    return out


def net_candidates(net, cloud, rng, k_top: int, n_sampled: int, device) -> list[dict]:
    """The network's best k_top points (both approach directions), plus n_sampled points drawn
    by probability with jittered poses (exploring around what it believes)."""
    import torch

    from grasp2show.learn.hooknet import OPEN_WIDTH, hook_pose, predict_hooks, split

    out = [{"T": h["T"], "width": h["width"], "source": "net", "wide": True}
           for h in predict_hooks(net, cloud, k=k_top, min_prob=0.0, device=device)]
    if n_sampled:
        with torch.no_grad():
            o = split(net(torch.as_tensor(cloud[None], device=device)))
        p = torch.sigmoid(o["logit"][0]).cpu().numpy()
        w = p ** 0.5
        for i in rng.choice(len(p), n_sampled, p=w / w.sum()):
            one = float(torch.sigmoid(o["one_logit"][0, i])) > 0.5 if rng.random() < 0.8 else rng.random() < 0.5
            wd = OPEN_WIDTH if one else 0.0
            a = o["approach"][0, i].cpu().numpy() * (1 if rng.random() < 0.5 else -1)
            T = hook_pose(cloud[i, :3], a, o["closing"][0, i].cpu().numpy(), o["offset"][0, i].cpu().numpy(), wd)
            out.append({"T": jitter(T, rng), "width": wd, "source": "net", "wide": True})
    return out


def round_job(args):
    """Worker: test the network's candidates on one object plus n_random exploration ones."""
    name, cands, n_random, seed = args
    from grasp2show.hookteach import _load, random_candidates

    t0 = time.time()
    try:
        obj, sim = _load(name)
        rnd = random_candidates(obj, np.random.default_rng(seed), n_random) if n_random else []
        t = test_candidates(sim, cands + rnd)
    except Exception as e:  # noqa: BLE001 (one bad object must not stop a 12-hour run)
        return name, None, time.time() - t0, f"{type(e).__name__}: {e}"
    return name, t, time.time() - t0, ""


def clusters(anchors: np.ndarray, r: float) -> np.ndarray:
    """Greedy cluster centres, r apart."""
    out = []
    for a in anchors:
        if all(np.linalg.norm(a - c) > r for c in out):
            out.append(a)
    return np.array(out).reshape(-1, 3)


def evaluate(net, data: Data, pool: Workers, k: int, device, seed: int) -> dict:
    import torch
    from train_scorer import auc

    from grasp2show.learn.hooknet import split

    if len(data) == 0:
        return {}
    from grasp2show.learn.hooknet import labels

    pp, yy, jobs, t_net = [], [], [], 0.0
    rng = np.random.default_rng(seed)
    for i in range(len(data)):
        c = fixed_cloud(data, i)
        yy.append(labels(c, data.T[i], data.one[i], data.mask[i])[0])
        t0 = time.perf_counter()
        with torch.no_grad():
            pp.append(torch.sigmoid(split(net(torch.as_tensor(c[None], device=device)))["logit"])[0].cpu().numpy())
        jobs.append((data.names[i], net_candidates(net, c, rng, k, 0, device), 0, 0))
        t_net += time.perf_counter() - t0
    res = {j[0]: (r if r is not None else (j[0], None, 0.0, "lost")) for j, r in pool.run(round_job, jobs)}
    n_c = n_l = hit = has = 0
    cov, disc = [], 0
    per = {}
    for i, n in enumerate(data.names):
        t = res[n][1]
        teacher = load_tested(n)
        t_lift = teacher.anchor[teacher.result == LIFTED]
        net_lift = t.anchor[t.result == LIFTED] if t is not None else np.zeros((0, 3))
        n_c += len(t) if t is not None else 0
        n_l += len(net_lift)
        hookable = len(t_lift) > 0 or len(net_lift) > 0
        has += hookable
        hit += hookable and len(net_lift) > 0
        cl = clusters(t_lift, 0.025)
        if len(cl):
            cov.append(np.mean([len(net_lift) > 0 and np.linalg.norm(net_lift - c_, axis=1).min() < 0.025 for c_ in cl]))
        nc = clusters(net_lift, 0.025)
        new = [c_ for c_ in nc if len(t_lift) == 0 or np.linalg.norm(t_lift - c_, axis=1).min() > 0.025]
        disc += len(new)
        per[n] = {"teacher_hooks": len(cl), "net_lifts": len(net_lift), "net_new": len(new)}
    yy_, pp_ = np.concatenate(yy), np.concatenate(pp)
    return {"objects": len(data), "hookable": int(has),
            "point_auc": float(auc(pp_, yy_)) if 0 < yy_.sum() < len(yy_) else float("nan"),
            "precision": n_l / max(n_c, 1), "hit_rate": hit / max(has, 1),
            "coverage": float(np.mean(cov)) if cov else float("nan"), "discoveries": int(disc),
            "net_s_per_object": t_net / len(data), "per_object": per}


# ---------------------------------------------------------------- main

def plot(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g for g in ("test_gen", "test_real_handles", "test_scans_other") if g in history[0]]
    keys = ("point_auc", "hit_rate", "coverage", "precision")
    fig, axs = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.4))
    r = [h["round"] for h in history]
    for ax, k in zip(axs, keys):
        for g in groups:
            ax.plot(r, [h[g].get(k, np.nan) for h in history], "o-", label=g)
        ax.set_title(k)
        ax.set_xlabel("round")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    p.add_argument("--epochs-first", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40, help="per later round (warm start)")
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--partial", type=float, default=0.3, help="share of single-camera views in training")
    p.add_argument("--k-top", type=int, default=6, help="network's best points proposed per object and round")
    p.add_argument("--n-sampled", type=int, default=8)
    p.add_argument("--n-random", type=int, default=6)
    p.add_argument("--eval-k", type=int, default=6)
    p.add_argument("--smoke", action="store_true", help="a few objects and epochs: check the pipeline")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        args.rounds, args.epochs_first, args.epochs = 2, 2, 1
    out = OUT / ("smoke" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "log.txt", "a", encoding="utf-8")  # noqa: SIM115 (open for the whole run)

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    pools = object_pools(args.smoke)
    log(f"== self-training {time.strftime('%Y-%m-%d %H:%M')}: {len(pools['train'])} train objects, "
        f"{len(pools['test_gen'])} held-out generated, {len(pools['test_real_handles'])} + "
        f"{len(pools['test_scans_other'])} held-out scans (with / mostly without handles); device {device}, "
        f"{args.workers} workers")
    skip_f = out / "skipped.txt"                          # objects whose worker crashed or hung: left out
    skipped = set(skip_f.read_text().split()) if skip_f.exists() else set()
    if skipped:
        log(f"leaving out {len(skipped)} objects that crashed or hung before: {', '.join(sorted(skipped))}")
    pools = {g: [n for n in ns if n not in skipped] for g, ns in pools.items()}

    def give_up(name: str) -> None:
        skipped.add(name)
        skip_f.write_text("\n".join(sorted(skipped)))
        log(f"  {name}: its worker crashed or hung; left out from now on")

    everything = download_missing(sorted(set().union(*pools.values())))
    pool = Workers(args.workers)
    try:
        t0 = time.time()
        bad = []
        for n, r in pool.run(cloud_job, everything):
            if r is None or r[1]:
                bad.append(n)
                log(f"  no cloud for {n}: {r[1] if r else 'worker lost'}")
        log(f"clouds ready ({time.time() - t0:.0f} s)")
        todo = [n for n in everything if not (STORE / f"{n}.npz").exists() and n not in bad]
        log(f"teacher: {len(todo)} objects to teach ({len(everything) - len(todo)} done before)")
        t0 = time.time()
        for i, ((n, _), r) in enumerate(pool.run(teach_job, [(n, seed_of(n)) for n in todo]), 1):
            if r is None:
                give_up(n)
                continue
            _, nt, nl, sec, err = r
            log(f"  [{i}/{len(todo)}] {n}: " + (f"FAILED {err}" if err else f"{nl}/{nt} lift ({sec:.0f} s)"))
        log(f"teacher done ({(time.time() - t0) / 60:.0f} min)")
        pools = {g: [n for n in ns if n not in skipped] for g, ns in pools.items()}

        state_f = out / "state.json"
        state = json.loads(state_f.read_text()) if state_f.exists() else {"round_done": 0, "history": []}
        test_sets = {g: Data(pools[g]) for g in ("test_gen", "test_real_handles", "test_scans_other")}
        init = None
        if state["round_done"]:
            init = torch.load(MODELS / f"hook_net_self_r{state['round_done']}.pt", map_location=device)["state"]
        for r in range(state["round_done"] + 1, args.rounds + 1):
            t_round = time.time()
            data = Data(pools["train"])
            n_l = sum(int(m.sum()) for m in data.mask)
            n_f = sum(len(f) for f in data.fell)
            log(f"== round {r}: {len(data)} objects, {n_l} lifting hooks, {n_f} failed tries in the labels")
            val_idx = list(np.random.default_rng(0).choice(len(data), max(2, len(data) // 10), replace=False))
            first = init is None
            net, val_auc = train(data, val_idx, args.epochs_first if first else args.epochs,
                                 args.lr if first else args.lr / 2, args.batch, device, init, args.partial, log)
            init = {k: v.detach().clone() for k, v in net.state_dict().items()}
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"state": net.state_dict(), "hops": 3, "round": r}, MODELS / f"hook_net_self_r{r}.pt")
            t_train = time.time() - t_round
            metrics = {"round": r, "val_point_auc": val_auc, "train_objects": len(data), "lifting_hooks": n_l,
                       "failed_tries": n_f}
            for g, d in test_sets.items():
                metrics[g] = evaluate(net, d, pool, args.eval_k, device, seed=r)
                m = metrics[g]
                log(f"  {g:18s} point AUC {m['point_auc']:.3f}  hit {m['hit_rate']:.0%} of {m['hookable']}  "
                    f"coverage {m['coverage']:.0%}  precision {m['precision']:.0%}  new hooks {m['discoveries']}  "
                    f"({m['net_s_per_object'] * 1000:.0f} ms/object)")
            if r < args.rounds:                                             # propose on the training objects
                rng = np.random.default_rng(1000 + r)
                net.eval()
                jobs = [(n, net_candidates(net, fixed_cloud(data, i), rng, args.k_top, args.n_sampled, device),
                         args.n_random, seed_of(n) + r) for i, n in enumerate(data.names)]
                t0 = time.time()
                found = tried = 0
                for i, (job, res) in enumerate(pool.run(round_job, jobs), 1):
                    if res is None:
                        give_up(job[0])
                        continue
                    n, t, _, err = res
                    if t is None:
                        log(f"  {n}: FAILED {err}")
                        continue
                    t.source[t.source == "net"] = f"net{r}"
                    save_tested(n, load_tested(n).merged(t))
                    found += int((t.result == LIFTED).sum())
                    tried += len(t)
                    if i % 100 == 0:
                        log(f"    proposed and tested {i}/{len(jobs)} objects")
                log(f"  tested {tried} new candidates on {len(jobs)} objects: {found} lift "
                    f"({(time.time() - t0) / 60:.0f} min)")
            metrics["minutes"] = {"train": t_train / 60, "round": (time.time() - t_round) / 60}
            state["round_done"] = r
            state["history"].append({k: ({kk: vv for kk, vv in v.items() if kk != "per_object"}
                                         if isinstance(v, dict) and "per_object" in v else v)
                                     for k, v in metrics.items()})
            (out / f"round_{r}.json").write_text(json.dumps(metrics, indent=1, default=float))
            state_f.write_text(json.dumps(state, indent=1, default=float))
            plot(state["history"], ROOT / "media" / ("self_train_hooks_smoke.png" if args.smoke else "self_train_hooks.png"))
            log(f"  round {r} took {(time.time() - t_round) / 60:.0f} min")
    finally:
        pool.close()
    hist = state["history"]
    score = lambda h: np.nanmean([h["test_gen"]["hit_rate"], h["test_real_handles"]["hit_rate"],
                                  h["test_gen"]["coverage"], h["test_real_handles"]["coverage"]])
    best = max(hist, key=score)
    if not args.smoke:
        shutil.copy(MODELS / f"hook_net_self_r{best['round']}.pt", MODELS / "hook_net_self.pt")
    log(f"best round {best['round']} -> data/models/hook_net_self.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
