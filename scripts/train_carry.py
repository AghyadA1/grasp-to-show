"""Train the carry network (grasp2show/learn/carry_net.py) in simulation.

Problems: data/carry/problems (scripts/carry_problems.py; training objects only; validation
objects held out). Each problem is run with two pose errors the robot does not know: its own, and
the same error mirrored, so a carry must hold either way (carry.evaluate_theta).

Rounds of search and learning (expert iteration, as AlphaZero learns from its own search; in
continuous control, guided policy search / reward-weighted regression):
  1. a batch of training problems; the network proposes M carries for each (its heads);
  2. each head's carry and --samples random variations of it (noise on the path coefficients and
     the durations) are run in MuJoCo from the saved grip (carry.CarrySim) and scored
     (carry.score: the object kept, no collision, within the robot's limits, then time, slip and
     smoothness);
  3. every head keeps, per problem, the best carry its own proposals have found so far (the
     archive), and is trained towards it; the value head towards the score its own carry got.
     A head never learns from another head's carry: each keeps to its own kind of motion, so two
     equally good motions are never averaged into a bad middle one;
  4. problems where every carry so far lost the object in the lift (the grasp itself does not
     hold; no motion can help) are set aside.
Every --val-every rounds: on the validation problems, the head the value head picks, against the
planner's own carry (and, less often, the best of all heads: what a perfect value head would get).

Resumable (data/carry/run/: network, archive, log). Rounds take minutes; run as long as the
validation keeps improving (--rounds).

Usage:  python scripts/train_carry.py --workers 11 --rounds 60
        python scripts/train_carry.py --smoke
Outputs: data/models/carry_net.pt, data/carry/run/{log.jsonl, archive.pkl, val.json}
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("G2S_OBJECT_SET", "v2")
from grasp2show.carry import N_IN, N_THETA, K
from grasp2show.learn.carry_net import CarryNet
from grasp2show.parallel import Workers

PROBLEMS = ROOT / "data" / "carry" / "problems"
RUN = ROOT / "data" / "carry" / "run"
SIGMA = np.r_[np.full(K * 7, 0.15), 0.15, 0.2].astype(np.float32)   # exploration noise per theta entry

_SIMS: dict = {}


def offsets(p) -> list:
    return [tuple(p.offset), tuple(-np.asarray(p.offset))]


def load_problems(root: Path) -> list:
    out = []
    for f in sorted(root.glob("*.pkl")):
        with open(f, "rb") as fh:
            for i, p in enumerate(pickle.load(fh)):
                out.append((f.name, i, p))
    return out


def sim_of(p):
    from grasp2show.carry import CarrySim

    key = (p.asset, p.yaw_deg, p.mu, p.mass_scale, p.kind, float(p.T_obj_tcp[0, 3]))
    if key not in _SIMS:
        if len(_SIMS) > 6:
            _SIMS.clear()
        _SIMS[key] = CarrySim(p)
    return _SIMS[key]


def features_job(args) -> np.ndarray:
    from grasp2show.carry import features

    _, p = args
    return features(p, sim_of(p).obj)


def rollout_job(args) -> list[dict]:
    """Every candidate theta of one problem, scored."""
    p, thetas = args
    from grasp2show.carry import evaluate_theta

    sim = sim_of(p)
    out = []
    for th in thetas:
        try:
            r = evaluate_theta(sim, th, offsets(p))
        except Exception as e:  # noqa: BLE001 (a simulation that blows up: the worst score)
            r = {"score": -1.0, "success": False, "lost": ["error"], "error": f"{type(e).__name__}: {e}"}
        out.append({k: r[k] for k in ("score", "success", "lost") if k in r} | {"T": r.get("T", 0.0)})
    return out


def baseline_job(args) -> dict:
    from grasp2show.carry import evaluate_baseline

    _, p = args
    try:
        r = evaluate_baseline(sim_of(p), offsets(p))
    except Exception as e:  # noqa: BLE001
        r = {"score": -1.0, "success": False, "T": 0.0, "lost": [f"error {type(e).__name__}"]}
    return {k: r[k] for k in ("score", "success", "T", "lost")}


def lift_lost(r: dict) -> bool:
    return any(x == "lift" for x in r["lost"])


def train_net(net: CarryNet, X: np.ndarray, arch: dict, vals: dict, steps: int, device, lr: float = 1e-3) -> float:
    """Each head towards its own best carry per problem; the value head towards each head's
    latest score."""
    idx = sorted(set(arch) | set(vals))
    if not idx:
        return float("nan")
    x = torch.as_tensor(X[idx], device=device)
    tgt = np.zeros((len(idx), net.M, N_THETA), np.float32)
    has = np.zeros((len(idx), net.M), np.float32)
    v_t = np.zeros((len(idx), net.M), np.float32)
    v_has = np.zeros((len(idx), net.M), np.float32)
    for j, i in enumerate(idx):
        for m, (s, th) in arch.get(i, {}).items():
            tgt[j, m], has[j, m] = th, 1.0
        for m, s in vals.get(i, {}).items():
            v_t[j, m], v_has[j, m] = s, 1.0
    tgt, has, v_t, v_has = (torch.as_tensor(a, device=device) for a in (tgt, has, v_t, v_has))
    w = torch.as_tensor(1.0 / SIGMA, device=device)          # errors in units of the exploration noise
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    tot = 0.0
    for _ in range(steps):
        b = torch.randint(0, len(idx), (min(256, len(idx)),), device=device)
        th, v = net(x[b])
        lt = ((((th - tgt[b]) * w) ** 2).mean(-1) * has[b]).sum() / has[b].sum().clamp(min=1)
        lv = (nn_huber(v - v_t[b]) * v_has[b]).sum() / v_has[b].sum().clamp(min=1)
        loss = lt * 0.1 + lv
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
    net.eval()
    return tot / steps


def nn_huber(e: torch.Tensor, d: float = 0.5) -> torch.Tensor:
    a = e.abs()
    return torch.where(a < d, 0.5 * e**2, d * (a - 0.5 * d))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--batch", type=int, default=200, help="training problems per round")
    p.add_argument("--samples", type=int, default=2, help="random variations of each head's carry per problem")
    p.add_argument("--steps", type=int, default=300, help="network updates per round")
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--val-problems", type=int, default=150)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--problems", type=Path, default=PROBLEMS)
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.problems, args.run = PROBLEMS.with_name("problems_smoke"), RUN.with_name("run_smoke")
        args.rounds, args.batch, args.steps, args.val_every, args.val_problems = 2, 8, 50, 1, 6
    args.run.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    probs = load_problems(args.problems)
    train_i = [i for i, (_, _, q) in enumerate(probs) if q.split == "train" or args.smoke]
    val_i = [i for i, (_, _, q) in enumerate(probs) if q.split == "val" or args.smoke][: args.val_problems]
    print(f"{len(probs)} problems: {len(train_i)} training ({sum(probs[i][2].kind == 'hook' for i in train_i)} hooks), "
          f"{len(val_i)} validation; device {device}", flush=True)
    pool = Workers(args.workers, stall=3600)
    feat_f = args.run / "features.npz"
    keys = [f"{f}:{i}" for f, i, _ in probs]
    X = None
    if feat_f.exists():
        z = np.load(feat_f)
        if list(z["keys"]) == keys:
            X = z["X"]
    if X is None:
        print("features ...", flush=True)
        X = np.zeros((len(probs), N_IN), np.float32)
        for job, r in pool.run(features_job, [(i, q) for i, (_, _, q) in enumerate(probs)]):
            X[job[0]] = r if r is not None else 0.0
        np.savez(feat_f, X=X, keys=np.array(keys))
    state_f = args.run / "state.pkl"
    net = CarryNet().to(device)
    st = {"round": 0, "arch": {}, "vals": {}, "hopeless": set(), "baseline": {}}
    if state_f.exists():
        with open(state_f, "rb") as fh:
            st = pickle.load(fh)
        net.load_state_dict(torch.load(args.run / "net.pt", map_location=device))
        print(f"resumed at round {st['round']}: {len(st['arch'])} problems in the archive, {len(st['hopeless'])} set "
              f"aside", flush=True)
    else:
        net.x_mean.copy_(torch.as_tensor(X[train_i].mean(0)))
        net.x_std.copy_(torch.as_tensor(X[train_i].std(0) + 1e-3))
    log_f = open(args.run / "log.jsonl", "a", encoding="utf-8")  # noqa: SIM115

    def propose(idx):
        with torch.no_grad():
            th, v = net(torch.as_tensor(X[idx], device=device))
        return th.cpu().numpy(), v.cpu().numpy()

    def validate(full: bool) -> dict:
        todo = [i for i in val_i if i not in st["baseline"]]
        for job, r in pool.run(baseline_job, [(i, probs[i][2]) for i in todo]):
            st["baseline"][job[0]] = r if r is not None else {"score": -1.0, "success": False, "T": 1.0, "lost": ["?"]}
        th, v = propose(val_i)
        pick = v.argmax(1)
        jobs = [(probs[i][2], list(th[j]) if full else [th[j, pick[j]]]) for j, i in enumerate(val_i)]
        res = {}
        for job, r in pool.run(rollout_job, jobs):
            res[id(job[0])] = r
        rows = []
        for j, i in enumerate(val_i):
            r = res.get(id(probs[i][2]))
            if r is None:
                continue
            chosen = r[pick[j]] if full else r[0]
            b = st["baseline"][i]
            rows.append({"kind": probs[i][2].kind, "net": chosen["success"], "net_T": chosen["T"], "net_score": chosen["score"],
                         "base": b["success"], "base_T": b["T"], "base_score": b["score"],
                         "oracle": max(x["success"] for x in r) if full else None})
        out = {}
        for kind in ("pinch", "hook", "all"):
            rs = [x for x in rows if kind == "all" or x["kind"] == kind]
            if not rs:
                continue
            both = [x for x in rs if x["net"] and x["base"]]
            out[kind] = {"n": len(rs), "net_success": float(np.mean([x["net"] for x in rs])),
                         "planner_success": float(np.mean([x["base"] for x in rs])),
                         "won": int(sum(x["net"] and not x["base"] for x in rs)),
                         "lost": int(sum(x["base"] and not x["net"] for x in rs)),
                         "duration_ratio_median": float(np.median([x["net_T"] / x["base_T"] for x in both])) if both else None,
                         "net_score": float(np.mean([x["net_score"] for x in rs])),
                         "planner_score": float(np.mean([x["base_score"] for x in rs]))}
            if full:
                out[kind]["best_head_success"] = float(np.mean([x["oracle"] for x in rs]))
        return out

    try:
        while st["round"] < args.rounds:
            t0 = time.time()
            st["round"] += 1
            pool_i = [i for i in train_i if i not in st["hopeless"]]
            batch = list(rng.choice(pool_i, min(args.batch, len(pool_i)), replace=False))
            th, _ = propose(batch)
            jobs, meta = [], []
            for j, i in enumerate(batch):
                cands, heads = [], []
                for m in range(net.M):
                    cands.append(th[j, m])
                    heads.append((m, True))
                    for _ in range(args.samples):
                        cands.append(th[j, m] + SIGMA * rng.standard_normal(N_THETA).astype(np.float32))
                        heads.append((m, False))
                jobs.append((probs[i][2], cands))
                meta.append((i, cands, heads))
            by = {}
            for job, r in pool.run(rollout_job, jobs):
                by[id(job[0])] = r
            n_succ, n_head_succ, n_new, n_lift = 0, 0, 0, 0
            for i, cands, heads in meta:
                r = by.get(id(probs[i][2]))
                if r is None:
                    continue
                arch = st["arch"].setdefault(i, {})
                vals = st["vals"].setdefault(i, {})
                for (m, is_mean), th_c, res in zip(heads, cands, r):
                    if is_mean:
                        vals[m] = res["score"]
                        n_head_succ += res["success"]
                    if m not in arch or res["score"] > arch[m][0]:
                        n_new += m in arch
                        arch[m] = (res["score"], np.asarray(th_c, np.float32))
                n_succ += any(x["success"] for x in r)
                if all(lift_lost(x) for x in r) and all(s < -0.5 for s, _ in arch.values()):
                    st["hopeless"].add(i)
                    n_lift += 1
            loss = train_net(net, X, st["arch"], st["vals"], args.steps, device)
            rec = {"round": st["round"], "problems": len(batch), "any_success": n_succ / len(batch),
                   "head_success": n_head_succ / (len(batch) * net.M), "improved": n_new, "set_aside": n_lift,
                   "loss": loss, "seconds": round(time.time() - t0)}
            if st["round"] % args.val_every == 0 or st["round"] == args.rounds:
                rec["val"] = validate(full=st["round"] % (2 * args.val_every) == 0 or args.smoke)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            msg = (f"round {rec['round']:3d}: a carry worked in {rec['any_success']:.0%} of {len(batch)} problems, "
                   f"heads {rec['head_success']:.0%}; {n_new} improved, {n_lift} set aside (lost in the lift); "
                   f"loss {loss:.3f}; {rec['seconds']} s")
            if "val" in rec:
                for kind, x in rec["val"].items():
                    msg += (f"\n   val {kind:5s} ({x['n']}): network {x['net_success']:.0%} vs planner "
                            f"{x['planner_success']:.0%} (won {x['won']}, lost {x['lost']}), duration x"
                            f"{x['duration_ratio_median'] or float('nan'):.2f}" +
                            (f", best head {x['best_head_success']:.0%}" if "best_head_success" in x else ""))
            print(msg, flush=True)
            torch.save(net.state_dict(), args.run / "net.pt")
            with open(state_f, "wb") as fh:
                pickle.dump(st, fh)
            torch.save({"state_dict": net.state_dict(), "M": net.M, "width": 256, "round": st["round"]},
                       ROOT / "data" / "models" / ("carry_net_smoke.pt" if args.smoke else "carry_net.pt"))
    finally:
        pool.close()
        log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
