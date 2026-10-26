"""Step 2: better labels for the problems the self-supervised network still gets wrong (DAgger-style).

Gradient training finds the best motion *near* the one the network already makes. Where the
only solutions are a different kind of motion (a detour that turns the object so its weight goes
where the grip is strong), it can stay stuck. Here the optimiser looks for those solutions,
starting both from the network's own motion and from the straight one, and the feasible result
becomes a label that pulls the network into the right region; the constraint loss keeps
optimising from there (scripts/train_transport_selfsup.py --labels).

Problems: every training problem (never the validation or test objects) at its recorded friction,
plus `--variants` random versions (fresh friction; half of them with the mass scaled 0.6-2.5x,
as in training). The network's raw motion is checked on all of them; the infeasible ones are
sent to the optimiser (no collision check, as in training).

Usage:  python scripts/refine_transport_labels.py [--net data/models/transport_net_ss_mass.pt] [--workers 4]
Output: data/transport_labels.npz (problem, friction, mass scale, coefficients, duration)
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_transport_selfsup import INERTIA_COLS, MASS_COL, MU_COL, MU_RANGE, load, obj_model, val_split

from grasp2show.learn.transport_net import TransportNet
from grasp2show.learn.transport_torch import Constraints, pack
from grasp2show.transport import friction_params
from grasp2show.transport_opt import Solution, TransportProblem


def scaled(obj, k: float):
    return dataclasses.replace(obj, mass=obj.mass * k, inertia=obj.inertia * k) if k != 1.0 else obj


def solve(job):
    i, name, T_obj_tcp, width, q0, q1, mu, k, c_net, T_net, seed = job
    pr = TransportProblem(scaled(obj_model(name), k), T_obj_tcp, width, q0, q1, friction_params(mu))
    best = None
    ret = pr.retime_only()
    for start in (Solution(T_net, c_net, False, {}), ret):
        s = pr.shape_and_time(start, seed=seed)
        if s.feasible and (best is None or s.T < best.T):
            best = s
    if best is None:
        return i, None, None, ret.feasible
    return i, best.c, best.T, ret.feasible


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", type=Path, default=ROOT / "data" / "models" / "transport_net_ss_mass.pt")
    p.add_argument("--data", type=Path, nargs="+", default=[ROOT / "data" / "transport_dataset",
                                                            ROOT / "data" / "transport_problems"])
    p.add_argument("--variants", type=int, default=2)
    p.add_argument("--mass-scale", type=float, nargs=2, default=(0.6, 2.5))
    p.add_argument("--out", type=Path, default=ROOT / "data" / "transport_labels.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    args = p.parse_args()
    rng = np.random.default_rng(args.seed + 7)
    data = [d for d in args.data if d.exists()]
    rows = load(data)
    val = val_split([r for r in rows if r["file"].startswith(data[0].name)], 5, 0)
    rows = [r for r in rows if r["name"] not in val]

    # problem variants: (row, mu, mass scale)
    var = [(r, r["mu"], 1.0) for r in rows]
    lo, hi = np.log(args.mass_scale)
    for _ in range(args.variants):
        for r in rows:
            k = float(np.exp(rng.uniform(lo, hi))) if rng.random() < 0.5 else 1.0
            var.append((r, float(rng.uniform(*MU_RANGE)), min(k, 1.5 / r["obj"].mass)))
    print(f"{len(var)} problems ({len(rows)} training problems x {1 + args.variants}); checking the network ...", flush=True)

    model = TransportNet()
    model.load_state_dict(torch.load(args.net, map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    con = Constraints()
    jobs, n_ok, t0 = [], 0, time.perf_counter()
    for s in range(0, len(var), 256):
        chunk = var[s:s + 256]
        probs = [{**r, "obj": scaled(r["obj"], k), "mu": mu} for r, mu, k in chunk]
        P = pack(probs)
        keep = P.pop("index").tolist()
        X = torch.as_tensor(np.stack([chunk[j][0]["x"] for j in keep]))
        X[:, MU_COL] = torch.as_tensor([chunk[j][1] for j in keep], dtype=torch.float32)
        ks = torch.as_tensor([chunk[j][2] for j in keep], dtype=torch.float32)
        X[:, MASS_COL] *= ks
        X[:, INERTIA_COLS] *= ks[:, None]
        with torch.no_grad():
            c, lt = model(X)
            c, T = c.double(), torch.exp(lt.double().clamp(np.log(0.2), np.log(30.0)))
            u_lim = torch.clamp(con.static_u(P) + 0.05, min=con.u_lim0)
            feas = con.feasible(con(c, T, P), u_lim).numpy()
        n_ok += int(feas.sum())
        for jj, j in enumerate(keep):
            if not feas[jj]:
                r, mu, k = chunk[j]
                jobs.append((s + j, r["name"], r["T_obj_tcp"], r["width"], r["q0"], r["q1"], mu, k,
                             c[jj].numpy(), float(T[jj]), int(rng.integers(1 << 30))))
    print(f"network feasible on {n_ok}/{len(var)} ({n_ok / len(var):.1%}); optimising the other {len(jobs)} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)

    out, n_found, n_ret_ok, t0 = [], 0, 0, time.perf_counter()
    with Pool(args.workers, maxtasksperchild=50) as pool:
        for m, (i, c, T, ret_ok) in enumerate(pool.imap_unordered(solve, jobs), 1):
            n_ret_ok += ret_ok
            if c is not None:
                n_found += 1
                r, mu, k = var[i]
                out.append((r["file"], mu, k, c, T, ret_ok))
            if m % 20 == 0 or m == len(jobs):
                el = time.perf_counter() - t0
                print(f"[{m}/{len(jobs)}] labels found {n_found} (straight motion feasible in {n_ret_ok})  "
                      f"({el / 60:.1f} min, ~{el / m * (len(jobs) - m) / 60:.0f} min left)", flush=True)
    np.savez_compressed(args.out, problem=np.array([o[0] for o in out]), mu=np.array([o[1] for o in out]),
                        mass_scale=np.array([o[2] for o in out]), c=np.array([o[3] for o in out]).reshape(-1, 4, 7),
                        T=np.array([o[4] for o in out]), retime_feasible=np.array([o[5] for o in out]))
    print(f"{len(out)} labels -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
