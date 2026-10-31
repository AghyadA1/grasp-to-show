"""Video: the transport network vs the optimiser on the same grasp, from how each makes its path
to how the gripper follows it.

Two acts, one problem of the unseen-object test set (scripts/transport_net_episodes.py):
  1  How each path is made. The optimiser searches: 60 rounds of 32 trial paths (a few of each
     round in grey) that close in on its answer (orange), about 25 s. The network answers in one
     pass (blue), about 1 ms.
  2  Both carries in full simulation, overlaid at half opacity: the planned path (thin) and the
     followed path (thick), with the grip use u(t) of each (the safety model: 1 = slipping, limit
     0.7), and a scoreboard: planning time, carry time, peak grip use, outcome.

Usage:  python scripts/demo_net_vs_optimiser.py --problem 00089 [--fps 30]
Writes media/net_vs_opt_<problem>_<object>.mp4 (and a .gif preview).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
import demo_joint_choice as J
from demo_comparison import plan
from demo_overlay import add_path, write_gif
from eval_way_b import kinematic_choices, way_b_grasps
from transport_net_episodes import net_predict

from grasp2show.experiment import Scenario
from grasp2show.transport import friction_params, inertial_load, robust_utilisation
from grasp2show.transport_opt import TransportProblem

OPT, NET = (235, 120, 40), (60, 140, 240)
J.RUNS = {"optimiser": ("optimiser (search)", OPT), "network": ("network (one pass)", NET)}
W, H, BG = J.W, J.H, J.BG


def setup(problem: str, model: Path, clouds: Path, net: Path):
    from grasp2show.learn.transport_net import features

    z = dict(np.load(ROOT / "data" / "transport_dataset_test" / f"{problem}.npz", allow_pickle=True))
    asset, yaw, xy, mu = str(z["object"]), float(z["yaw"]), tuple(z["start_xy"]), float(z["mu"])
    T, Wd, R, _ = way_b_grasps(asset, yaw, model, clouds, 10, start_xy=xy)
    sc = Scenario(asset, yaw, {"T_obj_tcp": T, "width": Wd, "success": np.ones(len(T), bool)}, start_xy=xy)
    grasp, paths = plan(sc, "way_b", kinematic_choices(sc, list(zip(range(len(T)), R.tolist()))))
    q0, q1 = paths["lift"][-1], paths["present"][-1]
    pr = TransportProblem(sc.obj, grasp.T_obj_tcp, grasp.width, q0, q1, friction_params(mu), checker=sc.fc)
    # the optimiser, with every trial recorded (same call and seed as in the experiments)
    trials, cost = [], pr.cost

    def recording_cost(c, T_):
        trials.append((np.array(c, float).copy(), float(T_)))
        return cost(c, T_)

    pr.cost = recording_cost
    t0 = time.perf_counter()
    ret = pr.retime_only()
    opt = pr.shape_and_time(ret, seed=0)
    t_opt = time.perf_counter() - t0
    pr.cost = cost
    t0 = time.perf_counter()
    c_net, T_net = net_predict(net, features(q0, q1, mu, grasp.T_obj_tcp, grasp.width, sc.obj.mass, sc.obj.com,
                                             sc.obj.inertia))
    t_net = time.perf_counter() - t0
    e_net, e_opt = pr.evaluate(c_net, T_net), pr.evaluate(opt.c, opt.T)
    return {"sc": sc, "mu": mu, "asset": asset, "grasp": grasp, "paths": paths, "pr": pr, "ret": ret,
            "sol": {"optimiser": (opt.c, opt.T, e_opt), "network": (c_net, T_net, e_net)},
            "plan_s": {"optimiser": t_opt, "network": t_net}, "trials": trials}


def tcp_path(S, c, T_) -> np.ndarray:
    sc, pr = S["sc"], S["pr"]
    m, d = sc.ep.model, mujoco.MjData(sc.ep.model)
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    out = []
    for qq in pr.derivs(c, T_)[0]:
        d.qpos[:7] = qq
        mujoco.mj_kinematics(m, d)
        out.append(d.site_xpos[tcp].copy())
    return np.array(out)


def record(S, key: str, fps: int):
    sc, pr = S["sc"], S["pr"]
    c, T_, _ = S["sol"][key]
    m = sc.ep.model
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    frames = []

    def on_frame(d):
        frames.append((d.time, d.qpos.copy(), d.site_xpos[tcp].copy()))

    p = dict(S["paths"])
    traj = pr.sample(c, T_, m.opt.timestep)
    p["transport"] = traj
    res = sc.ep.execute(S["grasp"], p, on_frame=on_frame, fps=fps, mu=S["mu"], log_every=5)
    t_end = frames[-1][0]
    log = res.transport
    t_log_end = t_end - 1.0
    t_log = t_log_end - (len(log.q) - 1 - np.arange(len(log.q))) * log.dt
    G = S["grasp"].T_obj_tcp
    F, Tq = inertial_load(log.T_world_tcp @ np.linalg.inv(G), log.dt, sc.obj.mass, sc.obj.com, sc.obj.inertia)
    u = robust_utilisation(pr.contacts[0], pr.contacts[1], G, sc.obj.com, F, Tq, pr.prm, pr.com_margin)
    u = np.maximum(u, np.abs(F @ G[:3, 1]) / pr.axial_cap)
    print(f"{key}: {res.summary()}  carry {T_:.2f} s")
    return {"t": np.array([f[0] for f in frames]), "q": np.array([f[1] for f in frames]),
            "tcp": np.array([f[2] for f in frames]), "res": res, "u_t": t_log, "u": u,
            "carry": (t_log_end - len(traj) * m.opt.timestep, t_log_end), "u_lim": pr.u_lim}


def fmt_time(s: float) -> str:
    return f"{s * 1e3:.1f} ms" if s < 0.1 else f"{s:.1f} s"


def scoreboard(S, runs, done: bool, size=(440, 300)) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), BG, np.uint8)
    cols = {"optimiser": 200, "network": 330}
    img = J.text(img, "same grasp, same start and end", (12, 6), 14, (170, 175, 185))
    for k, x in cols.items():
        cv2.rectangle(img, (x - 6, 34), (x + 104, 60), J.RUNS[k][1], -1)
        img = J.text(img, k, (x, 37), 15, (255, 255, 255), bold=True)
    rows = [("planning time", {k: fmt_time(S["plan_s"][k]) for k in cols}),
            ("carry time", {k: f"{S['sol'][k][1]:.2f} s" for k in cols}),
            ("peak grip use (sim)", {k: f"{runs[k]['u'].max():.2f}" for k in cols}),
            ("safety rule", {k: "kept" if S["sol"][k][2]["feasible"] else "broken" for k in cols})]
    if done:
        rows.append(("outcome", {k: ("QR read" if runs[k]["res"].success else "failed") +
                                 f", slip {runs[k]['res'].slip_mm:.0f} mm" for k in cols}))
    for i, (name, vals) in enumerate(rows):
        y = 76 + 40 * i
        img = J.text(img, name, (12, y), 15, (200, 200, 205))
        for k, x in cols.items():
            img = J.text(img, vals[k], (x, y), 15 if len(vals[k]) < 12 else 12, (240, 240, 240), bold=True)
        cv2.line(img, (12, y + 30), (w - 12, y + 30), (45, 48, 55), 1)
    return img


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", default="00089")
    p.add_argument("--verdict", default="", help="one line shown under the title of act 2")
    p.add_argument("--net", type=Path, default=ROOT / "data" / "models" / "transport_net_ss_mass.pt")
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    S = setup(args.problem, args.model, args.clouds, args.net)
    print(f"{S['asset']}: optimiser {S['sol']['optimiser'][1]:.2f} s in {S['plan_s']['optimiser']:.1f} s, "
          f"network {S['sol']['network'][1]:.2f} s in {S['plan_s']['network'] * 1e3:.1f} ms; "
          f"{len(S['trials'])} optimiser trials")
    runs = {k: record(S, k, args.fps) for k in J.RUNS}
    painter = J.Painter(S)
    frames = []
    name = f"{S['asset'].replace('_', ' ')[:34]}  ·  {S['sc'].obj.mass:.2f} kg  ·  friction {S['mu']:.2f}"
    t_max = max(rn["t"][-1] for rn in runs.values())

    # act 1: how each path is made (both robots at the start of the carry: the same pose)
    q_start = {k: rn["q"][np.searchsorted(rn["t"], rn["carry"][0])] for k, rn in runs.items()}
    pop = 32
    first, rounds = S["trials"][0], [S["trials"][1 + i * pop:1 + (i + 1) * pop] for i in range(60)]
    rounds = [r for r in rounds if r]
    net_path = tcp_path(S, *S["sol"]["network"][:2])
    opt_final = tcp_path(S, *S["sol"]["optimiser"][:2])
    pts = np.vstack([net_path, opt_final])                    # frame the carry: centre on the two paths
    painter.cam.lookat[:] = 0.5 * (pts.min(0) + pts.max(0)) + np.array([0.0, 0.0, -0.05])
    painter.cam.distance = float(np.clip(2.4 * np.ptp(pts, axis=0).max(), 1.0, 1.6))
    rng = np.random.default_rng(0)
    best, best_cost = first, None
    per = max(int(6.0 * args.fps / len(rounds)), 1)
    for i, rd in enumerate(rounds):
        show = [rd[j] for j in rng.choice(len(rd), min(6, len(rd)), replace=False)]
        greys = [tcp_path(S, c, T_) for c, T_ in show]
        # best so far among the first trials of each round (the first is the search's current best)
        for c, T_ in rd[:8]:
            f = S["pr"].cost(c, T_)[0]
            if best_cost is None or f < best_cost:
                best, best_cost = (c, T_), f
        best_path = tcp_path(S, *best)
        frac = (i + 1) / len(rounds)

        def deco(scn, greys=greys, best_path=best_path):
            for g in greys:
                add_path(scn, g, (200, 202, 210), radius=0.0013)
            add_path(scn, best_path, OPT, radius=0.0026)
            add_path(scn, net_path, NET, radius=0.0026)

        img = J.legend(painter.render(q_start, deco), {
            "optimiser": f"optimiser: round {i + 1}/{len(rounds)}, {fmt_time(frac * S['plan_s']['optimiser'])}",
            "network": f"network: done in {fmt_time(S['plan_s']['network'])}"})
        img = J.text(img, "grey: some of the optimiser's trial paths in this round", (14, img.shape[0] - 30), 15,
                     (230, 230, 230))
        side = scoreboard(S, runs, False)
        frame = J.compose(img, side, J.u_panel(runs, 0, t_max),
                          "1 · How each path is made: a search vs one pass of the network", name)
        frames += [frame] * per
    for _ in range(int(1.5 * args.fps)):
        def deco(scn):
            add_path(scn, opt_final, OPT, radius=0.0026)
            add_path(scn, net_path, NET, radius=0.0026)

        img = J.legend(painter.render(q_start, deco), {
            "optimiser": f"optimiser: {fmt_time(S['plan_s']['optimiser'])} -> carry {S['sol']['optimiser'][1]:.2f} s",
            "network": f"network: {fmt_time(S['plan_s']['network'])} -> carry {S['sol']['network'][1]:.2f} s"})
        frames.append(J.compose(img, scoreboard(S, runs, False), J.u_panel(runs, 0, t_max),
                                "1 · How each path is made: a search vs one pass of the network", name))

    # act 2: both carries in full simulation, real time
    n = max(len(rn["t"]) for rn in runs.values())
    planned = {"optimiser": opt_final, "network": net_path}
    for i in range(n + int(3 * args.fps)):
        t_now = min(i, n - 1) / args.fps
        qs, now = {}, {}
        for k, rn in runs.items():
            j = min(i, len(rn["t"]) - 1)
            qs[k], now[k] = rn["q"][j], rn["tcp"][:j + 1]

        def deco(scn, now=now):
            for k in J.RUNS:
                add_path(scn, planned[k], J.RUNS[k][1], radius=0.0012)
                add_path(scn, now[k], J.RUNS[k][1], radius=0.0034)

        lines = {}
        for k, rn in runs.items():
            a0, b0 = rn["carry"]
            st = "waiting to carry" if t_now < a0 else (f"carrying  {t_now - a0:.2f} s" if t_now < b0 else
                                                        f"carry done in {b0 - a0:.2f} s")
            lines[k] = f"{k}:  {st}"
        img = J.legend(painter.render(qs, deco), lines)
        frames.append(J.compose(img, scoreboard(S, runs, i >= n), J.u_panel(runs, t_now, t_max),
                                "2 · Both carries in full simulation (half opacity)",
                                args.verdict or "Thin: the planned path.  Thick: the path the gripper followed."))
    painter.close()
    stem = ROOT / "media" / f"net_vs_opt_{args.problem}_{S['asset'][:20]}"
    imageio.mimwrite(f"{stem}.mp4", frames, fps=args.fps, quality=8, macro_block_size=8)
    write_gif(frames, args.fps, Path(f"{stem}.gif"), gif_fps=8, width=640)
    print(f"wrote {stem}.mp4 ({len(frames) / args.fps:.0f} s) and .gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
