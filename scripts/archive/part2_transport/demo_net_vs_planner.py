"""Video: the planner's carry vs the transport network's, same grasp, same start and end.

The planner (episode.PickAndShow.plan_analytic, what every episode used before part 2): a joint-
space move whose duration comes from the largest joint step, then a straight line to the scanner,
so the hand stops in between. The network (part 2, self-supervised, transport_net_ss_mass.pt): one
smooth motion from the lift to the scanner in one pass (~1 ms). Both run in full simulation,
overlaid at half opacity, with the hand's speed over time underneath.

--pick: scan the unseen-object test problems (data/transport_dataset_test) and list those where
both succeed, with the network's carry time relative to the planner's; the demo should show a
typical case (near the median), not the most extreme.

Usage:  python scripts/archive/part2_transport/demo_net_vs_planner.py --pick
        python scripts/archive/part2_transport/demo_net_vs_planner.py --problem 00012
Writes media/net_vs_planner_<problem>_<object>.mp4 and .gif.
"""
from __future__ import annotations

import argparse
import json
import sys
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
from grasp2show.transport import count_stops, friction_params
from grasp2show.transport_opt import TransportProblem

PLAN, NET = (235, 120, 40), (60, 140, 240)
J.RUNS = {"planner": ("planner", PLAN), "network": ("network", NET)}
W, H, BG = J.W, J.H, J.BG
NETW = ROOT / "data" / "models" / "transport_net_ss_mass.pt"
MODEL = ROOT / "data" / "models" / "heatmap_net.pt"
CLOUDS = ROOT / "data" / "dataset" / "clouds.npz"


def setup(problem: str):
    from grasp2show.learn.transport_net import features

    z = dict(np.load(ROOT / "data" / "transport_dataset_test" / f"{problem}.npz", allow_pickle=True))
    asset, yaw, xy, mu = str(z["object"]), float(z["yaw"]), tuple(z["start_xy"]), float(z["mu"])
    T, Wd, R, _ = way_b_grasps(asset, yaw, MODEL, CLOUDS, 10, start_xy=xy)
    sc = Scenario(asset, yaw, {"T_obj_tcp": T, "width": Wd, "success": np.ones(len(T), bool)}, start_xy=xy)
    grasp, paths = plan(sc, "way_b", kinematic_choices(sc, list(zip(range(len(T)), R.tolist()))))
    q0, q1 = paths["lift"][-1], paths["present"][-1]
    pr = TransportProblem(sc.obj, grasp.T_obj_tcp, grasp.width, q0, q1, friction_params(mu), checker=sc.fc)
    c, T_net = net_predict(NETW, features(q0, q1, mu, grasp.T_obj_tcp, grasp.width, sc.obj.mass, sc.obj.com,
                                          sc.obj.inertia))
    seg = {n: d for n, _, d in sc.ep.SEGMENTS}
    dq = float(np.abs(np.diff(np.asarray(paths["transit"]), axis=0)).max())
    T_plan = max(seg["transit"], 1.7 * dq) + seg["present"]
    return {"sc": sc, "mu": mu, "asset": asset, "grasp": grasp, "paths": paths, "pr": pr, "c": c, "T_net": T_net,
            "T_plan": T_plan, "feasible": pr.evaluate(c, T_net)["feasible"]}


def record(S, key: str, fps: int):
    sc = S["sc"]
    m = sc.ep.model
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    frames = []

    def on_frame(d):
        frames.append((d.time, d.qpos.copy(), d.site_xpos[tcp].copy()))

    p = dict(S["paths"])
    if key == "network":
        p["transport"] = S["pr"].sample(S["c"], S["T_net"], m.opt.timestep)
    res = sc.ep.execute(S["grasp"], p, on_frame=on_frame, fps=fps, mu=S["mu"])
    t = np.array([f[0] for f in frames])
    xyz = np.array([f[2] for f in frames])
    speed = np.r_[0.0, np.linalg.norm(np.diff(xyz, axis=0), axis=1) * fps]
    carry_T = S["T_net"] if key == "network" else S["T_plan"]
    t_end = t[-1] - 1.0                                  # the 1 s hold at the scanner
    a = t_end - carry_T
    sel = (t >= a + 0.15) & (t <= t_end - 0.15)          # inside the carry: pauses on the way
    return {"t": t, "q": np.array([f[1] for f in frames]), "tcp": xyz, "speed": speed, "res": res,
            "carry": (a, t_end), "stops": count_stops(speed[sel]), "peak": float(speed[(t >= a) & (t <= t_end)].max())}


def speed_panel(runs, t_now, t_min, t_max, size=(440, 270)) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), BG, np.uint8)
    x0, x1, y0, y1 = 46, w - 12, 34, h - 34
    vmax = 1.1 * max(rn["peak"] for rn in runs.values())
    X = lambda t: int(x0 + (x1 - x0) * np.clip((t - t_min) / (t_max - t_min), 0, 1))
    Y = lambda v: int(y1 - (y1 - y0) * np.clip(v / vmax, 0, 1))
    cv2.line(img, (x0, y1), (x1, y1), (90, 90, 95), 1)
    cv2.line(img, (x0, y0), (x0, y1), (90, 90, 95), 1)
    for key, rn in runs.items():
        sel = (rn["t"] <= t_now) & (rn["t"] >= t_min)
        if sel.sum() > 1:
            pts = np.array([[X(t), Y(v)] for t, v in zip(rn["t"][sel], rn["speed"][sel])], np.int32)
            cv2.polylines(img, [pts], False, J.RUNS[key][1], 2, cv2.LINE_AA)
        b = rn["carry"][1]
        if t_now >= b:
            cv2.line(img, (X(b), y1 - 6), (X(b), y1 + 6), J.RUNS[key][1], 2)
    img = J.text(img, "speed of the hand (m/s): lift, then the carry", (x0, 8), 14, (200, 200, 205))
    img = J.text(img, f"{vmax / 1.1:.2f}", (x0 - 6, Y(vmax / 1.1) - 7), 11, (150, 150, 155), anchor="ra")
    img = J.text(img, "ticks: carry done", (x1, y1 + 8), 11, (150, 150, 155), anchor="ra")
    return img


def scoreboard(S, runs, done: bool, size=(440, 300)) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), BG, np.uint8)
    cols = {"planner": 200, "network": 330}
    img = J.text(img, "same grasp, same start and end", (12, 6), 14, (170, 175, 185))
    for k, x in cols.items():
        cv2.rectangle(img, (x - 6, 34), (x + 104, 60), J.RUNS[k][1], -1)
        img = J.text(img, k, (x, 37), 15, (255, 255, 255), bold=True)
    rows = [("carry time", {"planner": f"{S['T_plan']:.2f} s", "network": f"{S['T_net']:.2f} s"}),
            ("stops on the way", {k: str(runs[k]["stops"]) for k in cols})]
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


def pick() -> int:
    rows = []
    for f in sorted((ROOT / "data" / "transport_dataset_test").glob("*.npz")):
        try:
            S = setup(f.stem)
        except SystemExit:
            continue
        if not S["feasible"]:
            continue
        runs = {k: record(S, k, 30) for k in J.RUNS}
        if not all(r["res"].success for r in runs.values()):
            continue
        rows.append({"problem": f.stem, "object": S["asset"], "mass": round(S["sc"].obj.mass, 2), "mu": S["mu"],
                     "planner_s": round(S["T_plan"], 2), "network_s": round(S["T_net"], 2),
                     "ratio": round(S["T_net"] / S["T_plan"], 2), "stops": [runs[k]["stops"] for k in J.RUNS]})
        print(rows[-1], flush=True)
    r = np.array([x["ratio"] for x in rows])
    print(f"{len(rows)} problems where both succeed: network/planner carry time median {np.median(r):.2f} "
          f"(quartiles {np.quantile(r, 0.25):.2f}-{np.quantile(r, 0.75):.2f})")
    (ROOT / "data" / "transport_net_episodes" / "vs_planner.json").write_text(json.dumps(rows, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", default="00012")
    p.add_argument("--pick", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    if args.pick:
        return pick()
    S = setup(args.problem)
    runs = {k: record(S, k, args.fps) for k in J.RUNS}
    for k, rn in runs.items():
        print(f"{k}: {rn['res'].summary()}  carry {rn['carry'][1] - rn['carry'][0]:.2f} s, {rn['stops']} stops")
    painter = J.Painter(S)
    pts = np.vstack([rn["tcp"][(rn["t"] >= rn["carry"][0])] for rn in runs.values()])
    painter.cam.lookat[:] = 0.5 * (pts.min(0) + pts.max(0)) + np.array([0.0, 0.0, -0.05])
    painter.cam.distance = float(np.clip(3.0 * np.ptp(pts, axis=0).max(), 1.3, 1.9))
    name = f"{S['asset'].replace('_', ' ')[:34]}  ·  {S['sc'].obj.mass:.2f} kg  ·  friction {S['mu']:.2f}"
    t_min = min(rn["carry"][0] for rn in runs.values()) - 1.5
    t_max = max(rn["t"][-1] for rn in runs.values())
    n = max(len(rn["t"]) for rn in runs.values())
    i0 = int(np.searchsorted(runs["planner"]["t"], t_min))
    frames = []
    for i in range(i0, n + int(2 * args.fps)):
        t_now = min(i, n - 1) / args.fps
        qs, now = {}, {}
        for k, rn in runs.items():
            j = min(i, len(rn["t"]) - 1)
            qs[k], now[k] = rn["q"][j], rn["tcp"][i0:j + 1]

        def deco(scn, now=now):
            for k in J.RUNS:
                add_path(scn, now[k], J.RUNS[k][1], radius=0.0034)

        lines = {}
        for k, rn in runs.items():
            a, b = rn["carry"]
            st = "lifting" if t_now < a else (f"carrying  {t_now - a:.1f} s" if t_now < b else
                                              f"at the scanner after {b - a:.1f} s")
            lines[k] = f"{k}:  {st}"
        img = J.legend(painter.render(qs, deco), lines)
        frames.append(J.compose(img, scoreboard(S, runs, i >= n), speed_panel(runs, t_now, t_min, t_max),
                                "The planner's carry vs the network's (same grasp, full simulation)",
                                name + "  ·  the network plans the whole motion at once"))
    painter.close()
    stem = ROOT / "media" / f"net_vs_planner_{args.problem}_{S['asset'][:20]}"
    imageio.mimwrite(f"{stem}.mp4", frames, fps=args.fps, quality=8, macro_block_size=8)
    write_gif(frames, args.fps, Path(f"{stem}.gif"), gif_fps=10, width=720)
    print(f"wrote {stem}.mp4 ({len(frames) / args.fps:.0f} s) and .gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
