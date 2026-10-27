"""Video: what way B proposes, how the network turns each grasp into a carry, and why choosing the
grasp together with the motion is faster.

Three acts, one problem of the unseen-object test set (scripts/joint_grasp_motion.py):
  1  Way B's candidate grasps on the object (coloured by way B's score) and, for each, the carry
     time of the network's motion (about 1 ms each): way B's first choice (orange) and the joint
     choice (blue: nearly the same score, much shorter carry).
  2  How a carry is made: the network starts from the plain joint-space blend between the lifted
     pose and the presentation pose (grey) and bends it (coloured), so the load stays where the
     grasp is strong.
  3  Both episodes in full simulation, overlaid at half opacity: the planned path (thin) and the
     path the gripper actually followed (thick), with the grip use u(t) of each (the safety model:
     1 = slipping, limit 0.7) and the carry timers.

Usage:  python scripts/demo_joint_choice.py [--problem 00098] [--fps 30]
Writes media/joint_choice_<object>.mp4 (and a .gif preview).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from demo_overlay import add_path, robot_geoms, write_gif
from eval_way_b import _load, kinematic_choices, way_b_grasps
from joint_grasp_motion import candidates
from transport_net_episodes import net_predict
from visualize_way_b import draw, gripper_lines

from grasp2show.config import CameraConfig, ObjectConfig, SceneConfig
from grasp2show.experiment import Scenario
from grasp2show.learn.dense import condition, split_outputs
from grasp2show.objects import object_from_config
from grasp2show.scene import start_pose
from grasp2show.task import show_pose
from grasp2show.transport import friction_params, inertial_load, robust_utilisation

W, H = 1280, 720
BG = (17, 19, 23)
ORANGE, BLUE, GREY = (235, 120, 40), (60, 140, 240), (215, 218, 225)
RUNS = {"way_b": ("way B's first grasp", ORANGE), "joint": ("grasp chosen with the motion", BLUE)}


# -- text -----------------------------------------------------------------------------------------
def _font(size: int, bold: bool = False):
    names = (["Inter-SemiBold.otf", "Inter-Bold.otf"] if bold else ["Inter-Regular.otf"]) + \
        (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"])
    for n in names:
        for d in ("/usr/share/fonts/opentype/inter/", "/usr/share/fonts/truetype/dejavu/", ""):
            try:
                return ImageFont.truetype(d + n, size)
            except OSError:
                continue
    return ImageFont.load_default()


FONTS = {}


def text(img: np.ndarray, s: str, xy, size=18, color=(235, 235, 235), bold=False, anchor="la") -> np.ndarray:
    key = (size, bold)
    if key not in FONTS:
        FONTS[key] = _font(size, bold)
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text(xy, s, font=FONTS[key], fill=tuple(color), anchor=anchor)
    return np.asarray(im).copy()


# -- the problem ------------------------------------------------------------------------------------
def setup(problem: str, model: Path, clouds_path: Path, net: Path, delta: float):
    z = dict(np.load(ROOT / "data" / "transport_dataset_test" / f"{problem}.npz", allow_pickle=True))
    asset, yaw, xy, mu = str(z["object"]), float(z["yaw"]), tuple(z["start_xy"]), float(z["mu"])
    T, Wd, R, _, Q = way_b_grasps(asset, yaw, model, clouds_path, 10, start_xy=xy, with_scores=True)
    sc = Scenario(asset, yaw, {"T_obj_tcp": T, "width": Wd, "success": np.ones(len(T), bool)}, start_xy=xy)
    cands = candidates(sc, kinematic_choices(sc, list(zip(range(len(T)), R.tolist()))))
    from grasp2show.learn.transport_net import features
    from grasp2show.transport_opt import TransportProblem

    prm = friction_params(mu)
    info = []
    for c, g, paths in cands:
        q0, q1 = paths["lift"][-1], paths["present"][-1]
        pr = TransportProblem(sc.obj, g.T_obj_tcp, g.width, q0, q1, prm, checker=sc.fc)
        coef, T_pred = net_predict(net, features(q0, q1, mu, g.T_obj_tcp, g.width, sc.obj.mass, sc.obj.com,
                                                 sc.obj.inertia))
        e = pr.evaluate(coef, T_pred)
        info.append({"choice": c, "grasp": g, "paths": paths, "score": float(Q[c.grasp]), "T": T_pred, "c": coef,
                     "feasible": bool(e["feasible"]), "pr": pr})
    best_q = info[0]["score"]
    ok = [i for i, d in enumerate(info) if d["feasible"] and d["score"] >= best_q - delta]
    pick = {"way_b": 0, "joint": min(ok, key=lambda i: info[i]["T"]) if ok else 0}
    # way B's score map on the object: best over the 12 presentation rolls
    netB, clouds = _load(model, clouds_path)
    o = clouds.names.index(asset)
    cfg = SceneConfig(obj=ObjectConfig(asset=asset, start_yaw=np.radians(yaw), start_xy=xy))
    obj = object_from_config(cfg.obj)
    rolls = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    cond = np.stack([condition(start_pose(cfg, obj), show_pose(cfg, obj, r), CameraConfig().pos, clouds.mass[o],
                               clouds.com[o]) for r in rolls])
    import torch

    with torch.no_grad():
        pr_ = split_outputs(netB(torch.full((12,), o), torch.as_tensor(cond)))
    q = (1 / (1 + np.exp(-np.clip(pr_["logit"].numpy(), -50, 50))) * pr_["score"].numpy()).max(0)
    return {"sc": sc, "mu": mu, "info": info, "pick": pick, "delta": delta, "heat": q,
            "pts": clouds.feats[o, :, :3], "obj": obj, "asset": asset}


# -- act 1: way B's candidates and the network's carry time for each ----------------------------------
def candidates_figure(S, azim_offset: float, size=(12.8, 6.0), small=False) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    info, pick = S["info"], S["pick"]
    fig = plt.figure(figsize=size, dpi=100, facecolor=np.array(BG) / 255)
    ax = fig.add_axes([0.0, 0.0, 0.52, 1.0] if not small else [0, 0, 1, 1], projection="3d")
    ax.set_facecolor(np.array(BG) / 255)
    heat = (S["heat"] - S["heat"].min()) / max(np.ptp(S["heat"]), 1e-9)
    ang = np.radians(-60 + azim_offset)
    draw(ax, S["obj"], S["pts"], heat, np.array([np.cos(ang), np.sin(ang), 0.25]), cm.viridis)
    order = [i for i in range(len(info)) if i not in pick.values()] + [pick["way_b"], pick["joint"]]
    for i in order:
        col = ORANGE if i == pick["way_b"] else BLUE if i == pick["joint"] else (120, 125, 135)
        lw = 3.2 if i in pick.values() else 1.0
        for s in gripper_lines(info[i]["grasp"].T_obj_tcp, info[i]["grasp"].width):
            ax.plot(*s.T, c=np.array(col) / 255, lw=lw, zorder=8 if i in pick.values() else 6,
                    solid_capstyle="round", alpha=1.0 if i in pick.values() else 0.7)
    if small:
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img
    fig.text(0.03, 0.93, "object coloured by way B's score", color="0.75", fontsize=11)
    cb = fig.add_axes([0.03, 0.08, 0.18, 0.018])
    cb.imshow(np.linspace(0, 1, 256)[None], cmap="viridis", aspect="auto")
    cb.set_xticks([0, 255], ["low", "high"], color="0.7", fontsize=9)
    cb.set_yticks([])
    for sp in cb.spines.values():
        sp.set_visible(False)
    bx = fig.add_axes([0.62, 0.14, 0.34, 0.72], facecolor=np.array(BG) / 255)
    n = len(info)
    best_q = info[0]["score"]
    for i, d in enumerate(info):
        col = ORANGE if i == pick["way_b"] else BLUE if i == pick["joint"] else (120, 125, 135)
        near = d["score"] >= best_q - S["delta"]
        bx.barh(i, d["T"], color=np.array(col) / 255, alpha=1.0 if near else 0.35, height=0.7,
                hatch=None if d["feasible"] else "///", edgecolor=np.array(BG) / 255)
        bx.text(d["T"] + 0.05, i, f"{d['T']:.2f} s" + ("" if d["feasible"] else "  unsafe"), va="center",
                color="0.85", fontsize=10)
        bx.text(-0.08, i, f"#{i + 1}  {d['score']:.3f}", va="center", ha="right", color="0.85" if near else "0.45",
                fontsize=10, family="monospace")
    bx.set_ylim(n - 0.5, -0.5)
    bx.set_xlim(0, max(d["T"] for d in info) * 1.35)
    bx.set_yticks([])
    bx.tick_params(colors="0.6", labelsize=9)
    for sp in ("top", "right", "left"):
        bx.spines[sp].set_visible(False)
    bx.spines["bottom"].set_color("0.4")
    bx.set_xlabel("carry time of the network's motion (s)", color="0.75", fontsize=10)
    fig.text(0.52, 0.89, "rank  way-B score", color="0.6", fontsize=10, family="monospace")
    fig.text(0.62, 0.93, "faded: score more than 0.02 below the best;  hatched: fails the safety check",
             color="0.6", fontsize=9)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


# -- episodes ---------------------------------------------------------------------------------------
def record(S, key: str, fps: int):
    sc, d0 = S["sc"], S["info"][S["pick"][key]]
    m = sc.ep.model
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    frames = []

    def on_frame(d):
        frames.append((d.time, d.qpos.copy(), d.site_xpos[tcp].copy()))

    p = dict(d0["paths"])
    traj = d0["pr"].sample(d0["c"], d0["T"], m.opt.timestep)
    p["transport"] = traj
    res = sc.ep.execute(d0["grasp"], p, on_frame=on_frame, fps=fps, mu=S["mu"], log_every=5)
    t_end = frames[-1][0]
    log = res.transport
    t_log_end = t_end - 1.0                      # the episode ends with a 1 s hold at the presentation pose
    t_log = t_log_end - (len(log.q) - 1 - np.arange(len(log.q))) * log.dt
    pr = d0["pr"]
    T_rigid = log.T_world_tcp @ np.linalg.inv(d0["grasp"].T_obj_tcp)
    F, Tq = inertial_load(T_rigid, log.dt, sc.obj.mass, sc.obj.com, sc.obj.inertia)
    u = robust_utilisation(pr.contacts[0], pr.contacts[1], d0["grasp"].T_obj_tcp, sc.obj.com, F, Tq, pr.prm,
                           pr.com_margin)
    u = np.maximum(u, np.abs(F @ d0["grasp"].T_obj_tcp[:3, 1]) / pr.axial_cap)
    carry = (t_log_end - len(traj) * m.opt.timestep, t_log_end)
    print(f"{key}: {res.summary()}  carry {carry[1] - carry[0]:.2f} s")
    return {"t": np.array([f[0] for f in frames]), "q": np.array([f[1] for f in frames]),
            "tcp": np.array([f[2] for f in frames]), "res": res, "u_t": t_log, "u": u, "carry": carry,
            "u_lim": pr.u_lim}


def planned_path(S, key: str, alpha: float) -> np.ndarray:
    """TCP positions (world) along the planned carry with the network's bends scaled by alpha."""
    sc, d0 = S["sc"], S["info"][S["pick"][key]]
    m, d = sc.ep.model, mujoco.MjData(sc.ep.model)
    q = d0["pr"].derivs(alpha * d0["c"], d0["T"])[0]
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    out = []
    for qq in q:
        d.qpos[:7] = qq
        mujoco.mj_kinematics(m, d)
        out.append(d.site_xpos[tcp].copy())
    return np.array(out)


def add_dots(scn, pts, rgb, radius=0.0032, every=2):
    rgba = np.array([*np.asarray(rgb) / 255, 1.0], dtype=np.float32)
    for p in pts[::every]:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius), p, np.eye(3).ravel(), rgba)
        scn.ngeom += 1


class Painter:
    """Overlay renderer: each run's robot tinted in its colour, the two layers blended 50/50."""

    def __init__(self, S):
        self.sc = S["sc"]
        m = self.m = self.sc.ep.model
        self.d = mujoco.MjData(m)
        self.r = self.sc.ep.renderer
        self.geoms = robot_geoms(m)
        self.base_rgba, self.base_mat = m.geom_rgba[self.geoms].copy(), m.geom_matid[self.geoms].copy()
        paint = np.where((self.base_mat >= 0)[:, None], m.mat_rgba[np.maximum(self.base_mat, 0), :3],
                         self.base_rgba[:, :3])
        self.shade = np.clip(paint.mean(1, keepdims=True) * 1.3, 0.25, 1.0)
        m.geom_matid[self.geoms] = -1
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.45, -0.05, 0.35]
        cam.distance, cam.azimuth, cam.elevation = 1.55, 205.0, -22.0
        self.cam = cam

    def render(self, qs: dict, decorate) -> np.ndarray:
        layers = []
        for key, q in qs.items():
            self.m.geom_rgba[self.geoms, :3] = self.shade * np.asarray(RUNS[key][1]) / 255
            self.d.qpos[:] = q
            mujoco.mj_forward(self.m, self.d)
            self.r.update_scene(self.d, camera=self.cam)
            decorate(self.r.scene)
            layers.append(self.r.render().astype(np.float32))
        return (0.5 * layers[0] + 0.5 * layers[1]).astype(np.uint8)

    def close(self):
        self.m.geom_rgba[self.geoms], self.m.geom_matid[self.geoms] = self.base_rgba, self.base_mat


def u_panel(runs, t_now, t_max, size=(440, 270)) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), BG, np.uint8)
    x0, x1, y0, y1 = 46, w - 12, 34, h - 34
    umax = 1.1
    t_min = min(rn["u_t"][0] for rn in runs.values()) - 0.3       # from the lift on (the grip holds the object)
    X = lambda t: int(x0 + (x1 - x0) * np.clip((t - t_min) / (t_max - t_min), 0, 1))
    Y = lambda u: int(y1 - (y1 - y0) * np.clip(u / umax, 0, 1))
    for key, rn in runs.items():                                  # carry windows
        a, b = rn["carry"]
        ov = img.copy()
        cv2.rectangle(ov, (X(a), y0), (X(min(b, t_now)), y1), RUNS[key][1], -1)
        if t_now > a:
            img = cv2.addWeighted(ov, 0.13, img, 0.87, 0)
    cv2.line(img, (x0, y1), (x1, y1), (90, 90, 95), 1)
    cv2.line(img, (x0, y0), (x0, y1), (90, 90, 95), 1)
    for u in (0.5, 1.0):
        cv2.line(img, (x0 - 4, Y(u)), (x0, Y(u)), (90, 90, 95), 1)
    lim = min(rn["u_lim"] for rn in runs.values())
    for xx in range(x0, x1, 10):
        cv2.line(img, (xx, Y(lim)), (min(xx + 5, x1), Y(lim)), (200, 70, 70), 1)
    for key, rn in runs.items():
        sel = rn["u_t"] <= t_now
        if sel.sum() > 1:
            pts = np.array([[X(t), Y(u)] for t, u in zip(rn["u_t"][sel], rn["u"][sel])], np.int32)
            cv2.polylines(img, [pts], False, RUNS[key][1], 2, cv2.LINE_AA)
    img = text(img, "grip use u(t)   (safety model: 1 = slipping)", (x0, 8), 14, (200, 200, 205))
    img = text(img, f"limit {lim:.2f}", (x1, Y(lim) - 18), 12, (220, 110, 110), anchor="ra")
    img = text(img, "1.0", (x0 - 8, Y(1.0) - 7), 11, (150, 150, 155), anchor="ra")
    img = text(img, "0.5", (x0 - 8, Y(0.5) - 7), 11, (150, 150, 155), anchor="ra")
    img = text(img, "from the lift to the end  ·  shaded: carry", (x1, y1 + 8), 11, (150, 150, 155), anchor="ra")
    return img


def compose(main: np.ndarray, side_top: np.ndarray, side_bottom: np.ndarray, title: str, sub: str) -> np.ndarray:
    canvas = np.full((H, W, 3), BG, np.uint8)
    canvas = text(canvas, title, (24, 16), 24, bold=True)
    canvas = text(canvas, sub, (24, 50), 15, (170, 175, 185))
    mw, mh = 800, 600
    canvas[96:96 + mh, 16:16 + mw] = cv2.resize(main, (mw, mh), interpolation=cv2.INTER_AREA)
    sw = W - mw - 48
    st = cv2.resize(side_top, (sw, int(side_top.shape[0] * sw / side_top.shape[1])), interpolation=cv2.INTER_AREA)
    canvas[96:96 + st.shape[0], mw + 32:mw + 32 + sw] = st
    sb = cv2.resize(side_bottom, (sw, int(side_bottom.shape[0] * sw / side_bottom.shape[1])),
                    interpolation=cv2.INTER_AREA)
    y = H - 16 - sb.shape[0]
    canvas[y:y + sb.shape[0], mw + 32:mw + 32 + sw] = sb
    return canvas


def legend(img, runs_text: dict) -> np.ndarray:
    for k, (key, s) in enumerate(runs_text.items()):
        y = 14 + 28 * k
        cv2.rectangle(img, (14, y + 3), (30, y + 19), RUNS[key][1], -1)
        img = text(img, s, (38, y), 17, (240, 240, 240), bold=True)
    return img


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", default="00098")
    p.add_argument("--net", type=Path, default=ROOT / "data" / "models" / "transport_net_ss_mass.pt")
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--delta", type=float, default=0.02)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    S = setup(args.problem, args.model, args.clouds, args.net, args.delta)
    info, pick = S["info"], S["pick"]
    a, b = info[pick["way_b"]], info[pick["joint"]]
    print(f"{S['asset']}: {len(info)} plannable grasps; way B #1 carry {a['T']:.2f} s, "
          f"joint #{pick['joint'] + 1} carry {b['T']:.2f} s")
    from grasp2show.franka_ik import fk

    def turn(i):
        pr = info[i]["pr"]
        R = fk(pr.q0)[:3, :3].T @ fk(pr.q1)[:3, :3]
        return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))), float(np.abs(pr.dq).max())

    (ra, ja), (rb, jb) = turn(pick["way_b"]), turn(pick["joint"])
    same_spot = np.linalg.norm(a["grasp"].T_obj_tcp[:3, 3] - b["grasp"].T_obj_tcp[:3, 3]) < 0.005
    story1 = (f"#1 and #{pick['joint'] + 1} hold the object at the same spot; #{pick['joint'] + 1} has the hand "
              f"flipped and shows the code the other way up (still readable)." if same_spot else
              f"#{pick['joint'] + 1} holds the object elsewhere with nearly the same score.") + \
        f"  The hand turns {rb:.0f} deg on the way instead of {ra:.0f} deg."
    story2 = (f"Orange must turn the hand {ra:.0f} deg on the way (joints up to {ja:.1f} rad), blue {rb:.0f} deg "
              f"({jb:.1f} rad).  The bends are 28 numbers, predicted in one pass.")
    runs = {k: record(S, k, args.fps) for k in RUNS}
    frames = []

    # act 1: the candidates, object turning
    n1 = int(5.5 * args.fps)
    for k in range(0, n1, 2):
        img = candidates_figure(S, azim_offset=60 * k / n1)
        canvas = np.full((H, W, 3), BG, np.uint8)
        canvas = text(canvas, "1 · Way B proposes grasps; the network plans a carry for each in about 1 ms",
                      (24, 16), 24, bold=True)
        canvas = text(canvas, f"{S['asset'].replace('_', ' ')[:40]}  ·  {S['obj'].mass:.2f} kg  ·  friction "
                      f"{S['mu']:.2f}.   Orange: way B's first choice.   Blue: nearly the same score, "
                      f"{a['T'] / b['T']:.1f}x shorter carry.", (24, 50), 15, (170, 175, 185))
        canvas = text(canvas, story1, (24, H - 30), 15, (215, 215, 220))
        canvas[76:76 + min(img.shape[0], H - 120), :img.shape[1]] = img[:H - 120, :W]
        frames += [canvas, canvas]

    painter = Painter(S)
    small = candidates_figure(S, 30, size=(4.4, 3.3), small=True)
    t_max = max(rn["t"][-1] for rn in runs.values())
    # act 2: from the plain blend to the network's carry (both robots at the start of their carry)
    q_start = {k: rn["q"][np.searchsorted(rn["t"], rn["carry"][0])] for k, rn in runs.items()}
    plain = {k: planned_path(S, k, 0.0) for k in RUNS}
    n2 = int(4 * args.fps)
    for k in range(n2):
        al = float(np.clip((k - 0.25 * n2) / (0.55 * n2), 0, 1))
        al = al * al * (3 - 2 * al)
        bent = {kk: planned_path(S, kk, al) for kk in RUNS}

        def deco(scn, bent=bent):
            for kk in RUNS:
                add_dots(scn, plain[kk], GREY)
                add_path(scn, bent[kk], RUNS[kk][1], radius=0.0024)

        img = legend(painter.render(q_start, deco), {kk: RUNS[kk][0] for kk in RUNS})
        img = text(img, f"grey: plain joint-space blend   ·   colour: the network's carry ({int(100 * al)} % bent)",
                   (14, img.shape[0] - 30), 15, (230, 230, 230))
        frames.append(compose(img, small, u_panel(runs, 0, t_max),
                              "2 · The network bends the plain blend so the load stays where the grip is strong",
                              story2))

    # act 3: both episodes, real time
    n3 = max(len(rn["t"]) for rn in runs.values())
    planned = {k: planned_path(S, k, 1.0) for k in RUNS}
    for i in range(n3 + int(3 * args.fps)):
        t_now = min(i, n3 - 1) / args.fps
        qs, paths_now = {}, {}
        for k, rn in runs.items():
            j = min(i, len(rn["t"]) - 1)
            qs[k], paths_now[k] = rn["q"][j], rn["tcp"][:j + 1]

        def deco(scn, paths_now=paths_now):
            for kk in RUNS:
                add_path(scn, planned[kk], RUNS[kk][1], radius=0.0012)
                add_path(scn, paths_now[kk], RUNS[kk][1], radius=0.0034)

        lines = {}
        for k, rn in runs.items():
            a0, b0 = rn["carry"]
            tc = np.clip(t_now - a0, 0, b0 - a0)
            state = "waiting to carry" if t_now < a0 else (f"carrying  {tc:.2f} s" if t_now < b0 else
                                                           f"carry done in {b0 - a0:.2f} s")
            lines[k] = f"{RUNS[k][0]}:  {state}"
        img = legend(painter.render(qs, deco), lines)
        if i >= n3:
            for k2, (k, rn) in enumerate(runs.items()):
                res = rn["res"]
                ok = res.success
                s = (f"QR read, slip {res.slip_mm:.0f} mm" if ok else f"failed: slip {res.slip_mm:.0f} mm")
                img = text(img, s, (14 + 360 * k2, img.shape[0] - 34), 16, RUNS[k][1],
                           bold=True)
        frames.append(compose(img, small, u_panel(runs, t_now, t_max),
                              "3 · Both carries in full simulation (half opacity)",
                              "Thin: the planned path.  Thick: the path the gripper followed.  Same object, same "
                              "start, same scanner; only the grasp, and so the motion, differ."))
    painter.close()
    stem = ROOT / "media" / f"joint_choice_{S['asset'][:24]}"
    imageio.mimwrite(f"{stem}.mp4", frames, fps=args.fps, quality=8, macro_block_size=8)
    write_gif(frames, args.fps, Path(f"{stem}.gif"), gif_fps=8, width=640)
    print(f"wrote {stem}.mp4 ({len(frames) / args.fps:.0f} s) and .gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
