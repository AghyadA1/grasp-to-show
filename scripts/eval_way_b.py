"""Way B in full pick-and-show episodes on unseen objects, against way A and the earlier policies.

For each scenario (10 unseen objects x 8 start rotations):
  way_b          the heatmap network predicts, for each of the 12 target poses, a score and a grasp
                 at every surface point; the best (point, target) pairs become grasps; the first 10
                 whose open gripper clears the object (as for sampled candidates) go through the
                 exact IK/collision checks (start and target reach only, like every learned policy)
                 and the usual planning fallback
  way_a          all checks on ~1,000 dense candidates (the label generator; slow), best score
  aware          analytic checks on the labelled candidate set (week 1)
  stability_only the baseline
  way_a_smooth   (optional) way A ranking the same feasible candidates by the score averaged over
                 their neighbours: tests why the network can beat the search it imitates
Usage:  python scripts/eval_way_b.py [--workers 4]
        python scripts/eval_way_b.py --policies way_a way_a_smooth way_b --tag way_a_smooth
        python scripts/eval_way_b.py --policies way_b --model data/models/heatmap_net_s1.pt --tag way_b_s1
Outputs: data/episodes_<tag>/records.jsonl, summary.json, media/<tag>_policy.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import generate_dense_labels as gdl
from run_experiments import wilson

from grasp2show.config import CameraConfig
from grasp2show.experiment import Choice, Scenario, choose
from grasp2show.objects import load_object_set
from grasp2show.task import show_pose

GRASPS = ROOT / "data" / "grasps"
_net = {}


def _load(model_path: Path, clouds_path: Path):
    if not _net:
        import torch

        from grasp2show.learn.data import load_clouds
        from grasp2show.learn.dense import HeatmapNet

        torch.set_num_threads(1)
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        clouds = load_clouds(clouds_path)
        net = HeatmapNet(torch.as_tensor(clouds.feats))
        net.load_state_dict(ck["state_dict"])
        _net.update(net=net.eval(), clouds=clouds)
    return _net["net"], _net["clouds"]


def kinematic_choices(sc: Scenario, order: list[tuple[int, int]]) -> list[Choice]:
    """(grasp, target) pairs in preference order -> choices reachable at start and target (best twin)."""
    out = []
    for i, j in order:
        best, key = None, -np.inf
        for tw in sc.reports[i].twins:
            if tw.start_ok and j < len(tw.targets) and tw.targets[j].reach.ok:
                m = min(tw.start.limit_margin, tw.targets[j].reach.limit_margin)
                if m > key:
                    best, key = tw.twin, m
        if best is not None:
            out.append(Choice(i, best, j, sc.reports[i].ok))
    return out


def way_b_grasps(asset: str, yaw: float, model_path: Path, clouds_path: Path, k: int, min_sep: float = 0.01):
    """Top-k (grasp, target) pairs from the network: T_obj_tcp (k,4,4), width (k,), roll index (k,)."""
    import torch

    from grasp2show.config import ObjectConfig, SceneConfig
    from grasp2show.learn.dense import condition, grasp_from_prediction, split_outputs
    from grasp2show.objects import object_from_config
    from grasp2show.scene import start_pose

    net, clouds = _load(model_path, clouds_path)
    o = clouds.names.index(asset)
    cfg = SceneConfig(obj=ObjectConfig(asset=asset, start_yaw=np.radians(yaw)))
    obj = object_from_config(cfg.obj)
    rolls = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    T0 = start_pose(cfg, obj)
    cond = np.stack([condition(T0, show_pose(cfg, obj, r), CameraConfig().pos, clouds.mass[o], clouds.com[o])
                     for r in rolls])
    t = time.perf_counter()
    with torch.no_grad():
        pr = split_outputs(net(torch.full((12,), o), torch.as_tensor(cond)))
    ms = (time.perf_counter() - t) * 1e3
    pr = {kk: v.numpy() for kk, v in pr.items()}
    q = 1 / (1 + np.exp(-np.clip(pr["logit"], -50, 50))) * pr["score"]   # (12, P)
    pts = clouds.feats[o, :, :3]
    picks = []
    for flat in np.argsort(-q, axis=None):
        r, pi = divmod(int(flat), q.shape[1])
        if any(r == r2 and np.linalg.norm(pts[pi] - pts[p2]) < min_sep for r2, p2 in picks):
            continue
        picks.append((r, pi))
        if len(picks) == 3 * k:
            break
    T = np.stack([grasp_from_prediction(pts[pi], pr["approach"][r, pi], pr["closing"][r, pi], pr["offset"][r, pi])
                  for r, pi in picks])
    W = np.clip([pr["width"][r, pi] for r, pi in picks], 0.0, 0.08)
    # Same filter as for every sampled candidate: the open gripper must not intersect the object.
    from grasp2show.gripper_sim import GripperSim

    sim = GripperSim(cfg, obj)
    keep = [i for i in range(len(T)) if not sim.in_collision(T[i], float(W[i]))][:k]
    return T[keep], np.asarray(W)[keep], np.array([picks[i][0] for i in keep], int), ms


def run(job):
    asset, yaw, model_path, clouds_path, k, policies = job
    recs = []
    if {"stability_only", "aware"} & set(policies):
        z = np.load(GRASPS / f"{asset}.npz")
        sc = Scenario(asset, yaw, {kk: z[kk] for kk in ("T_obj_tcp", "width", "success")})
        for pol in ("stability_only", "aware"):
            if pol in policies:
                recs.append(asdict(sc.run(pol, choose(pol, sc.reports, sc.lift_margin))))
    if "way_b" in policies:   # network grasps, exact kinematic checks, full episode
        T, W, R, ms = way_b_grasps(asset, yaw, model_path, clouds_path, k)
        if len(T) == 0:
            r = {"object": asset, "yaw_deg": yaw, "policy": "way_b", "success": False, "reason": "no grasp"}
        else:
            sb = Scenario(asset, yaw, {"T_obj_tcp": T, "width": W, "success": np.ones(len(T), bool)})
            r = asdict(sb.run("way_b", kinematic_choices(sb, list(zip(range(len(T)), R.tolist())))))
        r["net_ms"] = round(ms, 1)
        recs.append(r)
    if {"way_a", "way_a_smooth"} & set(policies):
        recs += way_a(asset, yaw, k, [p for p in ("way_a", "way_a_smooth") if p in policies])
    return recs


def smoothed(d: dict, radius: float = 0.01, max_deg: float = 30.0) -> np.ndarray:
    """Each candidate's score averaged over its neighbours: TCP within `radius`, approach and
    closing axes within `max_deg` (twins count as the same grasp). Tests the "averaging" explanation."""
    from scipy.spatial import cKDTree

    T, S = d["T_obj_tcp"].astype(float), d["S"]
    z, y = T[:, :3, 2], T[:, :3, 1]
    c = np.cos(np.radians(max_deg))
    out = np.zeros_like(S)
    for g, nb in enumerate(cKDTree(T[:, :3, 3]).query_ball_point(T[:, :3, 3], radius)):
        nb = np.asarray(nb)
        nb = nb[(z[nb] @ z[g] > c) & (np.abs(y[nb] @ y[g]) > c)]
        out[g] = S[nb].mean(0)
    return out


def way_a(asset: str, yaw: float, k: int, variants: list[str]) -> list[dict]:
    """All checks on dense candidates (what the network was trained to imitate), best first;
    way_a_smooth ranks the same feasible candidates by their neighbourhood-averaged score."""
    t = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        gdl.OUT = Path(tmp)
        gdl.process((asset, [yaw], 400, 12, 1000 + load_object_set()["objects"].index(asset)))
        d = dict(np.load(Path(tmp) / f"{asset}__yaw{int(yaw)}.npz"))
    secs = time.perf_counter() - t
    out = []
    for name in variants:
        S = d["S"]
        rank = smoothed(d) * (S > 0) if name == "way_a_smooth" else S
        flat = [(int(f) // S.shape[1], int(f) % S.shape[1]) for f in np.argsort(-rank, axis=None, kind="stable")[:k]
                if rank.flat[f] > 0]
        gi = sorted({g for g, _ in flat})
        if gi:
            sa = Scenario(asset, yaw, {"T_obj_tcp": d["T_obj_tcp"][gi], "width": d["width"][gi],
                                       "success": np.ones(len(gi), bool)})
            ra = asdict(sa.run(name, kinematic_choices(sa, [(gi.index(g), j) for g, j in flat])))
        else:
            ra = {"object": asset, "yaw_deg": yaw, "policy": name, "success": False, "reason": "no grasp"}
        ra["select_s"] = round(secs, 1)
        out.append(ra)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaws", type=float, nargs="+", default=[0, 45, 90, 135, 180, 225, 270, 315])
    p.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "heatmap_net.pt")
    p.add_argument("--clouds", type=Path, default=ROOT / "data" / "dataset" / "clouds.npz")
    p.add_argument("--objects", type=int, default=None, help="only the first N test objects")
    p.add_argument("--top", type=int, default=10, help="grasps proposed per scenario")
    p.add_argument("--policies", nargs="+", default=["stability_only", "aware", "way_a", "way_b"],
                   choices=["stability_only", "aware", "way_a", "way_a_smooth", "way_b"])
    p.add_argument("--tag", default="way_b", help="output names: data/episodes_<tag>/, media/<tag>_policy.png")
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)),
                   help="each worker loads PyTorch; more than ~4 can exhaust Windows' paging file")
    args = p.parse_args()
    test = [n for n in load_object_set()["test"] if (GRASPS / f"{n}.npz").exists()][: args.objects]
    jobs = [(n, y, args.model, args.clouds, args.top, args.policies) for n in test for y in args.yaws]
    out_dir = ROOT / "data" / f"episodes_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} scenarios on unseen objects, {args.workers} workers")
    recs = []
    with Pool(args.workers) as pool:
        for kk, rs in enumerate(pool.imap_unordered(run, jobs), 1):
            recs += rs
            print(f"[{kk}/{len(jobs)}] {rs[0]['object'][:34]:34s} yaw {rs[0]['yaw_deg']:3.0f}  " +
                  "  ".join(f"{r['policy']}: {'ok' if r['success'] else (r.get('reason') or 'fail').split(':')[0]}"
                            for r in rs), flush=True)
    (out_dir / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    summary = {"scenarios": len(jobs), "policies": {}}
    by = {}
    for r in recs:
        by.setdefault((r["object"], r["yaw_deg"]), {})[r["policy"]] = r["success"]
    for pol in args.policies:
        rs = [r for r in recs if r["policy"] == pol]
        kk = sum(r["success"] for r in rs)
        lo, hi = wilson(kk, len(rs))
        fails = {}
        for r in rs:
            if not r["success"]:
                key = (r.get("reason") or "fail").split(":")[0]
                fails[key] = fails.get(key, 0) + 1
        row = {"success_rate": round(kk / len(rs), 3), "ci95": [round(lo, 3), round(hi, 3)], "failures": fails}
        if pol != "way_b" and "way_b" in args.policies:
            row["way_b_wins_losses"] = [sum(v["way_b"] and not v[pol] for v in by.values()),
                                        sum(v[pol] and not v["way_b"] for v in by.values())]
        summary["policies"][pol] = row
    for key, pol, f in (("way_b_net_ms_median", "way_b", "net_ms"), ("way_a_seconds_median", "way_a", "select_s"),
                        ("way_a_seconds_median", "way_a_smooth", "select_s")):
        v = [r[f] for r in recs if r["policy"] == pol and f in r]
        if v:
            summary[key] = float(np.median(v))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    plot(summary, ROOT / "media" / f"{args.tag}_policy.png")
    return 0


def plot(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = {"stability_only": "stability only\n(baseline)", "aware": "analytic checks\n(labelled set)",
           "way_a": "way A: checks on\ndense candidates", "way_a_smooth": "way A,\nneighbourhood-\naveraged score",
           "way_b": "way B: heatmap\nnetwork"}
    col = {"stability_only": "#9aa5b1", "aware": "#c1121f", "way_a": "#e07a2f", "way_a_smooth": "#f2b880",
           "way_b": "#1d4e6e"}
    ps = list(summary["policies"])
    r = [summary["policies"][p]["success_rate"] for p in ps]
    err = np.array([[ri - summary["policies"][p]["ci95"][0] for ri, p in zip(r, ps)],
                    [summary["policies"][p]["ci95"][1] - ri for ri, p in zip(r, ps)]])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(range(len(ps)), r, 0.6, yerr=err, capsize=4, color=[col[p] for p in ps],
           error_kw={"lw": 1, "ecolor": "0.3"})
    for i, (ri, p) in enumerate(zip(r, ps)):
        ax.text(i, summary["policies"][p]["ci95"][1] + 0.02, f"{ri:.0%}", ha="center", fontsize=9)
    ax.set_xticks(range(len(ps)), [lab[p] for p in ps], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("task success")
    ax.set_title(f"Unseen objects: {summary['scenarios']} scenarios, full episodes, 95 % CI", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
