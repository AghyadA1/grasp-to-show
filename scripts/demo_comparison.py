"""Side-by-side video: the baseline's grasp vs the target-aware grasp on the same scenario.

Usage:
  python scripts/demo_comparison.py                        # first scenario where they differ
  python scripts/demo_comparison.py --object Crayola_Crayons_120_crayons --yaw 0
Writes media/compare_<object>_<yaw>.mp4 and media/compare_<object>_<yaw>.gif.
Needs data/grasps/ (generate_grasps.py). Uses data/episodes/records.jsonl to find a scenario
where the policies disagree if --object is not given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from demo_pick_and_show import banner, compose, write_comparison_gif

from grasp2show.experiment import Scenario, choose
from grasp2show.feasibility import twin_of
from grasp2show.task import Grasp, show_pose

LABEL = {"stability_only": "baseline: most stable grasp", "aware": "target-aware grasp (ours)"}


def find_scenario() -> tuple[str, float]:
    """A scenario where the baseline fails and ours succeeds with a *different* grasp; prefer the
    most common win type, a baseline finger hiding the QR, then any other."""
    f = ROOT / "data" / "episodes" / "records.jsonl"
    by = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "policy" and r["policy"] in LABEL:
            by.setdefault((r["object"], r["yaw_deg"]), {})[r["policy"]] = r
    wins = [(k, v) for k, v in by.items()
            if v.get("aware", {}).get("success") and not v.get("stability_only", {}).get("success", True)
            and v["aware"]["grasp"] != v["stability_only"]["grasp"]]
    for want in ("qr", ""):
        for (obj, yaw), v in wins:
            if v["stability_only"]["reason"].startswith(want):
                return obj, yaw
    raise SystemExit("no scenario where the baseline fails and target-aware succeeds")


def plan(sc: Scenario, policy: str) -> tuple[Grasp, dict]:
    """The policy's first choice whose motion can be planned (as in the experiments)."""
    for c in choose(policy, sc.reports, sc.lift_margin):
        tw = sc.reports[c.grasp].twins[c.twin]
        t = tw.targets[c.roll_idx]
        T = sc.T[c.grasp] if c.twin == 0 else twin_of(sc.T[c.grasp])
        paths, _ = sc.ep.plan_analytic(T, show_pose(sc.cfg, sc.obj, sc.rolls[c.roll_idx]), tw.start.q,
                                       t.reach.q, checker=sc.fc, width=float(sc.W[c.grasp]))
        if paths is not None:
            return Grasp(policy, T, float(sc.W[c.grasp])), paths
    raise SystemExit(f"{policy}: no plannable grasp")


def outcome_text(res) -> tuple[str, bool]:
    if res.success:
        return f"QR decoded, slip {res.slip_mm:.0f} mm", True
    if not res.qr_ok and res.held:
        return "QR not readable", False
    return f"object slipped {res.slip_mm:.0f} mm / {res.slip_deg:.0f} deg", False


def film(sc: Scenario, policy: str, fps: int):
    grasp, paths = plan(sc, policy)
    r = sc.ep.renderer
    frames = []

    def on_frame(d):
        r.update_scene(d, camera="overview")
        ov = r.render().copy()
        r.update_scene(d, camera="scanner")
        frames.append(compose(ov, r.render(), LABEL[policy], d.time))

    res = sc.ep.execute(grasp, paths, on_frame=on_frame, fps=fps)
    text, ok = outcome_text(res)
    frames += [banner(frames[-1], text, ok)] * int(1.5 * fps)
    print(f"{policy}: {res.summary()}")
    return frames


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--object", default=None)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    obj, yaw = (args.object, args.yaw) if args.object else find_scenario()
    z = np.load(ROOT / "data" / "grasps" / f"{obj}.npz")
    sc = Scenario(obj, yaw, {k: z[k] for k in ("T_obj_tcp", "width", "success")})
    a, b = film(sc, "stability_only", args.fps), film(sc, "aware", args.fps)
    stem = ROOT / "media" / f"compare_{obj[:30]}_{int(yaw)}"
    n = max(len(a), len(b))
    pad = lambda f: f + [f[-1]] * (n - len(f))
    both = [np.hstack([x, y]) for x, y in zip(pad(a), pad(b))]
    both = [cv2.resize(f, (f.shape[1] // 2 * 2, f.shape[0] // 2 * 2)) for f in both]
    imageio.mimwrite(f"{stem}.mp4", both, fps=args.fps, quality=7, macro_block_size=8)
    write_comparison_gif(pad(a), pad(b), args.fps, Path(f"{stem}.gif"))
    print(f"wrote {stem}.mp4 and .gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
