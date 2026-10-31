"""Day 1-2 demo: scripted pick -> move -> show, judged by a QR decoder.

Runs the two hand-designed grasps on the same box and target pose:
  clear_qr  - fingers on the side faces: QR stays visible  -> expected SUCCESS
  covers_qr - fingers on the QR face: equally stable lift  -> expected FAIL (QR hidden)

Usage:
  python scripts/demo_pick_and_show.py                      # both grasps, videos to media/
  python scripts/demo_pick_and_show.py --grasp clear_qr --roll 30
  python scripts/demo_pick_and_show.py --viewer             # interactive MuJoCo viewer
Headless machines: set MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa (CPU).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # scripts/archive/<part>/<script>.py
sys.path.insert(0, str(ROOT))
for _d in (ROOT / "scripts", *sorted((ROOT / "scripts" / "archive").iterdir())):     # sibling scripts
    sys.path.insert(1, str(_d))
from grasp2show.config import SceneConfig
from grasp2show.episode import PickAndShow
from grasp2show.scene import build_scene
from grasp2show.task import scripted_grasps, show_pose

MEDIA = ROOT / "media"


def compose(overview: np.ndarray, scanner: np.ndarray, label: str, t: float) -> np.ndarray:
    """Overview view with the scanner camera as a picture-in-picture inset."""
    frame = overview.copy()
    h, w = frame.shape[:2]
    iw, ih = w // 3, h // 3
    inset = cv2.resize(scanner, (iw, ih), interpolation=cv2.INTER_AREA)
    frame[8:8 + ih, w - iw - 8:w - 8] = inset
    cv2.rectangle(frame, (w - iw - 9, 7), (w - 7, 8 + ih), (255, 255, 255), 1)
    cv2.putText(frame, "scanner", (w - iw - 4, 22 + ih), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{label}   t={t:4.1f}s", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def banner(frame: np.ndarray, text: str, ok: bool) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    color = (40, 170, 70) if ok else (200, 50, 50)
    cv2.rectangle(out, (0, h - 46), (w, h), color, -1)
    cv2.putText(out, text, (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)
    return out


def write_comparison_gif(a: list, b: list, fps: int, path: Path, gif_fps: int = 8,
                         size: tuple[int, int] = (320, 240)) -> None:
    """Side-by-side GIF for the README, small enough to load quickly (~2-3 MB)."""
    n, step = max(len(a), len(b)), max(fps // gif_fps, 1)
    from PIL import Image

    frames = [np.hstack([cv2.resize(f[min(i, len(f) - 1)], size, interpolation=cv2.INTER_AREA)
                         for f in (a, b)])
              for i in range(0, n, step)]
    # One shared 64-colour palette and no dithering: dithering noise defeats GIF's
    # compression and is the main reason naive GIFs are large.
    # Palette from first, middle and last frames, so the result banners keep their colour.
    sample = np.vstack([frames[0], frames[len(frames) // 2], frames[-1]])
    palette = Image.fromarray(sample).quantize(96, dither=Image.Dither.NONE)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / gif_fps),
                 loop=0, optimize=True)


def run_one(cfg, ep: PickAndShow, name: str, roll_deg: float, fps: int):
    grasp = scripted_grasps(cfg)[name]
    target = show_pose(cfg, ep.obj, np.radians(roll_deg))
    # Reuse the episode's renderer for both views. Several mujoco.Renderer objects (several
    # OpenGL contexts) in one process can render the wrong camera on some backends.
    r = ep.renderer
    frames = []

    def on_frame(d):
        r.update_scene(d, camera="overview")
        overview = r.render().copy()
        r.update_scene(d, camera="scanner")
        frames.append(compose(overview, r.render(), name, d.time))

    res = ep.run(grasp, target, on_frame=on_frame, fps=fps)
    text = (f"QR decoded: '{res.decoded}'  slip {res.slip_mm:.1f} mm" if res.success
            else ("QR not readable" if not res.qr_ok else "dropped / slipped / collided"))
    frames += [banner(frames[-1], text, res.success)] * int(1.5 * fps)
    return res, frames


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grasp", choices=["clear_qr", "covers_qr", "both"], default="both")
    p.add_argument("--roll", type=float, default=0.0, help="target roll about the viewing axis (deg)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--viewer", action="store_true", help="open the interactive viewer instead")
    args = p.parse_args()

    cfg = SceneConfig()
    scene = build_scene(cfg)
    model = scene.model

    if args.viewer:
        import mujoco.viewer

        from grasp2show.scene import reset_to_home
        d = mujoco.MjData(model)
        reset_to_home(model, d)
        mujoco.viewer.launch(model, d)
        return

    ep = PickAndShow(scene, cfg)
    names = ["clear_qr", "covers_qr"] if args.grasp == "both" else [args.grasp]
    MEDIA.mkdir(exist_ok=True)
    all_frames = {}
    for name in names:
        res, frames = run_one(cfg, ep, name, args.roll, args.fps)
        print(res.summary())
        cv2.imwrite(str(MEDIA / f"scanner_{name}.png"), cv2.cvtColor(res.scanner_rgb, cv2.COLOR_RGB2BGR))
        if not args.no_video:
            imageio.mimwrite(MEDIA / f"pick_and_show_{name}.mp4", frames, fps=args.fps,
                             quality=7, macro_block_size=8)
        all_frames[name] = frames

    if len(all_frames) == 2 and not args.no_video:
        write_comparison_gif(all_frames["clear_qr"], all_frames["covers_qr"], args.fps,
                             MEDIA / "clear_vs_covering.gif")
        print(f"wrote videos and GIF to {MEDIA}")


if __name__ == "__main__":
    main()
