"""Download the selected scanned objects (no git needed).

Usage:
  python scripts/get_objects.py            # all objects in grasp2show/object_set.json
  python scripts/get_objects.py --only 3   # just the first 3 (quick start)
Files come from raw.githubusercontent.com (kevinzakka/mujoco_scanned_objects; Google
Scanned Objects assets, CC-BY 4.0). Each object is ~1-3 MB.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp2show.objects import load_object_set, scanned_objects_dir

RAW = "https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/main/models"


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def download_object(name: str) -> str:
    d = scanned_objects_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    fetch(f"{RAW}/{name}/model.xml", d / "model.xml")
    xml = (d / "model.xml").read_text()
    files = sorted(set(re.findall(r'file="([^"]+)"', xml)))
    for f in files:
        fetch(f"{RAW}/{name}/{f}", d / f)
    return f"{name} ({len(files)} files)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=int, default=None, help="download only the first N objects")
    args = p.parse_args()
    names = load_object_set()["objects"][: args.only]
    print(f"downloading {len(names)} objects to {scanned_objects_dir()}")
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, msg in enumerate(ex.map(download_object, names), 1):
            print(f"[{i}/{len(names)}] {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
