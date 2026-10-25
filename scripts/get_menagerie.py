"""Fetch only the Franka Panda model from MuJoCo Menagerie (works on Windows, macOS, Linux).

Usage:  python scripts/get_menagerie.py
Needs git on the PATH. Uses a sparse checkout, so only a few MB are downloaded.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
DEST = Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie"


def main() -> int:
    if (DEST / "franka_emika_panda" / "panda.xml").exists():
        print(f"Menagerie Panda already present in {DEST}")
        return 0
    if shutil.which("git") is None:
        print("git not found. Install Git for Windows (https://git-scm.com) and retry.")
        return 1
    DEST.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", URL, str(DEST)],
                   check=True)
    subprocess.run(["git", "-C", str(DEST), "sparse-checkout", "set", "franka_emika_panda"], check=True)
    print(f"Done: {DEST / 'franka_emika_panda'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
