"""Speed of the C++ kernels vs the numpy reference (same inputs, same results).

  python scripts/bench_cpp.py        writes data/bench_cpp.json
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp2show._accel import HAVE_CPP
from grasp2show.franka_ik import Q_MAX, Q_MIN, fk, ik
from grasp2show.stability import StabilityParams, utilisation_wrench


def best_of(f, repeats=5):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts)


def main() -> int:
    if not HAVE_CPP:
        print("C++ extension not installed: pip install ./cpp")
        return 1
    rng = np.random.default_rng(0)
    out = {"machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"}

    # Stability: one kernel call per grasp x load (the shake-test model evaluates ~34 loads per grasp).
    n = 100_000
    y = rng.normal(size=(n, 3))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    c = rng.normal(scale=0.02, size=(n, 3))
    w = rng.uniform(0.01, 0.07, (n, 1))
    args = (c - 0.5 * w * y, c + 0.5 * w * y, y, np.zeros(3), rng.normal(scale=4.0, size=(n, 3)),
            rng.normal(scale=0.05, size=(n, 3)), StabilityParams())
    t_np = best_of(lambda: utilisation_wrench(*args, use_cpp=False))
    t_cpp = best_of(lambda: utilisation_wrench(*args, use_cpp=True))
    out["stability"] = {"evaluations": n, "numpy_us_each": round(t_np / n * 1e6, 3),
                        "cpp_us_each": round(t_cpp / n * 1e6, 3), "speedup": round(t_np / t_cpp, 1)}

    # IK: 16 q7 values x 4 branches per pose, as used by the feasibility checks.
    Q = Q_MIN + (Q_MAX - Q_MIN) * (0.05 + 0.9 * rng.random((5000, 7)))
    T = fk(Q)
    q7 = np.linspace(Q_MIN[6] + 0.02, Q_MAX[6] - 0.02, 16)
    t_np = best_of(lambda: ik(T, q7, use_cpp=False), 3)
    t_cpp = best_of(lambda: ik(T, q7, use_cpp=True), 3)
    out["ik"] = {"poses": len(T), "q7_per_pose": len(q7), "numpy_us_per_pose": round(t_np / len(T) * 1e6, 1),
                 "cpp_us_per_pose": round(t_cpp / len(T) * 1e6, 1), "speedup": round(t_np / t_cpp, 1)}
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "bench_cpp.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
