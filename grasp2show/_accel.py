"""Optional C++ kernels (package `grasp2show-cpp`, built from cpp/ with `pip install ./cpp`).

If the extension is installed, the stability model and the analytic IK use it; otherwise
the numpy implementations run. Both give the same results (tests/test_cpp.py).
Set the environment variable G2S_NO_CPP=1 to force numpy.
"""
from __future__ import annotations

import os

try:
    if os.environ.get("G2S_NO_CPP"):
        raise ImportError
    import g2s_cpp as cpp
    HAVE_CPP = True
except ImportError:
    cpp = None
    HAVE_CPP = False
