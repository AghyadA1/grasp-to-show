"""Small metrics shared by the training scripts."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve (the Mann-Whitney statistic); NaN if y has one class only."""
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
