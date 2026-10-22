"""Paired statistics across hardware seeds."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(x: Sequence[float], stat=np.median, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``stat(x)``."""
    arr = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = stat(arr[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_comparison(method: Sequence[float], baseline: Sequence[float], seed: int = 0) -> dict[str, float]:
    """Compare a method against a baseline on paired samples (same hardware seed).

    Reports the median relative gain ``method/baseline - 1`` with a bootstrap
    95 % CI, the number of wins, and a two-sided Wilcoxon signed-rank test on
    the paired log-ratios (rates span orders of magnitude).
    """
    m = np.asarray(method, dtype=float)
    b = np.asarray(baseline, dtype=float)
    floor = 1e-12
    log_ratio = np.log(np.maximum(m, floor)) - np.log(np.maximum(b, floor))
    gain = np.exp(log_ratio) - 1.0
    lo, hi = bootstrap_ci(gain, seed=seed)
    if np.allclose(log_ratio, 0.0):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(log_ratio, alternative="two-sided", zero_method="wilcox").pvalue)
    return {
        "n": int(len(m)),
        "median_rel_gain": float(np.median(gain)),
        "ci_low": lo,
        "ci_high": hi,
        "mean_rel_gain": float(np.mean(gain)),
        "wins": int(np.sum(log_ratio > 1e-9)),
        "losses": int(np.sum(log_ratio < -1e-9)),
        "wilcoxon_p": p_value,
    }
