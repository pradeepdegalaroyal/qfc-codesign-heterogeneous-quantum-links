"""Shared figure style (publication quality, 300 dpi, CVD-validated palette)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots in fixed order (validated adjacent-pair CVD-safe order).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a8a7", "#e34948", "#a32a2a"]
)


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.17, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK)
