"""Shared plotting style so every figure in the project looks consistent."""
from __future__ import annotations

import matplotlib.pyplot as plt

from . import config as C

# Colour-blind-safe palette (Okabe-Ito)
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#6b6b6b",
)
CLASS_COLORS = {0: BLUE, 1: RED}                 # no disease / disease
CLASS_NAMES = {0: "No disease", 1: "Disease"}
MODEL_COLORS = {"Logistic Regression": BLUE, "Random Forest": GREEN, "XGBoost": ORANGE}


def set_style() -> None:
    """Apply one clean matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "font.size": 10,
    })


def save_fig(fig, name: str) -> str:
    """Save a figure to reports/figures/<name>.png and return the path."""
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return str(path)
