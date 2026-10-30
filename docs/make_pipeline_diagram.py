"""Draw the project pipeline diagram used in the report.   python docs/make_pipeline_diagram.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src import viz

viz.set_style()
steps = [
    ("1  Data", "5,000 patients\n500 original +\n4,500 synthetic", viz.SKY),
    ("2  Validate and clean", "missing values,\nduplicates,\nvalue ranges", viz.SKY),
    ("3  Split", "80% train / 20% test\nstratified by label\nand data source", viz.SKY),
    ("4  Tune 3 models", "Logistic Regression,\nRandom Forest, XGBoost\n5-fold CV, train only", viz.ORANGE),
    ("5  Select + threshold", "best CV ROC-AUC\nscreening cut-off\nfrom out-of-fold data", viz.ORANGE),
    ("6  Test once + explain", "test metrics, bootstrap\nintervals, SHAP,\nsubgroup checks", viz.GREEN),
    ("7  Streamlit app", "risk estimate +\nSHAP explanation +\ndisclaimer", viz.GREEN),
]
# snake layout: row 1 goes left to right (steps 1-4), row 2 right to left (steps 5-7)
xs = [0.13, 0.38, 0.63, 0.88, 0.88, 0.63, 0.38]
ys = [0.74, 0.74, 0.74, 0.74, 0.24, 0.24, 0.24]
w, h = 0.215, 0.36

fig, ax = plt.subplots(figsize=(9.4, 4.4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
for (title, body, color), x, y in zip(steps, xs, ys):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=color, ec="none", alpha=0.22))
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="none", ec=color, lw=1.8))
    ax.text(x, y + 0.095, title, ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x, y - 0.045, body, ha="center", va="center", fontsize=8.6, linespacing=1.35)

def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.6, color=viz.GREY, shrinkA=0, shrinkB=0))

for i in range(3):                                   # row 1, left to right
    arrow((xs[i] + w / 2 + 0.012, ys[i]), (xs[i + 1] - w / 2 - 0.012, ys[i + 1]))
arrow((xs[3], ys[3] - h / 2 - 0.012), (xs[4], ys[4] + h / 2 + 0.012))   # down
for i in (4, 5):                                     # row 2, right to left
    arrow((xs[i] - w / 2 - 0.012, ys[i]), (xs[i + 1] + w / 2 + 0.012, ys[i + 1]))

ax.text(0.13, 0.24, "Disclaimer shown in every\nnotebook and in the app:\nnot for clinical diagnosis.",
        ha="center", va="center", fontsize=8.6, color=viz.RED, style="italic", linespacing=1.4)
viz.save_fig(fig, "pipeline_diagram")
print("saved reports/figures/pipeline_diagram.png")
