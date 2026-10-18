"""Legend strip for Figure 4 (umap_cond). Replaces the legend of figures/umap_compare.py, whose automatic shortening
truncated long CELLxGENE cell-type names. Colors and order follow umap_compare.py (PAL, six most frequent test types).

python figures/umap_cond_legend.py --out paper/figs/umap_cond_legend
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

_F = os.path.join(os.path.dirname(__file__), "..", "fonts", "GoogleSans-Regular.ttf")
if os.path.exists(_F):
    font_manager.fontManager.addfont(_F)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_F).get_name()

PAL = ["#4F7FB8", "#D9713F", "#23918F", "#6F5FB0", "#C9A227", "#B04F7F"]
# Readable names of the six coloured types, in the order of umap_cond.json "colored".
LABELS = ["B cell", "CD14-low CD16+ monocyte", "CD14+ monocyte",
          "CD16- CD56-bright NK cell", "CD4+ cytotoxic T cell", "central memory CD4+ T cell"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
fig = plt.figure(figsize=(6.5, 0.34))
h = [Line2D([], [], ls="", marker="o", ms=3.5, mew=0, color=c) for c in PAL]
fig.legend(h, LABELS, loc="center", ncol=3, fontsize=6.5, frameon=False, handletextpad=.2, columnspacing=1.6,
           labelspacing=.35)
fig.savefig(a.out + ".pdf")
fig.savefig(a.out + ".png", dpi=200)
