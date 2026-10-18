"""Paper figures from the auto-generated results JSON only (no hand-typed numbers).

python figures/make_figures.py --auto paper/auto --out paper/figs
Figures: fig_forest_norman (paired NLL difference vs perturbed mean, all/singles/pairs), fig_headroom (fraction of
oracle headroom captured + identity controls), fig_jurkat (if jurkat_val.json exists).
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito (colour-blind safe); fixed roles so a model has the same colour in every figure
COL = {"additive": "#E69F00", "ridge": "#56B4E9", "CAD-P": "#009E73", "CAD-P shuffled names": "#009E73",
       "CAD-P opaque ids": "#009E73", "CAD-N tree": "#D55E00", "CAD-N tree+tilt": "#D55E00", "CAD-R tree+tilt": "#CC79A7",
       "C2S-1B (ours, tutorial 10)": "#666666", "C2S-1B (matched budget)": "#555555", "control mean": "#BBBBBB"}
HOLLOW = {"CAD-P shuffled names", "CAD-P opaque ids", "CAD-N tree"}   # controls / failed variants drawn hollow
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
                     "font.family": "DejaVu Sans"})


def forest(out, groups, fname, title):
    models = [m for m in out["stats"]["all"] if m not in ("perturbed mean", "control mean", "oracle (in-sample)", "oracle (split-half)")]
    fig, axes = plt.subplots(1, len(groups), figsize=(7.2, 0.28 * len(models) + 1.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        xmax = 0.07
        for i, m in enumerate(models):
            s = out["stats"][g][m]["diff"]
            c = COL.get(m, "#444444")
            if s[0] > xmax:    # far outside the useful range: arrow at the edge + value
                ax.annotate(f"+{s[0]:.2f}", xy=(xmax, i), xytext=(xmax - 0.045, i), color=c, fontsize=7, va="center",
                            ha="right", arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
                continue
            ax.errorbar(s[0], i, xerr=[[s[0] - s[1]], [s[2] - s[0]]], fmt="o", color=c, mfc="white" if m in HOLLOW else c,
                        ms=4.5, capsize=2, lw=1)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(-0.12, xmax)
        ax.set_title(f"{g} (n={out['n_perts'][g]})", fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.invert_yaxis()
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.supxlabel("paired NLL/UMI difference to the perturbed mean (nats; lower = better; 95% CI over perturbations)", fontsize=7.5)
    fig.suptitle(title, fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def headroom(out, fname):
    models = [m for m in out["stats"]["all"] if m not in ("perturbed mean", "control mean", "oracle (in-sample)", "oracle (split-half)")]
    vals = [out["stats"]["all"][m]["headroom_frac"] for m in models]
    fig, ax = plt.subplots(figsize=(3.4, 0.26 * len(models) + 0.9))
    for i, (m, v) in enumerate(zip(models, vals)):
        ax.barh(i, 100 * v, color=COL.get(m, "#444"), alpha=0.35 if m in HOLLOW else 0.95, edgecolor=COL.get(m, "#444"))
        ax.text(100 * v + (1.5 if v >= 0 else -1.5), i, f"{100 * v:.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("% of in-sample oracle headroom captured")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


KF_LABEL = {"additive": "additive (partial)", "ridge": "ridge", "real": "CAD-P, real names", "rid": "CAD-P, opaque ids",
            "shuf": "CAD-P, shuffled names", "desc": "CAD-P + own description", "descother": "CAD-P + other gene's description",
            "oracle (split-half)": "split-half oracle"}
KF_COL = {"additive": "#E69F00", "ridge": "#56B4E9", "real": "#009E73", "rid": "#009E73", "shuf": "#009E73", "desc": "#0072B2",
          "descother": "#0072B2", "oracle (split-half)": "#555555"}
KF_HOLLOW = {"rid", "shuf", "descother"}


def kfold_forest(res, fname):
    """Pooled gene-level 5-fold: paired NLL difference to the perturbed mean (negative = better)."""
    groups = [g for g in ("unseen singles", "pairs (seen genes)", "pairs (one unseen gene)") if g in res]
    models = [m for m in KF_LABEL if m in res[groups[0]]]
    fig, axes = plt.subplots(1, len(groups), figsize=(7.2, 0.27 * len(models) + 1.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        for i, m in enumerate(models):
            s = res[g][m]
            c = KF_COL[m]
            ax.errorbar(s[0], i, xerr=[[s[0] - s[1]], [s[2] - s[0]]], fmt="o" if "oracle" not in m else "D", color=c,
                        mfc="white" if m in KF_HOLLOW else c, ms=4.2, capsize=2, lw=1)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{g} (n={res[g]['n']})", fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([KF_LABEL[m] for m in models])
        ax.invert_yaxis()
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.supxlabel("paired NLL/UMI difference to the perturbed mean (nats; lower = better; 95% CI over perturbations)", fontsize=7.5)
    fig.suptitle("Norman, gene-level 5-fold (unit: perturbation)", fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", default="paper/auto")
    ap.add_argument("--out", default="paper/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kp = os.path.join(a.auto, "norman_kfold.json")
    if os.path.exists(kp):
        kfold_forest(json.load(open(kp)), os.path.join(a.out, "fig_kfold_forest.pdf"))
    # validation-split figures (tree vs flat) are produced only once the corrected-split tree runs exist
    vp = os.path.join(a.auto, "norman_val.json")
    if os.path.exists(vp) and os.environ.get("CAD_VAL_FIGS"):
        n = json.load(open(vp))
        forest(n, ["all", "unseen singles", "pairs (seen genes)", "pairs (one unseen gene)"],
               os.path.join(a.out, "fig_forest_norman_val.pdf"), "Norman, held-out perturbations (validation)")
    print("figures written to", a.out)
