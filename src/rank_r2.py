"""How well does a rank sentence 'reconstruct' expression? The Cell2Sentence metric (R^2 of log1p-CP10k on log rank, their
Fig. 3 reports 0.815 on immune cells) next to the likelihood headroom kept by the same reconstruction (repr_ceiling.py).

python -m src.rank_r2 --name norman --split test
Fit (as in C2S / c2s_baseline.fit_rank_expression): log1p(CP10k) = a + b log(rank) on the top-200 expressed panel genes of 3000
training cells. Report R^2, Pearson and Spearman of predicted vs observed log1p(CP10k) over the top-200 genes of held-out cells.
"""
import argparse
import json
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr

from .common import OUT_DIR
from .cell_eval import World
from .c2s_baseline import fit_rank_expression


def pairs(C, k=200):
    xs, ys = [], []
    for c in C:
        o = np.argsort(-c, kind="stable")[:k]
        keep = c[o] > 0
        r = np.arange(1, len(o) + 1)[keep]
        xs.append(np.log(r)); ys.append(np.log1p(c[o][keep] / c.sum() * 1e4))
    return np.concatenate(xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="norman")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    task = "pert" if a.name == "norman" else "cond"
    w = World(task, a.name)
    tr = np.where((w.split == "train") & ~w.ctrl)[0]
    ra, rb = fit_rank_expression(w.X, w.panel, tr, k=200)
    te = np.where((w.split == a.split) & ~w.ctrl)[0]
    te = np.random.default_rng(0).choice(te, min(3000, len(te)), replace=False)
    x, y = pairs(w.counts(te))
    pred = ra + rb * x
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out = {"name": a.name, "split": a.split, "fit": [ra, rb], "n_cells": int(len(te)), "n_gene_values": int(len(y)),
           "r2": float(r2), "pearson": float(pearsonr(pred, y)[0]), "spearman": float(spearmanr(pred, y)[0])}
    os.makedirs(os.path.join(OUT_DIR, "auto_rev2"), exist_ok=True)
    json.dump(out, open(os.path.join(OUT_DIR, "auto_rev2", f"rank_r2_{a.name}_{a.split}.json"), "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
