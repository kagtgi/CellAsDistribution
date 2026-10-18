"""Balanced two-level gene tree (64 modules x 32 genes) from TRAIN cells only; random-tree control.

python -m src.gene_tree --name norman [--random]
Output: OUT_DIR/<name>/tree.json (or tree_random.json) with modules {Mxx: [gene,...]},
module descriptions, and gene->module map.
"""
import argparse
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from .common import SEED, N_MODULES, GENES_PER_MODULE, OUT_DIR, load_json, save_json


def gene_embedding(X_train, n_comp=50):
    """Gene embedding from log-normalised train coexpression (SVD of cells x genes, genes as rows)."""
    X = sp.csr_matrix(X_train).astype(np.float64)
    s = np.asarray(X.sum(1)).ravel().clip(min=1)
    X = (sp.diags(1e4 / s) @ X).tocsr()
    X.data = np.log1p(X.data)
    X = X - sp.csr_matrix(np.ones((X.shape[0], 1))) @ sp.csr_matrix(X.mean(0))  # centre genes
    svd = TruncatedSVD(n_components=n_comp, random_state=SEED).fit(X)
    emb = svd.components_.T * np.sqrt(svd.singular_values_)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


def balanced_modules(emb, n_modules=N_MODULES, size=GENES_PER_MODULE):
    km = KMeans(n_clusters=n_modules, n_init=4, random_state=SEED).fit(emb)
    d = ((emb[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)   # genes x modules
    cost = np.repeat(d, size, axis=1)                                         # genes x (modules*size) slots
    rows, cols = linear_sum_assignment(cost)
    assign = np.empty(len(emb), dtype=int)
    assign[rows] = cols // size
    return assign


def describe(genes_in_module, mean_expr, n_genes=1):
    top = sorted(genes_in_module, key=lambda g: -mean_expr[g])[:n_genes]
    return " ".join(top)


def build_tree(X_train, panel, random=False, desc_genes=1):
    n = len(panel)
    assert n == N_MODULES * GENES_PER_MODULE, f"panel must be {N_MODULES * GENES_PER_MODULE} genes, got {n}"
    if random:
        assign = np.repeat(np.arange(N_MODULES), GENES_PER_MODULE)
        np.random.default_rng(SEED).shuffle(assign)
    else:
        assign = balanced_modules(gene_embedding(X_train))
    mean_expr = dict(zip(panel, np.asarray(sp.csr_matrix(X_train).mean(0)).ravel()))
    modules = {f"M{m:02d}": [panel[i] for i in np.where(assign == m)[0]] for m in range(N_MODULES)}
    assert all(len(v) == GENES_PER_MODULE for v in modules.values())
    desc = {k: describe(v, mean_expr, desc_genes) for k, v in modules.items()}
    return {"modules": modules, "desc": desc, "gene_to_module": {g: k for k, v in modules.items() for g in v},
            "random": bool(random)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--desc-genes", type=int, default=1)
    a = ap.parse_args()
    d = os.path.join(OUT_DIR, a.name)
    panel = load_json(os.path.join(d, "panel.json"))["panel"]
    X = sp.load_npz(os.path.join(d, "counts.npz")).tocsr()
    obs = pd.read_csv(os.path.join(d, "obs.csv"), index_col=0)
    tr = np.where(obs["split"].values == "train")[0]
    tree = build_tree(X[tr], panel, random=a.random, desc_genes=a.desc_genes)
    save_json(tree, os.path.join(d, "tree_random.json" if a.random else "tree.json"))
    print("tree written; modules", len(tree["modules"]), "random", a.random)


if __name__ == "__main__":
    main()
