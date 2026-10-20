"""Raw counts -> train-only gene panel + perturbation-level splits (leakage-safe).

Usage: python -m src.preprocess --h5ad ~/data/NormanWeissman2019_filtered.h5ad --name norman \
           --pert-col perturbation [--control-label control]

Writes OUT_DIR/<name>/{panel.json, splits.json, counts.npz(sparse), obs.csv}.
Split protocol (md 5): unseen singles and unseen pairs are held out by *perturbation identity*;
singles of a held-out pair stay in train (pairs test), held-out single-target cells never appear
in train. Control cells are split by cell into train/val/test pools.
"""
import argparse
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

from .common import SEED, N_GENES, CONTROL_LABELS, OUT_DIR, save_json, split_pert


def is_control(p):
    return str(p) in CONTROL_LABELS or str(p).startswith("control")


def check_raw_counts(X, n=500):
    x = X[:n]
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    ok = bool(np.allclose(x, np.round(x))) and x.min() >= 0
    return ok, float(x.max())


def make_splits(obs, pert_col, rng, frac_test_single=0.2, frac_val_single=0.1, frac_test_pair=0.2, frac_val_pair=0.1):
    """Gene-level leakage-safe split.
    * Held-out single genes (val V / test T) appear in NO training perturbation (neither as singles nor inside pairs).
    * Pairs touching V (or T) with their other gene a train single -> held-out group 'val_pu' ('test_pu'): one unseen gene.
    * Pairs of two train singles -> held-out 'val_p' / 'test_p' (composition of seen genes).
    * Pairs that cannot be placed without leakage are 'excluded' (never trained on or scored)."""
    perts = obs[pert_col].astype(str)
    ctrl = perts.map(is_control)
    types = perts[~ctrl].unique()
    singles = sorted(p for p in types if len(split_pert(p)) == 1)
    pairs = sorted(p for p in types if len(split_pert(p)) == 2)

    rng.shuffle(singles)
    n_ts = int(round(frac_test_single * len(singles)))
    n_vs = int(round(frac_val_single * len(singles)))
    test_s, val_s = singles[:n_ts], singles[n_ts:n_ts + n_vs]
    train_s = singles[n_ts + n_vs:]
    T = {split_pert(p)[0] for p in test_s}
    V = {split_pert(p)[0] for p in val_s}
    train_genes = {split_pert(p)[0] for p in train_s}

    seen_seen = [p for p in pairs if all(g in train_genes for g in split_pert(p))]
    touches_V = [p for p in pairs if any(g in V for g in split_pert(p)) and not any(g in T for g in split_pert(p))]
    touches_T = [p for p in pairs if any(g in T for g in split_pert(p)) and not any(g in V for g in split_pert(p))]
    val_pu = [p for p in touches_V if sum(g in V for g in split_pert(p)) == 1
              and all(g in train_genes or g in V for g in split_pert(p))]
    test_pu = [p for p in touches_T if sum(g in T for g in split_pert(p)) == 1
               and all(g in train_genes or g in T for g in split_pert(p))]
    rng.shuffle(seen_seen)
    n_tp = int(round(frac_test_pair * len(pairs)))
    n_vp = int(round(frac_val_pair * len(pairs)))
    test_p, val_p = seen_seen[:n_tp], seen_seen[n_tp:n_tp + n_vp]
    train_p = [p for p in seen_seen if p not in set(test_p) | set(val_p)]
    placed = set(train_s) | set(val_s) | set(test_s) | set(train_p) | set(val_p) | set(test_p) | set(val_pu) | set(test_pu)
    excluded = [p for p in pairs if p not in placed]

    pert_split = {}
    for names, tag in ((train_s + train_p, "train"), (val_s + val_p + val_pu, "val"), (test_s + test_p + test_pu, "test"),
                       (excluded, "excluded")):
        for p in names:
            pert_split[p] = tag

    cell_split = np.array(["train"] * len(obs), dtype=object)
    for i, (p, c) in enumerate(zip(perts.values, ctrl.values)):
        cell_split[i] = pert_split[p] if not c else rng.choice(["train", "val", "test"], p=[0.8, 0.1, 0.1])
    return pert_split, cell_split, {"train_s": train_s, "val_s": val_s, "test_s": test_s, "train_p": train_p,
                                    "val_p": val_p, "test_p": test_p, "val_pu": val_pu, "test_pu": test_pu, "excluded": excluded}


def assert_no_leak(obs, pert_col, cell_split):
    perts = obs[pert_col].astype(str).values
    tr = {p for p, s in zip(perts, cell_split) if s == "train" and not is_control(p)}
    train_genes = {g for p in tr for g in split_pert(p)}
    for tag in ("val", "test"):
        te = {p for p, s in zip(perts, cell_split) if s == tag and not is_control(p)}
        assert not (tr & te), f"perturbation leak train/{tag}: {sorted(tr & te)[:5]}"
        te_single_genes = {split_pert(p)[0] for p in te if len(split_pert(p)) == 1}
        assert not (te_single_genes & train_genes), f"{tag} single gene seen in a train perturbation: {sorted(te_single_genes & train_genes)[:5]}"


def select_panel(X_train, var_names, extra_genes, n_genes=N_GENES):
    """Highest-variance genes of log-normalised TRAIN cells, plus perturbation targets, fixed order."""
    var_names = np.asarray(var_names)
    X_train = sp.csr_matrix(X_train)
    s = np.asarray(X_train.sum(1)).ravel().clip(min=1)
    Xl = (sp.diags(1e4 / s) @ X_train).tocsr()
    Xl.data = np.log1p(Xl.data)
    mean = np.asarray(Xl.mean(0)).ravel()
    sq = np.asarray(Xl.multiply(Xl).mean(0)).ravel()
    disp = (sq - mean ** 2) / (mean + 1e-6)
    expressed = np.asarray((X_train > 0).mean(0)).ravel() > 0.01
    forced = [i for i, g in enumerate(var_names) if g in set(extra_genes)]
    rank = np.argsort(-np.where(expressed, disp, -np.inf))
    chosen = list(dict.fromkeys(forced + list(rank)))[:n_genes]
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pert-col", default="perturbation")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--force-all-targets", action="store_true",
                    help="rev4 add-on panel: force every perturbed gene (train, val and test) into the panel; only gene identities "
                         "are used, never held-out expression, and the panel is still filled by training-cell dispersion")
    a = ap.parse_args()

    adata = ad.read_h5ad(a.h5ad)
    ok, mx = check_raw_counts(adata.X)
    print(f"raw-count check: integer-valued={ok} max={mx}")
    if not ok:
        raise SystemExit("X is not raw counts; check adata.layers / .raw before continuing")
    obs = adata.obs.copy()
    if a.max_cells and adata.n_obs > a.max_cells:
        keep = np.random.default_rng(SEED).choice(adata.n_obs, a.max_cells, replace=False)
        adata, obs = adata[np.sort(keep)].copy(), obs.iloc[np.sort(keep)]
    rng = np.random.default_rng(SEED)
    pert_split, cell_split, groups = make_splits(obs, a.pert_col, rng)
    assert_no_leak(obs, a.pert_col, cell_split)

    X = sp.csr_matrix(adata.X)
    train_idx = np.where(cell_split == "train")[0]
    targets = {g for p in obs[a.pert_col].astype(str).unique() if not is_control(p) for g in split_pert(p)}
    # only targets seen in TRAIN may force-enter the panel (test targets are chosen only by variance)
    train_targets = {g for p in obs[a.pert_col].astype(str).values[train_idx] if not is_control(p) for g in split_pert(p)}
    panel_idx = select_panel(X[train_idx], adata.var_names, targets if a.force_all_targets else train_targets)
    panel = [str(adata.var_names[i]) for i in panel_idx]

    out = os.path.join(OUT_DIR, a.name)
    os.makedirs(out, exist_ok=True)
    sp.save_npz(os.path.join(out, "counts.npz"), X[:, panel_idx].tocsr())
    obs_out = obs[[a.pert_col]].copy()
    obs_out["split"] = cell_split
    obs_out["total_counts_full"] = np.asarray(X.sum(1)).ravel()
    obs_out.to_csv(os.path.join(out, "obs.csv"))
    save_json({"panel": panel, "n_targets_in_panel": len(targets & set(panel))}, os.path.join(out, "panel.json"))
    save_json({"pert_split": pert_split, "groups": groups,
               "cells": {t: int((cell_split == t).sum()) for t in ("train", "val", "test")}},
              os.path.join(out, "splits.json"))
    print(a.name, "cells", adata.n_obs, "panel", len(panel), "splits", {t: int((cell_split == t).sum()) for t in ("train", "val", "test")})
    print({k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    main()
