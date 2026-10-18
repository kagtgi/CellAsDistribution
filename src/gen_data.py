"""Conditional-generation dataset: one CELLxGENE Census dataset of raw counts, split by DONOR (no cell of a held-out donor in train).

python -m src.gen_data --name pbmc --dataset-id ed5d841d-6346-47d4-ab2f-7119ad7e3a35

Default dataset: Hao et al. 2021 (Cell), human PBMC, 10x 3' v3, 8 donors, 30 cell types (Census 2025-11-08).
Writes OUT_DIR/<name>/{counts.npz, obs.csv, panel.json, splits.json} with the same layout as src.preprocess, with
`cell_type` in place of `perturbation`.
"""
import argparse
import os
import numpy as np
import scipy.sparse as sp

from .common import SEED, OUT_DIR, save_json
from .preprocess import check_raw_counts, select_panel

CENSUS_VERSION = "2025-11-08"
HAO = "ed5d841d-6346-47d4-ab2f-7119ad7e3a35"
SPLIT_BY_DONOR = {"P1": "train", "P2": "train", "P3": "train", "P4": "train", "P5": "train", "P6": "val", "P7": "test", "P8": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pbmc")
    ap.add_argument("--dataset-id", default=HAO)
    ap.add_argument("--max-cells", type=int, default=0)
    a = ap.parse_args()
    import cellxgene_census as cc
    with cc.open_soma(census_version=CENSUS_VERSION) as c:
        adata = cc.get_anndata(c, organism="Homo sapiens", obs_value_filter=f"dataset_id=='{a.dataset_id}'", X_name="raw",
                               obs_column_names=["cell_type", "donor_id", "assay", "disease", "tissue"])
        ds = c["census_info"]["datasets"].read(value_filter=f"dataset_id=='{a.dataset_id}'").concat().to_pandas()
    ok, mx = check_raw_counts(adata.X)
    print(f"raw-count check: integer-valued={ok} max={mx}; cells={adata.n_obs}")
    if not ok:
        raise SystemExit("Census X['raw'] is not integer counts")
    obs = adata.obs.copy()
    for col in ("cell_type", "donor_id"):
        obs[col] = obs[col].astype(str)
    keep = np.where(obs["cell_type"].values != "unknown")[0]
    if a.max_cells and len(keep) > a.max_cells:
        keep = np.sort(np.random.default_rng(SEED).choice(keep, a.max_cells, replace=False))
    adata, obs = adata[keep].copy(), obs.iloc[keep]
    donors = sorted(obs["donor_id"].unique())
    split_map = SPLIT_BY_DONOR if set(donors) == set(SPLIT_BY_DONOR) else {
        d: ("train" if i < len(donors) - 3 else ("val" if i == len(donors) - 3 else "test")) for i, d in enumerate(donors)}
    cell_split = np.array([split_map[d] for d in obs["donor_id"]], dtype=object)
    X = sp.csr_matrix(adata.X)
    train_idx = np.where(cell_split == "train")[0]
    names = np.asarray(adata.var["feature_name"].astype(str).values)
    panel_idx = select_panel(X[train_idx], names, set())
    panel = [str(names[i]) for i in panel_idx]
    out = os.path.join(OUT_DIR, a.name)
    os.makedirs(out, exist_ok=True)
    sp.save_npz(os.path.join(out, "counts.npz"), X[:, panel_idx].tocsr())
    o = obs[["cell_type", "donor_id"]].copy()
    o["split"] = cell_split
    o["total_counts_full"] = np.asarray(X.sum(1)).ravel()
    o.to_csv(os.path.join(out, "obs.csv"))
    save_json({"panel": panel}, os.path.join(out, "panel.json"))
    save_json({"donor_split": split_map, "dataset_id": a.dataset_id, "census_version": CENSUS_VERSION,
               "collection": str(ds["collection_name"].iloc[0]), "doi": str(ds["collection_doi"].iloc[0]),
               "cells": {t: int((cell_split == t).sum()) for t in ("train", "val", "test")},
               "cell_types": {k: int(v) for k, v in o["cell_type"].value_counts().items()}},
              os.path.join(out, "splits.json"))
    print(a.name, {t: int((cell_split == t).sum()) for t in ("train", "val", "test")}, "panel", len(panel))


if __name__ == "__main__":
    main()
