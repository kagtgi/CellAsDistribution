"""Cells -> Unsloth decision records (md 4.3), schema verified in experiments/e0_smoke.

Record = {"state": {...}, "questions": {name: {"type":"choice","instructions":..,"criteria":{key:desc}}},
          "gold": {name: {"probabilities": {key: p}}}}
One root question ("module", 64 options) + R leaf questions for modules sampled proportional to mass.
Loss weighting (verified): trainer averages soft-CE over question rows, so the record loss is
(CE_root + sum_r CE_leaf_r)/(R+1); the optimum of each factor is unchanged (md 4.2 exact decomposition).
"""
import argparse
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import SEED, OUT_DIR, load_json, split_pert


def _context(name):
    name = "norman" if name.startswith("norman") else ("jurkat" if name.startswith("jurkat") else name)
    return {"norman": "K562 cells, CRISPRa perturbation", "jurkat": "Jurkat cells, CRISPRi perturbation"}.get(name, name)


def make_units(obs, pert_col, split, bag_size, rng, max_cells=0):
    """Index groups; each unit = bag_size cells of ONE perturbation (bag_size=1: single cells)."""
    idx = np.where(obs["split"].values == split)[0]
    if max_cells and len(idx) > max_cells:
        idx = np.sort(rng.choice(idx, max_cells, replace=False))
    labels = obs[pert_col].astype(str).values[idx]
    units = []
    for p in np.unique(labels):
        members = idx[labels == p]
        rng.shuffle(members)
        for s in range(0, len(members), bag_size):
            chunk = members[s:s + bag_size]
            if len(chunk) == bag_size or (bag_size > 1 and len(chunk) >= max(2, bag_size // 2)) or bag_size == 1:
                units.append(chunk)
    rng.shuffle(units)
    return units


def tilt_target(counts, prior, shrink):
    """Residual-on-prior target: t_g proportional to p_g / prior_g, with p shrunk toward the prior
    (empirical Bayes, `shrink` pseudo-UMIs). Generation recovers p exactly: q = prior * t / sum(prior * t)."""
    p = (counts + shrink * prior) / (counts.sum() + shrink)
    t = p / prior
    return t / t.sum()


def make_records(X, obs, tree, panel, pert_col, split, name, R=4, root_only=False, seed=SEED,
                 max_cells=0, bag_size=1, prior=None, shrink=2000.0):
    rng = np.random.default_rng(seed)
    units = make_units(obs, pert_col, split, bag_size, rng, max_cells)
    mod_keys = list(tree["modules"].keys())
    panel_pos = {g: i for i, g in enumerate(panel)}
    mod_cols = {k: np.array([panel_pos[g] for g in v]) for k, v in tree["modules"].items()}
    desc = tree["desc"]
    X = sp.csr_matrix(X)
    root_q = {"type": "choice", "instructions": "Which gene module does the next transcript come from?",
              "criteria": {k: desc[k] for k in mod_keys}}
    recs = []
    for unit in units:
        i = int(unit[0])
        dense = np.asarray(X[unit].sum(0)).ravel()
        total = dense.sum()
        if total <= 0:
            continue
        if prior is not None:           # tilt mode: all quantities below are on the tilt distribution t
            dense = tilt_target(dense, prior, shrink)
            total = 1.0
        mod_mass = np.array([dense[mod_cols[k]].sum() for k in mod_keys])
        p_mod = mod_mass / total
        pert = str(obs[pert_col].iloc[i])
        state = {"context": _context(name), "perturbation": "+".join(split_pert(pert)) or "control"}
        qs = {"module": root_q}
        gold = {"module": {"probabilities": {k: float(p) for k, p in zip(mod_keys, p_mod)}}}
        if not root_only:
            nz = np.where(mod_mass > 0)[0]
            n_pick = min(R, len(nz))
            picks = rng.choice(nz, size=n_pick, replace=False, p=mod_mass[nz] / mod_mass[nz].sum())
            for m in picks:
                k = mod_keys[m]
                c = dense[mod_cols[k]]
                genes = tree["modules"][k]
                qs[f"leaf_{k}"] = {"type": "choice", "instructions": f"Which gene of module {k}?",
                                   "criteria": {g: "" for g in genes}}
                gold[f"leaf_{k}"] = {"probabilities": {g: float(v) for g, v in zip(genes, c / c.sum())}}
        recs.append({"state": state, "questions": qs, "gold": gold, "meta": {"cell": i, "n_cells": int(len(unit)), "pert": pert, "total": float(total)}})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pert-col", default="perturbation")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tree", default="tree.json")
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--max-cells", type=int, default=5000)
    ap.add_argument("--bag-size", type=int, default=1)
    a = ap.parse_args()
    d = os.path.join(OUT_DIR, a.name)
    X = sp.load_npz(os.path.join(d, "counts.npz")).tocsr()
    obs = pd.read_csv(os.path.join(d, "obs.csv"), index_col=0)
    tree = load_json(os.path.join(d, a.tree))
    panel = load_json(os.path.join(d, "panel.json"))["panel"]
    recs = make_records(X, obs, tree, panel, a.pert_col, a.split, a.name, R=a.R, max_cells=a.max_cells, bag_size=a.bag_size)
    import json
    out = os.path.join(d, f"records_{a.split}_{a.tree.replace('.json', '')}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(recs), "records ->", out)


if __name__ == "__main__":
    main()
