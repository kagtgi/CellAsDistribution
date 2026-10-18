"""Baselines for zero-shot cell types (Stage 3): the requested type has no training cells.

python -m src.zs_baselines --split val      (all 5 folds of splits/pbmc_zs_folds.json)

global mean    pooled frequency of all training cells of the training types of the fold (smoothed, pseudo-count 0.5)
ontology NN    the type mean of the training type closest in the Cell Ontology: largest Jaccard overlap of the two
               sets {term} union {CL ancestors} (splits/pbmc_cell_ontology.json, EBI OLS); ties broken by the larger type
name NN        the type mean of the training type whose NAME shares the most words (Jaccard of word sets): string matching
CL parent      pooled frequency of all training types under the held-out type's direct Cell Ontology parents
All see only training donors and only the training types of the fold; the held-out types are the evaluation units.
"""
import argparse
import os
import numpy as np

from .common import load_json
from .cell_eval import World, eval_units
from .cell_dist import to_dist

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--min-cells", type=int, default=30)
    a = ap.parse_args()
    w = World("cond", "pbmc")
    folds = load_json(os.path.join(REPO, "splits", "pbmc_zs_folds.json"))["folds"]
    CL = load_json(os.path.join(REPO, "splits", "pbmc_cell_ontology.json"))["terms"]
    anc = {t: set([v["id"]] + v.get("ancestors_cl", [])) for t, v in CL.items() if "id" in v}
    log = {}
    for k, held in enumerate(folds):
        tr = np.where((w.split == "train") & ~np.isin(w.lab, held))[0]
        types = sorted(set(w.lab[tr]))
        size = {t: int((w.lab[tr] == t).sum()) for t in types}
        tmean = {t: to_dist(w.counts(tr[w.lab[tr] == t]).sum(0), "smooth", 0.5) for t in types if size[t] >= 5}
        glob = to_dist(w.counts(tr).sum(0), "smooth", 0.5)

        def nn(u):
            cand = [t for t in tmean if t in anc]
            sc = {t: len(anc[u] & anc[t]) / len(anc[u] | anc[t]) for t in cand}
            return max(cand, key=lambda t: (sc[t], size[t]))
        def words(t):
            return set(t.lower().replace(",", " ").replace("-", " ").split())

        def name_nn(u):          # string matching: training type whose name shares the most words (Jaccard)
            return max(tmean, key=lambda t: (len(words(u) & words(t)) / len(words(u) | words(t)), size[t]))

        def parent_mean(u):      # pooled training cells of every training type under the held-out type's direct CL parents
            par = {p[0] for p in CL[u].get("parents", [])} if u in CL else set()
            kids = [t for t in types if t in anc and par & anc[t]]
            if not kids:
                return glob, []
            return to_dist(w.counts(tr[np.isin(w.lab[tr], kids)]).sum(0), "smooth", 0.5), kids
        units = [u for u in w.units(a.split, a.min_cells) if u in held]
        match = {u: nn(u) for u in units}
        nmatch = {u: name_nn(u) for u in units}
        pmean = {u: parent_mean(u) for u in units}
        log[k] = {u: {"ontology_nn": match[u], "name_nn": nmatch[u], "parent_types": pmean[u][1]} for u in units}
        eval_units(w, a.split, lambda u, c: glob[None], f"cb_zs_global_f{k}", min_cells=a.min_cells, seed=0, units=units)
        eval_units(w, a.split, lambda u, c: tmean[match[u]][None], f"cb_zs_ontology_f{k}", min_cells=a.min_cells, seed=0, units=units)
        eval_units(w, a.split, lambda u, c: tmean[nmatch[u]][None], f"cb_zs_namenn_f{k}", min_cells=a.min_cells, seed=0, units=units)
        eval_units(w, a.split, lambda u, c: pmean[u][0][None], f"cb_zs_parent_f{k}", min_cells=a.min_cells, seed=0, units=units)
        for u in units:
            print(f"fold {k}: {u} -> CL nn: {match[u]} | name nn: {nmatch[u]} | parent types: {pmean[u][1]}", flush=True)
    print(log)


if __name__ == "__main__":
    main()
