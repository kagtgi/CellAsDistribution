"""GEARS baseline (Roohani et al. 2024) on Norman, scored by the same eval_units as every other method.

Run in the separate environment ~/gears (cell-gears, torch_geometric, scanpy):
    python -m src.run_gears --epochs 15 --seed 1

GEARS is trained on the TRAIN perturbations of our gene-level split (ctrl cells included), uses the validation perturbations for its own
checkpoint choice, and predicts one mean expression profile per held-out perturbation (no control cell is shown). Its gene set is our
2048-gene panel plus every perturbation target gene present in the data, so unseen target genes have a GO-graph embedding. Predicted
log1p(CP10k) expression is converted back to a frequency over the panel (expm1, renormalise, +0.5 pseudo-count at the mean training library
size, the same smoothing as the frequency baselines) and scored on the validation and the test split of the same trained model.
"""
import argparse
import os
import pickle
import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sp

from .cell_eval import World, eval_units
from .baselines import split_pert, profile, perturbed_mean

H5AD = os.path.join(os.environ.get("CAD_DATA", os.path.expanduser("~/data")), "NormanWeissman2019_filtered.h5ad")
GEARS_DIR = os.path.expanduser("~/gears_data")


def build(w, force=False, train_ctrl_only=False):
    # train_ctrl_only: drop validation/test control cells, so GEARS sees only training controls
    name = "norman_cad_trctrl" if train_ctrl_only else "norman_cad"
    keep = ((w.split == "train") | ~w.ctrl) if train_ctrl_only else np.ones(len(w.split), bool)
    done = os.path.join(GEARS_DIR, name, "data_pyg")
    from gears import PertData
    os.makedirs(GEARS_DIR, exist_ok=True)
    pd_ = PertData(GEARS_DIR)
    lab, ctrl = w.lab, w.ctrl
    cond = np.array(["ctrl" if c else (l if "+" in l else l + "+ctrl") for l, c in zip(lab, ctrl)])
    split = {"train": sorted(set(cond[w.split == "train"])), "val": sorted(set(cond[(w.split == "val") & ~ctrl])),
             "test": sorted(set(cond[(w.split == "test") & ~ctrl]))}
    assert not (set(split["train"]) & set(split["val"])) and not (set(split["train"]) & set(split["test"])), "perturbation leak"
    if os.path.isdir(done) and not force:
        pd_.load(data_path=os.path.join(GEARS_DIR, name))
    else:
        full = ad.read_h5ad(H5AD)
        assert (full.obs_names[:3] == w.obs.index[:3]).all() or set(w.obs.index[:100]) <= set(full.obs_names), "cell ids do not match the h5ad"
        genes = list(dict.fromkeys(list(w.panel) + [g for l in set(lab[~ctrl]) for g in split_pert(l)]))
        genes = [g for g in genes if g in full.var_names]
        a = full[w.obs.index.values[keep], genes].copy()
        a.X = sp.csr_matrix(a.X).astype(np.float32)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.obs["condition"] = cond[keep]
        a.obs["cell_type"] = "K562"
        a.var["gene_name"] = a.var_names
        pd_.new_data_process(dataset_name=name, adata=a)
    path = os.path.join(GEARS_DIR, name, "split.pkl")
    pickle.dump(split, open(path, "wb"))
    pd_.prepare_split(split="custom", seed=1, split_dict_path=path)
    pd_.get_dataloader(batch_size=32, test_batch_size=128)
    return pd_, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--train-ctrl-only", action="store_true")
    a = ap.parse_args()
    from gears import GEARS
    w = World("pert", "norman")
    G = len(w.panel)
    pd_, split = build(w, a.force, a.train_ctrl_only)
    model = GEARS(pd_, device="cuda", weight_bias_track=False, proj_name="cad", exp_name="gears_cad")
    model.model_initialize(hidden_size=a.hidden)
    try:
        model.train(epochs=a.epochs)
    except KeyError as e:   # cell-gears 0.1.2 fails in its final DE analysis (after best_model is stored); training is complete
        print("GEARS post-training analysis failed, using the stored best model:", repr(e), flush=True)
    assert hasattr(model, "best_model"), "GEARS training did not finish"
    gnames = list(pd_.gene_names)
    pos = [gnames.index(g) for g in w.panel]
    smean = float(w.lib.mean())
    tr = (w.split == "train") & ~w.ctrl
    pm = perturbed_mean({p: profile(w.counts(np.where(tr & (w.lab == p))[0])) for p in np.unique(w.lab[tr]) if (tr & (w.lab == p)).sum() >= 5})
    cache, miss = {}, []

    def qfn(u, cidx):
        if u not in cache:
            try:
                y = model.predict([split_pert(u)])
                v = np.asarray(list(y.values())[0], dtype=np.float64)[pos]
                c = np.expm1(np.clip(v, 0, None))
                c = c / max(c.sum(), 1e-12)
                q = (c * smean + 0.5) / (smean + 0.5 * G)
            except Exception as e:      # gene unknown to GEARS: fall back to the perturbed mean, and count it
                miss.append(u)
                q = pm(u)
            cache[u] = (q / q.sum())[None]
        return cache[u]

    for sp_ in ("val", "test"):
        per = eval_units(w, sp_, qfn, "cb_GEARS_trctrl" if a.train_ctrl_only else "cb_GEARS", min_cells=a.min_cells, seed=0)
        print(f"GEARS {sp_}: nll_pooled={np.mean([v['nll_pooled'] for v in per.values()]):.4f} "
              f"mmd_dm={np.mean([v['mmd_dm'] for v in per.values()]):.4f} auc_mult={np.mean([v['auc_mult'] for v in per.values()]):.3f} "
              f"units={len(per)} fallback_to_perturbed_mean={len(set(miss))}", flush=True)


if __name__ == "__main__":
    main()
