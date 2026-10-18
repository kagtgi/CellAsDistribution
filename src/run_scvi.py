"""NB-likelihood generative baselines (scvi-tools) scored by the same eval_units as the language-model decoders.

python -m src.run_scvi --task cond --name pbmc --split val     # scVI and scANVI: decoder frequencies averaged over training cells of the cell type
python -m src.run_scvi --task pert --name norman --split val   # scGen-style: scVI latent shift of known single genes (partial additive), decoded from the control cell

Everything is trained on TRAIN cells only, on the same 2048-gene panel; the decoder mean rho (sums to 1) is the model's frequency vector q, so the
shared sampler and the multinomial NLL apply unchanged. Run in the separate environment ~/scvi (scvi-tools).
"""
import argparse
import numpy as np
import anndata as ad
import scipy.sparse as sp
import torch
import scvi

from .cell_eval import World, eval_units
from .baselines import split_pert


def make_adata(w, idx, labels=False):
    a = ad.AnnData(sp.csr_matrix(w.X[idx]).astype(np.float32))
    if labels:
        a.obs["label"] = w.lab[idx]
    return a


def rho(model, adata, idx=None, bs=2048):
    out = []
    n = adata.n_obs if idx is None else len(idx)
    ii = np.arange(adata.n_obs) if idx is None else idx
    for s in range(0, n, bs):
        out.append(np.asarray(model.get_normalized_expression(adata, indices=ii[s:s + bs], library_size=1, return_numpy=True)))
    return np.concatenate(out)


def type_mean_fns(w, epochs, n_latent, seed):
    scvi.settings.seed = seed
    tr = np.where((w.split == "train") & ~w.ctrl)[0]
    a = make_adata(w, tr, labels=True)
    scvi.model.SCVI.setup_anndata(a)
    vae = scvi.model.SCVI(a, n_latent=n_latent, gene_likelihood="nb")
    vae.train(max_epochs=epochs, accelerator="gpu", batch_size=512)
    fns = {}
    types = np.unique(w.lab[tr])
    q_scvi = {}
    for u in types:
        ii = np.where(w.lab[tr] == u)[0]
        if len(ii) >= 5:
            q_scvi[u] = rho(vae, a, ii[:3000]).mean(0)
    sc = scvi.model.SCANVI.from_scvi_model(vae, labels_key="label", unlabeled_category="__none__")
    sc.train(max_epochs=max(10, epochs // 4), accelerator="gpu", batch_size=512)
    q_sc = {}
    for u in types:
        ii = np.where(w.lab[tr] == u)[0]
        if len(ii) >= 5:
            q_sc[u] = rho(sc, a, ii[:3000]).mean(0)
    glob_v, glob_s = np.mean(list(q_scvi.values()), 0), np.mean(list(q_sc.values()), 0)
    fns["scVI (NB, type-mean decoder frequency)"] = lambda u, c: q_scvi.get(u, glob_v)[None]
    fns["scANVI (NB, type-mean decoder frequency)"] = lambda u, c: q_sc.get(u, glob_s)[None]
    return fns


def scgen_style_fn(w, epochs, n_latent, seed):
    scvi.settings.seed = seed
    tr = np.where(w.split == "train")[0]
    a = make_adata(w, tr)
    scvi.model.SCVI.setup_anndata(a)
    vae = scvi.model.SCVI(a, n_latent=n_latent, gene_likelihood="nb")
    vae.train(max_epochs=epochs, accelerator="gpu", batch_size=512)
    z = vae.get_latent_representation(a)
    lab, ctrl = w.lab[tr], w.ctrl[tr]
    z_ctrl = z[ctrl].mean(0)
    shift = {}
    for p in np.unique(lab[~ctrl]):
        if len(split_pert(p)) == 1 and (lab == p).sum() >= 5:
            shift[p] = z[lab == p].mean(0) - z_ctrl
    mean_shift = np.mean([z[lab == p].mean(0) - z_ctrl for p in np.unique(lab[~ctrl]) if (lab == p).sum() >= 5], 0)
    dev = vae.device

    @torch.no_grad()
    def decode(zz):
        zz = torch.tensor(zz, dtype=torch.float32, device=dev)
        out = vae.module.generative(z=zz, library=torch.zeros(len(zz), 1, device=dev),
                                    batch_index=torch.zeros(len(zz), 1, dtype=torch.long, device=dev))
        r = out["px"].scale if hasattr(out["px"], "scale") else out["px_scale"]
        return r.double().cpu().numpy()

    def fn(u, cidx):
        known = [g for g in split_pert(u) if g in shift]
        s = np.sum([shift[g] for g in known], 0) if known else mean_shift
        ac = make_adata(w, cidx)
        zc = vae.get_latent_representation(ac)
        q = decode(zc + s[None])
        return q / q.sum(1, keepdims=True)
    return {"scGen-style (scVI latent shift, partial additive)": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cond", "pert"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-latent", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-cells", type=int, default=30)
    a = ap.parse_args()
    w = World(a.task, a.name)
    fns = type_mean_fns(w, a.epochs, a.n_latent, a.seed) if a.task == "cond" else scgen_style_fn(w, a.epochs, a.n_latent, a.seed)
    for label, fn in fns.items():
        tag = "cb_" + label.split(" (")[0].replace(" ", "_").replace("-", "_")
        per = eval_units(w, a.split, fn, tag, min_cells=a.min_cells, seed=a.seed)
        print(f"{label:55s} nll_pooled={np.mean([v['nll_pooled'] for v in per.values()]):.4f} "
              f"nll_mix={np.mean([v['nll_mix'] for v in per.values()]):.4f} auc_mult={np.mean([v['auc_mult'] for v in per.values()]):.3f} "
              f"mmd_dm={np.mean([v['mmd_dm'] for v in per.values()]):.4f} units={len(per)}", flush=True)


if __name__ == "__main__":
    main()
