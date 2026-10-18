"""scVI / scANVI evaluated as their own generative models, next to the type-mean frequency used in the main tables.

python -m src.run_scvi_native --name pbmc --split test --seed 0      (environment ~/scvi)

scVI has no cell-type-conditional prior, so cells of type u are generated from the empirical posterior: for each of M=300
training cells of type u (random), z ~ q(z | x) is sampled and decoded to rho (sums to 1). Each generated cell uses its own rho,
so q has M rows and the per-cell NLL is the mixture -log (1/M) sum_m Mult(x | rho_m); counts are drawn from scVI's own
negative binomial with the learned gene-wise inverse dispersion theta_g = exp(px_r) (eval_units thfn hook, metrics *_nb),
and also through the shared multinomial / Dirichlet-multinomial samplers for comparison.
scANVI: the same with the scANVI encoder and decoder after semi-supervised training with the cell-type labels.
"""
import argparse
import numpy as np
import torch
import scvi

from .cell_eval import World, eval_units
from .run_scvi import make_adata


def posterior_rho(model, X, n_draw, rng):
    m = model.module
    dev = model.device
    x = torch.tensor(X, dtype=torch.float32, device=dev)
    bi = torch.zeros(len(X), 1, dtype=torch.long, device=dev)
    with torch.no_grad():
        inf = m.inference(x=x, batch_index=bi, cont_covs=None, cat_covs=None)
        qz = inf.get("qz")
        if qz is not None:
            z = qz.sample()
        else:
            z = inf["qz_m"] + inf["qz_v"].sqrt() * torch.randn_like(inf["qz_m"])
        gen = m.generative(z=z, library=torch.zeros(len(X), 1, device=dev), batch_index=bi)
        px = gen["px"]
        r = px.scale if hasattr(px, "scale") else gen["px_scale"]
    r = r.double().cpu().numpy()
    return r / r.sum(1, keepdims=True)


def theta_of(model):
    return torch.exp(model.module.px_r.detach()).double().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pbmc")
    ap.add_argument("--split", default="test")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-latent", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--min-cells", type=int, default=30)
    a = ap.parse_args()
    scvi.settings.seed = a.seed
    rng = np.random.default_rng(a.seed)
    w = World("cond", a.name)
    tr = np.where(w.split == "train")[0]
    ad_ = make_adata(w, tr, labels=True)
    scvi.model.SCVI.setup_anndata(ad_)
    vae = scvi.model.SCVI(ad_, n_latent=a.n_latent, gene_likelihood="nb")
    vae.train(max_epochs=a.epochs, accelerator="gpu", batch_size=512)
    sc = scvi.model.SCANVI.from_scvi_model(vae, labels_key="label", unlabeled_category="__none__")
    sc.train(max_epochs=max(10, a.epochs // 4), accelerator="gpu", batch_size=512)
    for name, model in (("scVI_native", vae), ("scANVI_native", sc)):
        th = theta_of(model)
        cache = {}

        def qfn(u, cidx, model=model):
            if u not in cache:
                ii = tr[w.lab[tr] == u]
                pick = rng.choice(ii, a.draws, replace=len(ii) < a.draws)
                cache[u] = posterior_rho(model, w.counts(pick), a.draws, rng)
            return cache[u]
        per = eval_units(w, a.split, qfn, f"cb_{name}_s{a.seed}", min_cells=a.min_cells, seed=a.seed,
                         thfn=lambda u, c, th=th: th[None])
        print(f"{name}: nll_pooled={np.mean([v['nll_pooled'] for v in per.values()]):.4f} "
              f"nll_mix={np.mean([v['nll_mix'] for v in per.values()]):.4f} nll_nb_own={np.mean([v['nll_nb_own'] for v in per.values()]):.4f} "
              f"mmd_nb={np.mean([v['mmd_nb'] for v in per.values()]):.4f} auc_nb={np.mean([v['auc_nb'] for v in per.values()]):.3f} "
              f"mmd_dm={np.mean([v['mmd_dm'] for v in per.values()]):.4f}", flush=True)


if __name__ == "__main__":
    main()
