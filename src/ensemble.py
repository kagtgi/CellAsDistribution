"""Test-time tricks on saved q's: seed ensembles and expert mixing (geometric / arithmetic), with the mixing
weight chosen leave-one-perturbation-out (LOO) so the reported NLL is not tuned on the perturbation it scores.

python -m src.ensemble --name norman --runs g1c_cadf_08b g1c_cadf_08b_s1 g1c_cadf_08b_s2 [--split val]
"""
import argparse
import os
import numpy as np

from .eval_common import load_world
from .eval_nll import nll_per_umi
from .evaluate import bootstrap_ci
from .gene_tree import gene_embedding
from . import baselines as B


def geo(qs, w=None):
    w = np.ones(len(qs)) / len(qs) if w is None else np.asarray(w)
    l = sum(wi * np.log(np.clip(q, 1e-12, None)) for wi, q in zip(w, qs))
    e = np.exp(l - l.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    d, X, obs, panel, lab, ctrl, split = load_world(a.name)
    cnt = lambda m: np.asarray(X[np.where(m)[0]].todense())
    tcells = {p: np.where((lab == p) & (split == "train"))[0] for p in set(lab[~ctrl & (split == "train")])}
    tcells = {p: i for p, i in tcells.items() if len(i) >= 5}
    train_f = {p: B.profile(np.asarray(X[i].todense())) for p, i in tcells.items()}
    f_ctrl = B.profile(cnt(ctrl & (split == "train")))
    pm = B.perturbed_mean(train_f)
    add = B.additive(train_f, f_ctrl, pm)
    rdg = B.ridge(train_f, gene_embedding(X[np.where(split == "train")[0]]), panel, f_ctrl)

    zs = [np.load(os.path.join(d, "runs", r, "q_by_pert.npz")) for r in a.runs]
    perts = zs[0]["perts"].tolist()
    real = {p: cnt((lab == p) & (split == a.split)) for p in perts}
    Q = [{p: z["q"][i] for i, p in enumerate(perts)} for z in zs]
    nll = lambda qd: {p: nll_per_umi(real[p], qd[p]) for p in perts}
    ref = nll({p: pm(p) for p in perts})
    out = {"perturbed_mean": ref, "additive": nll({p: add(p) for p in perts}), "ridge": nll({p: rdg(p) for p in perts})}
    for r, q in zip(a.runs, Q):
        out[r] = nll(q)
    ens_a = {p: np.mean([q[p] for q in Q], 0) for p in perts}
    ens_g = {p: geo([q[p] for q in Q]) for p in perts}
    out["ENS arithmetic"], out["ENS geometric"] = nll(ens_a), nll(ens_g)

    # expert mixing: q = geo(ens, expert; w) ; w chosen LOO over perturbations from a grid
    grid = np.linspace(0, 1, 11)
    for ename, ef in (("additive", add), ("ridge", rdg)):
        E = {p: ef(p) for p in perts}
        table = np.array([[nll_per_umi(real[p], geo([ens_g[p], E[p]], [1 - w, w])) for w in grid] for p in perts])
        loo = []
        for i in range(len(perts)):
            others = np.delete(table, i, axis=0).mean(0)
            loo.append(table[i, int(np.argmin(others))])
        out[f"ENS x {ename} (LOO weight)"] = dict(zip(perts, loo))
        print(f"best global w for {ename}: {grid[int(np.argmin(table.mean(0)))]:.1f} (in-sample, optimistic)")
    print(f"split={a.split} n_perts={len(perts)}")
    print(f"{'model':34s} {'NLL/UMI':>8s}  paired diff vs perturbed_mean [95% CI]")
    for n, v in out.items():
        diff = bootstrap_ci([v[p] - ref[p] for p in perts])
        print(f"{n:34s} {np.mean(list(v.values())):8.4f}  {diff[0]:+.4f} [{diff[1]:+.4f}, {diff[2]:+.4f}]")


if __name__ == "__main__":
    main()
