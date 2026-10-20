"""Co-expression realism of generated cells (rev3 E5-lite, post hoc, CPU).

python -m src.coexpr_eval --split test --out ~/cad_out/auto_rev3 --stem test_coexpr \
    --runs "Additive=cb_additive" "CAD, no control cell=t2_g_noctrl_s0" ...

A distribution over genes for one state says nothing by itself about how genes co-vary from cell to cell; that variation comes
from the control cell (residual), from a latent (low-rank head) or only from the shared sampler. For each held-out perturbation u
with >= --min-cells real cells, on H genes selected on TRAIN cells only (largest variance of log1p CP10k):
  corr_err   mean |r_gen(g,h) - r_real(g,h)| over gene pairs g<h, where r is the Pearson correlation across the cells of u
             (real cells vs the method's generated cells, multinomial sampler, saved gen_<split>.npz)
  var_ratio  sum of per-gene variances of generated / real log1p CP10k cells on the H genes (1 = same total variation)
  pc1_share  |share of variance on the first principal component, generated - real| (structure of the variation)
Differences between methods are paired over perturbations (bootstrap CI).
"""
import argparse
import os
import numpy as np

from .common import save_json
from .cell_eval import World
from .evaluate import log_cp10k, bootstrap_ci
from .de_eval import load_gen


def corr(L):
    L = L - L.mean(0)
    sd = np.sqrt((L ** 2).mean(0))
    ok = sd > 0
    C = np.zeros((L.shape[1], L.shape[1]))
    Z = L[:, ok] / sd[ok]
    C[np.ix_(ok, ok)] = Z.T @ Z / len(L)
    return C


def pc1_share(L):
    ev = np.linalg.eigvalsh(np.cov(L, rowvar=False))
    return float(ev[-1] / max(ev.sum(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="norman")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--H", type=int, default=100)
    ap.add_argument("--min-cells", type=int, default=100)
    a = ap.parse_args()
    w = World("pert", a.name)
    G = len(w.panel)
    tr = np.where((w.split == "train") & ~w.ctrl)[0]
    sub = np.sort(np.random.default_rng(0).choice(tr, min(len(tr), 10000), replace=False))
    hv = np.argsort(-log_cp10k(w.counts(sub)).var(0))[: a.H]
    runs = dict(r.split("=", 1) for r in a.runs)
    gens = {lab: load_gen(os.path.join(w.dir, "runs", t, f"gen_{a.split}.npz"), G) for lab, t in runs.items()}
    units = sorted(u for u in set.intersection(*[set(g) for g in gens.values()])
                   if ((w.lab == u) & (w.split == a.split) & ~w.ctrl).sum() >= a.min_cells)
    iu = np.triu_indices(a.H, 1)
    per = {lab: {} for lab in runs}
    for u in units:
        Lr = log_cp10k(w.counts(np.where((w.lab == u) & (w.split == a.split) & ~w.ctrl)[0]))[:, hv]
        Cr, vr, pr = corr(Lr), Lr.var(0).sum(), pc1_share(Lr)
        for lab in runs:
            Lg = log_cp10k(gens[lab][u])[:, hv]
            per[lab][u] = {"corr_err": float(np.abs(corr(Lg)[iu] - Cr[iu]).mean()),
                           "var_ratio": float(Lg.var(0).sum() / max(vr, 1e-12)),
                           "pc1_err": float(abs(pc1_share(Lg) - pr))}
    ref = next(iter(runs))
    summ = {}
    for lab in runs:
        summ[lab] = {}
        for m in ("corr_err", "var_ratio", "pc1_err"):
            s = {"mean": float(np.mean([per[lab][u][m] for u in units]))}
            if lab != ref:
                s["diff_vs_" + ref] = bootstrap_ci([per[lab][u][m] - per[ref][u][m] for u in units])
            summ[lab][m] = s
        print(f"{lab:40s} " + "  ".join(f"{m}={v['mean']:.3f}" + (f" ({v['diff_vs_' + ref][0]:+.3f} [{v['diff_vs_' + ref][1]:+.3f},"
                                                                    f"{v['diff_vs_' + ref][2]:+.3f}])" if lab != ref else "")
                                         for m, v in summ[lab].items()), flush=True)
    os.makedirs(a.out, exist_ok=True)
    save_json({"split": a.split, "H": a.H, "genes": [w.panel[i] for i in hv], "units": units, "runs": runs, "ref": ref,
               "summary": summ, "per_unit": per}, os.path.join(a.out, f"{a.stem}.json"))


if __name__ == "__main__":
    main()
