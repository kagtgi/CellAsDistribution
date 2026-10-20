"""Perturbation-effect (delta) and differential-expression metrics, computed post hoc from saved generations (CPU only).

python -m src.de_eval --name norman --split test --out ~/cad_out/auto_rev2 --stem norman_test_de \
    --runs "Perturbed mean=cb_perturbed_mean" "Additive=cb_additive" "CAD (Gemma), no control cell=t_g_noctrl_s0"

All profiles are mean log1p(CP10k) over the panel (CP10k with the panel total as denominator, as everywhere in the paper).
For a held-out perturbation u with real perturbed cells R_u and held-out control cells C (same split):
  delta_real = mean(R_u) - mean(C),  delta_pred = mean(G_u) - mean(C), where G_u are the cells the method generated for u
  (multinomial sampler, saved as gen_<split>.npz, so real and generated profiles go through the same log-CP10k map).
Metrics per unit:
  pearson_delta_ctrl   Pearson(delta_pred, delta_real) over all panel genes (control-referenced; can be inflated by
                       systematic variation shared by all perturbations, Vinas Torne et al. 2025)
  pearson_delta_pert   the same with the TRAIN perturbed-cell centroid as reference (removes the shared perturbation effect)
  pearson_de20         Pearson of delta_pred and delta_real on the top-20 DE genes of u
  sign_de20            fraction of the top-20 DE genes whose predicted change has the real sign
  mse_de20             mean squared error of the predicted change on the top-20 DE genes
  prec_de20            |top-20 by |delta_pred|  intersect  top-20 DE| / 20
Top-20 DE genes of u: largest |Welch t| of log-CP10k between R_u and C (real cells only, never model output).
"""
import argparse
import os
import numpy as np
import scipy.sparse as sp
from scipy.stats import ttest_ind

from .common import OUT_DIR, save_json, split_pert
from .cell_eval import World
from .evaluate import log_cp10k, safe_pearson


def load_gen(path, G):
    z = np.load(path, allow_pickle=True)
    out = {}
    for u in z["units"]:
        ptr = z[f"{u}__ptr"]
        out[str(u)] = sp.csr_matrix((z[f"{u}__data"], z[f"{u}__ind"], ptr), shape=(len(ptr) - 1, G)).toarray()
    return out


def unit_metrics(Lr, Lg, mu_c, mu_tp, top):
    mr, mg = Lr.mean(0), Lg.mean(0)
    dr, dp = mr - mu_c, mg - mu_c
    pred_top = np.argsort(-np.abs(dp))[: len(top)]
    return {
        "pearson_delta_ctrl": safe_pearson(dp, dr),
        "pearson_delta_pert": safe_pearson(mg - mu_tp, mr - mu_tp),
        "pearson_de20": safe_pearson(dp[top], dr[top]),
        "sign_de20": float((np.sign(dp[top]) == np.sign(dr[top])).mean()),
        "mse_de20": float(((dp[top] - dr[top]) ** 2).mean()),
        "prec_de20": float(len(set(pred_top) & set(top)) / len(top)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="norman")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", nargs="+", required=True, help='"Label=tag"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--exclude-targets", action="store_true", help="qbar only: drop the perturbed genes from all metrics")
    ap.add_argument("--source", default="gen", choices=["gen", "qbar"],
                    help="gen: mean log-CP10k of generated cells; qbar: pseudobulk log1p(1e4 q) against log1p(1e4 x pooled real "
                         "frequency), no sampling involved")
    a = ap.parse_args()
    if a.source == "qbar":
        return main_qbar(a)
    w = World("pert", a.name)
    G = len(w.panel)
    C = log_cp10k(w.counts(np.where(w.ctrl & (w.split == a.split))[0]))
    mu_c = C.mean(0)
    tp = np.where(~w.ctrl & (w.split == "train"))[0]
    tp = np.sort(np.random.default_rng(0).choice(tp, min(len(tp), 20000), replace=False))
    mu_tp = log_cp10k(w.counts(tp)).mean(0)
    runs = dict(r.split("=", 1) for r in a.runs)
    gens = {lab: load_gen(os.path.join(w.dir, "runs", t, f"gen_{a.split}.npz"), G) for lab, t in runs.items()}
    units = sorted(set.intersection(*[set(g) for g in gens.values()]))
    per = {lab: {} for lab in runs}
    de = {}
    for u in units:
        Lr = log_cp10k(w.counts(np.where((w.lab == u) & (w.split == a.split) & ~w.ctrl)[0]))
        t = ttest_ind(Lr, C, equal_var=False, axis=0).statistic
        t = np.nan_to_num(t, nan=0.0)
        top = np.argsort(-np.abs(t))[: a.k]
        de[u] = [w.panel[i] for i in top]
        for lab in runs:
            per[lab][u] = unit_metrics(Lr, log_cp10k(gens[lab][u]), mu_c, mu_tp, top)
    os.makedirs(a.out, exist_ok=True)
    save_json({"split": a.split, "units": units, "runs": runs, "de_genes": de, "per_unit": per},
              os.path.join(a.out, f"{a.stem}.json"))
    for lab in runs:
        m = {k: float(np.mean([per[lab][u][k] for u in units])) for k in per[lab][units[0]]}
        print(f"{lab:45s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))


def main_qbar(a):
    """Pseudobulk version: every profile is log1p(1e4 * pooled frequency), so no sampler or Jensen gap enters the comparison."""
    w = World("pert", a.name)
    pb = lambda C: np.log1p(1e4 * C.sum(0) / max(C.sum(), 1))
    Cidx = np.where(w.ctrl & (w.split == a.split))[0]
    Ccells = w.counts(Cidx)
    C, mu_c = log_cp10k(Ccells), pb(Ccells)
    tp = np.where(~w.ctrl & (w.split == "train"))[0]
    tp = np.sort(np.random.default_rng(0).choice(tp, min(len(tp), 20000), replace=False))
    mu_tp = pb(w.counts(tp))
    runs = dict(r.split("=", 1) for r in a.runs)
    qb = {}
    for lab, t in runs.items():
        z = np.load(os.path.join(w.dir, "runs", t, f"qbar_{a.split}.npz"), allow_pickle=True)
        qb[lab] = {str(u): q for u, q in zip(z["units"], z["q"])}
    units = sorted(set.intersection(*[set(v) for v in qb.values()]))
    pos = {g: i for i, g in enumerate(w.panel)}
    per = {lab: {} for lab in runs}
    targets = {}
    for u in units:
        R = w.counts(np.where((w.lab == u) & (w.split == a.split) & ~w.ctrl)[0])
        t = np.nan_to_num(ttest_ind(log_cp10k(R), C, equal_var=False, axis=0).statistic, nan=0.0)
        # --exclude-targets: the CRISPRa target genes of u are removed from the DE ranking and from every correlation, so
        # the metrics measure the downstream response only; the targets are scored separately (target_*).
        tgt = [pos[g] for g in split_pert(u) if g in pos] if a.exclude_targets else []
        targets[u] = [w.panel[i] for i in tgt]
        keep = np.ones(len(w.panel), bool)
        keep[tgt] = False
        top = [i for i in np.argsort(-np.abs(t)) if keep[i]][: a.k]
        mr = pb(R)
        for lab in runs:
            mg = np.log1p(1e4 * qb[lab][u] / qb[lab][u].sum())
            dr, dp = mr - mu_c, mg - mu_c
            pred_top = [i for i in np.argsort(-np.abs(dp)) if keep[i]][: a.k]
            m = {"pearson_delta_ctrl": safe_pearson(dp[keep], dr[keep]),
                 "pearson_delta_pert": safe_pearson((mg - mu_tp)[keep], (mr - mu_tp)[keep]),
                 "pearson_de20": safe_pearson(dp[top], dr[top]), "sign_de20": float((np.sign(dp[top]) == np.sign(dr[top])).mean()),
                 "mse_de20": float(((dp[top] - dr[top]) ** 2).mean()), "prec_de20": float(len(set(pred_top) & set(top)) / a.k)}
            if tgt:   # direct target induction: change of the target gene(s) over the held-out control mean
                m["target_abs_err"] = float(np.mean(np.abs(dp[tgt] - dr[tgt])))
                m["target_sign"] = float((np.sign(dp[tgt]) == np.sign(dr[tgt])).mean())
            per[lab][u] = m
    os.makedirs(a.out, exist_ok=True)
    save_json({"split": a.split, "source": "qbar", "exclude_targets": bool(a.exclude_targets), "units": units, "runs": runs,
               "targets_in_panel": targets, "per_unit": per}, os.path.join(a.out, f"{a.stem}.json"))
    for lab in runs:
        keys = sorted({k for u in units for k in per[lab][u]})
        m = {k: float(np.mean([per[lab][u][k] for u in units if k in per[lab][u]])) for k in keys}
        print(f"{lab:45s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
