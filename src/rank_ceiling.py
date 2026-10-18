"""Rank-representation likelihood ceiling (CPU). Question: how much held-out likelihood can ANY model reach if the cell
is first written as a C2S rank sentence and reconstructed with the tutorial's rank->expression adapter?

Take REAL held-out cells -> top-k sentence -> adapter profile -> average over the perturbation's cells -> mix with the
train-control prior (eps fit on TRAIN cells only) -> held-out NLL against the same cells' pooled counts.
Compare with: the same cells through the mass representation (smoothed own frequency = in-sample oracle),
the perturbed mean, and CAD-P. A rank ceiling near/above the perturbed mean means the representation, not the model,
caps C2S-style likelihoods.

python -m src.rank_ceiling --name norman --split val --ks 200 2048
"""
import argparse
import numpy as np

from .eval_common import load_world
from .eval_nll import nll_per_umi
from .evaluate import bootstrap_ci
from .c2s_baseline import sentence_from_counts, fit_rank_expression, profile_from_sentence
from . import baselines as B


def rank_profile(C, panel, pos, a, b, k, max_cells=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(C), min(len(C), max_cells), replace=False)
    prof = np.zeros(len(panel))
    for i in rows:
        prof += profile_from_sentence(sentence_from_counts(C[i], panel, k), pos, a, b, len(panel))
    return prof / max(prof.sum(), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ks", type=int, nargs="+", default=[200, 2048])
    ap.add_argument("--cadf-tag", default="", help="run tag whose q_by_pert.npz gives CAD-P for the paired comparison")
    a = ap.parse_args()
    import os
    d, X, obs, panel, lab, ctrl, spl = load_world(a.name)
    pos = {g: i for i, g in enumerate(panel)}
    cnt = lambda m: np.asarray(X[np.where(m)[0]].todense())
    tr = np.where(spl == "train")[0]
    f_ctrl = B.profile(cnt(ctrl & (spl == "train")))
    tcells = {p: np.where((lab == p) & (spl == "train"))[0] for p in set(lab[~ctrl & (spl == "train")])}
    tcells = {p: i for p, i in tcells.items() if len(i) >= 5}
    pm = B.perturbed_mean({p: B.profile(np.asarray(X[i].todense())) for p, i in tcells.items()})
    perts = sorted(p for p in set(lab[~ctrl & (spl == a.split)]) if ((lab == p) & (spl == a.split)).sum() >= 20)
    real = {p: cnt((lab == p) & (spl == a.split)) for p in perts}
    ref = {p: nll_per_umi(real[p], pm(p)) for p in perts}
    rng = np.random.default_rng(0)
    train_perts = sorted(tcells)
    fit_perts = [train_perts[i] for i in rng.choice(len(train_perts), min(40, len(train_perts)), replace=False)]
    print(f"{a.name} {a.split}: n_perts={len(perts)}; perturbed-mean NLL={np.mean(list(ref.values())):.4f}; "
          f"mass oracle (in-sample)={np.mean([nll_per_umi(real[p], (real[p].sum(0)+.5)/(real[p].sum()+.5*len(panel))) for p in perts]):.4f}")
    for k in a.ks:
        ra, rb = fit_rank_expression(X, panel, tr, k=min(k, 200) if k <= 200 else 2048)
        # eps fit on TRAIN-perturbation cells only (one scalar)
        tr_prof = {p: (rank_profile(np.asarray(X[tcells[p]].todense()), panel, pos, ra, rb, k), np.asarray(X[tcells[p]].todense())) for p in fit_perts}
        grid = [0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.8]
        fit = [np.mean([nll_per_umi(c, (1 - e) * pr + e * f_ctrl) for pr, c in tr_prof.values()]) for e in grid]
        eps = grid[int(np.argmin(fit))]
        res = {}
        for p in perts:
            q = (1 - eps) * rank_profile(real[p], panel, pos, ra, rb, k) + eps * f_ctrl
            res[p] = nll_per_umi(real[p], q / q.sum())
        diff = bootstrap_ci([res[p] - ref[p] for p in perts])
        cr = cnt(ctrl & (spl == a.split))
        qc = (1 - eps) * rank_profile(cr, panel, pos, ra, rb, k) + eps * f_ctrl
        print(f"rank ceiling top-{k}: eps(train-fit)={eps}  NLL={np.mean(list(res.values())):.4f}  "
              f"diff vs perturbed mean {diff[0]:+.4f} [{diff[1]:+.4f},{diff[2]:+.4f}]  control NLL={nll_per_umi(cr, qc/qc.sum()):.4f} "
              f"(f_ctrl={nll_per_umi(cr, f_ctrl):.4f})")
        # IN-SAMPLE comparison (the rank ceiling rewrites the very cells it scores): compare with the in-sample MASS oracle
        mass = {p: nll_per_umi(real[p], (real[p].sum(0) + .5) / (real[p].sum() + .5 * len(panel))) for p in perts}
        out = {"k": k, "eps": eps, "n": len(perts),
               "rank_ceiling_vs_pm": bootstrap_ci([res[p] - ref[p] for p in perts]),
               "mass_oracle_insample_vs_pm": bootstrap_ci([mass[p] - ref[p] for p in perts]),
               "rank_minus_mass_oracle": bootstrap_ci([res[p] - mass[p] for p in perts]),
               "share_of_insample_headroom_kept_by_rank": float(np.mean([ref[p] - res[p] for p in perts]) / np.mean([ref[p] - mass[p] for p in perts])),
               "per_pert": {p: {"rank": res[p], "mass_oracle": mass[p], "perturbed_mean": ref[p]} for p in perts}}
        if a.cadf_tag:
            import os
            from .common import OUT_DIR
            z = np.load(os.path.join(OUT_DIR, a.name, "runs", a.cadf_tag, "q_by_pert.npz"))
            qm = dict(zip(z["perts"].tolist(), z["q"]))
            cf = {p: nll_per_umi(real[p], qm[p]) for p in perts if p in qm}
            out["cadf_vs_pm"] = bootstrap_ci([cf[p] - ref[p] for p in cf])
            out["cadf_minus_rank_ceiling"] = bootstrap_ci([cf[p] - res[p] for p in cf])
            for p in cf:
                out["per_pert"][p]["cadf"] = cf[p]
        print("  in-sample mass oracle vs pm %+.4f [%+.4f,%+.4f]; rank keeps %.0f%% of in-sample headroom" % (*out["mass_oracle_insample_vs_pm"], 100 * out["share_of_insample_headroom_kept_by_rank"]))
        if a.cadf_tag:
            print("  CAD-P vs pm %+.4f [%+.4f,%+.4f]; CAD-P minus rank ceiling %+.4f [%+.4f,%+.4f]" % (*out["cadf_vs_pm"], *out["cadf_minus_rank_ceiling"]))
        from .common import save_json as _sj, OUT_DIR as _OD
        _sj(out, os.path.join(_OD, "auto", f"{a.name}_rank_ceiling_k{k}.json")) if os.path.isdir(os.path.join(_OD, "auto")) else None


if __name__ == "__main__":
    main()
