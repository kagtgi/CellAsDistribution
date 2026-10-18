"""Held-out multinomial NLL per UMI (nats) for baselines and any model's q_by_pert.npz  -- the G1 criterion.

python -m src.eval_nll --name norman --runs g1_cadn_08b [--split test]
Lower is better. Reference: uniform = log(G); `oracle` = real test frequencies (smoothed), not a model.
"""
import argparse
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import OUT_DIR, load_json
from .gene_tree import gene_embedding
from .preprocess import is_control
from .run_baselines import norm_label
from . import baselines as B


def nll_per_umi(counts, q):
    q = np.clip(q, 1e-9, None)
    q = q / q.sum()
    c = counts.sum(0)
    return float(-(c * np.log(q)).sum() / c.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pert-col", default="perturbation")
    ap.add_argument("--split", default="val")   # val only until the metrics freeze
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args()
    d = os.path.join(OUT_DIR, a.name)
    X = sp.load_npz(os.path.join(d, "counts.npz")).tocsr()
    obs = pd.read_csv(os.path.join(d, "obs.csv"), index_col=0)
    panel = load_json(os.path.join(d, "panel.json"))["panel"]
    lab = obs[a.pert_col].astype(str).map(norm_label).values
    ctrl = np.array([is_control(p) or l == "control" for p, l in zip(obs[a.pert_col].astype(str), lab)])
    split = obs["split"].values
    cnt = lambda m: np.asarray(X[np.where(m)[0]].todense())
    tr_ctrl = cnt(ctrl & (split == "train"))
    tcells = {p: np.where((lab == p) & (split == "train"))[0] for p in set(lab[~ctrl & (split == "train")])}
    tcells = {p: i for p, i in tcells.items() if len(i) >= 5}
    train_pb = {p: B.profile(np.asarray(X[i].todense())) for p, i in tcells.items()}
    f_ctrl = B.profile(tr_ctrl)
    emb = gene_embedding(X[np.where(split == "train")[0]])
    pm = B.perturbed_mean(train_pb)
    models = {"control_mean": B.control_mean(f_ctrl), "perturbed_mean": pm,
              "additive": B.additive(train_pb, f_ctrl, pm), "ridge": B.ridge(train_pb, emb, panel, f_ctrl)}
    perts = sorted(p for p in set(lab[~ctrl & (split == a.split)]) if ((lab == p) & (split == a.split)).sum() >= 20)
    real = {p: cnt((lab == p) & (split == a.split)) for p in perts}
    res = {"uniform": {p: float(np.log(len(panel))) for p in perts}}
    for n, fn in models.items():
        res[n] = {p: nll_per_umi(real[p], fn(p)) for p in perts}
    train_mean_freq = X[np.where(split == "train")[0]].sum(0).A1
    res["train_pooled_freq"] = {p: nll_per_umi(real[p], train_mean_freq) for p in perts}
    for r in a.runs:
        z = np.load(os.path.join(d, "runs", r, "q_by_pert.npz"))
        qm = dict(zip(z["perts"].tolist(), z["q"]))
        res[r] = {p: nll_per_umi(real[p], qm[p]) for p in perts}
    res["oracle_smoothed"] = {p: nll_per_umi(real[p], (real[p].sum(0) + 0.5) / (real[p].sum() + 0.5 * len(panel))) for p in perts}
    from .evaluate import bootstrap_ci
    print(f"split={a.split} n_perts={len(perts)}")
    print(f"{'model':22s} {'NLL/UMI':>8s}  win-rate  paired diff vs perturbed_mean [95% CI] (negative = better)")
    for n, v in res.items():
        m = np.mean(list(v.values()))
        wr = np.mean([v[p] < res['perturbed_mean'][p] for p in perts])
        diff = bootstrap_ci([v[p] - res['perturbed_mean'][p] for p in perts])
        print(f"{n:22s} {m:8.4f}  {wr:6.2f}   {diff[0]:+.4f} [{diff[1]:+.4f}, {diff[2]:+.4f}]")
    from .common import split_pert
    for grp, ps in (("SINGLES (unseen gene)", [p for p in perts if len(split_pert(p)) == 1]),
                    ("PAIRS (both genes seen as train singles)", [p for p in perts if len(split_pert(p)) == 2])):
        if not ps:
            continue
        print(f"\n--- {grp}: n={len(ps)}   (paired diff vs perturbed_mean, negative = better)")
        for n, v in res.items():
            diff = bootstrap_ci([v[p] - res['perturbed_mean'][p] for p in ps])
            print(f"{n:22s} {np.mean([v[p] for p in ps]):8.4f}   {diff[0]:+.4f} [{diff[1]:+.4f}, {diff[2]:+.4f}]")
    # control-state fit check (held-out control cells of the same split)
    ctrl_real = cnt(ctrl & (split == a.split))
    print(f"\ncontrol cells ({a.split}): {len(ctrl_real)} | NLL/UMI  f_ctrl(train controls)={nll_per_umi(ctrl_real, f_ctrl):.4f}", end="")
    for r in a.runs:
        z = np.load(os.path.join(d, "runs", r, "q_by_pert.npz"))
        if "q_control" in z.files:
            print(f" | {r} q_control={nll_per_umi(ctrl_real, z['q_control']):.4f}", end="")
    print()


if __name__ == "__main__":
    main()
