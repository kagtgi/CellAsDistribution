"""Run mean / additive / ridge baselines on the held-out split and write metrics JSON.

python -m src.run_baselines --name norman --pert-col perturbation [--split test] [--n-cells 100]
Output: OUT_DIR/<name>/metrics_baselines_<split>.json  {model: {metric: [mean, lo, hi]}, per_pert: ...}
"""
import argparse
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import SEED, OUT_DIR, load_json, save_json, split_pert
from .evaluate import Scorer, summarize
from .gene_tree import gene_embedding
from . import baselines as B
from .preprocess import is_control


def norm_label(p):
    return "+".join(split_pert(p)) or "control"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pert-col", default="perturbation")
    ap.add_argument("--split", default="val")   # val only until the metrics freeze
    ap.add_argument("--n-cells", type=int, default=100)
    ap.add_argument("--min-real-cells", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    d = os.path.join(OUT_DIR, a.name)
    X = sp.load_npz(os.path.join(d, "counts.npz")).tocsr()
    obs = pd.read_csv(os.path.join(d, "obs.csv"), index_col=0)
    panel = load_json(os.path.join(d, "panel.json"))["panel"]
    lab = obs[a.pert_col].astype(str).map(norm_label).values
    ctrl = np.array([is_control(p) or l == "control" for p, l in zip(obs[a.pert_col].astype(str), lab)])
    split = obs["split"].values

    def counts(mask):
        return np.asarray(X[np.where(mask)[0]].todense())

    tr_ctrl = counts(ctrl & (split == "train"))
    train_cells = {p: np.where((lab == p) & (split == "train"))[0] for p in set(lab[~ctrl & (split == "train")])}
    train_cells = {p: i for p, i in train_cells.items() if len(i) >= 5}
    train_pb = {p: B.profile(np.asarray(X[i].todense())) for p, i in train_cells.items()}
    from .eval_common import sample_rows
    tr_pert = sample_rows(counts, ~ctrl & (split == "train"))
    scorer = Scorer(tr_ctrl, tr_pert)

    test_perts = [p for p in set(lab[~ctrl & (split == a.split)]) if (lab == p)[split == a.split].sum() >= a.min_real_cells]
    real = {p: counts((lab == p) & (split == a.split)) for p in sorted(test_perts)}
    lib = np.asarray(X[np.where(split == "train")[0]].sum(1)).ravel()
    f_ctrl = B.profile(tr_ctrl)
    emb = gene_embedding(X[np.where(split == "train")[0]])

    pm = B.perturbed_mean(train_pb)
    models = {"control_mean": B.control_mean(f_ctrl), "perturbed_mean": pm,
              "additive": B.additive(train_pb, f_ctrl, pm), "ridge": B.ridge(train_pb, emb, panel, f_ctrl)}
    results, per_all = {}, {}
    for name, fn in models.items():
        pred = {p: B.sample_cells(fn(p), a.n_cells, lib, rng) for p in real}
        per = scorer.score_all(real, pred)
        results[name] = {m: list(v) for m, v in summarize(per).items()}
        per_all[name] = per
        print(f"{name:15s} pearson_delta_pert={results[name]['pearson_delta_pert'][0]:+.3f} "
              f"energy={results[name]['energy'][0]:.3f} centroid_acc={results[name]['centroid_acc'][0]:.2f}")
    save_json({"n_test_perts": len(real), "models": results, "per_pert": per_all},
              os.path.join(d, f"metrics_baselines_{a.split}.json"))
    print("wrote metrics; test perts:", len(real))


if __name__ == "__main__":
    main()
