"""Reference baselines that replay REAL training cells (no model): an empirical upper reference for sample-quality metrics.

python -m src.run_resample_baseline --task cond --name pbmc   --split test
python -m src.run_resample_baseline --task pert --name norman --split test

cond: for a cell type u, q has one row per training cell of type u (300 drawn at random, training donors only):
      q_m = 0.9 * (the cell's own frequency x_m / N_m) + 0.1 * (pooled training frequency of type u), a fixed weight
      (a per-gene pseudo-count would swamp cells with ~300 panel UMIs); the per-cell NLL is the mixture over these 300 cells and each
      generated cell is a resampled training cell (shared samplers). This is what latent-variable models that encode training
      cells approach, so it calibrates how much of a low MMD/AUC needs any model at all.
pert: the perturbation is unseen, so the analogue draws 300 random TRAINING perturbed cells of any perturbation, mixed
      0.9 / 0.1 with the pooled training perturbed frequency.
"""
import argparse
import numpy as np

from .cell_eval import World, eval_units
from .cell_dist import to_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cond", "pert"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--min-cells", type=int, default=30)
    a = ap.parse_args()
    w = World(a.task, a.name)
    rng = np.random.default_rng(0)
    tr = np.where((w.split == "train") & ~w.ctrl)[0]
    pooled = {}

    def qfn(u, cidx):
        ii = tr[w.lab[tr] == u] if a.task == "cond" else tr
        if len(ii) == 0:
            ii = tr
        key = u if a.task == "cond" else "__all__"
        if key not in pooled:
            pooled[key] = to_dist(w.counts(ii).sum(0), "smooth", 0.5)
        pick = rng.choice(ii, a.draws, replace=len(ii) < a.draws)
        Q = 0.9 * to_dist(w.counts(pick), "freq") + 0.1 * pooled[key][None]
        if a.task == "pert" and cidx is not None:
            return Q[: len(cidx)] if len(cidx) <= len(Q) else Q
        return Q

    tag = "cb_train_cell_resampling"
    per = eval_units(w, a.split, qfn, tag, min_cells=a.min_cells, seed=0)
    print(f"{tag} {a.name} {a.split}: nll_pooled={np.mean([v['nll_pooled'] for v in per.values()]):.4f} "
          f"nll_mix={np.mean([v['nll_mix'] for v in per.values()]):.4f} mmd_dm={np.mean([v['mmd_dm'] for v in per.values()]):.4f} "
          f"auc_mult={np.mean([v['auc_mult'] for v in per.values()]):.3f} mmd_mult={np.mean([v['mmd_mult'] for v in per.values()]):.4f}", flush=True)


if __name__ == "__main__":
    main()
