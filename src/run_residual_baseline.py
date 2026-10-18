"""Non-language-model control-cell residual: the additive baseline applied to each control cell in log space.

python -m src.run_residual_baseline --split test

For a held-out perturbation u and each of its M matched control cells x0_m (cell_eval.eval_units draws),
    q_m  proportional to  b_m * q_add(u) / b_bar,
where b_m = smooth(x0_m) = (x0_m + 0.5) / (N_m + 0.5 G) is exactly the log-prior b that CAD's control residual uses,
q_add(u) is the partial-additive baseline's frequency for u (runs/cb_additive/qbar_<split>.npz), and b_bar is the mean of
b over the training control cells (equal weight per cell). Averaged over control cells the shift recovers q_add, and each
control cell moves q in the direction of its own deviation from the average control. It is CAD's control residual with the
language model replaced by the additive shift, so it separates the effect of the residual construction from what the
language model learns. (A first version divided by the pooled control frequency instead of b_bar; with ~216 UMIs per cell
the 0.5 pseudo-count makes b mostly uniform, so that version was not a fair analogue and was discarded.)
"""
import argparse
import os
import numpy as np

from .cell_eval import World, eval_units
from .cell_dist import to_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-cells", type=int, default=30)
    a = ap.parse_args()
    w = World("pert", "norman")
    z = np.load(os.path.join(w.dir, "runs", "cb_additive", f"qbar_{a.split}.npz"), allow_pickle=True)
    q_add = {str(u): q.astype(np.float64) for u, q in zip(z["units"], z["q"])}
    b_bar = to_dist(w.counts(np.where(w.ctrl & (w.split == "train"))[0]), "smooth", 0.5).mean(0)
    shift = {u: np.log(q / q.sum()) - np.log(b_bar) for u, q in q_add.items()}

    def qfn(u, cidx):
        L = np.log(to_dist(w.counts(cidx), "smooth", 0.5)) + shift[u][None]
        L -= L.max(1, keepdims=True)
        Q = np.exp(L)
        return Q / Q.sum(1, keepdims=True)

    per = eval_units(w, a.split, qfn, "cb_additive_residual", min_cells=a.min_cells, seed=0, units=sorted(q_add))
    print(f"additive residual {a.split}: nll_pooled={np.mean([v['nll_pooled'] for v in per.values()]):.4f} "
          f"nll_mix={np.mean([v['nll_mix'] for v in per.values()]):.4f} mmd_dm={np.mean([v['mmd_dm'] for v in per.values()]):.4f} "
          f"auc_dm={np.mean([v['auc_dm'] for v in per.values()]):.3f}", flush=True)


if __name__ == "__main__":
    main()
