"""Biology-aware cheap baseline: partial additive + CRISPRa target induction.

python -m src.run_ti_baseline --split test

CRISPRa raises the expression of its own target. The partial-additive baseline gives an unseen gene no effect, so it also
misses this induction. Here every target gene of u that was never perturbed alone in training, and that lies in the panel,
has its additive frequency multiplied by ONE scalar s, then q is renormalized.
s = exp(median over TRAIN single perturbations whose target is in the panel of log(f_pert[target] / f_ctrl[target])), the
same estimator as src/target_induction.py. Nothing is fitted on held-out cells. The run is scored by cell_eval.eval_units
with the matched control cells, so its rows are directly comparable with every other method (tag cb_additive_ti).
"""
import argparse
import os
import numpy as np

from .cell_eval import World, eval_units
from .common import split_pert
from . import baselines as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--name", default="norman")
    ap.add_argument("--base-run", default="cb_additive", help="run whose q gets the target-induction factor (rev4: also CAD)")
    ap.add_argument("--tag", default="", help="output tag (default: <base-run>_ti, or cb_additive_ti for the additive baseline)")
    ap.add_argument("--mode", default="unseen", choices=["unseen", "all"])
    a = ap.parse_args()
    w = World("pert", a.name)
    tag = a.tag or ("cb_additive_ti" if a.base_run == "cb_additive" else a.base_run + "_ti")
    pos = {g: i for i, g in enumerate(w.panel)}
    f_ctrl = B.profile(w.counts(np.where(w.ctrl & (w.split == "train"))[0]))
    tr_lab = set(w.lab[~w.ctrl & (w.split == "train")])
    singles = {p for p in tr_lab if len(split_pert(p)) == 1 and ((w.lab == p) & (w.split == "train")).sum() >= 5}
    selfl = [np.log(B.profile(w.counts(np.where((w.lab == p) & (w.split == "train"))[0]))[pos[p]] / f_ctrl[pos[p]])
             for p in singles if p in pos]
    s = float(np.exp(np.median(selfl)))
    print(f"target-induction scale s={s:.3f} from {len(selfl)} train singles", flush=True)
    # --mode all (rev4, "target prior"): a target gene seen alone in training takes the frequency it reached in its own
    # training single perturbation; an unseen target is multiplied by s. --mode unseen: only unseen targets (the baseline).
    f_self = {p: B.profile(w.counts(np.where((w.lab == p) & (w.split == "train"))[0]))[pos[p]] for p in singles if p in pos}
    z = np.load(os.path.join(w.dir, "runs", a.base_run, f"qbar_{a.split}.npz"), allow_pickle=True)
    q_ti = {}
    for u, q in zip(z["units"], z["q"]):
        q = q.astype(np.float64) / q.sum()
        for g in split_pert(str(u)):
            if g not in pos:
                continue
            if g not in singles:
                q[pos[g]] *= s
            elif a.mode == "all":
                q[pos[g]] = max(q[pos[g]], f_self[g])
        q_ti[str(u)] = q / q.sum()

    def qfn(u, cidx):
        return np.repeat(q_ti[u][None], len(cidx), 0)

    per = eval_units(w, a.split, qfn, tag, min_cells=a.min_cells, seed=0, units=sorted(q_ti))
    print(f"{tag} {a.split}: nll_pooled={np.mean([v['nll_pooled'] for v in per.values()]):.4f} "
          f"mmd_dm={np.mean([v['mmd_dm'] for v in per.values()]):.4f} auc_dm={np.mean([v['auc_dm'] for v in per.values()]):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
