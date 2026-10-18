"""Baselines for the per-cell use cases, scored by the same eval_units as the language-model decoders.

python -m src.run_cell_baselines --task cond --name pbmc --split val
python -m src.run_cell_baselines --task pert --name norman --split val
Writes OUT_DIR/<name>/runs/<baseline-tag>/cell_<split>.json and gen_<split>.npz.
"""
import argparse
import numpy as np

from .cell_eval import World, eval_units
from .cell_dist import to_dist
from .gene_tree import gene_embedding
from . import baselines as B


def build_qfns(w):
    tr = (w.split == "train") & ~w.ctrl
    G = len(w.panel)
    fns = {}
    if w.task == "cond":
        glob = B.profile(w.counts(np.where(tr)[0]))
        tm = {}
        for u in np.unique(w.lab[tr]):
            i = np.where(tr & (w.lab == u))[0]
            if len(i) >= 5:
                tm[u] = B.profile(w.counts(i))
        fns["global mean"] = lambda u, c: glob[None]
        fns["type mean"] = lambda u, c: tm.get(u, glob)[None]
        return fns
    ctr = w.ctrl & (w.split == "train")
    f_ctrl = B.profile(w.counts(np.where(ctr)[0]))
    train_f = {}
    for p in np.unique(w.lab[tr]):
        i = np.where(tr & (w.lab == p))[0]
        if len(i) >= 5:
            train_f[p] = B.profile(w.counts(i))
    pm = B.perturbed_mean(train_f)
    add = B.additive(train_f, f_ctrl, pm)
    rd = B.ridge(train_f, gene_embedding(w.X[np.where(w.split == "train")[0]]), w.panel, f_ctrl)
    fns["control mean"] = lambda u, c: f_ctrl[None]
    fns["perturbed mean"] = lambda u, c: pm(u)[None]
    fns["additive"] = lambda u, c: add(u)[None]
    fns["ridge"] = lambda u, c: rd(u)[None]
    # copy-control: the control cell's own (smoothed) distribution mixed with the perturbed mean; lambda fit on TRAIN cells
    rng = np.random.default_rng(0)
    pi = rng.choice(np.where(tr)[0], 3000, replace=False)
    ci = rng.choice(np.where(ctr)[0], 3000)
    Cp, Cc = w.counts(pi), w.counts(ci)
    Pc = to_dist(Cc, "smooth", 0.5)
    best = (1e9, 0.0)
    for lam in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
        mix = lam * Pc + (1 - lam) * pm("x")[None]
        nll = float(-(Cp * np.log(mix)).sum() / Cp.sum())
        best = min(best, (nll, lam))
    lam = best[1]
    print("copy-control lambda (fit on train cells):", lam)

    def copy_ctrl(u, cidx):
        P = to_dist(w.counts(cidx), "smooth", 0.5)
        q = lam * P + (1 - lam) * pm(u)[None]
        return q / q.sum(1, keepdims=True)
    fns["control cell + perturbed mean"] = copy_ctrl

    # dispersion controls: a mixture over random control cells gives an overdispersed model with NO learned control information.
    # lambda / alpha are fit on TRAIN cells by the same per-cell mixture NLL used for evaluation (unpaired random controls).
    from scipy.special import logsumexp
    pm0 = pm("x")
    xs = w.counts(rng.choice(np.where(tr)[0], 300, replace=False))
    Pm = to_dist(w.counts(rng.choice(np.where(ctr)[0], 50, replace=False)), "smooth", 0.5)

    def mixnll(Q):
        L = xs @ np.log(np.clip(Q, 1e-12, None)).T
        return float(-(logsumexp(L, axis=1) - np.log(Q.shape[0])).sum() / xs.sum())
    lam2 = min((mixnll((1 - l) * pm0[None] + l * Pm), l) for l in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5))[1]
    alpha = min((mixnll(np.exp(np.log(pm0)[None] + a * (np.log(Pm) - np.log(f_ctrl)[None])) / np.exp(np.log(pm0)[None] + a * (np.log(Pm) - np.log(f_ctrl)[None])).sum(1, keepdims=True)), a)
                for a in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0))[1]
    print("control-mixture lambda:", lam2, "control-state tilt alpha:", alpha)

    def ctrl_mix(u, cidx):
        P = to_dist(w.counts(cidx), "smooth", 0.5)
        return (1 - lam2) * pm(u)[None] + lam2 * P

    def ctrl_tilt(u, cidx):
        P = to_dist(w.counts(cidx), "smooth", 0.5)
        q = np.exp(np.log(pm(u))[None] + alpha * (np.log(P) - np.log(f_ctrl)[None]))
        return q / q.sum(1, keepdims=True)
    fns["control-mixture (dispersion only)"] = ctrl_mix
    fns["control-state tilt (alpha fit)"] = ctrl_tilt
    return fns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cond", "pert"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--min-cells", type=int, default=30)
    a = ap.parse_args()
    w = World(a.task, a.name)
    print("DM kappa:", w.kappa)
    for label, fn in build_qfns(w).items():
        tag = "cb_" + label.replace(" ", "_").replace("+", "plus")
        per = eval_units(w, a.split, fn, tag, min_cells=a.min_cells)
        print(f"{label:32s} nll_pooled={np.mean([v['nll_pooled'] for v in per.values()]):.4f} "
              f"nll_mix={np.mean([v['nll_mix'] for v in per.values()]):.4f} auc_mult={np.mean([v['auc_mult'] for v in per.values()]):.3f} "
              f"mmd_dm={np.mean([v['mmd_dm'] for v in per.values()]):.4f} units={len(per)}", flush=True)


if __name__ == "__main__":
    main()
