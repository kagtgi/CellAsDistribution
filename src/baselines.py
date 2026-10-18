"""Mandatory baselines (Systema / Ahlmann-Eltze): control mean, perturbed mean, additive, ridge.

All baselines work in gene-FREQUENCY space (smoothed count frequencies, logs only for fold-changes),
so each outputs a valid probability vector q per perturbation. Averaging log-CP10k and mapping back with
expm1 distorts frequencies (found in G1 check, 13 Oct) and is NOT used. Cells for every model come from the
SAME sampler (`sample_cells`), so distributional metrics are comparable.
"""
import numpy as np
from sklearn.linear_model import Ridge

from .common import split_pert
from .evaluate import log_cp10k  # re-exported for callers


def profile(counts, alpha=0.5):
    """Smoothed gene frequency of a counts matrix [cells x genes] (pooled over cells)."""
    c = np.asarray(counts, dtype=np.float64).sum(0)
    f = c + alpha
    return f / f.sum()


def renorm(f):
    f = np.clip(f, 1e-12, None)
    return f / f.sum()


def sample_cells(q, n, lib_sizes, rng, kappa=None):
    """Multinomial(s, q) or Dirichlet-multinomial when kappa is given; s drawn from train library sizes."""
    s = rng.choice(lib_sizes, n)
    out = np.zeros((n, len(q)), dtype=np.int32)
    for i in range(n):
        p = rng.dirichlet(kappa * q + 1e-6) if kappa else q
        out[i] = rng.multinomial(int(s[i]), p / p.sum())
    return out


def control_mean(f_ctrl):
    return lambda pert: f_ctrl


def perturbed_mean(train_f):
    """Equal weight per training perturbation (Systema perturbed-centroid reference), frequency space."""
    m = renorm(np.mean(list(train_f.values()), axis=0))
    return lambda pert: m


def additive(train_f, f_ctrl, fallback):
    """pair = f_ctrl * exp(sum of log-fold-changes of its singles); unseen singles -> fallback."""
    lfc = {p: np.log(f) - np.log(f_ctrl) for p, f in train_f.items()}

    def pred(pert):
        # partial additive: sum the log-fold-changes of the genes whose single effect is known; an unseen gene
        # contributes 0 (so a pair with one unseen gene is predicted from its seen partner, not from the mean)
        known = [g for g in split_pert(pert) if g in lfc]
        if not known:
            return fallback(pert)
        return renorm(f_ctrl * np.exp(np.sum([lfc[g] for g in known], axis=0)))
    return pred


def ridge(train_f, gene_emb, panel, f_ctrl, alpha=10.0):
    """Perturbation embedding = sum of panel-gene embeddings of its targets; target = log f - log f_ctrl."""
    pos = {g: i for i, g in enumerate(panel)}

    def emb(pert):
        v = np.zeros(gene_emb.shape[1])
        for g in split_pert(pert):
            if g in pos:
                v += gene_emb[pos[g]]
        return v
    keys = sorted(train_f)
    Xtr = np.stack([emb(p) for p in keys])
    Ytr = np.stack([np.log(train_f[p]) - np.log(f_ctrl) for p in keys])
    model = Ridge(alpha=alpha).fit(Xtr, Ytr)
    mean_lfc = Ytr.mean(0)

    def pred(pert):
        e = emb(pert)
        return renorm(f_ctrl * np.exp(model.predict(e[None])[0] if e.any() else mean_lfc))
    return pred
