"""Cell -> distribution transformations, text serialization of a cell, and the shared sampler.

Every transformation returns rows on the probability simplex (nonnegative, sum 1), so it can be a training target for the
softmax head. Evaluation never uses a transformed target: held-out cells are always scored by the multinomial NLL of their raw
counts (so transformations are compared on one scale).
"""
import numpy as np

TRANSFORMS = ("freq", "smooth", "log1p", "sqrt")


def to_dist(counts, kind="freq", alpha=0.1):
    """counts [n, G] (or [G]) -> rows on the simplex.
    freq   c / sum(c)                         (maximum likelihood)
    smooth (c + alpha) / (sum(c) + alpha G)   (pseudo-count)
    log1p  log(1+c) / sum log(1+c)            (compresses the dynamic range)
    sqrt   sqrt(c) / sum sqrt(c)              (Hellinger-style)"""
    c = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    G = c.shape[1]
    if kind == "freq":
        v = c
    elif kind == "smooth":
        v = c + alpha
    elif kind == "log1p":
        v = np.log1p(c)
    elif kind == "sqrt":
        v = np.sqrt(c)
    else:
        raise ValueError(kind)
    s = v.sum(1, keepdims=True)
    out = np.where(s > 0, v / np.where(s > 0, s, 1.0), 1.0 / G)
    return out[0] if np.ndim(counts) == 1 else out


def serialize_cell(counts, panel, k=48, levels=14):
    """Mass-preserving text of one cell: its top-k genes with a quantized log-mass level (0..levels-1) of the CP10k value.
    This keeps magnitude (unlike a rank sentence) but is lossy: only k of the panel genes are listed."""
    c = np.asarray(counts, dtype=np.float64).ravel()
    s = c.sum()
    if s <= 0:
        return ""
    top = np.argsort(-c)[:k]
    top = top[c[top] > 0]
    lvl = np.minimum(levels - 1, np.floor(np.log2(1.0 + 1e4 * c[top] / s))).astype(int)
    return " ".join(f"{panel[i]}:{l}" for i, l in zip(top, lvl))


def sample_counts(q, n, lib_sizes, rng, kappa=None):
    """Multinomial(s, q) (or Dirichlet-multinomial with concentration kappa); s resampled from `lib_sizes`. One sampler for all methods."""
    q = np.asarray(q, dtype=np.float64)
    q = q / q.sum()
    s = rng.choice(lib_sizes, n)
    out = np.zeros((n, len(q)), dtype=np.int32)
    for i in range(n):
        p = rng.dirichlet(kappa * q + 1e-8) if kappa else q
        out[i] = rng.multinomial(int(s[i]), p / p.sum())
    return out


def nll_per_umi(counts, q):
    """Multinomial NLL of pooled counts [n,G] (or [G]) under q, per UMI."""
    c = np.asarray(counts, dtype=np.float64)
    c = c.sum(0) if c.ndim == 2 else c
    return float(-(c * np.log(np.clip(q, 1e-12, None))).sum() / c.sum())


def serialize_rank(counts, panel, k=200):
    """Native Cell2Sentence text: the top-k expressed panel genes ordered by decreasing expression, no mass information."""
    c = np.asarray(counts, dtype=np.float64).ravel()
    top = np.argsort(-c)[:k]
    top = top[c[top] > 0]
    return " ".join(panel[i] for i in top)


def serialize_contrast(counts, panel, ref_freq, k=48, max_level=9):
    """Differential-state text of one cell: its k genes most over-represented relative to a reference frequency
    (the mean training control cell), as GENE:level with level = min(max_level, floor(log2(p_g / ref_g))), p = c / sum(c).
    Only expressed genes (c_g > 0) with p_g > ref_g are listed, so the text says what is unusually high in THIS cell."""
    c = np.asarray(counts, dtype=np.float64).ravel()
    s = c.sum()
    if s <= 0:
        return ""
    nz = np.where(c > 0)[0]
    r = np.log2((c[nz] / s) / np.asarray(ref_freq)[nz])
    keep = r > 0
    nz, r = nz[keep], r[keep]
    o = np.argsort(-r, kind="stable")[:k]
    return " ".join(f"{panel[nz[i]]}:{int(min(max_level, np.floor(r[i])))}" for i in o)
