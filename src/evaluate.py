"""Metrics for held-out perturbations (md 6/9). Unit of analysis = perturbation, never cells.

All metrics work on log1p(CP10k) over the fixed panel. `real` and `pred` are dicts
{perturbation: counts matrix [cells x genes]} so every model (baselines, CAD, C2S) is scored identically.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


def log_cp10k(C):
    C = np.asarray(C, dtype=np.float64)
    s = C.sum(1, keepdims=True).clip(min=1)
    return np.log1p(C / s * 1e4)


def centroid(C):
    return log_cp10k(C).mean(0)


def safe_pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(pearsonr(a, b)[0])


def energy_distance(A, B, max_n=1000, seed=0):
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(len(A), max_n), replace=False)]
    B = B[rng.choice(len(B), min(len(B), max_n), replace=False)]

    def md(X, Y):
        d = np.sqrt(np.maximum(((X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None] - 2 * X @ Y.T), 0))
        return d.mean()
    return float(2 * md(A, B) - md(A, A) - md(B, B))


class Scorer:
    """Fit once on TRAIN cells: control centroid, perturbed centroid, PCA embedding for distances."""

    def __init__(self, train_ctrl_counts, train_pert_counts, n_pcs=30, topk=50, seed=0):
        self.ctrl = centroid(train_ctrl_counts)
        self.pert = centroid(train_pert_counts)
        allc = np.vstack([log_cp10k(train_ctrl_counts), log_cp10k(train_pert_counts)])
        self.pca = PCA(n_components=n_pcs, random_state=seed).fit(allc[np.random.default_rng(seed).choice(len(allc), min(len(allc), 20000), replace=False)])
        self.topk = topk

    def score_one(self, real_counts, pred_counts):
        r, p = centroid(real_counts), centroid(pred_counts)
        out = {}
        for ref_name, ref in (("ctrl", self.ctrl), ("pert", self.pert)):
            out[f"pearson_delta_{ref_name}"] = safe_pearson(p - ref, r - ref)
        out["mae"] = float(np.abs(p - r).mean())
        out["mse"] = float(((p - r) ** 2).mean())
        d_real, d_pred = r - self.ctrl, p - self.ctrl
        top = np.argsort(-np.abs(d_real))[: self.topk]
        out["de_sign_acc"] = float((np.sign(d_pred[top]) == np.sign(d_real[top])).mean())
        out["de_overlap"] = float(len(set(top) & set(np.argsort(-np.abs(d_pred))[: self.topk])) / self.topk)
        out["energy"] = energy_distance(self.pca.transform(log_cp10k(real_counts)), self.pca.transform(log_cp10k(pred_counts)))
        out["var_ratio"] = float(log_cp10k(pred_counts).var(0).mean() / (log_cp10k(real_counts).var(0).mean() + 1e-12))
        return out

    def score_all(self, real, pred):
        perts = sorted(real)
        per = {p: self.score_one(real[p], pred[p]) for p in perts}
        # centroid accuracy (Systema): is the predicted centroid closest to its own real centroid?
        rc = np.stack([centroid(real[p]) for p in perts])
        hit = []
        for i, p in enumerate(perts):
            pc = centroid(pred[p])
            hit.append(float(np.argmin(((rc - pc) ** 2).sum(1)) == i))
        for i, p in enumerate(perts):
            per[p]["centroid_acc"] = hit[i]
        return per


def bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(per, metrics=None):
    metrics = metrics or sorted(next(iter(per.values())).keys())
    return {m: bootstrap_ci([per[p][m] for p in per]) for m in metrics}


def paired_diff(per_a, per_b, metric, n_boot=2000, seed=0):
    """Mean paired difference a-b over shared perturbations with bootstrap CI."""
    ps = sorted(set(per_a) & set(per_b))
    return bootstrap_ci([per_a[p][metric] - per_b[p][metric] for p in ps], n_boot, seed)
