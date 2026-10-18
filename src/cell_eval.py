"""Shared scoring for the two per-cell use cases (no torch import, runs on CPU).

task "cond": unit = cell type, held-out cells = cells of held-out DONORS; a model gives q(cell type).
task "pert": unit = perturbation, held-out cells = perturbed cells of held-out perturbations; a model gives q(perturbation, control cell)
             for M random held-out control cells (the C2S-tutorial pairing: a random control cell in, a perturbed cell out).

Per unit and per model: pooled NLL per UMI under the mean of the q's, per-cell mixture NLL per UMI (log mean_m Mult(x | q_m)),
and generation metrics on cells sampled from q (energy distance, RBF MMD, real-vs-generated classifier AUC, gene mean/variance
correlation, cell-type accuracy). Unit of analysis for statistics = unit (cell type or perturbation), never cells.
"""
import os
import time
import zlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

from .common import OUT_DIR, load_json, save_json
from .cell_dist import sample_counts, nll_per_umi
from .evaluate import log_cp10k, energy_distance
from .preprocess import is_control
from .run_baselines import norm_label


class World:
    def __init__(self, task, name, seed=0):
        self.task, self.name = task, name
        d = os.path.join(OUT_DIR, name)
        self.dir = d
        self.X = sp.load_npz(os.path.join(d, "counts.npz")).tocsr()
        self.obs = pd.read_csv(os.path.join(d, "obs.csv"), index_col=0)
        self.panel = load_json(os.path.join(d, "panel.json"))["panel"]
        self.split = self.obs["split"].values
        if task == "cond":
            self.lab = self.obs["cell_type"].astype(str).values
            self.ctrl = np.zeros(len(self.lab), bool)
        else:
            raw = self.obs["perturbation"].astype(str)
            self.lab = raw.map(norm_label).values
            self.ctrl = np.array([is_control(p) or l == "control" for p, l in zip(raw, self.lab)])
        tr = np.where((self.split == "train") & ~self.ctrl)[0]
        self.lib = np.asarray(self.X[tr].sum(1)).ravel()
        rng = np.random.default_rng(seed)
        sub = np.sort(rng.choice(tr, min(len(tr), 20000), replace=False))
        Lt = log_cp10k(np.asarray(self.X[sub].todense()))
        self.pca = PCA(n_components=50, random_state=seed).fit(Lt)
        self._sub, self._seed = sub, seed
        self.kappa = self.fit_kappa()
        self._clf = None

    def counts(self, idx):
        return np.asarray(self.X[np.asarray(idx)].todense())

    def units(self, split, min_cells):
        m = (self.split == split) & ~self.ctrl
        u, n = np.unique(self.lab[m], return_counts=True)
        return sorted(x for x, k in zip(u, n) if k >= min_cells)

    def fit_kappa(self, top=300):
        """Dirichlet-multinomial concentration from within-unit variance of train cells (method of moments, median over top genes)."""
        idx = np.where((self.split == "train") & ~self.ctrl)[0]
        if len(idx) > 40000:
            idx = np.sort(np.random.default_rng(1).choice(idx, 40000, replace=False))
        C = self.counts(idx)
        s = C.sum(1)
        P = C / s[:, None]
        labs = self.lab[idx]
        res = np.zeros(C.shape[1]); mu_all = np.zeros(C.shape[1]); n = 0
        for u in np.unique(labs):
            m = labs == u
            if m.sum() < 30:
                continue
            res += ((P[m] - P[m].mean(0)) ** 2).sum(0)
            mu_all += P[m].sum(0); n += m.sum()
        var, mu = res / max(n, 1), mu_all / max(n, 1)
        g = np.argsort(-mu)[:top]
        sbar = float(np.mean(1.0 / s))
        ratio = var[g] / (mu[g] * (1 - mu[g]))
        kap = (1 - sbar) / np.maximum(ratio - sbar, 1e-6) - 1
        return float(np.clip(np.median(kap), 5, 1e5))

    def type_classifier(self):
        if self._clf is None:
            Z = self.pca.transform(log_cp10k(self.counts(self._sub)))
            self._clf = LogisticRegression(max_iter=300, C=1.0).fit(Z, self.lab[self._sub])
        return self._clf


def mmd_rbf(A, B, max_n=500, seed=0):
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(len(A), max_n), replace=False)]
    B = B[rng.choice(len(B), min(len(B), max_n), replace=False)]
    Z = np.vstack([A, B])
    d2 = ((Z ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None] - 2 * Z @ Z.T).clip(min=0)
    sig2 = np.median(d2[d2 > 0])
    K = np.exp(-d2 / sig2)
    n = len(A)
    Kaa, Kbb, Kab = K[:n, :n], K[n:, n:], K[:n, n:]
    return float((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1)) + (Kbb.sum() - np.trace(Kbb)) / (len(B) * (len(B) - 1)) - 2 * Kab.mean())


def real_vs_gen_auc(R, G, seed=0):
    """Cross-validated AUC of a logistic classifier separating real from generated cells in PCA space (0.5 = indistinguishable)."""
    n = min(len(R), len(G), 400)
    rng = np.random.default_rng(seed)
    X = np.vstack([R[rng.choice(len(R), n, replace=False)], G[rng.choice(len(G), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    p = cross_val_predict(LogisticRegression(max_iter=300, C=1.0), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def gen_metrics(world, real, gen, unit):
    Lr, Lg = log_cp10k(real), log_cp10k(gen)
    Zr, Zg = world.pca.transform(Lr), world.pca.transform(Lg)
    out = {"energy": energy_distance(Zr, Zg), "mmd": mmd_rbf(Zr, Zg), "auc": real_vs_gen_auc(Zr, Zg),
           "mean_corr": float(np.corrcoef(Lr.mean(0), Lg.mean(0))[0, 1]), "var_corr": float(np.corrcoef(Lr.var(0), Lg.var(0))[0, 1]),
           "var_ratio": float(Lg.var(0).mean() / (Lr.var(0).mean() + 1e-12))}
    if world.task == "cond":
        clf = world.type_classifier()
        out["type_acc_gen"] = float((clf.predict(Zg) == unit).mean())
        out["type_acc_real"] = float((clf.predict(Zr) == unit).mean())
    return out


def nb_nll_np(x, mu, theta):
    from scipy.special import gammaln
    return -(gammaln(x + theta) - gammaln(theta) - gammaln(x + 1) + theta * (np.log(theta) - np.log(theta + mu))
             + x * (np.log(np.clip(mu, 1e-300, None)) - np.log(theta + mu)))


def eval_units(world, split, qfn, tag, M=50, n_gen=300, min_cells=30, seed=0, sampler=("mult", "dm"), save=True, units=None,
               thfn=None, save_qrows=False):
    """qfn(unit, ctrl_idx) -> q [M, G] (cond: ctrl_idx is None and q has one row). Returns {unit: metrics}.
    thfn (optional, NB head): thfn(unit, ctrl_idx) -> inverse dispersion theta [rows, G]; adds the NB NLL per UMI of the real
    cells under the head's own dispersion (mu = N_i * qbar) and generation metrics from native NB sampling (suffix _nb)."""
    rng = np.random.default_rng(seed)
    units = units or world.units(split, min_cells)
    pool = np.where(world.ctrl & (world.split == split))[0]
    per, gens, qbars, qrows = {}, {}, {}, {}
    for u in units:
        ridx = np.where((world.lab == u) & (world.split == split) & ~world.ctrl)[0]
        real = world.counts(ridx)
        # matched controls: the M control cells of a unit depend only on the unit name, so every method sees the same ones
        cidx = (np.random.default_rng(zlib.crc32(u.encode())).choice(pool, M, replace=False)
                if world.task == "pert" else None)
        q = np.atleast_2d(qfn(u, cidx))
        qbar = q.mean(0)
        qbars[u] = qbar.astype(np.float32)
        if save_qrows:
            qrows[u] = q[:M].astype(np.float16)          # one row per control cell (pert) or per draw; for the appendix demo
        r = {"nll_pooled": nll_per_umi(real, qbar)}
        L = real @ np.log(np.clip(q, 1e-12, None)).T          # [n_real, M]
        r["nll_mix"] = float(-(logsumexp(L, axis=1) - np.log(q.shape[0])).sum() / real.sum())
        n = min(n_gen, len(real))
        th = None
        if thfn is not None:
            th = np.atleast_2d(thfn(u, cidx))
            N = real.sum(1, keepdims=True)
            r["nll_nb_own"] = float(nb_nll_np(real, N * qbar[None], th.mean(0)[None]).sum() / real.sum())
        for smp in tuple(sampler) + (("nb",) if th is not None else ()):
            kap = world.kappa if smp == "dm" else None
            rows = []
            for i in range(n):
                if smp == "nb":
                    qi = q[i % q.shape[0]]
                    mu = rng.choice(world.lib) * qi / qi.sum()
                    t_i = th[i % th.shape[0]]
                    rows.append(rng.poisson(rng.gamma(t_i, mu / t_i)).astype(np.int32))
                else:
                    rows.append(sample_counts(q[i % q.shape[0]], 1, world.lib, rng, kap)[0])
            g = np.stack(rows)
            gens[(u, smp)] = g
            for k, v in gen_metrics(world, real[rng.choice(len(real), n, replace=False)] if len(real) > n else real, g, u).items():
                r[f"{k}_{smp}"] = v
        per[u] = r
    if save:
        run = os.path.join(world.dir, "runs", tag)
        os.makedirs(run, exist_ok=True)
        save_json({"task": world.task, "split": split, "kappa": world.kappa, "per_unit": per}, os.path.join(run, f"cell_{split}.json"))
        np.savez_compressed(os.path.join(run, f"qbar_{split}.npz"), units=np.array(list(units)), q=np.stack([qbars[u] for u in units]))
        if save_qrows:
            np.savez_compressed(os.path.join(run, f"qrows_{split}.npz"), units=np.array(list(units)),
                                **{f"{u}__q": qrows[u] for u in units})
            if world.task == "pert":
                np.savez_compressed(os.path.join(run, f"ctrl_idx_{split}.npz"), units=np.array(list(units)),
                                    **{f"{u}__c": np.random.default_rng(zlib.crc32(u.encode())).choice(
                                        np.where(world.ctrl & (world.split == split))[0], M, replace=False) for u in units})
        pack = {}
        for (u, smp), g in gens.items():
            if smp == "mult":
                s = sp.csr_matrix(g)
                pack[f"{u}__data"], pack[f"{u}__ind"], pack[f"{u}__ptr"] = s.data, s.indices, s.indptr
        np.savez_compressed(os.path.join(run, f"gen_{split}.npz"), units=np.array(list(units)), **pack)
    return per
