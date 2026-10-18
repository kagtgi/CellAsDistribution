"""Identical generation + scoring for every model that produces a gene distribution q per state.

score_model(name, tag, q_fn, ...): q_fn(pert_label) -> q[2048]. Used by CAD-N, CAD-P, CAD-R, C2S adapters.
Writes runs/<tag>/{metrics.json, q_by_pert.npz}; returns summary dict.
"""
import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import OUT_DIR, load_json, save_json
from .evaluate import Scorer, summarize
from . import baselines as B
from .preprocess import is_control
from .run_baselines import norm_label


def load_world(name, pert_col="perturbation"):
    d = os.path.join(OUT_DIR, name)
    X = sp.load_npz(os.path.join(d, "counts.npz")).tocsr()
    obs = pd.read_csv(os.path.join(d, "obs.csv"), index_col=0)
    panel = load_json(os.path.join(d, "panel.json"))["panel"]
    lab = obs[pert_col].astype(str).map(norm_label).values
    ctrl = np.array([is_control(p) or l == "control" for p, l in zip(obs[pert_col].astype(str), lab)])
    return d, X, obs, panel, lab, ctrl, obs["split"].values


def score_model(name, tag, q_fn, eval_split="val", n_gen=100, seed=3407, pert_col="perturbation", min_cells=20,
                perts_subset=None):
    d, X, obs, panel, lab, ctrl, split = load_world(name, pert_col)
    run = os.path.join(d, "runs", tag)
    os.makedirs(run, exist_ok=True)
    cnt = lambda m: np.asarray(X[np.where(m)[0]].todense())
    scorer = Scorer(cnt(ctrl & (split == "train")), sample_rows(cnt, ~ctrl & (split == "train")))
    perts = sorted(p for p in set(lab[~ctrl & (split == eval_split)]) if ((lab == p) & (split == eval_split)).sum() >= min_cells)
    if perts_subset is not None:
        perts = [p for p in perts if p in set(perts_subset)]
    real = {p: cnt((lab == p) & (split == eval_split)) for p in perts}
    lib = np.asarray(X[np.where(split == "train")[0]].sum(1)).ravel()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    qs = {p: q_fn(p) for p in perts}
    q_control = q_fn("control")
    pred = {p: B.sample_cells(qs[p], n_gen, lib, rng) for p in perts}
    per = scorer.score_all(real, pred)
    summ = {m: list(v) for m, v in summarize(per).items()}
    save_json({"n_perts": len(perts), "split": eval_split, "models": {tag: summ}, "per_pert": {tag: per}},
              os.path.join(run, "metrics.json"))
    np.savez_compressed(os.path.join(run, "q_by_pert.npz"), perts=np.array(perts),
                        q=np.stack([qs[p] for p in perts]), q_control=q_control)
    summ["_gen_seconds"] = time.time() - t0
    summ["_sum_q_range"] = [float(min(q.sum() for q in qs.values())), float(max(q.sum() for q in qs.values()))]
    return summ


def sample_rows(counts_fn, mask, n=20000, seed=0):
    """Seeded random subsample of the rows selected by `mask` (never the first n rows: file order is grouped)."""
    idx = np.where(mask)[0]
    if len(idx) > n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    return counts_fn(np.isin(np.arange(len(mask)), idx))
