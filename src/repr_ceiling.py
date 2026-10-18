"""Representation ceiling (CPU): how much likelihood can survive if REAL held-out cells are first written in a representation
and then reconstructed? Extends rank_ceiling.py to both tasks and to more representations.

Per held-out unit (cell type for cond, perturbation for pert): every real cell is encoded, decoded to a profile, the profiles
are averaged and mixed with the train reference by a single epsilon fit on TRAIN cells, and the pooled counts are scored by the
multinomial NLL per UMI. Reference = type mean (cond) / perturbed mean (pert). Representations:
  frequency  the cell's own smoothed frequency (in-sample oracle, the distribution view)
  rank-200   C2S rank sentence of the top 200 genes, tutorial log-rank reconstruction
  rank-2048  rank sentence over the whole panel
  bins-51    scGPT-style per-cell value bins (51 bins of log1p CP10k of the non-zero genes), bin -> train mean value
  levels-48  our mass-level text (top 48 genes, 14 quantized log-mass levels), level -> bin centre
Share of headroom kept = (reference - representation) / (reference - frequency oracle).

python -m src.repr_ceiling --task cond --name pbmc --split val
python -m src.repr_ceiling --task pert --name norman --split val
"""
import argparse
import os
import numpy as np

from .cell_eval import World
from .cell_dist import nll_per_umi
from .common import OUT_DIR, save_json
from .evaluate import bootstrap_ci
from .c2s_baseline import sentence_from_counts, fit_rank_expression, profile_from_sentence
from . import baselines as B

NB = 51


def bin_cell(c, nb=NB):
    """scGPT-style: non-zero values of log1p(CP10k) are split into nb-1 quantile bins (1..nb-1); zeros get bin 0."""
    c = np.asarray(c, dtype=np.float64)
    v = np.log1p(c / max(c.sum(), 1) * 1e4)
    b = np.zeros(len(c), dtype=int)
    nz = np.where(c > 0)[0]
    if len(nz):
        edges = np.quantile(v[nz], np.linspace(0, 1, nb))[1:-1]
        b[nz] = 1 + np.searchsorted(edges, v[nz], side="right")
    return b


def fit_bin_values(cells):
    """mean CP10k value per bin index, from train cells."""
    tot, cnt = np.zeros(NB + 1), np.zeros(NB + 1)
    for c in cells:
        b = bin_cell(c)
        v = c / max(c.sum(), 1) * 1e4
        np.add.at(tot, b, v)
        np.add.at(cnt, b, 1)
    return tot / np.maximum(cnt, 1)


def levels_profile(c, k=48, levels=14):
    c = np.asarray(c, dtype=np.float64)
    s = c.sum()
    p = np.zeros(len(c))
    top = np.argsort(-c)[:k]
    top = top[c[top] > 0]
    lv = np.minimum(levels - 1, np.floor(np.log2(1.0 + 1e4 * c[top] / s)))
    p[top] = 1.5 * 2.0 ** lv - 1.0
    return p


def encode_decode(rep, C, ctx):
    G = C.shape[1]
    out = np.zeros(G)
    for c in C:
        if rep in ("frequency", "frequency-matched"):
            out += c / max(c.sum(), 1)
            continue
        if rep.startswith("rank"):
            k = int(rep.split("-")[1])
            a, b = ctx["rank"][k]
            p = profile_from_sentence(sentence_from_counts(c, ctx["panel"], k), ctx["pos"], a, b, G)
        elif rep == "bins-51":
            p = ctx["binval"][bin_cell(c)]
        else:
            p = levels_profile(c)
        out += p / max(p.sum(), 1e-12)
    return out / len(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cond", "pert"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-cells", type=int, default=300)
    ap.add_argument("--min-cells", type=int, default=30)
    a = ap.parse_args()
    w = World(a.task, a.name)
    G, panel = len(w.panel), w.panel
    rng = np.random.default_rng(0)
    tr = np.where((w.split == "train") & ~w.ctrl)[0]
    train_units = [u for u in np.unique(w.lab[tr]) if (w.lab[tr] == u).sum() >= 30]
    tf = {u: B.profile(w.counts(tr[w.lab[tr] == u])) for u in train_units}
    glob = B.renorm(np.mean(list(tf.values()), 0))
    ref_q = (lambda u: tf.get(u, glob)) if a.task == "cond" else (lambda u: glob)
    ctx = {"panel": panel, "pos": {g: i for i, g in enumerate(panel)}, "rank": {}}
    for k in (200, 2048):
        ctx["rank"][k] = fit_rank_expression(w.X, panel, tr, k=k)
    sub = rng.choice(tr, 3000, replace=False)
    ctx["binval"] = fit_bin_values(w.counts(sub))
    reps = ["frequency", "frequency-matched", "rank-200", "rank-2048", "bins-51", "levels-48"]
    # frequency-matched: the same estimator as every other representation (equal-weight mean of per-cell profiles over the
    # same <= max-cells subsample, mixed with the reference by an epsilon fit on train units), so ratios compare like with like.
    units = w.units(a.split, a.min_cells)
    real = {u: w.counts(np.where((w.lab == u) & (w.split == a.split) & ~w.ctrl)[0]) for u in units}
    ref = {u: nll_per_umi(real[u], ref_q(u)) for u in units}
    # epsilon fit on TRAIN units only
    fit_units = [train_units[i] for i in rng.choice(len(train_units), min(30, len(train_units)), replace=False)]
    fit_cells = {u: w.counts(rng.choice(tr[w.lab[tr] == u], min(a.max_cells, (w.lab[tr] == u).sum()), replace=False)) for u in fit_units}
    grid = [0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.8]
    res, eps_by = {}, {}
    for rep in reps:
        if rep == "frequency":
            eps_by[rep] = 0.0
        else:
            prof = {u: encode_decode(rep, fit_cells[u], ctx) for u in fit_units}
            fit = [np.mean([nll_per_umi(fit_cells[u], (1 - e) * prof[u] + e * ref_q(u)) for u in fit_units]) for e in grid]
            eps_by[rep] = grid[int(np.argmin(fit))]
        e = eps_by[rep]
        res[rep] = {}
        for u in units:
            C = real[u][rng.choice(len(real[u]), min(a.max_cells, len(real[u])), replace=False)]
            p = encode_decode(rep, C, ctx)
            if rep == "frequency":   # in-sample oracle on ALL cells of the unit, smoothed like the mass oracle
                q = (real[u].sum(0) + .5) / (real[u].sum() + .5 * G)
            else:
                q = (1 - e) * p + e * ref_q(u)
            res[rep][u] = nll_per_umi(real[u], q / q.sum())
        print(f"{rep:10s} eps={e:<5} NLL={np.mean(list(res[rep].values())):.4f}  reference={np.mean(list(ref.values())):.4f}", flush=True)
    head = np.mean([ref[u] - res["frequency"][u] for u in units])
    head_m = np.mean([ref[u] - res["frequency-matched"][u] for u in units])
    out = {"task": a.task, "split": a.split, "n_units": len(units), "reference_nll": float(np.mean(list(ref.values()))), "eps": eps_by, "reps": {}}
    for rep in reps:
        d = bootstrap_ci([res[rep][u] - ref[u] for u in units])
        keep = float(np.mean([ref[u] - res[rep][u] for u in units]) / head)
        keep_m = float(np.mean([ref[u] - res[rep][u] for u in units]) / head_m)
        out["reps"][rep] = {"nll": float(np.mean(list(res[rep].values()))), "diff_vs_reference": d, "headroom_kept": keep,
                            "headroom_kept_matched": keep_m, "per_unit": res[rep]}
        print(f"  {rep:18s} diff vs reference {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]  keeps {100 * keep:.0f}% of the pooled-oracle "
              f"headroom, {100 * keep_m:.0f}% of the matched-frequency headroom")
    os.makedirs(os.path.join(OUT_DIR, "auto_cell"), exist_ok=True)
    save_json(out, os.path.join(OUT_DIR, "auto_cell", f"{a.name}_{a.split}_repr_ceiling_v2.json"))


if __name__ == "__main__":
    main()
