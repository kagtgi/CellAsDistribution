"""Negative-binomial log-likelihood evaluation (CPU) next to the multinomial NLL.

For every saved run (qbar_<split>.npz written by eval_units) the held-out cells of a unit are scored with
    x_ig ~ NB(mean = s_i * q_g, dispersion phi_g),   var = mu + phi_g mu^2,
where s_i is the cell's own library size and q the pooled model frequency of the unit, and phi_g is fit on TRAIN cells only
(method of moments on library-normalised counts, pooled within-unit variance). Reported per UMI so that it is comparable across
methods; the unit of analysis is the cell type / perturbation with paired bootstrap against a reference tag.

python -m src.nb_eval --task pert --name norman --split val --ref cb_perturbed_mean --tags cb_additive v2_x ...
Writes OUT_DIR/auto_cell/<name>_<split>_nb.json
"""
import argparse
import os
import numpy as np
from scipy.special import gammaln

from .cell_eval import World
from .common import OUT_DIR, save_json
from .evaluate import bootstrap_ci


def fit_dispersion(w, min_cells=30, cap=(1e-3, 1e2)):
    idx = np.where((w.split == "train") & ~w.ctrl)[0]
    if len(idx) > 40000:
        idx = np.sort(np.random.default_rng(1).choice(idx, 40000, replace=False))
    C = w.counts(idx)
    s = C.sum(1)
    Y = C * (s.mean() / s)[:, None]                      # library-normalised counts
    labs = w.lab[idx]
    ss, n, k = np.zeros(C.shape[1]), 0, 0
    mu_sum = np.zeros(C.shape[1])
    for u in np.unique(labs):
        m = labs == u
        if m.sum() < min_cells:
            continue
        ss += ((Y[m] - Y[m].mean(0)) ** 2).sum(0)
        mu_sum += Y[m].sum(0)
        n += m.sum()
        k += 1
    var, mu = ss / max(n - k, 1), mu_sum / max(n, 1)
    phi = np.where(mu > 0, (var - mu) / np.maximum(mu, 1e-9) ** 2, cap[0])
    return np.clip(phi, *cap)


def nb_nll_per_umi(X, s, q, phi):
    """X [n,G] counts, s [n] library sizes, q [G] frequency, phi [G]."""
    mu = np.clip(s[:, None] * np.clip(q, 1e-12, None)[None], 1e-12, None)
    r = (1.0 / phi)[None]
    ll = gammaln(X + r) - gammaln(r) - gammaln(X + 1) + r * np.log(r / (r + mu)) + X * np.log(mu / (r + mu))
    return float(-ll.sum() / X.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cond", "pert"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ref", required=True, help="reference tag for paired differences")
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    w = World(a.task, a.name)
    phi = fit_dispersion(w)
    print("dispersion phi: median %.3f, 90%% %.3f" % (np.median(phi), np.percentile(phi, 90)))
    out = {"phi_median": float(np.median(phi)), "tags": {}}
    for tag in [a.ref] + [t for t in a.tags if t != a.ref]:
        f = os.path.join(w.dir, "runs", tag, f"qbar_{a.split}.npz")
        if not os.path.exists(f):
            print("missing", f)
            continue
        z = np.load(f)
        per = {}
        for u, q in zip(z["units"].tolist(), z["q"]):
            ridx = np.where((w.lab == u) & (w.split == a.split) & ~w.ctrl)[0]
            R = w.counts(ridx)
            q = q.astype(np.float64)
            q /= q.sum()
            per[u] = {"nb": nb_nll_per_umi(R, R.sum(1), q, phi),
                      "mult": float(-(R.sum(0) * np.log(np.clip(q, 1e-12, None))).sum() / R.sum())}
        out["tags"][tag] = per
        print(f"{tag:45s} NB={np.mean([v['nb'] for v in per.values()]):.4f}  multinomial={np.mean([v['mult'] for v in per.values()]):.4f}  units={len(per)}", flush=True)
    ref = out["tags"].get(a.ref)
    diffs = {}
    if ref:
        for tag, per in out["tags"].items():
            common = [u for u in per if u in ref]
            diffs[tag] = {k: bootstrap_ci([per[u][k] - ref[u][k] for u in common]) for k in ("nb", "mult")}
            if tag != a.ref:
                d = diffs[tag]
                print(f"  {tag:43s} d(NB) {d['nb'][0]:+.4f} [{d['nb'][1]:+.4f},{d['nb'][2]:+.4f}]  d(mult) {d['mult'][0]:+.4f} [{d['mult'][1]:+.4f},{d['mult'][2]:+.4f}]")
    out["diff_vs_ref"] = diffs
    os.makedirs(os.path.join(OUT_DIR, "auto_cell"), exist_ok=True)
    save_json(out, os.path.join(OUT_DIR, "auto_cell", f"{a.name}_{a.split}_nb.json"))


if __name__ == "__main__":
    main()
