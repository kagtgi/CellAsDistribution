"""Single source of truth for paper numbers: held-out NLL tables (paired CI vs perturbed mean) -> JSON + LaTeX.

python -m src.results --name norman --split val --out paper/auto \
    --runs "CAD-P=g1c_cadf_08b" "CAD-P shuffled=g1c_cadf_08b_shuffled" "CAD-N tree=g1c_cadn_08b" "CAD-N tree+tilt=g2_cadn_tilt_08b"
Writes <out>/<name>_<split>.json (per-pert NLL, headline stats) and <out>/<name>_<split>_table.tex (booktabs).
"""
import argparse
import json
import os
import numpy as np

from .common import split_pert, load_json
from .eval_common import load_world
from .eval_nll import nll_per_umi
from .evaluate import bootstrap_ci
from .gene_tree import gene_embedding
from . import baselines as B


def build(name, runs, split="val", pert_col="perturbation"):
    d, X, obs, panel, lab, ctrl, spl = load_world(name, pert_col)
    cnt = lambda m: np.asarray(X[np.where(m)[0]].todense())
    tcells = {p: np.where((lab == p) & (spl == "train"))[0] for p in set(lab[~ctrl & (spl == "train")])}
    tcells = {p: i for p, i in tcells.items() if len(i) >= 5}
    train_f = {p: B.profile(np.asarray(X[i].todense())) for p, i in tcells.items()}
    f_ctrl = B.profile(cnt(ctrl & (spl == "train")))
    pm = B.perturbed_mean(train_f)
    models = {"control mean": B.control_mean(f_ctrl), "perturbed mean": pm, "additive": B.additive(train_f, f_ctrl, pm),
              "ridge": B.ridge(train_f, gene_embedding(X[np.where(spl == "train")[0]]), panel, f_ctrl)}
    perts = sorted(p for p in set(lab[~ctrl & (spl == split)]) if ((lab == p) & (spl == split)).sum() >= 20)
    # restrict to perturbations every run covers (e.g. C2S subset)
    qmaps = {}
    for label, tag in runs.items():
        z = np.load(os.path.join(d, "runs", tag, "q_by_pert.npz"))
        qmaps[label] = (dict(zip(z["perts"].tolist(), z["q"])), z["q_control"])
        perts = [p for p in perts if p in qmaps[label][0]]
    real = {p: cnt((lab == p) & (spl == split)) for p in perts}
    res = {n: {p: nll_per_umi(real[p], fn(p)) for p in perts} for n, fn in models.items()}
    for label, (qm, _) in qmaps.items():
        res[label] = {p: nll_per_umi(real[p], qm[p]) for p in perts}
    res["oracle (in-sample)"] = {p: nll_per_umi(real[p], (real[p].sum(0) + .5) / (real[p].sum() + .5 * len(panel))) for p in perts}
    # split-half oracle: smoothed frequency of one random half of the held-out cells scored on the other half (and back);
    # a realistic, noisier reference than the in-sample oracle (which fits and scores on the same cells)
    rng = np.random.default_rng(0)
    sh = {}
    for p in perts:
        perm = rng.permutation(len(real[p]))
        A, Bh = real[p][perm[: len(perm) // 2]], real[p][perm[len(perm) // 2:]]
        fa, fb = (A.sum(0) + .5) / (A.sum() + .5 * len(panel)), (Bh.sum(0) + .5) / (Bh.sum() + .5 * len(panel))
        sh[p] = 0.5 * (nll_per_umi(Bh, fa) + nll_per_umi(A, fb))
    res["oracle (split-half)"] = sh
    ref = res["perturbed mean"]
    from .run_baselines import norm_label
    pu = {norm_label(p) for p in load_json(os.path.join(d, "splits.json"))["groups"].get("val_pu", []) + load_json(os.path.join(d, "splits.json"))["groups"].get("test_pu", [])}
    groups = {"all": perts, "unseen singles": [p for p in perts if len(split_pert(p)) == 1],
              "pairs (seen genes)": [p for p in perts if len(split_pert(p)) == 2 and p not in pu],
              "pairs (one unseen gene)": [p for p in perts if p in pu]}
    stats = {}
    for g, ps in groups.items():
        if not ps:
            continue
        stats[g] = {}
        for m, v in res.items():
            diff = bootstrap_ci([v[p] - ref[p] for p in ps])
            head = np.mean([ref[p] - res["oracle (in-sample)"][p] for p in ps]); head_sh = np.mean([ref[p] - res["oracle (split-half)"][p] for p in ps])
            stats[g][m] = {"nll": float(np.mean([v[p] for p in ps])), "diff": diff, "win": float(np.mean([v[p] < ref[p] for p in ps])),
                           "headroom_frac": float(np.mean([ref[p] - v[p] for p in ps]) / head) if head > 0 else None,
                           "headroom_frac_sh": float(np.mean([ref[p] - v[p] for p in ps]) / head_sh) if head_sh > 0 else None}
    ctrl_real = cnt(ctrl & (spl == split))
    stats["control_nll"] = {"f_ctrl": nll_per_umi(ctrl_real, f_ctrl), **{l: nll_per_umi(ctrl_real, q[1]) for l, q in qmaps.items()}}
    return {"name": name, "split": split, "n_perts": {g: len(p) for g, p in groups.items()}, "stats": stats, "per_pert": res}


def latex_table(out, caption, label, groups=("all", "unseen singles", "pairs (seen genes)", "pairs (one unseen gene)"), skip=("oracle (in-sample)", "control mean")):
    ms = [m for m in out["stats"]["all"] if m not in skip]
    gs = [g for g in groups if g in out["stats"]]
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "rr" * len(gs) + "}", r"\toprule",
             "Model & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{g} ($n={out['n_perts'][g]}$)}}" for g in gs) + r" \\",
             " & " + " & ".join("NLL & $\\Delta$ [95\\% CI]" for _ in gs) + r" \\", r"\midrule"]
    best = {g: min((m for m in ms if not m.startswith("oracle")), key=lambda m: out["stats"][g][m]["nll"]) for g in gs}   # references excluded
    for m in ms:
        cells = []
        for g in gs:
            s = out["stats"][g][m]
            d = s["diff"]
            nll, dif = f"{s['nll']:.3f}", "---" if m == "perturbed mean" else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
            if best[g] == m:
                nll, dif = r"\textbf{" + nll + "}", (r"\textbf{" + dif + "}" if dif != "---" else dif)
            cells += [nll, dif]
        lines.append((r"\rowcolor{ours}" if m.startswith("CAD") else "") + m.replace("&", r"\&") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", default=[], help='"Label=tag" pairs')
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs)
    out = build(a.name, runs, a.split)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"{a.name}_{a.split}")
    json.dump(out, open(base + ".json", "w"), indent=1)
    open(base + "_table.tex", "w").write(latex_table(
        out, f"Held-out {a.name.capitalize()} perturbations ({a.split}). NLL: multinomial negative log-likelihood per UMI (lower is better); "
             r"$\Delta$: paired difference to the perturbed-mean baseline, bootstrap over perturbations.", f"tab:{a.name}-{a.split}"))
    print("wrote", base + ".json", "and _table.tex;", out["n_perts"])


if __name__ == "__main__":
    main()
