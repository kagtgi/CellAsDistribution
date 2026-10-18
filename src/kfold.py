"""Gene-level K-fold over perturbations (train+val perturbations only; the frozen TEST split is never touched).

Why: Norman validation has only 10 unseen single genes. Holding out each single gene once (5 folds) evaluates ~84 genes.
Rules per fold (same leak rule as preprocess.make_splits):
  * held-out single genes V appear in NO training perturbation (no train single, no train pair);
  * pairs with exactly one gene in V (partner a train single) -> val 'pairs (one unseen gene)';
  * pairs of two train singles: 20% -> val 'pairs (seen genes)', rest train;
  * pairs with two unseen genes, and everything in the frozen test split -> 'excluded'.
The 2048-gene panel and library sizes are kept from the base split (disclosed; they were fit on base-train cells).

python -m src.kfold make --name norman --folds 5          # writes OUT_DIR/norman_f0..f4 (+ trees)
python -m src.kfold summarize --name norman --folds 5 --runs real=kf_cadf rid=kf_cadf_rid   # pooled paired stats
"""
import argparse
import os
import shutil
import numpy as np
import pandas as pd

from .common import OUT_DIR, SEED, load_json, save_json, split_pert
from .preprocess import is_control


def make_fold(name, k, n_folds=5, seed=SEED, frac_val_pair=0.2, pert_col="perturbation"):
    d = os.path.join(OUT_DIR, name)
    obs = pd.read_csv(os.path.join(d, "obs.csv"), index_col=0)
    perts = obs[pert_col].astype(str)
    ctrl = perts.map(is_control).values
    pool = sorted(p for p, s in obs.assign(_p=perts).groupby("_p")["split"].agg(lambda x: x.mode().iloc[0]).items()
                  if s in ("train", "val") and not is_control(p))
    singles = sorted(p for p in pool if len(split_pert(p)) == 1)
    pairs = sorted(p for p in pool if len(split_pert(p)) == 2)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(singles))
    fold_of = {singles[i]: j % n_folds for j, i in enumerate(order)}      # identical for every k
    val_s = [p for p in singles if fold_of[p] == k]
    train_s = [p for p in singles if fold_of[p] != k]
    V = {split_pert(p)[0] for p in val_s}
    train_genes = {split_pert(p)[0] for p in train_s}
    seen_seen = [p for p in pairs if all(g in train_genes for g in split_pert(p))]
    one_unseen = [p for p in pairs if sum(g in V for g in split_pert(p)) == 1 and all(g in train_genes or g in V for g in split_pert(p))]
    rng2 = np.random.default_rng(seed + 1 + k)
    rng2.shuffle(seen_seen)
    n_vp = int(round(frac_val_pair * len(seen_seen)))
    val_p, train_p = seen_seen[:n_vp], seen_seen[n_vp:]
    tag = {p: "train" for p in train_s + train_p}
    tag.update({p: "val" for p in val_s + val_p + one_unseen})
    split = np.array(["excluded"] * len(obs), dtype=object)
    for i, (p, c, s0) in enumerate(zip(perts.values, ctrl, obs["split"].values)):
        split[i] = (s0 if s0 in ("train", "val") else "excluded") if c else tag.get(p, "excluded")
    out = os.path.join(OUT_DIR, f"{name}_f{k}")
    os.makedirs(out, exist_ok=True)
    o2 = obs.copy()
    o2["split"] = split
    o2.to_csv(os.path.join(out, "obs.csv"))
    for f in ("counts.npz", "panel.json"):
        shutil.copy(os.path.join(d, f), os.path.join(out, f))
    save_json({"pert_split": {p: t for p, t in tag.items()}, "groups": {
        "train_s": train_s, "val_s": val_s, "train_p": train_p, "val_p": val_p, "val_pu": one_unseen, "test_s": [], "test_p": [], "test_pu": []},
        "cells": {t: int((split == t).sum()) for t in ("train", "val")}}, os.path.join(out, "splits.json"))
    assert not ({split_pert(p)[0] for p in val_s} & {g for p in train_s + train_p for g in split_pert(p)}), "fold leak"
    print(f"fold {k}: val singles {len(val_s)}, val pairs seen {len(val_p)}, val pairs one-unseen {len(one_unseen)}, "
          f"train singles {len(train_s)}, train pairs {len(train_p)}; cells {int((split == 'train').sum())}/{int((split == 'val').sum())}")


def summarize(name, n_folds, runs, pert_col="perturbation"):
    """Pool per-perturbation NLL over folds (unit = perturbation; a pair seen in several folds keeps its first fold)."""
    from .results import build
    from .evaluate import bootstrap_ci
    rows = {}
    for k in range(n_folds):
        r = build(f"{name}_f{k}", {lab: f"{tag}_f{k}" for lab, tag in runs.items()}, "val", pert_col)
        spl = load_json(os.path.join(OUT_DIR, f"{name}_f{k}", "splits.json"))["groups"]
        from .run_baselines import norm_label
        pu = {norm_label(p) for p in spl["val_pu"]}
        for m, v in r["per_pert"].items():
            for p, x in v.items():
                grp = "unseen singles" if len(split_pert(p)) == 1 else ("pairs (one unseen gene)" if p in pu else "pairs (seen genes)")
                rows.setdefault((p, grp), {}).setdefault(m, x)           # keep the first fold's value
    groups = {"all": list(rows), "unseen singles": [x for x in rows if x[1] == "unseen singles"],
              "pairs (seen genes)": [x for x in rows if x[1] == "pairs (seen genes)"],
              "pairs (one unseen gene)": [x for x in rows if x[1] == "pairs (one unseen gene)"]}
    models = [m for m in next(iter(rows.values())) if m != "perturbed mean"]
    out = {}
    for g, keys in groups.items():
        if not keys:
            continue
        out[g] = {"n": len(keys)}
        for m in models:
            out[g][m] = bootstrap_ci([rows[x][m] - rows[x]["perturbed mean"] for x in keys])
        # Pre-specified comparison list (fixed before the 5-fold runs): PRIMARY real-rid on unseen singles;
        # secondary: real-shuf (added after validation showed opaque ids encode identity worse: tokenisation confound),
        # desc-descother, real-additive/ridge, desc-real.
        for a, b in (("real", "rid"), ("real", "shuf"), ("desc", "descother"), ("desc", "real"),
                     ("real", "additive"), ("real", "ridge"), ("rid", "shuf")):
            if a in models and b in models:
                out[g][f"{a} - {b}"] = bootstrap_ci([rows[x][a] - rows[x][b] for x in keys])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "summarize"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--runs", nargs="*", default=[], help="label=tag_prefix (tag_f{k} is read)")
    a = ap.parse_args()
    if a.cmd == "make":
        for k in range(a.folds):
            make_fold(a.name, k, a.folds)
    else:
        import json
        res = summarize(a.name, a.folds, dict(r.split("=", 1) for r in a.runs))
        print(json.dumps(res, indent=1))
        save_json(res, os.path.join(OUT_DIR, "auto", f"{a.name}_kfold.json"))
        open(os.path.join(OUT_DIR, "auto", f"{a.name}_kfold_table.tex"), "w").write(__import__("src.kfold", fromlist=["kfold_table_tex"]).kfold_table_tex(res))   # defined below the main block


def kfold_table_tex(res, label="tab:kfold", caption=None):
    """LaTeX table of the pooled 5-fold paired differences (nats/UMI, negative = better than the perturbed mean)."""
    gs = [g for g in ("unseen singles", "pairs (seen genes)", "pairs (one unseen gene)") if g in res]
    names = {"additive": "additive (partial)", "ridge": "ridge", "real": "CAD-P, real gene names", "rid": "CAD-P, opaque ids",
             "shuf": "CAD-P, shuffled names", "desc": "CAD-P + own description", "descother": "CAD-P + other gene's description",
             "oracle (split-half)": "split-half oracle (reference)"}
    cmp_ = [("real - rid", "real $-$ opaque ids (primary)"), ("real - shuf", "real $-$ shuffled names"),
            ("desc - descother", "description $-$ other description"), ("desc - real", "description $-$ real names"),
            ("real - additive", "CAD-P $-$ additive"), ("real - ridge", "CAD-P $-$ ridge")]
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "r" * len(gs) + "}", r"\toprule",
         "Paired NLL/UMI difference [95\% CI] & " + " & ".join(f"{g} ($n={res[g]['n']}$)" for g in gs) + r" \\", r"\midrule",
         r"\multicolumn{" + str(len(gs) + 1) + r"}{l}{\emph{to the perturbed mean}} \\"]
    best = {g: min((k for k in names if k in res[g] and not k.startswith("oracle")), key=lambda k: res[g][k][0]) for g in gs}   # lowest mean, oracle excluded
    for k, nm in names.items():
        if k in res[gs[0]]:
            cell = lambda g, k=k: (r"\textbf{%s}" if best[g] == k else "%s") % f"{res[g][k][0]:+.3f} [{res[g][k][1]:+.3f}, {res[g][k][2]:+.3f}]"
            L.append((r"\rowcolor{ours}" if nm.startswith("CAD-P") else "") + nm + " & " + " & ".join(cell(g) if k in res[g] else "---" for g in gs) + r" \\")
    L += [r"\midrule", r"\multicolumn{" + str(len(gs) + 1) + r"}{l}{\emph{between decoders}} \\"]
    for k, nm in cmp_:
        if k in res[gs[0]]:
            L.append(nm + " & " + " & ".join(f"{res[g][k][0]:+.3f} [{res[g][k][1]:+.3f}, {res[g][k][2]:+.3f}]" if k in res[g] else "---" for g in gs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{" + (caption or "Gene-level 5-fold analysis on Norman: pooled paired NLL differences (nats per UMI; negative favours the first term). Unit: perturbation.") + "}",
          rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(L)
