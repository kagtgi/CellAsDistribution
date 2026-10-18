"""Is the unseen-gene 'headroom' mostly the perturbed gene's own induction (CRISPRa raises its target)?

For each held-out perturbation with an unseen gene g (single, or pair with one unseen gene) compute, in the SAME frame as the paper
(held-out NLL per UMI, paired, unit = perturbation):
  pm        perturbed mean (singles reference)            add   partial-additive (pairs reference)
  ti        target-induction baseline: reference with the unseen gene's frequency multiplied by ONE scalar s,
            s = exp(median self-log-fold-change of the target over TRAIN single perturbations whose target is in the panel)
  ot        'oracle target': reference with ONLY the unseen gene's frequency set to its held-out value (upper bound of what
            the target gene alone can give)
  cadf      the saved CAD-P q;     oracle   in-sample oracle (all genes)
Stratified by whether the unseen gene is in the 2048-gene panel.

python -m src.target_induction --name norman --run v2_cadf_s0 --split val
python -m src.target_induction --kfold 5 --run kf_cadf          # pooled over folds
"""
import argparse
import json
import os
import numpy as np

from .common import OUT_DIR, split_pert, load_json, save_json
from .eval_common import load_world
from .eval_nll import nll_per_umi
from .evaluate import bootstrap_ci
from . import baselines as B


def analyse(name, run, split="val", pert_col="perturbation"):
    d, X, obs, panel, lab, ctrl, spl = load_world(name, pert_col)
    pos = {g: i for i, g in enumerate(panel)}
    cnt = lambda m: np.asarray(X[np.where(m)[0]].todense())
    f_ctrl = B.profile(cnt(ctrl & (spl == "train")))
    tr = {p: np.where((lab == p) & (spl == "train"))[0] for p in set(lab[~ctrl & (spl == "train")])}
    tr = {p: i for p, i in tr.items() if len(i) >= 5}
    train_f = {p: B.profile(np.asarray(X[i].todense())) for p, i in tr.items()}
    train_singles = {p for p in train_f if len(split_pert(p)) == 1}
    selfl = [np.log(train_f[p][pos[p]] / f_ctrl[pos[p]]) for p in train_singles if p in pos]
    s = float(np.exp(np.median(selfl)))
    pm = B.perturbed_mean(train_f)
    add = B.additive(train_f, f_ctrl, pm)
    z = np.load(os.path.join(d, "runs", run, "q_by_pert.npz"))
    qm = dict(zip(z["perts"].tolist(), z["q"]))
    spl_groups = load_json(os.path.join(d, "splits.json"))["groups"]
    from .run_baselines import norm_label
    pu = {norm_label(p) for p in spl_groups.get("val_pu", []) + spl_groups.get("test_pu", [])}
    perts = sorted(p for p in qm if ((lab == p) & (spl == split)).sum() >= 20)

    def boost(q, i, scale):
        q = q.copy(); q[i] *= scale; return q / q.sum()

    def set_gene(q, i, val):
        q = q.copy(); rest = 1.0 - q[i]
        q = q * ((1.0 - val) / rest); q[i] = val; return q

    rows = {}
    for p in perts:
        genes = split_pert(p)
        unseen = [g for g in genes if g not in train_singles]
        if len(genes) == 1:
            kind, ref = "single", pm(p)
        elif p in pu and len(unseen) == 1:
            kind, ref = "pair1", add(p)
        else:
            continue
        g = unseen[0]
        real = cnt((lab == p) & (spl == split))
        in_panel = g in pos
        f_real = real.sum(0) / real.sum()
        q_ti = boost(ref, pos[g], s) if in_panel else ref
        q_ot = set_gene(ref, pos[g], max(f_real[pos[g]], 1e-9)) if in_panel else ref
        sm = (real.sum(0) + .5) / (real.sum() + .5 * len(panel))
        rows[p] = {"kind": kind, "in_panel": in_panel, "ref": nll_per_umi(real, ref), "pm": nll_per_umi(real, pm(p)),
                   "ti": nll_per_umi(real, q_ti), "ot": nll_per_umi(real, q_ot), "cadf": nll_per_umi(real, qm[p]),
                   "oracle": nll_per_umi(real, sm)}
    return {"scale": s, "n_train_selfLFC": len(selfl), "rows": rows}


def summarize(rows_list):
    out = {}
    allrows = {}
    for r in rows_list:
        for p, v in r["rows"].items():
            allrows.setdefault(p, v)                    # a perturbation keeps its first fold
    for kind in ("single", "pair1"):
        for ip in (True, False):
            ks = [p for p, v in allrows.items() if v["kind"] == kind and v["in_panel"] == ip]
            if not ks:
                continue
            head = np.mean([allrows[p]["ref"] - allrows[p]["oracle"] for p in ks])
            o = {"n": len(ks), "oracle_headroom": float(head)}
            for m in ("ti", "ot", "cadf"):
                d = [allrows[p][m] - allrows[p]["ref"] for p in ks]
                o[m] = {"diff_vs_ref": bootstrap_ci(d), "share_of_headroom": float(-np.mean(d) / head) if head > 0 else None}
            out[f"{kind}|target in panel={ip}"] = o
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="norman")
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--kfold", type=int, default=0)
    a = ap.parse_args()
    if a.kfold:
        rs = [analyse(f"{a.name}_f{k}", f"{a.run}_f{k}", "val") for k in range(a.kfold)]
        print("scales per fold:", [round(r["scale"], 2) for r in rs])
    else:
        rs = [analyse(a.name, a.run, a.split)]
        print("scale:", round(rs[0]["scale"], 2), "from", rs[0]["n_train_selfLFC"], "train singles")
    res = summarize(rs)
    for k, v in res.items():
        print(f"== {k}  n={v['n']}  in-sample oracle headroom {v['oracle_headroom']:.4f}")
        for m in ("ti", "ot", "cadf"):
            d = v[m]["diff_vs_ref"]
            print(f"   {m:5s} vs reference {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]  share of headroom {v[m]['share_of_headroom']:.2f}")
    save_json(res, os.path.join(OUT_DIR, "auto", f"target_induction_{a.name}_{a.run}_{a.split}{'_kfold' if a.kfold else ''}.json"))
