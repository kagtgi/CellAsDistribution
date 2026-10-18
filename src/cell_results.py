"""Tables for the per-cell use cases: paired statistics over units + LaTeX with the best value bold and our rows highlighted.

python -m src.cell_results --task cond --name pbmc --split val --out paper/auto --ref "Cell-type mean=cb_type_mean" \
    --runs "Global mean=cb_global_mean" "Cell-type mean=cb_type_mean" "CAD (ours)=c_cond_freq" --ours "CAD"
Reads OUT_DIR/<name>/runs/<tag>/cell_<split>.json. Unit of analysis = cell type (cond) or perturbation (pert).
"""
import argparse
import json
import os
import numpy as np

from .common import OUT_DIR, load_json
from .evaluate import bootstrap_ci

SPECS = {   # key: (label, higher_is_better, decimals)
    "nll_pooled": ("NLL", False, 3), "nll_mix": ("cell NLL", False, 3),
    "mmd_dm": ("MMD", False, 4), "auc_dm": ("AUC", False, 3), "energy_dm": ("Energy", False, 3),
    "mean_corr_dm": ("$r_\\mu$", True, 3), "var_corr_dm": ("$r_{\\sigma^2}$", True, 3),
    "type_acc_gen_dm": ("Type acc.", True, 3),
    "mmd_mult": ("MMD", False, 4), "auc_mult": ("AUC", False, 3), "energy_mult": ("Energy", False, 3),
    "mean_corr_mult": ("$r_\\mu$", True, 3), "var_corr_mult": ("$r_{\\sigma^2}$", True, 3),
    "type_acc_gen_mult": ("Type acc.", True, 3),
}


def load_runs(name, split, runs):
    out = {}
    for label, tag in runs.items():
        out[label] = load_json(os.path.join(OUT_DIR, name, "runs", tag, f"cell_{split}.json"))["per_unit"]
    return out


def summarize(per, ref_label, metrics):
    units = sorted(set.intersection(*[set(v) for v in per.values()]))
    stats = {}
    for lab, d in per.items():
        stats[lab] = {}
        for m in metrics:
            vals = [d[u][m] for u in units if m in d[u]]
            if not vals:
                continue
            s = {"mean": float(np.mean(vals))}
            if ref_label in per and lab != ref_label and all(m in per[ref_label][u] for u in units):
                s["diff"] = bootstrap_ci([d[u][m] - per[ref_label][u][m] for u in units])
            stats[lab][m] = s
    return units, stats


def fmt_table(stats, metrics, ours, ref_label, caption, label, extra_cols=None):
    labs = list(stats)
    best = {}
    for m in metrics:
        vals = {l: stats[l][m]["mean"] for l in labs if m in stats[l]}
        if vals:
            best[m] = (max if SPECS[m][1] else min)(vals, key=vals.get)
    cols = "l" + "r" * len(metrics)
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{" + cols + "}", r"\toprule",
             "Model & " + " & ".join(SPECS[m][0] + (r" $\uparrow$" if SPECS[m][1] else r" $\downarrow$") for m in metrics) + r" \\", r"\midrule"]
    for l in labs:
        row = []
        for m in metrics:
            if m not in stats[l]:
                row.append("---"); continue
            v = f"{stats[l][m]['mean']:.{SPECS[m][2]}f}"
            if m == "nll_pooled" and "diff" in stats[l][m]:
                d = stats[l][m]["diff"]
                v += f" ({d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}])"
            if best.get(m) == l:
                v = r"\textbf{" + v + "}"
            row.append(v)
        pre = r"\rowcolor{ours}" if l.startswith(ours) else ""
        lines.append(pre + l.replace("&", r"\&") + " & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True, help='"Label=tag" of the reference for paired differences')
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ours", default="CAD")
    ap.add_argument("--sampler", default="dm", choices=["dm", "mult"])
    ap.add_argument("--stem", default="")
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs)
    ref_label = a.ref.split("=", 1)[0]
    per = load_runs(a.name, a.split, runs)
    s = a.sampler
    metrics = ["nll_pooled", "nll_mix", f"mmd_{s}", f"auc_{s}", f"mean_corr_{s}"] + ([f"type_acc_gen_{s}"] if a.task == "cond" else [])
    units, stats = summarize(per, ref_label, metrics)
    stem = a.stem or f"{a.name}_{a.split}_cell_{s}"
    os.makedirs(a.out, exist_ok=True)
    unit = "cell types" if a.task == "cond" else "perturbations"
    cap = (f"{ {'pbmc': 'Human PBMC (Hao et al.)', 'norman': 'Norman K562'}.get(a.name, a.name)}, {a.split} split: {len(units)} held-out {unit}. NLL: multinomial negative log-likelihood per UMI of the pooled real cells under the model "
           f"(Delta: paired difference to {ref_label} [95\\% bootstrap over {unit}]). "
           + ("cell NLL: per-cell mixture over $M$ random control cells. " if a.task == "pert" else "")
           + "MMD, AUC (real vs generated classifier; 0.5 is ideal), $r_\\mu$ (gene-mean correlation)" + (", type acc.: a classifier trained on real cells recovers the requested type" if a.task == "cond" else "")
           + f". The generation metrics use cells from the shared {'Dirichlet-multinomial' if s == 'dm' else 'multinomial'} sampler. Best in bold; our rows shaded.")
    open(os.path.join(a.out, stem + "_table.tex"), "w").write(fmt_table(stats, metrics, a.ours, ref_label, cap, f"tab:{stem}"))
    json.dump({"units": units, "stats": stats, "per_unit": per}, open(os.path.join(a.out, stem + ".json"), "w"), indent=1)
    for l in stats:
        print(l, {m: round(v["mean"], 4) for m, v in stats[l].items()})


if __name__ == "__main__":
    main()
