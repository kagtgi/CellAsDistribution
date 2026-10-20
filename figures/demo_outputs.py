"""Appendix demo: the exact prompt given to CAD and the distribution it outputs, against the held-out ground truth.

python figures/demo_outputs.py --out paper/figs/demo_outputs --split test \
   --case "cond|PBMC, conditional generation|pbmc|t2_g_cond_s0|cb_type_mean|all" \
   --case "pert|K562, no control cell|norman|t2_g_noctrl_s0|cb_additive|pu" ...

No cherry-picking: in each case the demo unit is the held-out unit whose paired NLL difference to the reference
(CAD minus reference, from cell_<split>.json) is the median over the units of the case's stratum (all, s, pu or p).
Ground truth = pooled frequency of all held-out real cells of the unit. Panel (a): the decoded distribution on a FIXED gene
axis (the --top genes with the highest mean frequency in training cells, in panel order = the order of the decision head's
outputs), shared by every unit of a dataset, so only the distribution changes between units; log frequency of ground truth,
CAD and the reference. For a per-cell (control-residual) model, the distributions decoded for three individual control
cells are drawn as hollow markers next to their mean. Panel (b), perturbations only: the top-20
DE genes of the unit (Welch t, real vs held-out control cells), log2 fold change over the held-out control mean of
ground truth, CAD and the reference (pseudobulk, pseudo-count 1e-5). For a control-cell model the plotted CAD
distribution is the mean over the M matched control cells and the prompt shown is the one for the first control cell.
Writes <out>.pdf/.png and <out>.json (chosen units, prompts, numbers quoted in the caption).
"""
import argparse
import json
import os
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import ttest_ind

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cell_eval import World  # noqa: E402
from src.common import load_json, split_pert  # noqa: E402
from src.evaluate import log_cp10k  # noqa: E402
from src.run_baselines import norm_label  # noqa: E402

_F = os.path.join(os.path.dirname(__file__), "..", "fonts", "GoogleSans-Regular.ttf")
if os.path.exists(_F):
    font_manager.fontManager.addfont(_F)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_F).get_name()
C_GT, C_CAD, C_REF = "#3A3A3A", "#23918F", "#D9713F"


def qbar_of(w, tag, split):
    z = np.load(os.path.join(w.dir, "runs", tag, f"qbar_{split}.npz"), allow_pickle=True)
    return {str(u): q.astype(np.float64) for u, q in zip(z["units"], z["q"])}


def qrows_of(w, tag, split):
    """Per-control-cell decoded distributions [M, G] saved by train_cell (empty dict if the run did not save them)."""
    f = os.path.join(w.dir, "runs", tag, f"qrows_{split}.npz")
    if not os.path.exists(f):
        return {}
    z = np.load(f, allow_pickle=True)
    return {str(u): z[f"{u}__q"].astype(np.float64) for u in z["units"]}


def fixed_axis(w, k):
    """The k panel genes with the highest mean frequency in TRAINING cells (no held-out data), returned in panel order,
    i.e. the order of the decision head's outputs. One axis per dataset, shared by every unit and case."""
    tr = np.where((w.split == "train") & ~w.ctrl)[0]
    tr = np.sort(np.random.default_rng(0).choice(tr, min(len(tr), 20000), replace=False))
    C = w.counts(tr)
    f = (C / np.clip(C.sum(1, keepdims=True), 1, None)).mean(0)
    return np.sort(np.argsort(-f)[:k])


def pick_median(w, tag, ref, split, stratum):
    a = load_json(os.path.join(w.dir, "runs", tag, f"cell_{split}.json"))["per_unit"]
    b = load_json(os.path.join(w.dir, "runs", ref, f"cell_{split}.json"))["per_unit"]
    units = sorted(set(a) & set(b))
    if stratum != "all":
        g = {norm_label(p) for p in load_json(os.path.join(w.dir, "splits.json"))["groups"][f"{split}_{stratum}"]}
        units = [u for u in units if u in g]
    d = {u: a[u]["nll_pooled"] - b[u]["nll_pooled"] for u in units}
    order = sorted(units, key=d.get)
    u = order[(len(order) - 1) // 2]
    return u, d[u], len(units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--case", action="append", required=True, help='"task|title|name|cad_tag|ref_tag|stratum"')
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--compact", action="store_true", help="main-text version: no prompt column, larger type (>= 7.5 pt at 6.5 in)")
    a = ap.parse_args()
    cases = [c.split("|") for c in a.case]
    if a.compact:   # (prompt, frequencies, DE) font sizes in pt at the printed width of 6.5 in
        FS = dict(tick=7.5, lab=8.0, title=8.5, leg=7.5, ms=4.0)
        fig, axs3 = plt.subplots(len(cases), 2, figsize=(6.5, 2.75 * len(cases)), gridspec_kw={"width_ratios": [1.45, 1.0]})
        axs3 = np.atleast_2d(axs3)
        axs = np.concatenate([np.array([[None]] * len(cases)), axs3], axis=1)
    else:
        FS = dict(tick=4.2, lab=5.5, title=5.5, leg=4.8, ms=2.6)
        fig, axs = plt.subplots(len(cases), 3, figsize=(7.2, 2.25 * len(cases)), gridspec_kw={"width_ratios": [1.15, 1.6, 1.0]})
        axs = np.atleast_2d(axs)
    worlds, record, axes, plotdata = {}, [], {}, {}
    for r, (task, title, name, tag, ref, stratum) in enumerate(cases):
        w = worlds.setdefault((task, name), World(task, name))
        u, dnll, n_units = pick_median(w, tag, ref, a.split, stratum)
        real = w.counts(np.where((w.lab == u) & (w.split == a.split) & ~w.ctrl)[0])
        gt = real.sum(0) / real.sum()
        q_cad, q_ref = qbar_of(w, tag, a.split)[u], qbar_of(w, ref, a.split)[u]
        prompts = os.path.join(w.dir, "runs", tag, f"prompts_{a.split}.json")
        ptxt = load_json(prompts)[u][0] if os.path.exists(prompts) else "(prompt file missing)"
        # (left) prompt
        if not a.compact:
            ax = axs[r, 0]
            ax.axis("off")
            wrapped = "\n".join(textwrap.fill(line, 46) for line in ptxt.split("\n"))
            lines = wrapped.split("\n")
            if len(lines) > 16:
                lines = lines[:15] + ["..."]
            ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=4.6, transform=ax.transAxes)
            ax.set_title(f"{title}\nunit: {u}", fontsize=6.5, loc="left")
        # (middle) the decoded distribution on a FIXED gene axis: the same genes, in the panel's own order (the order of the
        # decision head's outputs), for every unit of a dataset; only the distribution changes from unit to unit
        ax = axs[r, 1]
        if (task, name) not in axes:
            axes[(task, name)] = fixed_axis(w, a.top)
        top = axes[(task, name)]
        x = np.arange(len(top))
        refl = {"cb_additive": "additive", "cb_type_mean": "type mean", "cb_perturbed_mean": "perturbed mean"}.get(ref, ref.replace("cb_", "").replace("_", " "))
        ax.bar(x, gt[top], color=C_GT, alpha=.25, width=.8, label="held-out real (pooled)")
        qr = qrows_of(w, tag, a.split).get(u)
        if qr is not None and qr.shape[0] > 1 and np.abs(qr - qr[:1]).max() > 1e-6:   # a per-cell model: one q per control cell
            for j in range(min(3, qr.shape[0])):
                ax.plot(x + (j - 1) * 0.22, qr[j][top], "o", ms=FS["ms"] * .6, mfc="none", mec=C_CAD, mew=.6,
                        label="single control cells" if j == 0 else None)
        ax.plot(x, q_cad[top], "o", ms=FS["ms"], color=C_CAD, label="CellAsDistribution" if a.compact else "CAD")
        ax.plot(x, q_ref[top], "_", ms=FS["ms"] * 2, mew=1.4, color=C_REF, label=refl)
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * (12 if a.compact else 6))      # headroom so the legend never covers a bar
        plotdata[f"{r}_axis"], plotdata[f"{r}_gt"], plotdata[f"{r}_cad"], plotdata[f"{r}_ref"] = top, gt, q_cad, q_ref
        if qr is not None:
            plotdata[f"{r}_qrows"] = qr[:3]
        ax.set_xticks(x)
        ax.set_xticklabels([w.panel[i] for i in top], rotation=90, fontsize=FS["tick"])
        ax.tick_params(axis="y", labelsize=FS["tick"])
        ax.set_ylabel("share of transcripts", fontsize=FS["lab"])
        if a.compact:
            ax.set_title(f"({chr(97 + 2 * r)}) {title}: {u} (median of {n_units})", fontsize=FS["title"], loc="left")
        else:
            ax.set_title(f"fixed {a.top}-gene axis; NLL CAD - ref = {dnll:+.3f} (median of {n_units})", fontsize=5.5)
        if r == 0 or (qr is not None and qr.shape[0] > 1 and np.abs(qr - qr[:1]).max() > 1e-6):
            ax.legend(fontsize=FS["leg"], frameon=False, loc="upper right", ncol=2 if a.compact else 1)
        rec = {"title": title, "task": task, "name": name, "unit": u, "stratum": stratum, "n_units": n_units, "cad": tag, "ref": ref,
               "nll_diff_cad_minus_ref": dnll, "prompt": ptxt, "top_genes": [w.panel[i] for i in top]}
        # (right) DE genes, perturbations only
        ax = axs[r, 2]
        if task == "pert":
            C = w.counts(np.where(w.ctrl & (w.split == a.split))[0])
            t = np.nan_to_num(ttest_ind(log_cp10k(real), log_cp10k(C), equal_var=False, axis=0).statistic)
            de = np.argsort(-np.abs(t))[:20]
            pc = C.sum(0) / C.sum()
            lfc = lambda q: np.log2((q + 1e-5) / (pc + 1e-5))
            o = de[np.argsort(lfc(gt)[de])]
            y = np.arange(len(o))
            ax.axvline(0, color="#BBBBBB", lw=.5)
            ms = FS["ms"] * .92
            ax.plot(lfc(gt)[o], y, "s", ms=ms, color=C_GT, label="real")
            ax.plot(lfc(q_cad)[o], y, "o", ms=ms, color=C_CAD, label="CAD")
            ax.plot(lfc(q_ref)[o], y, "|", ms=ms * 1.7, mew=1.2, color=C_REF, label="reference")
            ax.set_yticks(y)
            tg = set(split_pert(u))
            ax.set_yticklabels([w.panel[i] for i in o], fontsize=FS["tick"] - (0.5 if a.compact else 0))
            for lbl in ax.get_yticklabels():     # the perturbed (CRISPRa-targeted) genes in red
                if lbl.get_text() in tg:
                    lbl.set_color("#B0303A")
            ax.tick_params(axis="x", labelsize=FS["tick"])
            ax.set_xlabel("log2 fold change vs control", fontsize=FS["lab"])
            ax.set_title(f"({chr(98 + 2 * r)}) top-20 DE genes" if a.compact else "top-20 DE genes", fontsize=FS["title"], loc="left" if a.compact else "center")
            gt_l, cad_l, ref_l = lfc(gt)[de], lfc(q_cad)[de], lfc(q_ref)[de]
            rec["de_genes"] = [w.panel[i] for i in de]
            rec["de_pearson_cad"] = float(np.corrcoef(gt_l, cad_l)[0, 1])
            rec["de_pearson_ref"] = float(np.corrcoef(gt_l, ref_l)[0, 1])
            rec["targets_in_panel"] = [g for g in split_pert(u) if g in w.panel]
        else:
            ax.axis("off")
        for k in (1, 2):
            for s in ("top", "right"):
                axs[r, k].spines[s].set_visible(False)
        record.append(rec)
    fig.tight_layout(h_pad=1.2, w_pad=.6)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=200)
    json.dump(record, open(a.out + ".json", "w"), indent=1)
    np.savez_compressed(a.out + "_plotdata.npz", **{k: np.asarray(v) for k, v in plotdata.items()})   # everything drawn, for local re-plotting
    # LaTeX snippet for the appendix text (every name and number comes from this run; nothing is typed by hand)
    esc = lambda t: t.replace("\\", "").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    REF = {"cb_type_mean": "the cell-type mean", "cb_additive": "the additive baseline", "cb_perturbed_mean": "the perturbed mean"}
    refname = lambda r: REF.get(r, "the nearest Cell Ontology type" if r.startswith("cb_zs_ontology") else r)
    lines = [r"\begin{itemize}"]
    for k, rec in enumerate(record):
        s = (rf"\item[({chr(97 + k)})] \emph{{{esc(rec['title'])}}}: {esc(rec['unit'])}, the median of {rec['n_units']} units "
             rf"(NLL difference to {refname(rec['ref'])}: ${rec['nll_diff_cad_minus_ref']:+.3f}$ nats per UMI)")
        if "de_pearson_cad" in rec:
            s += (rf"; Pearson correlation with the real log fold change on its top-20 DE genes: ${rec['de_pearson_cad']:.2f}$ for CAD and "
                  rf"${rec['de_pearson_ref']:.2f}$ for the reference")
        lines.append(s + ".")
    lines.append(r"\end{itemize}")
    open(a.out + "_text.tex", "w").write("\n".join(lines) + "\n")
    for rec in record:
        print({k: v for k, v in rec.items() if k not in ("prompt", "top_genes", "de_genes")})


if __name__ == "__main__":
    main()
