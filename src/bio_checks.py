"""Biology sanity checks of the panel and of the control-cell text (CPU only).

python -m src.bio_checks --out ~/cad_out/auto_rev2/biology_checks.json

For each dataset: panel composition (ribosomal RPL/RPS, mitochondrial MT-, hemoglobin HB*), share of panel UMIs carried by the
top-48 genes of a cell, which genes fill the top-48 control text most often, and how similar the top-48 lists of two random
control cells are (Jaccard). Also prints one real serialized control cell (used as the example in Figure 2).
"""
import argparse
import collections
import json
import os
import numpy as np

from .cell_eval import World
from .cell_dist import serialize_cell


def check(task, name, n=1000, k=48, seed=0):
    w = World(task, name)
    panel = np.array(w.panel)
    rp = [g for g in panel if g.startswith(("RPL", "RPS"))]
    mt = [g for g in panel if g.startswith("MT-")]
    hb = [g for g in panel if g.startswith("HB")]
    rng = np.random.default_rng(seed)
    pool = np.where(w.ctrl & (w.split == "train"))[0] if task == "pert" else np.where(w.split == "train")[0]
    idx = np.sort(rng.choice(pool, min(n, len(pool)), replace=False))
    C = w.counts(idx).astype(np.float64)
    s = C.sum(1)
    tops, share = [], []
    for c, si in zip(C, s):
        t = np.argsort(-c)[:k]
        t = t[c[t] > 0]
        tops.append(set(t.tolist()))
        share.append(c[t].sum() / si)
    freq = collections.Counter(g for t in tops for g in t)
    pairs = rng.choice(len(tops), (500, 2))
    jac = [len(tops[a] & tops[b]) / max(len(tops[a] | tops[b]), 1) for a, b in pairs if a != b]
    mean_p = C.sum(0) / C.sum()
    return {
        "dataset": name, "cells_checked": int(len(idx)), "cells_from": "train control cells" if task == "pert" else "train cells",
        "panel_size": int(len(panel)), "panel_ribosomal_RPL_RPS": rp, "panel_mitochondrial_MT": mt, "panel_hemoglobin_HB": hb,
        "median_panel_umis_per_cell": float(np.median(s)),
        "top48_share_of_panel_umis_median": float(np.median(share)),
        "top48_share_of_panel_umis_iqr": [float(np.percentile(share, 25)), float(np.percentile(share, 75))],
        "top48_jaccard_between_cells_median": float(np.median(jac)),
        "top48_jaccard_iqr": [float(np.percentile(jac, 25)), float(np.percentile(jac, 75))],
        "genes_most_often_in_top48": [[str(panel[g]), round(c / len(tops), 3)] for g, c in freq.most_common(30)],
        "genes_highest_mean_frequency": [[str(panel[g]), round(float(mean_p[g]), 4)] for g in np.argsort(-mean_p)[:20]],
        "example_control_text": serialize_cell(C[0], list(panel), k=k)[:400],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = [check("pert", "norman"), check("cond", "pbmc")]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k not in ("genes_most_often_in_top48", "genes_highest_mean_frequency")}, indent=1))
        print("most often in top48:", r["genes_most_often_in_top48"][:20])
        print("highest mean freq:", r["genes_highest_mean_frequency"][:12])
