"""Zero-shot cell types: pool the 5 leave-types-out folds and compare paired over held-out cell types (Stage 3).

python -m src.zs_results --split val --out ~/cad_out/auto_rev2 [--gate]
Runs: zs_q_{def,name,opaque}_f{k} (CAD, Qwen) and cb_zs_{global,ontology}_f{k} (baselines). Each held-out type appears in
exactly one fold, so pooling the folds gives one value per type. --gate writes zs_gate.json: pass if CAD with Cell Ontology
definitions beats the nearest-ontology-type mean on pooled NLL with a 95% CI that excludes zero (DECISIONS_rev2.md).
"""
import argparse
import json
import os
import numpy as np

from .common import OUT_DIR, load_json
from .evaluate import bootstrap_ci

RUNS = {"CAD, CL definition": "zs_q_def", "CAD, type name only": "zs_q_name", "CAD, opaque type id": "zs_q_opaque",
        "Global mean": "cb_zs_global", "Nearest CL type mean": "cb_zs_ontology",
        "Name-overlap type mean": "cb_zs_namenn", "CL-parent mean": "cb_zs_parent"}
METRICS = ["nll_pooled", "nll_mix", "mmd_dm", "auc_dm", "mean_corr_dm", "type_acc_gen_dm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="", help="tag prefix for test runs, e.g. t_")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    per = {}
    for lab, stem in RUNS.items():
        per[lab] = {}
        for k in range(5):
            f = os.path.join(OUT_DIR, "pbmc", "runs", f"{a.prefix if not stem.startswith('cb_') else ''}{stem}_f{k}", f"cell_{a.split}.json")
            if os.path.exists(f):
                per[lab].update(load_json(f)["per_unit"])
    per = {k: v for k, v in per.items() if v}
    units = sorted(set.intersection(*[set(v) for v in per.values()]))
    out = {"split": a.split, "n_units": len(units), "units": units, "compare": {}}
    for ref in ("Nearest CL type mean", "Name-overlap type mean", "CL-parent mean", "Global mean", "CAD, opaque type id"):
        if ref not in per:
            continue
        out["compare"][ref] = {}
        for lab in per:
            if lab == ref:
                continue
            out["compare"][ref][lab] = {m: {"mean": float(np.mean([per[lab][u][m] for u in units])),
                                            "diff": bootstrap_ci([per[lab][u][m] - per[ref][u][m] for u in units])}
                                        for m in METRICS if all(m in per[lab][u] and m in per[ref][u] for u in units)}
    if a.gate:
        d = out["compare"]["Nearest CL type mean"]["CAD, CL definition"]["nll_pooled"]["diff"]
        out["gate"] = {"rule": "CAD (CL definition) - nearest CL type mean, pooled NLL, CI upper < 0", "diff": d, "pass": d[2] < 0}
        json.dump(out["gate"], open(os.path.join(a.out, "zs_gate.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(a.out, f"pbmc_zs_{a.split}.json"), "w"), indent=1)
    print(f"zero-shot {a.split}: {len(units)} held-out types")
    for ref, d in out["compare"].items():
        print(f"== paired difference to {ref}")
        for lab, ms in d.items():
            print(f"  {lab:24s} " + "  ".join(f"{m}={v['mean']:.3f} ({v['diff'][0]:+.3f} [{v['diff'][1]:+.3f},{v['diff'][2]:+.3f}])" for m, v in ms.items()))
    if a.gate:
        print("GATE", out["gate"])


if __name__ == "__main__":
    main()
