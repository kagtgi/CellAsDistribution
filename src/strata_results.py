"""Stratified perturbation results: paired differences to a reference within each regime of the gene-level split.

python -m src.strata_results --name norman --split test --out ~/cad_out/auto_rev2 --stem norman_test_strata \
    --ref "Perturbed mean=cb_perturbed_mean" --runs "Additive=cb_additive" "CAD (Gemma), no control cell=t_g_noctrl_s0" \
    --de-json ~/cad_out/auto_rev2/norman_test_de.json

Strata come from splits.json["groups"] of the split (preprocess.make_splits):
  <split>_s   single perturbations whose gene occurs in no training perturbation (unseen single gene)
  <split>_pu  pairs of one unseen gene and one gene seen as a training single
  <split>_p   pairs of two genes both seen as training singles (the pair itself is held out)
Pairs of two unseen genes are excluded by construction. Unit of analysis = perturbation; 95% paired bootstrap over units.
"""
import argparse
import json
import os
import numpy as np

from .common import OUT_DIR, load_json
from .evaluate import bootstrap_ci
from .run_baselines import norm_label

STRATA = (("all", "all held-out perturbations"), ("s", "unseen single gene"), ("pu", "pair, one unseen gene"),
          ("p", "pair, both genes seen"))
LOWER_BETTER = {"nll_pooled", "nll_mix", "mmd_dm", "auc_dm", "energy_dm", "mse_de20"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="norman")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--de-json", default="")
    ap.add_argument("--metrics", nargs="+", default=["nll_pooled", "nll_mix", "mmd_dm", "auc_dm"])
    a = ap.parse_args()
    groups = load_json(os.path.join(OUT_DIR, a.name, "splits.json"))["groups"]
    runs = dict(r.split("=", 1) for r in [a.ref] + a.runs)
    ref = a.ref.split("=", 1)[0]
    per = {lab: load_json(os.path.join(OUT_DIR, a.name, "runs", t, f"cell_{a.split}.json"))["per_unit"] for lab, t in runs.items()}
    if a.de_json:
        de = load_json(a.de_json)["per_unit"]
        for lab in per:
            for u, m in de.get(lab, {}).items():
                if u in per[lab]:
                    per[lab][u].update(m)
    out = {"split": a.split, "ref": ref, "strata": {}}
    for key, desc in STRATA:
        shared = set.intersection(*[set(v) for v in per.values()])
        pool = shared if key == "all" else {norm_label(p) for p in groups[f"{a.split}_{key}"]}
        units = sorted(pool & shared)
        st = {"description": desc, "n": len(units), "units": units, "stats": {}}
        for lab in per:
            st["stats"][lab] = {}
            for m in a.metrics:
                if not all(m in per[lab][u] and m in per[ref][u] for u in units):
                    continue
                s = {"mean": float(np.mean([per[lab][u][m] for u in units]))}
                if lab != ref:
                    s["diff"] = bootstrap_ci([per[lab][u][m] - per[ref][u][m] for u in units])
                st["stats"][lab][m] = s
        out["strata"][key] = st
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, a.stem + ".json"), "w"), indent=1)
    for key, st in out["strata"].items():
        print(f"== {st['description']} (n={st['n']}), paired difference to {ref} [95% CI]")
        for lab, ms in st["stats"].items():
            print(f"  {lab:45s} " + "  ".join(
                f"{m}={v['mean']:.3f}" + (f" ({v['diff'][0]:+.3f} [{v['diff'][1]:+.3f},{v['diff'][2]:+.3f}])" if "diff" in v else "")
                for m, v in ms.items()))


if __name__ == "__main__":
    main()
