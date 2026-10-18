"""Recompute all secondary metrics (Pearson-delta, energy, centroid acc, ...) from SAVED q's with the fixed (seeded random)
Scorer reference. NLL is unaffected by the fix; the distance metrics are.

python -m src.rescore --name norman --tags g1c_cadf_08b g1c_cadn_08b ... [--split val]
"""
import argparse
import os
import numpy as np

from .eval_common import score_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    from .common import OUT_DIR
    for t in a.tags:
        z = np.load(os.path.join(OUT_DIR, a.name, "runs", t, "q_by_pert.npz"))
        qm = dict(zip(z["perts"].tolist(), z["q"]))
        qm["control"] = z["q_control"]
        s = score_model(a.name, t, lambda p: qm[p], a.split, perts_subset=sorted(z["perts"].tolist()))
        print(f"{t}: pearson_delta_pert={s['pearson_delta_pert'][0]:+.3f} energy={s['energy'][0]:.3f} centroid={s['centroid_acc'][0]:.2f}")


if __name__ == "__main__":
    main()
