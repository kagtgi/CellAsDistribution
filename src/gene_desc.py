"""Fetch short gene descriptions (NCBI RefSeq summary via MyGene.info) for every perturbed gene symbol of a dataset.

python -m src.gene_desc --name norman      # -> OUT_DIR/gene_desc.json  {symbol: "name. summary (truncated)"}
Used by train_flat --gene-desc (real vs a fixed permutation of OTHER genes' descriptions as the control).
"""
import argparse
import json
import os
import urllib.parse
import urllib.request

import pandas as pd

from .common import OUT_DIR, split_pert, save_json


def fetch(symbols, max_words=70):
    out = {}
    for i in range(0, len(symbols), 200):
        chunk = symbols[i:i + 200]
        data = urllib.parse.urlencode({"q": ",".join(chunk), "scopes": "symbol", "species": "human", "fields": "name,summary"}).encode()
        req = urllib.request.Request("https://mygene.info/v3/query", data=data)
        for hit in json.load(urllib.request.urlopen(req, timeout=60)):
            if hit.get("notfound") or "query" not in hit:
                continue
            q = hit["query"]
            if q in out and out[q]:
                continue
            txt = (hit.get("name", "") + ". " + hit.get("summary", "")).strip()
            out[q] = " ".join(txt.split()[:max_words])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pert-col", default="perturbation")
    a = ap.parse_args()
    obs = pd.read_csv(os.path.join(OUT_DIR, a.name, "obs.csv"), index_col=0)
    genes = sorted({g for p in obs[a.pert_col].astype(str).unique() for g in split_pert(p)})
    desc = fetch(genes)
    save_json(desc, os.path.join(OUT_DIR, a.name, "gene_desc.json"))
    print(f"{a.name}: {len(desc)}/{len(genes)} genes with a description; example:", next(iter(desc.items())))


if __name__ == "__main__":
    main()
