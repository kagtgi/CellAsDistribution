"""Shared constants and small helpers for CellAsDecision."""
import json
import os
import re
import numpy as np

SEED = 3407
N_GENES = 2048          # gene panel size (md 4.1)
N_MODULES = 64          # tree: 64 modules x 32 genes (md 4.2)
GENES_PER_MODULE = N_GENES // N_MODULES
CONTROL_LABELS = ("control", "non-targeting", "ctrl", "NT")

DATA_DIR = os.environ.get("CAD_DATA", os.path.expanduser("~/data"))
OUT_DIR = os.environ.get("CAD_OUT", os.path.expanduser("~/cad_out"))


def dense(x):
    """Rows of an AnnData X (sparse or dense) as a float64 ndarray."""
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float64)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_pert(p):
    """'CBL+CNN1' -> ('CBL','CNN1'); 'CBL+ctrl' -> ('CBL',) ; singles -> (gene,)."""
    return tuple(g for g in re.split(r"[+_]", str(p)) if g and g not in CONTROL_LABELS)
