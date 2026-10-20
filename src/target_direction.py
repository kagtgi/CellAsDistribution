"""Signed target error (rev3, post hoc): pred - real log fold change of the perturbed gene(s) over the held-out control mean,
qbar pseudobulk, test split. Run from the repo root on the VM: python -m src.target_direction (output: paper/auto_rev3/test_target_direction.txt)."""
import os, json, numpy as np
from src.cell_eval import World
from src.common import split_pert
w = World("pert", "norman")
pb = lambda C: np.log1p(1e4 * C.sum(0) / max(C.sum(), 1))
mu_c = pb(w.counts(np.where(w.ctrl & (w.split == "test"))[0]))
pos = {g: i for i, g in enumerate(w.panel)}
groups = json.load(open(os.path.join(w.dir, "splits.json")))["groups"]
from src.run_baselines import norm_label
strat = {norm_label(p): k for k in ("s", "pu", "p") for p in groups[f"test_{k}"]}
out = {}
for t in ["cb_additive", "cb_additive_ti", "t2_g_noctrl_s0", "t3_nolm_s0", "cb_scGen_style"]:
    z = np.load(os.path.join(w.dir, "runs", t, "qbar_test.npz"), allow_pickle=True)
    res = {}
    for u, q in zip(z["units"], z["q"]):
        u = str(u)
        tg = [pos[g] for g in split_pert(u) if g in pos]
        if not tg or u not in strat:
            continue
        R = w.counts(np.where((w.lab == u) & (w.split == "test") & ~w.ctrl)[0])
        dr = pb(R)[tg] - mu_c[tg]
        dp = np.log1p(1e4 * q / q.sum())[tg] - mu_c[tg]
        res.setdefault(strat[u], []).extend(list(dp - dr))
        res.setdefault(strat[u] + "_real", []).extend(list(dr))
    out[t] = {k: (round(float(np.mean(v)), 3), len(v)) for k, v in res.items()}
    print(t, out[t], flush=True)
