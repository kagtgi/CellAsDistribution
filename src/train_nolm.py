"""No-language-model decoder of the gene distribution (rev3 E1b): the matched control for "does the LM matter?".

python -m src.train_nolm --tag e1b_nolm_s0 --seed 0 --eval-split val

Same task, panel, training cells (20,000), fixed training-mean log-prior (CAD --base mean), target (cell frequency), loss (soft cross-entropy per cell), softmax output over the 2048
panel genes and scoring (cell_eval.eval_units, matched control cells, no control cell shown) as CAD without a control cell;
only the encoder of the state changes. A perturbation is the SUM of learned embeddings of its genes (multi-hot over the genes
that occur in training perturbation labels), followed by a 2-layer MLP and a softmax. A gene never seen in training gets a zero
embedding, the same convention as the partial-additive baseline, so an unseen single gene is decoded from the empty state.
Settings were fixed before any run (experiments/PREREGISTRATION_rev3.md): embedding 128, hidden 512, 5 epochs, batch 64,
AdamW lr 1e-3, one-cycle schedule, output layer zero-initialized (starts at the uniform q, like CAD's head).
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn

from .common import save_json, split_pert
from .cell_eval import World, eval_units
from .cell_dist import to_dist


class NoLM(nn.Module):
    def __init__(self, n_vocab, G, d=128, hidden=512, fixed=None):
        super().__init__()
        # learned: multi-hot @ E = sum of learned gene embeddings. fixed (rev4): E is a frozen external gene-embedding
        # matrix [n_vocab, d_ext] (GenePT or co-expression), followed by a learned shared projection to d.
        self.fixed = None if fixed is None else nn.Parameter(torch.as_tensor(fixed, dtype=torch.float32), requires_grad=False)
        self.emb = nn.Linear(n_vocab if fixed is None else fixed.shape[1], d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, G))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, mh):
        return self.mlp(self.emb(mh if self.fixed is None else mh @ self.fixed))


def load_genept(path):
    """GenePT gene embeddings (Chen & Zou; Zenodo 10833191): a pickled dict {GENE SYMBOL: ada-002 embedding of the NCBI
    summary}. Unpickled with a restricted loader that only rebuilds numpy arrays and plain containers (no code execution)."""
    import pickle

    class Safe(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] == "numpy" and name in ("_reconstruct", "ndarray", "dtype", "scalar", "_frombuffer"):
                return super().find_class(module, name)
            if module == "builtins" and name in ("dict", "list", "float", "tuple"):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
    with open(path, "rb") as f:
        d = Safe(f).load()
    return {str(k).upper(): np.asarray(v, dtype=np.float32) for k, v in d.items()}


def coexpr_embedding(w, k=64, n=10000):
    """Gene embedding from TRAINING cells only: the gene's loadings on the top-k principal components of standardized
    log-CP10k expression (panel genes; genes outside the panel get no embedding)."""
    tr = np.where(w.split == "train")[0]
    tr = np.sort(np.random.default_rng(0).choice(tr, min(len(tr), n), replace=False))
    C = w.counts(tr).astype(np.float64)
    L = np.log1p(1e4 * C / np.clip(C.sum(1, keepdims=True), 1, None))
    L = (L - L.mean(0)) / (L.std(0) + 1e-6)
    _, S, Vt = np.linalg.svd(L, full_matrices=False)
    E = (Vt[:k].T * S[:k] / np.sqrt(len(L))).astype(np.float32)          # [G, k]
    return {g: E[i] for i, g in enumerate(w.panel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="norman")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--max-train-cells", type=int, default=20000)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--gene-emb", default="learned", choices=["learned", "genept", "coexpr"])
    ap.add_argument("--genept", default=os.path.expanduser("~/genept/GenePT_gene_embedding_ada_text.pickle"))
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = World("pert", a.name)
    G = len(w.panel)
    tr = np.where((w.split == "train") & ~w.ctrl)[0]
    if len(tr) > a.max_train_cells:
        tr = np.sort(rng.choice(tr, a.max_train_cells, replace=False))
    fixed = None
    if a.gene_emb == "learned":
        vocab = sorted({g for lab in set(w.lab[tr]) for g in split_pert(lab)})
    else:
        # a frozen external embedding exists for genes never perturbed in training, so the vocabulary covers every
        # perturbed gene; no gene-specific parameter is learned, only the shared projection and MLP
        vocab = sorted({g for lab in set(w.lab[~w.ctrl]) for g in split_pert(lab)})
        E = load_genept(a.genept) if a.gene_emb == "genept" else coexpr_embedding(w)
        dim = len(next(iter(E.values())))
        fixed = np.stack([E.get(g.upper(), np.zeros(dim, np.float32)) for g in vocab])
        fixed /= np.maximum(np.linalg.norm(fixed, axis=1, keepdims=True), 1e-8)    # unit norm; a missing gene stays zero
        print(f"{a.gene_emb}: {int((np.abs(fixed).sum(1) > 0).sum())}/{len(vocab)} perturbed genes have an embedding", flush=True)
    vid = {g: i for i, g in enumerate(vocab)}

    def multihot(labels):
        m = np.zeros((len(labels), len(vocab)), np.float32)
        for n, lab in enumerate(labels):
            for g in split_pert(lab):
                if g in vid:
                    m[n, vid[g]] = 1.0
        return torch.tensor(m, device=dev)

    model = NoLM(len(vocab), G, fixed=fixed).to(dev)
    # fixed log-prior = smoothed training mean, exactly CAD's --base mean, so the network learns only a correction
    log_b = torch.log(torch.tensor(to_dist(w.counts(tr[:20000]).sum(0), "smooth", 0.5), dtype=torch.float32, device=dev))[None]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    n_steps = int(a.epochs * len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(n_steps, 2), pct_start=0.05)
    order = np.concatenate([rng.permutation(len(tr)) for _ in range(int(np.ceil(a.epochs)))])
    hist, t0 = [], time.time()
    model.train()
    for step in range(n_steps):
        idx = order[step * a.bs:(step + 1) * a.bs]
        tgt = torch.tensor(to_dist(w.counts(tr[idx]), "freq"), dtype=torch.float32, device=dev)
        loss = -(tgt * torch.log_softmax(model(multihot(w.lab[tr[idx]])) + log_b, -1)).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        hist.append(float(loss))
        if step % 200 == 0:
            print(f"step {step}/{n_steps} loss {loss.item():.4f}", flush=True)
    train_s = time.time() - t0
    model.eval()
    timing = {"seconds": 0.0, "prompts": 0}

    @torch.no_grad()
    def qfn(unit, cidx):
        t_ = time.time()
        q = torch.softmax(model(multihot([unit])) + log_b, -1).double().cpu().numpy()
        timing["seconds"] += time.time() - t_; timing["prompts"] += 1
        return np.repeat(q, len(cidx), 0) if cidx is not None else q

    per = eval_units(w, a.eval_split, qfn, a.tag, M=a.M, min_cells=a.min_cells, seed=a.seed, save_qrows=True)
    unseen = sorted({g for u in per for g in split_pert(u) if g not in vid})
    save_json({"args": vars(a), "n_train_cells": len(tr), "train_seconds": train_s, "vocab_size": len(vocab),
               "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "gene_emb": a.gene_emb, "loss_history": hist,
               "unseen_genes_in_eval": unseen,
               "decode_seconds_per_prompt": timing["seconds"] / max(timing["prompts"], 1)},
              os.path.join(w.dir, "runs", a.tag, "log.json"))
    print(f"{a.tag}: nll_pooled={np.mean([v['nll_pooled'] for v in per.values()]):.4f} "
          f"mmd_dm={np.mean([v['mmd_dm'] for v in per.values()]):.4f} train_s={train_s:.0f}", flush=True)


if __name__ == "__main__":
    main()
