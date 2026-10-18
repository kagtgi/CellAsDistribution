"""CAD-P: same Qwen3.5 LoRA backbone and text state as CAD-N, but a FLAT linear 2048-way softmax head
(soft cross-entropy on bagged gene frequencies). Isolates the tree / decision-structure contribution.

python -m src.train_flat --name norman --tag cadf_08b --epochs 3 --bag-size 20
No tree, no Clef head. Plain PyTorch loop (HF + PEFT), bf16.
"""
import unsloth  # noqa: F401  (import first: avoids the torchao/torch 2.8 clash when peft loads)
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

from .common import SEED, OUT_DIR, load_json, save_json, split_pert
from .records import make_units, _context
from .eval_common import load_world, score_model
from . import baselines as B


NAME_MAP = {}   # filled by --shuffle-names: gene symbol -> random other symbol (identity ablation)
DESC = {}       # filled by --gene-desc-file: gene symbol -> description text shown in the prompt


def prompt(name, pert):
    genes = [NAME_MAP.get(g, g) for g in split_pert(pert)]
    desc = "".join(f"Gene {g}: {DESC[g]}\n" for g in split_pert(pert) if g in DESC)
    return (f"Cell context: {_context(name)}.\nPerturbation: {'+'.join(genes) or 'control'}.\n{desc}"
            f"Gene expression frequency distribution of the cell:")


class FlatModel(nn.Module):
    def __init__(self, backbone, hidden, n_genes, bias_init=None):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden, n_genes)
        nn.init.zeros_(self.head.weight)
        if bias_init is None:
            nn.init.zeros_(self.head.bias)
        else:
            self.head.bias.data = torch.log(torch.tensor(bias_init, dtype=torch.float32))

    def pooled(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(1) - 1  # right padding: index of last real token
        return h[torch.arange(h.size(0)), last].float()

    def forward(self, input_ids, attention_mask):
        return self.head(self.pooled(input_ids, attention_mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pert-col", default="perturbation")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
    ap.add_argument("--bag-size", type=int, default=20)
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--bias-init", action="store_true", help="init head bias to log pooled train freq (NOT matched to CAD-N)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--shuffle-names", action="store_true", help="identity ablation: permute gene symbols in prompts")
    ap.add_argument("--random-ids", action="store_true", help="identity ablation: replace every gene symbol by an opaque id GENE_####")
    ap.add_argument("--gene-desc-file", default="", help="JSON {symbol: description}; adds knowledge text to the prompt")
    ap.add_argument("--desc-mode", default="real", choices=["real", "other"],
                    help="real: the gene's own description; other: a fixed permutation of OTHER genes' descriptions (control)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    d, X, obs, panel, lab, ctrl, split = load_world(a.name, a.pert_col)
    rng = np.random.default_rng(a.seed)
    if a.gene_desc_file:
        raw = json.load(open(os.path.expanduser(a.gene_desc_file)))
        keys = sorted(raw)
        if a.desc_mode == "real":
            DESC.update(raw)
        else:   # control: every gene gets another gene's description (derangement-like fixed permutation)
            perm = np.random.default_rng(0).permutation(len(keys))
            DESC.update({k: raw[keys[j]] for k, j in zip(keys, perm)})
    if a.shuffle_names or a.random_ids:
        genes = sorted({g for p in set(lab) for g in split_pert(p)})
        perm = np.random.default_rng(0).permutation(len(genes))
        if a.random_ids:   # opaque ids: no real-gene meaning at all (shuffled names are wrong-but-real names)
            NAME_MAP.update({g: f"GENE_{j:04d}" for g, j in zip(genes, perm)})
        else:
            NAME_MAP.update({g: genes[j] for g, j in zip(genes, perm)})
    run = os.path.join(d, "runs", a.tag)
    os.makedirs(run, exist_ok=True)

    units = make_units(obs, a.pert_col, "train", a.bag_size, rng, a.max_cells)
    texts = [prompt(a.name, obs[a.pert_col].iloc[u[0]]) for u in units]
    targets = []
    for u in units:
        c = np.asarray(X[u].sum(0)).ravel().astype(np.float32)
        targets.append(c / c.sum())
    targets = np.stack(targets)
    pooled = np.asarray(X[np.where(split == "train")[0]].sum(0)).ravel() + 0.5
    pooled = (pooled / pooled.sum()).astype(np.float32)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda()
    backbone = get_peft_model(backbone, LoraConfig(r=16, lora_alpha=16, target_modules="all-linear", lora_dropout=0.0))
    cfg = backbone.config
    hidden = getattr(getattr(cfg, "text_config", cfg), "hidden_size")
    model = FlatModel(backbone, hidden, len(panel), pooled if a.bias_init else None).cuda()
    opt = torch.optim.AdamW([{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.head_lr}], weight_decay=0.0)
    n_steps = int(a.epochs * len(units) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr], total_steps=max(n_steps, 2), pct_start=0.05)
    model.train()
    hist, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    order = np.concatenate([rng.permutation(len(units)) for _ in range(int(np.ceil(a.epochs)))])
    for step in range(n_steps):
        idx = order[step * a.bs:(step + 1) * a.bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        logits = model(enc["input_ids"], enc["attention_mask"])
        tgt = torch.tensor(targets[idx], device="cuda")
        loss = -(tgt * torch.log_softmax(logits, -1)).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        hist.append(float(loss))
        if step % 20 == 0:
            print(f"step {step}/{n_steps} loss {loss.item():.4f}", flush=True)
    train_s = time.time() - t0
    model.eval()

    @torch.no_grad()
    def q_fn(p):
        enc = tok([prompt(a.name, p)], return_tensors="pt").to("cuda")
        return torch.softmax(model(enc["input_ids"], enc["attention_mask"])[0], -1).double().cpu().numpy()

    summ = score_model(a.name, a.tag, q_fn, a.eval_split, seed=a.seed, pert_col=a.pert_col)
    log = {"args": vars(a), "n_train_records": len(units), "train_seconds": train_s,
           "peak_vram_gb": torch.cuda.max_memory_reserved() / 1e9, "loss_history": hist,
           "sum_q_range": summ.pop("_sum_q_range"), "gen_seconds": summ.pop("_gen_seconds")}
    save_json(log, os.path.join(run, "log.json"))
    print(f"{a.tag}: pearson_delta_pert={summ['pearson_delta_pert'][0]:+.3f} energy={summ['energy'][0]:.3f} "
          f"centroid_acc={summ['centroid_acc'][0]:.2f} mae={summ['mae'][0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
