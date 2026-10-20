"""CAD: language-model decoder trained on SINGLE CELLS, for the two per-cell use cases.

task cond: prompt = cell type            -> q over the 2048-gene panel (conditional cell generation; held-out donors)
task pert: prompt = perturbation + the serialized control cell -> q for the perturbed cell (control cell in, perturbed cell out;
           each training perturbed cell is paired with a random training control cell, as in the C2S tutorial)

Every cell is turned into a distribution by a transformation (src.cell_dist.to_dist) that defines the soft-CE target; evaluation
always scores raw counts by multinomial NLL. Head: linear softmax over the panel on the last-token state, with an optional fixed
log-prior added to the logits (--base none | mean | ctrl: none = plain head, mean = log perturbed/cell mean, ctrl = log of the
control cell's own smoothed distribution, i.e. a residual on the control cell).

Additional options (defaults reproduce the original runs exactly):
  --ctrl-text off        keep the control cell (e.g. as --base ctrl residual) but do not write it into the prompt (B1)
  --ctrl-format contrast control text = genes most over-represented relative to the mean training control cell (B2)
  --pert-desc real|other|opaque   perturbation prompt with NCBI RefSeq gene summaries (MyGene.info, gene_desc.json),
                         with another gene's summary (fixed derangement), or with opaque gene ids instead of symbols (B3)
  --weight umi           weight each cell's loss by its library size, matching the UMI-weighted evaluation (B5)
  --head nb              negative-binomial head (q plus per-gene dispersion), NB likelihood loss (B6)
  --head lowrank         low-rank logistic-normal head trained as a VAE; generation draws eps from the prior (B7)
  --zs-fold k            cond: zero-shot cell types; the types of fold k (splits/pbmc_zs_folds.json) are removed from
                         training and are the only evaluation units (Stage 3)
  --type-text def|opaque cond: add the formal Cell Ontology definition of the type (splits/pbmc_cell_ontology.json), or
                         replace type names by opaque ids

python -m src.train_cell --task cond --name pbmc --tag c_cond_freq --transform freq
python -m src.train_cell --task pert --name norman --tag c_pert_ctrl --base ctrl
"""
import unsloth  # noqa: F401  (import first)
import argparse
import os
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

from .common import SEED, save_json, load_json, split_pert
from .cell_eval import World, eval_units
from .cell_dist import to_dist, serialize_cell, serialize_rank, serialize_contrast
from .train_flat import FlatModel
from .heads import NBHead, LowRankHead

CTRL_LINE = {"mass": "Control cell (top genes with log-mass level)", "rank": "Control cell (top genes with log-mass level)",   # wording of the original rank runs, kept for reproducibility
             
             "contrast": "Control cell (genes over-represented relative to an average control cell, log2 fold level)"}


def prompt(task, label, ctrl_text=None, ctrl_line=CTRL_LINE["mass"], pert_text=None, desc=""):
    if task == "cond":
        return (f"Cell context: human peripheral blood mononuclear cells (PBMC).\nCell type: {pert_text or label}.\n{desc}"
                f"Gene expression frequency distribution of the cell:")
    s = f"Cell context: K562 cells, CRISPRa perturbation.\nPerturbation: {pert_text or label}.\n{desc}"
    if ctrl_text is not None:
        s += f"{ctrl_line}: {ctrl_text}\n"
    return s + "Gene expression frequency distribution of the perturbed cell:"


def pert_text_fns(w, mode):
    """Perturbation field and description lines for --pert-desc."""
    if mode == "none":
        return (lambda lab: lab), (lambda lab: "")
    genes = sorted({g for lab in set(w.lab[~w.ctrl]) for g in split_pert(lab)})
    if mode == "opaque":
        ids = {g: f"GENE{i:03d}" for i, g in enumerate(genes)}
        return (lambda lab: "+".join(ids[g] for g in split_pert(lab))), (lambda lab: "")
    desc = load_json(os.path.join(w.dir, "gene_desc.json"))
    have = [g for g in genes if g in desc]
    if mode == "other":                           # fixed derangement: every gene gets the summary of the next gene
        src = {g: desc[have[(i + 1) % len(have)]] for i, g in enumerate(have)}
    else:
        src = desc
    return (lambda lab: lab), (lambda lab: "".join(f"Gene {g}: {src[g]}\n" for g in split_pert(lab) if g in src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cond", "pert"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
    ap.add_argument("--transform", default="freq")
    ap.add_argument("--lora-r", type=int, default=16, help="0 = frozen backbone, only the head trains (rev3 E1a)")
    ap.add_argument("--head-arch", default="linear", choices=["linear", "mlp"], help="mlp: LayerNorm-Linear(1024)-GELU-Linear head (rev3 E1a)")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (needed for Gemma-2 2B at bs 16 on 40GB)")
    ap.add_argument("--attn", default=None, help="attn_implementation, e.g. eager for Gemma-2")
    ap.add_argument("--base", default="none", choices=["none", "mean", "ctrl"])
    ap.add_argument("--no-ctrl", action="store_true", help="pert: do not show the control cell (population-level conditional)")
    ap.add_argument("--ctrl-text", default="on", choices=["on", "off"])
    ap.add_argument("--topk", type=int, default=48)
    ap.add_argument("--ctrl-format", default="mass", choices=["mass", "rank", "contrast"])
    ap.add_argument("--pert-desc", default="none", choices=["none", "real", "other", "opaque"])
    ap.add_argument("--weight", default="cell", choices=["cell", "umi"])
    ap.add_argument("--head", default="softmax", choices=["softmax", "nb", "lowrank"])
    ap.add_argument("--zs-fold", type=int, default=-1)
    ap.add_argument("--type-text", default="name", choices=["name", "def", "opaque"])
    ap.add_argument("--rank-k", type=int, default=16)
    ap.add_argument("--kl-warmup", type=float, default=0.3, help="lowrank: fraction of steps over which beta rises 0 -> 1")
    ap.add_argument("--draws", type=int, default=300, help="lowrank, cond task: prior draws per cell type at evaluation")
    ap.add_argument("--max-train-cells", type=int, default=40000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--min-cells", type=int, default=30)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    w = World(a.task, a.name)
    G = len(w.panel)
    use_ctrl = a.task == "pert" and not a.no_ctrl
    show_ctrl = use_ctrl and a.ctrl_text == "on"
    if a.base == "ctrl" and not use_ctrl:
        raise SystemExit("--base ctrl needs the control cell")
    if a.head != "softmax" and a.transform != "freq":
        raise SystemExit("--head nb/lowrank are likelihood heads on raw counts (use --transform freq)")
    ctr_pool = np.where(w.ctrl & (w.split == "train"))[0]
    if a.ctrl_format == "contrast":
        ref_ctrl = to_dist(w.counts(ctr_pool).sum(0), "smooth", 0.5)
        SER = lambda c: serialize_contrast(c, w.panel, ref_ctrl, a.topk)
    elif a.ctrl_format == "rank":
        SER = lambda c: serialize_rank(c, w.panel, a.topk)
    else:
        SER = lambda c: serialize_cell(c, w.panel, a.topk)
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if a.task == "pert":
        pert_field, pert_desc = pert_text_fns(w, a.pert_desc)
    elif a.type_text == "def":
        CL = load_json(os.path.join(repo, "splits", "pbmc_cell_ontology.json"))["terms"]
        pert_field, pert_desc = (lambda l: l), (lambda l: f"Definition (Cell Ontology {CL[l]['id']}): {CL[l]['definition']}\n" if l in CL else "")
    elif a.type_text == "opaque":
        TID = {t: f"TYPE{i:02d}" for i, t in enumerate(sorted(set(w.lab)))}
        pert_field, pert_desc = (lambda l: TID[l]), (lambda l: "")
    else:
        pert_field, pert_desc = (lambda l: l), (lambda l: "")
    held = []
    if a.zs_fold >= 0:
        if a.task != "cond":
            raise SystemExit("--zs-fold is for the cond task")
        held = load_json(os.path.join(repo, "splits", "pbmc_zs_folds.json"))["folds"][a.zs_fold]

    def make_prompt(label, c=None):
        return prompt(a.task, label, SER(c) if (show_ctrl and c is not None) else None, CTRL_LINE[a.ctrl_format],
                      pert_field(label), pert_desc(label))

    tr = np.where((w.split == "train") & ~w.ctrl & ~np.isin(w.lab, held))[0]
    if len(tr) > a.max_train_cells:
        tr = np.sort(rng.choice(tr, a.max_train_cells, replace=False))
    pair = rng.choice(ctr_pool, len(tr)) if use_ctrl else None
    mean_q = to_dist(w.counts(tr[:20000]).sum(0), "smooth", 0.5) if a.base == "mean" else None
    mean_lib = float(w.lib.mean())

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16, **({"attn_implementation": a.attn} if a.attn else {})).cuda()
    if a.lora_r > 0:
        backbone = get_peft_model(backbone, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r, target_modules="all-linear", lora_dropout=0.0))
    else:                                         # frozen language model: features only
        for p_ in backbone.parameters():
            p_.requires_grad_(False)
    if a.grad_ckpt and a.lora_r > 0:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.enable_input_require_grads()
    cfg = backbone.config
    hidden = getattr(getattr(cfg, "text_config", cfg), "hidden_size")
    model = FlatModel(backbone, hidden, G, None).cuda()
    if a.head_arch == "mlp":
        out_l = torch.nn.Linear(1024, G)
        torch.nn.init.zeros_(out_l.weight); torch.nn.init.zeros_(out_l.bias)    # starts at the uniform q, like the linear head
        model.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, 1024), torch.nn.GELU(), out_l).cuda()
    extra = None
    if a.head == "nb":
        extra = NBHead(hidden, G).cuda()
    elif a.head == "lowrank":
        extra = LowRankHead(hidden, G, k=a.rank_k).cuda()
    head_params = list(model.head.parameters()) + (list(extra.parameters()) if extra is not None else [])
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = ([{"params": bb_params, "lr": a.lr}] if bb_params else []) + [{"params": head_params, "lr": a.head_lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    n_steps = int(a.epochs * len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=max(n_steps, 2), pct_start=0.05)

    def base_logits(counts_ctrl):
        if a.base == "none":
            return 0.0
        if a.base == "mean":
            return torch.log(torch.tensor(mean_q, dtype=torch.float32, device="cuda"))[None]
        return torch.log(torch.tensor(to_dist(counts_ctrl, "smooth", 0.5), dtype=torch.float32, device="cuda"))

    model.train()
    hist, diag, t0 = [], [], time.time()
    torch.cuda.reset_peak_memory_stats()
    order = np.concatenate([rng.permutation(len(tr)) for _ in range(int(np.ceil(a.epochs)))])
    for step in range(n_steps):
        idx = order[step * a.bs:(step + 1) * a.bs]
        Xc = w.counts(tr[idx])
        Cc = w.counts(pair[idx]) if use_ctrl else None
        texts = [make_prompt(w.lab[tr[k]], Cc[n] if use_ctrl else None) for n, k in enumerate(idx)]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        h = model.pooled(enc["input_ids"], enc["attention_mask"])
        logits = model.head(h) + base_logits(Cc)
        X = torch.tensor(Xc, dtype=torch.float32, device="cuda")
        N = X.sum(1)
        if a.head == "softmax":
            tgt = torch.tensor(to_dist(Xc, a.transform), dtype=torch.float32, device="cuda")
            per_cell = -(tgt * torch.log_softmax(logits, -1)).sum(-1)              # nats per UMI of each cell
            wts = N / mean_lib if a.weight == "umi" else torch.ones_like(N)
            loss = (wts * per_cell).sum() / wts.sum()
        elif a.head == "nb":
            nll = extra.loss(logits, h, X)                                         # nats per cell
            loss = nll.sum() / N.sum() if a.weight == "umi" else (nll / N.clamp(min=1)).mean()
        else:
            beta = min(1.0, (step + 1) / max(1, a.kl_warmup * n_steps))
            nll, d = extra.loss(logits, h, X, beta)
            loss = nll.sum() / N.sum() if a.weight == "umi" else (nll / N.clamp(min=1)).mean()
            if step % 50 == 0:
                diag.append({"step": step, "beta": beta, **d})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + (list(extra.parameters()) if extra is not None else []), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        hist.append(float(loss))
        if step % 50 == 0:
            print(f"step {step}/{n_steps} loss {loss.item():.4f}" + (f" {diag[-1]}" if diag else ""), flush=True)
    train_s = time.time() - t0
    model.eval()
    if extra is not None:
        extra.eval()
    gen = torch.Generator(device="cuda").manual_seed(a.seed)

    @torch.no_grad()
    def heads_out(texts, Cc):
        qs, ths = [], []
        for s in range(0, len(texts), 25):
            enc = tok(texts[s:s + 25], return_tensors="pt", padding=True).to("cuda")
            h = model.pooled(enc["input_ids"], enc["attention_mask"])
            lg = model.head(h) + base_logits(Cc[s:s + 25] if Cc is not None else None)
            if a.head == "lowrank":
                n = a.draws if (a.task == "cond" or not use_ctrl) else 1
                lg = extra.sample_logits(lg, h, n, generator=gen)
            qs.append(torch.softmax(lg, -1).double().cpu().numpy())
            if a.head == "nb":
                ths.append(torch.exp(extra.log_theta(h)).double().cpu().numpy())
        return np.concatenate(qs), (np.concatenate(ths) if ths else None)

    prompts_used = {}
    timing = {"seconds": 0.0, "prompts": 0}       # decoder wall-clock (GPU synchronized), excluding sampling and scoring

    def qfn(unit, cidx, want_theta=False):
        if a.task == "cond" or not use_ctrl:
            if a.base == "ctrl":
                raise SystemExit("ctrl base needs control cells")
            prompts_used[unit] = [make_prompt(unit)]
            torch.cuda.synchronize(); t_ = time.time()
            q, th = heads_out(prompts_used[unit], None)
            torch.cuda.synchronize(); timing["seconds"] += time.time() - t_; timing["prompts"] += 1
            if a.task == "pert" and cidx is not None and a.head != "lowrank":
                q = np.repeat(q, len(cidx), 0)                 # no control shown: the same q for every control cell
                th = np.repeat(th, len(cidx), 0) if th is not None else None
            if a.task == "pert" and a.head == "lowrank" and cidx is not None:
                q = q[: len(cidx)]                             # one prior draw per control slot
        else:
            Cc = w.counts(cidx)
            prompts_used[unit] = [make_prompt(unit, c) for c in Cc]
            torch.cuda.synchronize(); t_ = time.time()
            q, th = heads_out(prompts_used[unit], Cc)
            torch.cuda.synchronize(); timing["seconds"] += time.time() - t_; timing["prompts"] += len(prompts_used[unit])
        return (q, th) if want_theta else q

    thfn = (lambda u, c: qfn(u, c, True)[1]) if a.head == "nb" else None
    if a.head == "lowrank" and a.task == "pert" and not use_ctrl:
        a.draws = a.M
    eval_u = [u for u in w.units(a.eval_split, a.min_cells) if u in held] if held else None
    per = eval_units(w, a.eval_split, qfn, a.tag, M=a.M, min_cells=a.min_cells, seed=a.seed, thfn=thfn, units=eval_u,
                     save_qrows=True)
    save_json({u: p[:3] for u, p in prompts_used.items()}, os.path.join(w.dir, "runs", a.tag, f"prompts_{a.eval_split}.json"))
    n_trainable = sum(p.numel() for p in bb_params) + sum(p.numel() for p in head_params)
    save_json({"args": vars(a), "n_train_cells": len(tr), "train_seconds": train_s, "trainable_params": int(n_trainable),
               "peak_vram_gb": torch.cuda.max_memory_reserved() / 1e9, "loss_history": hist, "head_diag": diag,
               "decode_seconds_per_prompt": timing["seconds"] / max(timing["prompts"], 1), "decode_prompts": timing["prompts"]},
              os.path.join(w.dir, "runs", a.tag, "log.json"))
    print(f"{a.tag}: nll_pooled={np.mean([v['nll_pooled'] for v in per.values()]):.4f} "
          f"nll_mix={np.mean([v['nll_mix'] for v in per.values()]):.4f} auc_mult={np.mean([v['auc_mult'] for v in per.values()]):.3f} "
          f"mmd_dm={np.mean([v['mmd_dm'] for v in per.values()]):.4f} train_s={train_s:.0f}", flush=True)


if __name__ == "__main__":
    main()
