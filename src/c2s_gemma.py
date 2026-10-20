"""Autoregressive Cell2Sentence baseline on the SAME backbone and budget as CAD (isolates the representation from the backbone).

Backbone C2S-Scale-Gemma-2-2B with LoRA r16 on all linear layers (as CAD); 20,000 training perturbed cells, one epoch,
batch 16, lr 2e-4, one-cycle schedule (as CAD). Each training perturbed cell is paired with a random training control cell
(as CAD and the C2S tutorial). Input: control-cell rank sentence (top-200 panel genes) + perturbation name; output: the
perturbed cell's rank sentence (top-200 panel genes); loss on the response only.

Generation (subcommand gen): for every held-out unit and each of its M matched control cells (cell_eval.eval_units draws),
one sampled sentence (top-k 50, top-p 0.95). Sentences are reconstructed to expression with the tutorial's log-rank
regression (c2s_baseline.fit_rank_expression, fit on training cells) and cached as profiles [M, G] per unit.
Scoring (subcommand score): q_m = (1 - eps) * profile_m + eps * training perturbed mean, then the identical
cell_eval.eval_units as every other method (pooled NLL, per-cell mixture NLL over the M controls, generation metrics).
eps is chosen on the validation split and then fixed for test.

python -m src.c2s_gemma train --tag c2sg_s0 --seed 0
python -m src.c2s_gemma gen   --tag c2sg_s0 --split val
python -m src.c2s_gemma score --tag c2sg_s0 --split val --eps 0.1
"""
import unsloth  # noqa: F401
import argparse
import os
import time
import zlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from .common import save_json, load_json
from .cell_eval import World, eval_units
from .cell_dist import to_dist
from .c2s_baseline import sentence_from_counts, fit_rank_expression, profile_from_sentence

MODEL = "vandijklab/C2S-Scale-Gemma-2-2B"
TOPK = 200
PROMPT = ("Given the following cell sentence of {n} expressed genes representing a cell's basal state, predict the cell "
          "sentence after applying the perturbation: {pert}.\nControl cell sentence: {ctrl}.\n\nPerturbed cell sentence:")


def run_dir(w, tag):
    d = os.path.join(w.dir, "runs", tag)
    os.makedirs(d, exist_ok=True)
    return d


def train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    w = World("pert", a.name)
    tr = np.where((w.split == "train") & ~w.ctrl)[0]
    tr = np.sort(rng.choice(tr, min(a.max_train_cells, len(tr)), replace=False))
    pool = np.where(w.ctrl & (w.split == "train"))[0]
    pair = rng.choice(pool, len(tr))
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, target_modules="all-linear", lora_dropout=0.0, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    n_steps = int(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=n_steps, pct_start=0.05)
    order = rng.permutation(len(tr))
    hist, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(n_steps):
        idx = order[step * a.bs:(step + 1) * a.bs]
        X, C = w.counts(tr[idx]), w.counts(pair[idx])
        loss_acc = 0.0
        for s in range(0, len(idx), a.micro):            # gradient accumulation over micro-batches (long sequences)
            prompts, fulls = [], []
            for n in range(s, min(s + a.micro, len(idx))):
                cs = sentence_from_counts(C[n], w.panel, TOPK)
                pr = PROMPT.format(n=len(cs), pert=w.lab[tr[idx[n]]], ctrl=" ".join(cs))
                prompts.append(pr)
                fulls.append(pr + " " + " ".join(sentence_from_counts(X[n], w.panel, TOPK)) + "." + tok.eos_token)
            enc = tok(fulls, return_tensors="pt", padding=True, truncation=True, max_length=a.max_len, add_special_tokens=True).to("cuda")
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            for j, pr in enumerate(prompts):
                labels[j, : len(tok(pr, add_special_tokens=True)["input_ids"])] = -100
            loss = model(**enc, labels=labels).loss * (len(prompts) / len(idx))
            loss.backward()
            loss_acc += float(loss)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        hist.append(loss_acc)
        if step % 50 == 0:
            print(f"step {step}/{n_steps} loss {loss_acc:.4f} {time.time() - t0:.0f}s", flush=True)
    train_s = time.time() - t0
    d = run_dir(w, a.tag)
    model.save_pretrained(os.path.join(d, "adapter"))
    save_json({"args": vars(a), "n_train_cells": len(tr), "train_seconds": train_s, "steps": n_steps,
               "peak_vram_gb": torch.cuda.max_memory_reserved() / 1e9, "loss_history": hist}, os.path.join(d, "train_log.json"))
    print(f"trained {a.tag} in {train_s:.0f}s", flush=True)


def gen(a):
    torch.manual_seed(a.seed)
    w = World("pert", a.name)
    G, pos = len(w.panel), {g: i for i, g in enumerate(w.panel)}
    d = run_dir(w, a.tag)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda()
    model = PeftModel.from_pretrained(base, os.path.join(d, "adapter")).merge_and_unload().eval()
    tr = np.where((w.split == "train") & ~w.ctrl)[0]
    ra, rb = fit_rank_expression(w.X, w.panel, tr, k=TOPK)
    pool = np.where(w.ctrl & (w.split == a.split))[0]
    units = w.units(a.split, a.min_cells)
    if a.n_units:
        units = units[: a.n_units]
    prof, valid, lens, texts, t0 = {}, [], [], {}, time.time()
    for k, u in enumerate(units):
        cidx = np.random.default_rng(zlib.crc32(u.encode())).choice(pool, a.M, replace=False)   # = eval_units draws
        C = w.counts(cidx)
        prompts = []
        for c in C:
            cs = sentence_from_counts(c, w.panel, TOPK)
            prompts.append(PROMPT.format(n=len(cs), pert=u, ctrl=" ".join(cs)))
        P = np.zeros((len(prompts), G))
        for s in range(0, len(prompts), a.gen_bs):
            enc = tok(prompts[s:s + a.gen_bs], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, top_k=50, top_p=0.95,
                                     pad_token_id=tok.pad_token_id)
            for j, o in enumerate(out):
                text = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                texts.setdefault(u, []).append(text)
                # the training response ends with "." + EOS; gene names may contain dots (e.g. RP11-887P2.5), so only a
                # final period is removed and the sentence is cut at the first line break, never at the first "."
                text = text.split("\n")[0].strip()
                genes = (text[:-1] if text.endswith(".") else text).split()
                lens.append(len(genes))
                valid.append(float(np.mean([g in pos for g in genes])) if genes else 0.0)
                P[s + j] = profile_from_sentence(genes, pos, ra, rb, G)
        prof[u] = P.astype(np.float32)
        print(f"{k + 1}/{len(units)} {u} {time.time() - t0:.0f}s", flush=True)
    gen_s = time.time() - t0
    np.savez_compressed(os.path.join(d, f"profiles_{a.split}.npz"), units=np.array(units), **{f"{u}__P": p for u, p in prof.items()})
    save_json({"split": a.split, "gen_seconds": gen_s, "seconds_per_cell": gen_s / max(len(lens), 1),
               "gen_len_mean": float(np.mean(lens)), "valid_gene_frac_mean": float(np.mean(valid)), "rank_fit": [ra, rb]},
              os.path.join(d, f"gen_log_{a.split}.json"))
    save_json(texts, os.path.join(d, f"gen_texts_{a.split}.json"))


def redecode(a):
    """rev3 (post hoc): decode the SAVED generated sentences with the learned rank-to-frequency decoder of repr_ceiling
    (E[frequency | gene, rank-position bin], fit on training cells) instead of the C2S log-rank regression. Writes
    runs/<tag>_learned/profiles_<split>.npz, scored by the unchanged `score` subcommand (eps chosen on validation)."""
    from .repr_ceiling import fit_learned_rank, RBINS
    w = World("pert", a.name)
    G, pos = len(w.panel), {g: i for i, g in enumerate(w.panel)}
    d = os.path.join(w.dir, "runs", a.tag)
    texts = load_json(os.path.join(d, f"gen_texts_{a.split}.json"))
    tr = np.where((w.split == "train") & ~w.ctrl)[0]
    sub = np.random.default_rng(0).choice(tr, 3000, replace=False)          # the same training subsample as repr_ceiling
    L = fit_learned_rank(w.counts(sub), G, k=TOPK)
    prof = {}
    for u, ts in texts.items():
        P = np.zeros((len(ts), G))
        for j, text in enumerate(ts):
            text = text.split("\n")[0].strip()
            seen, o = set(), []
            for g in (text[:-1] if text.endswith(".") else text).split():
                if g in pos and g not in seen:
                    seen.add(g); o.append(pos[g])
            o = np.array(o[:TOPK], dtype=int)
            if len(o):
                P[j, o] = L["gene"][o, np.searchsorted(RBINS, np.arange(len(o)), side="right") - 1]
        prof[u] = P.astype(np.float32)
    out = os.path.join(w.dir, "runs", f"{a.tag}_learned")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f"profiles_{a.split}.npz"), units=np.array(sorted(prof)),
                        **{f"{u}__P": prof[u] for u in sorted(prof)})
    print(f"redecoded {len(prof)} units -> {out}", flush=True)


def score(a):
    w = World("pert", a.name)
    d = os.path.join(w.dir, "runs", a.tag)
    z = np.load(os.path.join(d, f"profiles_{a.split}.npz"), allow_pickle=True)
    prof = {str(u): z[f"{u}__P"].astype(np.float64) for u in z["units"]}
    tr = np.where((w.split == "train") & ~w.ctrl)[0]
    prior = to_dist(w.counts(tr[:20000]).sum(0), "smooth", 0.5)   # same training mean as CAD --base mean

    def qfn(u, cidx):
        P = prof[u]
        s = P.sum(1, keepdims=True)
        P = np.where(s > 0, P / np.where(s > 0, s, 1), prior)       # an empty sentence falls back to the prior
        return (1 - a.eps) * P + a.eps * prior[None]
    tag = f"{a.tag}_eps{a.eps:g}"
    eval_units(w, a.split, qfn, tag, M=a.M, min_cells=a.min_cells, seed=0, units=sorted(prof))
    print(f"scored {tag}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "gen", "score", "redecode"])
    ap.add_argument("--name", default="norman")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-train-cells", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--gen-bs", type=int, default=50)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--n-units", type=int, default=0, help="smoke test: only the first n units")
    a = ap.parse_args()
    {"train": train, "gen": gen, "score": score, "redecode": redecode}[a.cmd](a)
