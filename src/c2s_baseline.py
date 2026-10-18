"""C2S baseline for the perturbation task, following Cell2Sentence tutorial 10 (perturbation response prediction).

Protocol (tutorial): control cell sentence (top-K genes, descending expression) + perturbation name -> perturbed cell
sentence; each perturbed train cell is paired with a random control cell; fine-tune C2S-Scale-Pythia-1b-pt with loss on
the response only. Differences we state in the paper:
  * sentences use the SAME 2,048-gene panel as every other model (not the full transcriptome);
  * held-out PERTURBATIONS (tutorial splits cells at random, leaking perturbations into test);
  * C2S has no likelihood: generated sentences are reconstructed to expression via the tutorial's rank->expression
    linear fit (log expression vs log rank), averaged over generated cells into a frequency vector, and mixed with the
    train-control prior (eps) so every gene has support. Scored by the identical code as CAD (src/eval_common.score_model).

python -m src.c2s_baseline --name norman --tag c2s_1b --steps 1500 --n-eval-perts 0 --n-gen 20
"""
import unsloth  # noqa: F401  (import first; keeps the shared torch/peft stack consistent)
import argparse
import os
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import SEED, load_json, save_json, split_pert
from .eval_common import load_world, score_model
from . import baselines as B

TOPK = 200
PROMPT = ("Given the following cell sentence of {n} expressed genes representing a cell's basal state, predict the cell "
          "sentence after applying the perturbation: {pert}.\nControl cell sentence: {ctrl}.\n\nPerturbed cell sentence:")


def sentence_from_counts(c, panel, k=TOPK):
    """Top-k nonzero genes by library-normalised expression, descending (ties broken by panel order)."""
    c = np.asarray(c, dtype=np.float64)
    nz = np.where(c > 0)[0]
    order = nz[np.argsort(-c[nz], kind="stable")][:k]
    return [panel[i] for i in order]


def fit_rank_expression(X, panel, idx, k=TOPK, n=3000, seed=0):
    """Tutorial's reconstruction: linear fit log1p(CP10k) ~ a + b*log(rank+1) on train cells (top-k genes)."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(idx, min(n, len(idx)), replace=False)
    xs, ys = [], []
    for i in rows:
        c = np.asarray(X[i].todense()).ravel()
        o = np.argsort(-c, kind="stable")[:k]
        v = np.log1p(c[o] / c.sum() * 1e4)
        keep = c[o] > 0
        xs.append(np.log(np.arange(1, len(o) + 1)[keep]))
        ys.append(v[keep])
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    b, a = np.polyfit(xs, ys, 1)
    return float(a), float(b)


def profile_from_sentence(genes, pos, a, b, G):
    seen, v = set(), np.zeros(G)
    r = 0
    for g in genes:
        if g in pos and g not in seen:
            seen.add(g)
            r += 1
            v[pos[g]] = max(np.expm1(a + b * np.log(r)), 0.0)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pert-col", default="perturbation")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Pythia-1b-pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--ga", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-gen", type=int, default=20)
    ap.add_argument("--n-eval-perts", type=int, default=0, help="0 = all validation perts; else random subset")
    ap.add_argument("--max-new", type=int, default=800)
    ap.add_argument("--eps", type=float, default=0.3, help="weight of train-control prior mixed into the C2S profile")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eval-split", default="val")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d, X, obs, panel, lab, ctrl, split = load_world(a.name, a.pert_col)
    G, pos = len(panel), {g: i for i, g in enumerate(panel)}
    tr = np.where(split == "train")[0]
    tr_ctrl = np.where(ctrl & (split == "train"))[0]
    tr_pert = np.where(~ctrl & (split == "train"))[0]
    run = os.path.join(d, "runs", a.tag)
    os.makedirs(run, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    # fp32 master weights + bf16 autocast: bf16 weights would swallow lr=1e-5 updates
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    def ctrl_sentence():
        i = int(rng.choice(tr_ctrl))
        return sentence_from_counts(np.asarray(X[i].todense()).ravel(), panel)

    def pert_name(p):
        return "+".join(split_pert(p)) or "control"

    # ---- fine-tune: loss on the response only (as tutorial: loss_on_response_only=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    model.train()
    hist, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for step in range(a.steps):
        for _ in range(a.ga):
            idx = rng.choice(tr_pert, a.bs, replace=False)
            prompts, fulls = [], []
            for i in idx:
                cs = ctrl_sentence()
                ps = sentence_from_counts(np.asarray(X[int(i)].todense()).ravel(), panel)
                pr = PROMPT.format(n=len(cs), pert=pert_name(obs[a.pert_col].iloc[int(i)]), ctrl=" ".join(cs))
                prompts.append(pr)
                fulls.append(pr + " " + " ".join(ps) + ".")
            enc = tok(fulls, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda")
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            for j, pr in enumerate(prompts):
                labels[j, : len(tok(pr)["input_ids"])] = -100
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc, labels=labels).loss / a.ga
            loss.backward()
            hist.append(float(loss) * a.ga)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            print(f"step {step}/{a.steps} loss {np.mean(hist[-a.ga:]):.4f}", flush=True)
    train_s = time.time() - t0
    model.eval(); model.config.use_cache = True
    model.save_pretrained(os.path.join(run, "ckpt"))

    # ---- generation -> profile
    ra, rb = fit_rank_expression(X, panel, tr)
    f_ctrl = B.profile(np.asarray(X[tr_ctrl].todense()))
    perts_all = sorted(p for p in set(lab[~ctrl & (split == a.eval_split)]) if ((lab == p) & (split == a.eval_split)).sum() >= 20)
    if a.n_eval_perts and len(perts_all) > a.n_eval_perts:
        keep = set(rng.choice(perts_all, a.n_eval_perts, replace=False).tolist())
    else:
        keep = set(perts_all)
    cache, diag = {}, {"gen_valid_frac": [], "gen_len": []}

    @torch.no_grad()
    def q_fn(p):
        if p in cache:
            return cache[p]
        prompts = []
        for _ in range(a.n_gen):
            cs = ctrl_sentence()
            prompts.append(PROMPT.format(n=len(cs), pert=pert_name(p), ctrl=" ".join(cs)))
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, top_k=50, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        tok.padding_side = "right"
        prof = np.zeros(G)
        for o in out:
            text = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().rstrip(".")
            genes = text.split()
            diag["gen_len"].append(len(genes))
            diag["gen_valid_frac"].append(float(np.mean([g in pos for g in genes])) if genes else 0.0)
            prof += profile_from_sentence(genes, pos, ra, rb, G)
        prof = prof / max(prof.sum(), 1e-12) if prof.sum() > 0 else f_ctrl
        q = (1 - a.eps) * prof + a.eps * f_ctrl
        cache[p] = q / q.sum()
        return cache[p]

    summ = score_model(a.name, a.tag, q_fn, a.eval_split, seed=a.seed, pert_col=a.pert_col, perts_subset=sorted(keep))
    log = {"args": vars(a), "train_seconds": train_s, "peak_vram_gb": torch.cuda.max_memory_reserved() / 1e9,
           "loss_history": hist, "rank_fit": [ra, rb], "gen_valid_frac_mean": float(np.mean(diag["gen_valid_frac"])) if diag["gen_valid_frac"] else None,
           "gen_len_mean": float(np.mean(diag["gen_len"])) if diag["gen_len"] else None,
           "n_generated_perts": len(keep)}
    save_json(log, os.path.join(run, "log.json"))
    print(f"{a.tag}: pearson_delta_pert={summ['pearson_delta_pert'][0]:+.3f} energy={summ['energy'][0]:.3f} "
          f"centroid_acc={summ['centroid_acc'][0]:.2f} mae={summ['mae'][0]:.4f} | valid-gene frac {log['gen_valid_frac_mean']}",
          flush=True)


if __name__ == "__main__":
    main()
