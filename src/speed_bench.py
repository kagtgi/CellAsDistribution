"""Decoding cost, measured alone on the GPU with the same batch size (decision head vs autoregressive generation).

python -m src.speed_bench --out ~/cad_out/auto_rev2/speed_bench.json

Speed does not depend on trained weights, so CAD is timed with the C2S-Scale-Gemma-2-2B backbone, fresh LoRA r16 adapters
and a linear head over the 2048 panel genes (forward pass + softmax). Three CAD prompt types, all built by train_cell.prompt
from real test units and real control cells: no control cell (one prompt = one cell state), control cell as mass text,
and control cell as residual only (prompt without control text). C2S: the trained C2S-Gemma adapter (runs/c2sg_s0/adapter),
sampling one perturbed-cell sentence (top-k 50, top-p 0.95, up to 1024 new tokens) per control-cell prompt.
Same batch size for both; torch.cuda.synchronize around each batch; 3 repeats; median reported.
"""
import unsloth  # noqa: F401
import argparse
import json
import os
import time
import zlib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

from .cell_eval import World
from .cell_dist import serialize_cell
from .train_cell import prompt, CTRL_LINE
from .c2s_gemma import PROMPT as C2S_PROMPT, TOPK, sentence_from_counts

MODEL = "vandijklab/C2S-Scale-Gemma-2-2B"


def timed(fn, reps=3):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.time()
        fn()
        torch.cuda.synchronize(); out.append(time.time() - t)
    return float(np.median(out)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=25)
    ap.add_argument("--n", type=int, default=100, help="prompts per CAD condition")
    ap.add_argument("--n-c2s", type=int, default=50, help="generated sentences for C2S")
    a = ap.parse_args()
    torch.manual_seed(0)
    w = World("pert", "norman")
    units = w.units("test", 30)
    pool = np.where(w.ctrl & (w.split == "test"))[0]
    rng = np.random.default_rng(0)
    pick_u = [units[i % len(units)] for i in range(a.n)]
    C = w.counts(rng.choice(pool, a.n, replace=False))
    prompts = {
        "no control cell": [prompt("pert", u) for u in pick_u],
        "control cell as mass text": [prompt("pert", u, serialize_cell(c, w.panel, 48), CTRL_LINE["mass"]) for u, c in zip(pick_u, C)],
        "control cell as residual only": [prompt("pert", u) for u in pick_u],
    }
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    base = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda()
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=16, target_modules="all-linear", lora_dropout=0.0)).eval()
    head = nn.Linear(model.config.hidden_size, len(w.panel)).cuda()
    res = {"gpu": torch.cuda.get_device_name(0), "batch_size": a.bs, "cad": {}, "c2s": {}}
    logb = torch.log(torch.full((a.bs, len(w.panel)), 1.0 / len(w.panel), device="cuda"))

    @torch.no_grad()
    def cad(ps, residual):
        for s in range(0, len(ps), a.bs):
            enc = tok(ps[s:s + a.bs], return_tensors="pt", padding=True).to("cuda")
            h = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            last = enc["attention_mask"].sum(1) - 1
            z = head(h[torch.arange(h.size(0)), last].float())
            if residual:
                z = z + logb[: z.size(0)]
            torch.softmax(z, -1)
    cad(prompts["no control cell"][: a.bs], False)                     # warm-up
    for name, ps in prompts.items():
        med, all_ = timed(lambda: cad(ps, name != "no control cell"))
        ntok = float(np.mean([len(tok(p)["input_ids"]) for p in ps]))
        res["cad"][name] = {"seconds_per_prompt": med / len(ps), "repeats_seconds": all_, "prompt_tokens_mean": ntok}
        print(name, res["cad"][name], flush=True)
    del model, base
    torch.cuda.empty_cache()

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gbase = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda()
    gen = PeftModel.from_pretrained(gbase, os.path.join(w.dir, "runs", "c2sg_s0", "adapter")).merge_and_unload().eval()
    cps = []
    for u, c in zip(pick_u[: a.n_c2s], C[: a.n_c2s]):
        cs = sentence_from_counts(c, w.panel, TOPK)
        cps.append(C2S_PROMPT.format(n=len(cs), pert=u, ctrl=" ".join(cs)))
    lens = []

    @torch.no_grad()
    def c2s():
        for s in range(0, len(cps), a.bs):
            enc = tok(cps[s:s + a.bs], return_tensors="pt", padding=True).to("cuda")
            out = gen.generate(**enc, max_new_tokens=1024, do_sample=True, top_k=50, top_p=0.95, pad_token_id=tok.pad_token_id)
            lens.extend((out[:, enc["input_ids"].shape[1]:] != tok.pad_token_id).sum(1).tolist())
    med, all_ = timed(c2s)
    res["c2s"] = {"seconds_per_generated_cell": med / len(cps), "repeats_seconds": all_, "new_tokens_mean": float(np.mean(lens)),
                  "prompt_tokens_mean": float(np.mean([len(tok(p)["input_ids"]) for p in cps]))}
    print("c2s", res["c2s"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
