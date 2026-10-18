"""CAD-N / CAD-R: native Unsloth decision model on the gene tree. Train, generate, evaluate.

python -m src.train_native --name norman --tree tree.json --tag cadn_08b --epochs 1 --bag-size 20 --max-cells 40000
Writes OUT_DIR/<name>/runs/<tag>/{metrics.json, q_by_pert.npz, log.json}
Generation (md 4.3): q(g|pert) = q(module|pert) * q(gene|module,pert) from decision passes (the state holds
only the perturbation, so q is per perturbation); cells = Multinomial(s, q), s from train library sizes.
"""
import unsloth  # noqa: F401  (must be imported first)
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from transformers import TrainingArguments
from unsloth import FastDecisionModel, DecisionTrainer

from .common import SEED, OUT_DIR, load_json, save_json, split_pert
from .records import make_records, _context
from .evaluate import Scorer, summarize
from . import baselines as B
from .preprocess import is_control


def q_for_pert(model, tok, tree, pert_state, panel_pos, group=8, prior=None):
    """Full gene distribution for one state: root pass + all leaf questions in groups."""
    mods = list(tree["modules"].keys())
    root_q = {"type": "choice", "instructions": "Which gene module does the next transcript come from?",
              "criteria": {k: tree["desc"][k] for k in mods}}
    out = FastDecisionModel.predict(model, tok, pert_state, {"module": root_q})
    p_mod = np.array([out["module"]["probabilities"][k] for k in mods], dtype=np.float64)
    q = np.zeros(len(panel_pos))
    for s in range(0, len(mods), group):
        qs = {}
        for k in mods[s:s + group]:
            qs[f"leaf_{k}"] = {"type": "choice", "instructions": f"Which gene of module {k}?",
                               "criteria": {g: "" for g in tree["modules"][k]}}
        res = FastDecisionModel.predict(model, tok, pert_state, qs)
        for k in mods[s:s + group]:
            probs = res[f"leaf_{k}"]["probabilities"]
            w = np.array([probs[g] for g in tree["modules"][k]], dtype=np.float64)
            w = w / w.sum()
            for g, wi in zip(tree["modules"][k], w):
                q[panel_pos[g]] = p_mod[mods.index(k)] * wi
    q = q / q.sum()
    if prior is not None:               # tilt mode: tree predicts p/prior; multiply the prior back in
        q = prior * q
        q = q / q.sum()
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pert-col", default="perturbation")
    ap.add_argument("--tree", default="tree.json")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--bag-size", type=int, default=20)
    ap.add_argument("--max-cells", type=int, default=40000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eval-split", default="val")   # tune on val only; test is frozen until the metrics freeze
    ap.add_argument("--tilt", action="store_true", help="residual-on-prior: tree predicts p/f_ctrl, q = f_ctrl * tilt")
    ap.add_argument("--shrink", type=float, default=2000.0, help="empirical-Bayes pseudo-UMIs toward the prior (tilt mode)")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--wd", type=float, default=0.01)
    a = ap.parse_args()

    d = os.path.join(OUT_DIR, a.name)
    run = os.path.join(d, "runs", a.tag)
    os.makedirs(run, exist_ok=True)
    X = sp.load_npz(os.path.join(d, "counts.npz")).tocsr()
    obs = pd.read_csv(os.path.join(d, "obs.csv"), index_col=0)
    tree = load_json(os.path.join(d, a.tree))
    panel = load_json(os.path.join(d, "panel.json"))["panel"]
    panel_pos = {g: i for i, g in enumerate(panel)}
    log = {"args": vars(a)}

    prior = None
    if a.tilt:
        sp_ = obs["split"].values
        is_c = np.array([is_control(p) for p in obs[a.pert_col].astype(str)])
        prior = B.profile(np.asarray(X[np.where(is_c & (sp_ == "train"))[0]].todense()))   # train controls only
    train = make_records(X, obs, tree, panel, a.pert_col, "train", a.name, R=a.R, bag_size=a.bag_size,
                         max_cells=a.max_cells, seed=a.seed, prior=prior, shrink=a.shrink)
    log["n_train_records"] = len(train)
    model, tok = FastDecisionModel.from_pretrained(a.model, max_seq_length=a.max_len, load_in_4bit=a.load_4bit)
    model = FastDecisionModel.get_peft_model(model, r=16, lora_alpha=16, use_gradient_checkpointing="unsloth",
                                             random_state=a.seed)
    model.decision_config["max_len"] = a.max_len
    items, report = FastDecisionModel.build_dataset([{k: r[k] for k in ("state", "questions", "gold")} for r in train], tok, model)[:2]
    log["build_report"] = report
    print("BUILD", report, "items", len(items), flush=True)
    args = TrainingArguments(output_dir=os.path.join(run, "ckpt"), per_device_train_batch_size=2,
                             gradient_accumulation_steps=8, num_train_epochs=a.epochs, learning_rate=a.lr,
                             logging_steps=5, save_strategy="no", report_to="none", bf16=True, warmup_steps=5,
                             lr_scheduler_type="cosine", seed=a.seed, weight_decay=a.wd)
    trainer = DecisionTrainer(model=model, args=args, train_dataset=items, tokenizer=tok)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    log["train_seconds"] = time.time() - t0
    log["peak_vram_gb"] = torch.cuda.max_memory_reserved() / 1e9
    log["loss_history"] = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print("TRAINED", log["train_seconds"], log["peak_vram_gb"], flush=True)
    try:
        model.save_pretrained(os.path.join(run, "adapter"))
        tok.save_pretrained(os.path.join(run, "adapter"))
    except Exception as e:  # saving must never lose the run
        print("adapter save failed:", repr(e), flush=True)

    # ---- generation + evaluation on held-out perturbations (same scorer/sampler as baselines)
    from .run_baselines import norm_label
    lab = obs[a.pert_col].astype(str).map(norm_label).values
    ctrl = np.array([is_control(p) or l == "control" for p, l in zip(obs[a.pert_col].astype(str), lab)])
    split = obs["split"].values
    cnt = lambda m: np.asarray(X[np.where(m)[0]].todense())
    from .eval_common import sample_rows
    tr_ctrl, tr_pert = cnt(ctrl & (split == "train")), sample_rows(cnt, ~ctrl & (split == "train"))
    scorer = Scorer(tr_ctrl, tr_pert)
    perts = sorted(p for p in set(lab[~ctrl & (split == a.eval_split)]) if ((lab == p) & (split == a.eval_split)).sum() >= 20)
    real = {p: cnt((lab == p) & (split == a.eval_split)) for p in perts}
    lib = np.asarray(X[np.where(split == "train")[0]].sum(1)).ravel()
    rng = np.random.default_rng(a.seed)
    qs, pred = {}, {}
    t1 = time.time()
    FastDecisionModel.for_inference(model)      # as in the Unsloth Decision notebook
    for p in perts:
        state = {"context": _context(a.name), "perturbation": "+".join(split_pert(p))}
        qs[p] = q_for_pert(model, tok, tree, state, panel_pos, prior=prior)
        pred[p] = B.sample_cells(qs[p], a.n_gen, lib, rng)
    # control state (most-trained): separates "cannot fit" from "cannot generalise to unseen perturbations"
    q_control = q_for_pert(model, tok, tree, {"context": _context(a.name), "perturbation": "control"}, panel_pos, prior=prior)
    log["gen_seconds"] = time.time() - t1
    per = scorer.score_all(real, pred)
    summ = {m: list(v) for m, v in summarize(per).items()}
    log["sum_q_min_max"] = [float(min(q.sum() for q in qs.values())), float(max(q.sum() for q in qs.values()))]
    save_json({"n_test_perts": len(perts), "models": {a.tag: summ}, "per_pert": {a.tag: per}}, os.path.join(run, "metrics.json"))
    np.savez_compressed(os.path.join(run, "q_by_pert.npz"), perts=np.array(perts), q=np.stack([qs[p] for p in perts]), q_control=q_control)
    save_json(log, os.path.join(run, "log.json"))
    print(f"{a.tag}: pearson_delta_pert={summ['pearson_delta_pert'][0]:+.3f} energy={summ['energy'][0]:.3f} "
          f"centroid_acc={summ['centroid_acc'][0]:.2f} mae={summ['mae'][0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
