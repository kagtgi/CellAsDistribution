#!/usr/bin/env bash
# rev3 E1 (experiments/PREREGISTRATION_rev3.md): frozen LM + head, and no-LM gene-embedding decoder. Val (Qwen, seed 0) then test (Gemma, seeds 0-2).
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
S=~/r3_e1.status; O=~/cad_out/auto_rev3; mkdir -p $O
log() { echo "$(date +%H:%M) $*" >> $S; }
G="--model vandijklab/C2S-Scale-Gemma-2-2B --attn eager"
N="--task pert --name norman --base mean --no-ctrl --epochs 1 --max-train-cells 20000"
run() { tag=$1; shift
  [ -f ~/cad_out/norman/runs/$tag/cell_${SPLIT}.json ] && { log "skip $tag"; return; }
  python -m src.train_cell --tag $tag $N "$@" --eval-split $SPLIT > ~/r3_$tag.log 2>&1 && log "done $tag $(grep -a 'nll_pooled=' ~/r3_$tag.log | tail -1)" || log "FAILED $tag"; }
nolm() { tag=$1; shift
  [ -f ~/cad_out/norman/runs/$tag/cell_${SPLIT}.json ] && { log "skip $tag"; return; }
  python -m src.train_nolm --tag $tag "$@" --eval-split $SPLIT > ~/r3_$tag.log 2>&1 && log "done $tag $(grep -a 'nll_pooled=' ~/r3_$tag.log | tail -1)" || log "FAILED $tag"; }
log "start"
SPLIT=val
nolm v3_nolm_s0 --seed 0
run v3_q_frozen_mlp --lora-r 0 --head-arch mlp --seed 0
run v3_q_frozen_lin --lora-r 0 --head-arch linear --seed 0
M="nll_pooled nll_mix mmd_dm auc_dm pearson_delta_pert pearson_de20 sign_de20"
V=(v2_q_noctrl v3_nolm_s0 v3_q_frozen_mlp v3_q_frozen_lin cb_perturbed_mean cb_additive)
A=(); for t in "${V[@]}"; do A+=("$t=$t"); done
python -m src.de_eval --source qbar --exclude-targets --name norman --split val --out $O --stem val_e1_downstream --runs "${A[@]}" > $O/val_e1_downstream.txt 2>&1
python -m src.strata_results --name norman --split val --out $O --stem val_e1_vs_cad --ref "v2_q_noctrl=v2_q_noctrl" \
  --runs "v3_nolm_s0=v3_nolm_s0" "v3_q_frozen_mlp=v3_q_frozen_mlp" "v3_q_frozen_lin=v3_q_frozen_lin" "cb_perturbed_mean=cb_perturbed_mean" "cb_additive=cb_additive" \
  --de-json $O/val_e1_downstream.json --metrics $M > $O/val_e1_vs_cad.txt 2>&1
log "val done"
SPLIT=test
for s in 0 1 2; do nolm t3_nolm_s$s --seed $s; done
for s in 0 1 2; do run t3_g_frozen_mlp_s$s $G --lora-r 0 --head-arch mlp --seed $s; done
T=(t2_g_noctrl_s0 t2_g_noctrl_s1 t2_g_noctrl_s2 t3_nolm_s0 t3_nolm_s1 t3_nolm_s2 t3_g_frozen_mlp_s0 t3_g_frozen_mlp_s1 t3_g_frozen_mlp_s2
   cb_perturbed_mean cb_additive cb_additive_ti)
A=(); for t in "${T[@]}"; do A+=("$t=$t"); done
python -m src.de_eval --source qbar --exclude-targets --name norman --split test --out $O --stem test_e1_downstream --runs "${A[@]}" > $O/test_e1_downstream.txt 2>&1
for s in 0 1 2; do
  python -m src.strata_results --name norman --split test --out $O --stem test_e1_vs_cad_s$s --ref "t2_g_noctrl_s$s=t2_g_noctrl_s$s" \
    --runs "t3_nolm_s$s=t3_nolm_s$s" "t3_g_frozen_mlp_s$s=t3_g_frozen_mlp_s$s" "cb_additive=cb_additive" "cb_additive_ti=cb_additive_ti" \
    --de-json $O/test_e1_downstream.json --metrics $M > $O/test_e1_vs_cad_s$s.txt 2>&1
done
python - > $O/e1_params_speed.txt 2>&1 <<'PY'
import json,os
for t in ["t2_g_noctrl_s0","t3_nolm_s0","t3_g_frozen_mlp_s0","v2_q_noctrl","v3_nolm_s0","v3_q_frozen_mlp","v3_q_frozen_lin"]:
    f=os.path.expanduser(f"~/cad_out/norman/runs/{t}/log.json")
    if os.path.exists(f):
        L=json.load(open(f)); print(t, {k:L.get(k) for k in ["trainable_params","train_seconds","decode_seconds_per_prompt","peak_vram_gb"]})
PY
log "ALL_DONE"
exit 0
