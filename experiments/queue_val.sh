#!/usr/bin/env bash
# One queued GPU script for the 15 Oct runs (VALIDATION split only; test is run once from queue_test.sh after the 22 Oct freeze).
# Run on the VM:  nohup bash ~/cad_repo/experiments/queue_val.sh > ~/queue_val.log 2>&1 < /dev/null &
# Each line is independent: a failure is logged and the queue continues. Marker files: ~/qv_done_<stage>.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
G="--model vandijklab/C2S-Scale-Gemma-2-2B --attn eager --grad-ckpt"
C="--epochs 1 --max-train-cells 20000 --eval-split val"
run() { tag=$1; shift; echo "== $(date +%T) $tag"; python -m src.train_cell "$@" --tag $tag $C > ~/ql_$tag.log 2>&1 || echo "FAILED $tag"; tail -1 ~/ql_$tag.log; }

# S1: Gemma main model (seed 0)
run g_cond_freq     --task cond --name pbmc   $G
run g_pert_ctrl     --task pert --name norman $G --base ctrl
run g_pert_noctrl   --task pert --name norman $G --base mean --no-ctrl
touch ~/qv_done_S1

# S2: representation test on the same backbone (control cell as native rank sentence vs mass levels)
run g_pert_rank_ctrl --task pert --name norman $G --base ctrl --ctrl-format rank --topk 200
run g_pert_mass_ctrl --task pert --name norman $G --base ctrl --ctrl-format mass --topk 48
touch ~/qv_done_S2

# S3: seeds 1 and 2, both backbones
for s in 1 2; do
  run q_cond_freq_s$s   --task cond --name pbmc   --seed $s
  run q_pert_ctrl_s$s   --task pert --name norman --seed $s --base ctrl
  run q_pert_noctrl_s$s --task pert --name norman --seed $s --base mean --no-ctrl
  run g_cond_freq_s$s   --task cond --name pbmc   $G --seed $s
  run g_pert_ctrl_s$s   --task pert --name norman $G --seed $s --base ctrl
  run g_pert_noctrl_s$s --task pert --name norman $G --seed $s --base mean --no-ctrl
done
touch ~/qv_done_S3

# S4: perturbation under log1p / sqrt targets (main backbone: Gemma; Qwen kept for the comparison)
for t in log1p sqrt; do
  run g_pert_noctrl_$t --task pert --name norman $G --base mean --no-ctrl --transform $t
  run g_pert_ctrl_$t   --task pert --name norman $G --base ctrl --transform $t
  run q_pert_noctrl_$t --task pert --name norman --base mean --no-ctrl --transform $t
done
touch ~/qv_done_S4

# S5: LoRA rank sensitivity (conditional task, Gemma)
for r in 8 64; do run g_cond_freq_r$r --task cond --name pbmc $G --lora-r $r; done
touch ~/qv_done_S5

# S6: scVI / scANVI (PBMC) and scGen-style latent shift (Norman), separate environment
source ~/scvi/bin/activate
python -m src.run_scvi --task cond --name pbmc   --split val > ~/ql_scvi_cond.log 2>&1 || echo "FAILED scvi cond"
python -m src.run_scvi --task pert --name norman --split val > ~/ql_scvi_pert.log 2>&1 || echo "FAILED scvi pert"
touch ~/qv_done_S6
echo ALLDONE > ~/qv_done.txt
exit 0
