#!/usr/bin/env bash
# Re-run the two Gemma runs that finished before eval_units saved qbar_<split>.npz (same config and seed as g_cond_freq / g_pert_ctrl),
# then score them with the NB evaluation. Runs next to queue_val.sh (about 9 GB VRAM each).
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
G="--model vandijklab/C2S-Scale-Gemma-2-2B --attn eager --grad-ckpt"
C="--epochs 1 --max-train-cells 20000 --eval-split val"
python -m src.train_cell --task cond --name pbmc   $G --tag g_cond_freq_rr $C > ~/ql_g_cond_freq_rr.log 2>&1 || echo "FAILED cond"
python -m src.train_cell --task pert --name norman $G --base ctrl --tag g_pert_ctrl_rr $C > ~/ql_g_pert_ctrl_rr.log 2>&1 || echo "FAILED pert"
python -m src.nb_eval --task cond --name pbmc --split val --ref cb_type_mean --tags cb_global_mean cb_type_mean g_cond_freq_rr > ~/nb_cond.log 2>&1
python -m src.nb_eval --task pert --name norman --split val --ref cb_perturbed_mean --tags cb_control_mean cb_perturbed_mean cb_additive cb_ridge g_pert_ctrl_rr > ~/nb_pert.log 2>&1
echo DONE > ~/nb_done.txt
exit 0
