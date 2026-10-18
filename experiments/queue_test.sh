#!/usr/bin/env bash
# TEST split, each chosen configuration evaluated once per seed (seed 0 of the Qwen configurations was evaluated on 14 Oct and is re-run
# here only to save qbar_test.npz for the NB table; the numbers are identical because training is deterministic for a given seed).
# Configurations were chosen on validation (queue_val.sh). Run on the VM:
#   nohup bash ~/cad_repo/experiments/queue_test.sh > ~/queue_test.log 2>&1 < /dev/null &
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
G="--model vandijklab/C2S-Scale-Gemma-2-2B --attn eager --grad-ckpt"
C="--epochs 1 --max-train-cells 20000 --eval-split test"
run() { tag=$1; shift; echo "== $(date +%T) $tag"; python -m src.train_cell "$@" --tag $tag $C > ~/qt_$tag.log 2>&1 || echo "FAILED $tag"; tail -1 ~/qt_$tag.log; }

# frequency-space baselines (also write qbar_test.npz for the NB table)
python -m src.run_cell_baselines --task cond --name pbmc   --split test > ~/qt_cb_cond.log 2>&1 || echo "FAILED cb cond"
python -m src.run_cell_baselines --task pert --name norman --split test > ~/qt_cb_pert.log 2>&1 || echo "FAILED cb pert"

# seed 0 of both backbones first, then seeds 1 and 2
for s in 0 1 2; do
  run t_g_cond_s$s   --task cond --name pbmc   $G --seed $s
  run t_g_noctrl_s$s --task pert --name norman $G --seed $s --base mean --no-ctrl
  run t_g_ctrl_s$s   --task pert --name norman $G --seed $s --base ctrl
  run t_q_cond_s$s   --task cond --name pbmc   --seed $s
  run t_q_noctrl_s$s --task pert --name norman --seed $s --base mean --no-ctrl
  run t_q_ctrl_s$s   --task pert --name norman --seed $s --base ctrl
done
touch ~/qt_done_decoders

# NB-likelihood baselines (separate environment)
source ~/scvi/bin/activate
python -m src.run_scvi --task cond --name pbmc   --split test > ~/qt_scvi_cond.log 2>&1 || echo "FAILED scvi cond"
python -m src.run_scvi --task pert --name norman --split test > ~/qt_scvi_pert.log 2>&1 || echo "FAILED scvi pert"
echo ALLDONE > ~/qt_done.txt
exit 0
