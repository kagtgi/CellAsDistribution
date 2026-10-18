#!/usr/bin/env bash
# B4: autoregressive C2S on C2S-Scale-Gemma-2-2B, LoRA r16, budget matched to CAD. Train once (seed 0), generate on val,
# choose eps on val (grid), then generate on test with the saved adapter (run later by queue_rev2_test.sh after the gate).
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
python -m src.c2s_gemma train --tag c2sg_s0 --seed 0 --micro 2 > ~/c2sg_train.log 2>&1 || echo FAILED train >> ~/c2sg.status
python -m src.c2s_gemma gen --tag c2sg_s0 --split val --seed 0 > ~/c2sg_gen_val.log 2>&1 || echo FAILED gen >> ~/c2sg.status
for e in 0.01 0.03 0.1 0.2 0.3 0.5; do
  python -m src.c2s_gemma score --tag c2sg_s0 --split val --eps $e > ~/c2sg_score_val_$e.log 2>&1 || echo FAILED score $e >> ~/c2sg.status
done
echo DONE >> ~/c2sg.status
exit 0
