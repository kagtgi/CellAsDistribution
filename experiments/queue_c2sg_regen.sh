#!/usr/bin/env bash
# B4 re-generation on validation after the parsing fix (gene names with dots were cut at the first "."), with the raw
# generated texts saved; then the eps grid (extended to 0.9). Waits for the stage-2 validation lane.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
rm -f ~/c2sg.status
while [ ! -f ~/q2_fill.done ]; do sleep 120; done
D=~/cad_out/norman/runs; mkdir -p $D/c2sg_s0/bug_truncated && mv $D/c2sg_s0/profiles_val.npz $D/c2sg_s0/gen_log_val.json $D/c2sg_s0/bug_truncated/ 2>/dev/null
for e in 0.01 0.03 0.1 0.2 0.3 0.5; do rm -rf $D/c2sg_s0_eps$e; done
python -m src.c2s_gemma gen --tag c2sg_s0 --split val --seed 0 > ~/c2sg_gen_val2.log 2>&1 || echo FAILED gen >> ~/c2sg.status
for e in 0.01 0.03 0.1 0.2 0.3 0.5 0.7 0.9; do
  python -m src.c2s_gemma score --tag c2sg_s0 --split val --eps $e > ~/c2sg_score_val_$e.log 2>&1 || echo FAILED score $e >> ~/c2sg.status
done
echo DONE >> ~/c2sg.status
exit 0
