#!/usr/bin/env bash
# Smoke tests of the additional options (Qwen, 640 training cells = 40 steps). Not results.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
S="--max-train-cells 640 --eval-split val"
run() { tag=$1; shift; python -m src.train_cell --tag smoke_$tag "$@" $S > ~/smoke_$tag.log 2>&1 || echo FAILED $tag >> ~/smoke_rev2.status; echo "done $tag: $(grep -a "nll_pooled=" ~/smoke_$tag.log | tail -1)" >> ~/smoke_rev2.status; }
run b1a  --task pert --name norman --base ctrl --ctrl-text off
run b2   --task pert --name norman --base ctrl --ctrl-format contrast
run b3   --task pert --name norman --base mean --no-ctrl --pert-desc real
run b3o  --task pert --name norman --base mean --no-ctrl --pert-desc opaque
run b5   --task cond --name pbmc --weight umi
run b6   --task cond --name pbmc --head nb
run b7   --task cond --name pbmc --head lowrank
run b7p  --task pert --name norman --base ctrl --head lowrank
echo ALL_DONE >> ~/smoke_rev2.status
exit 0
