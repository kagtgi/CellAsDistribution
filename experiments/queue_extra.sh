#!/usr/bin/env bash
# Everything else that had not been run, started next to queue_test.sh (VM, unattended):
#   CPU   representation ceilings on the TEST split
#   GPU   test evaluation of the log1p / sqrt targets and of the rank-sentence control text (Gemma), LoRA-rank sensitivity on Norman (validation)
#   GEARS (separate env) trained on the train perturbations, scored on validation and test
# Marker ~/qx_done.txt when finished. A failed step is logged and the queue continues.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out CAD_DATA=~/data
G="--model vandijklab/C2S-Scale-Gemma-2-2B --attn eager --grad-ckpt"
runv() { split=$1; tag=$2; shift 2; echo "== $(date +%T) $tag"; python -m src.train_cell "$@" --tag $tag --epochs 1 --max-train-cells 20000 --eval-split $split > ~/qx_$tag.log 2>&1 || echo "FAILED $tag"; tail -1 ~/qx_$tag.log; }

python -m src.repr_ceiling --task pert --name norman --split test > ~/rc_norman_test.log 2>&1 || echo "FAILED rc norman test"
python -m src.repr_ceiling --task cond --name pbmc   --split test > ~/rc_pbmc_test.log 2>&1 || echo "FAILED rc pbmc test"

# GEARS first (its data processing needs CPU and downloads); the GPU part is small
while pgrep -f "run_gears --epochs 1$" > /dev/null; do sleep 30; done      # let the 1-epoch smoke test finish and cache the processed data
FORCE=""; grep -q "GEARS test" ~/gears_smoke.log 2>/dev/null || FORCE="--force"   # smoke failed: rebuild the cache
~/gears/bin/python -m src.run_gears --epochs 15 $FORCE > ~/qx_gears.log 2>&1 || echo "FAILED gears"
tail -2 ~/qx_gears.log | cut -c1-200

for t in log1p sqrt; do runv test t_g_noctrl_$t --task pert --name norman $G --base mean --no-ctrl --transform $t; done
for r in 8 64; do runv val v_g_noctrl_r$r --task pert --name norman $G --base mean --no-ctrl --lora-r $r; done
runv test t_g_rank_ctrl --task pert --name norman $G --base ctrl --ctrl-format rank --topk 200
echo DONE > ~/qx_done.txt
exit 0
