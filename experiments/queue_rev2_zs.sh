#!/usr/bin/env bash
# Stage 3 (VALIDATION): zero-shot cell types on PBMC, 5 leave-types-out folds, Qwen exploration, seed 0.
# Waits for the stage-2 validation lane to finish (one GPU lane next to the C2S job).
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
while [ ! -f ~/q2_fill.done ]; do sleep 120; done
python -m src.zs_baselines --split val > ~/qz_baselines_val.log 2>&1 && echo "done baselines val" >> ~/qz.status || echo "FAILED baselines val" >> ~/qz.status
B="--task cond --name pbmc --epochs 1 --max-train-cells 20000 --eval-split val --seed 0"
for k in 0 1 2 3 4; do
  for tt in def name opaque; do
    tag=zs_q_${tt}_f$k
    [ -f ~/cad_out/pbmc/runs/$tag/cell_val.json ] && continue
    python -m src.train_cell --tag $tag --zs-fold $k --type-text $tt $B > ~/qz_$tag.log 2>&1 || echo "FAILED $tag" >> ~/qz.status
    echo "done $tag $(grep -a 'nll_pooled=' ~/qz_$tag.log | tail -1)" >> ~/qz.status
  done
done
echo ALL_DONE >> ~/qz.status
exit 0
