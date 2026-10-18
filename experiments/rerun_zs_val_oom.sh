#!/usr/bin/env bash
# Re-run the three zero-shot validation runs that failed with OOM next to C2S generation (gate G3 was applied on the 8
# types of folds 0-1 before Stage 4; this completes the validation record). Then pool, copy, stop the VM.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
for k in 2 3 4; do
  rm -rf ~/cad_out/pbmc/runs/zs_q_def_f$k
  python -m src.train_cell --tag zs_q_def_f$k --zs-fold $k --type-text def --task cond --name pbmc --epochs 1 --max-train-cells 20000 --eval-split val --seed 0 > ~/qz_zs_q_def_f${k}_rerun.log 2>&1 || echo "FAILED rerun $k" >> ~/qz.status
done
python -m src.zs_results --split val --out ~/cad_out/auto_rev2_complete > /dev/null 2>&1; mkdir -p ~/cad_out/auto_rev2_complete
python -m src.zs_results --split val --out ~/cad_out/auto_rev2_complete > ~/cad_out/auto_rev2_complete/pbmc_zs_val_all_folds.txt 2>&1
echo DONE > ~/zs_rerun.done
sudo shutdown -h +5
exit 0
