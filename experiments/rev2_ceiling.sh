#!/usr/bin/env bash
# Representation ceiling with the matched-frequency estimator (CPU), test and validation, both datasets.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
for s in test val; do
  python -m src.repr_ceiling --task pert --name norman --split $s > ~/cad_out/auto_rev2/norman_${s}_ceiling_v2.txt 2>&1
  python -m src.repr_ceiling --task cond --name pbmc --split $s > ~/cad_out/auto_rev2/pbmc_${s}_ceiling_v2.txt 2>&1
done
echo DONE > ~/cad_out/auto_rev2/ceiling_v2.done
exit 0
