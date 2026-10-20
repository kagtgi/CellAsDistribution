#!/usr/bin/env bash
# rev3 E3 (post hoc, CPU): representation ceiling with learned rank-to-frequency decoders, val then test.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
mkdir -p ~/cad_out/auto_rev3
for s in val test; do
  python -m src.repr_ceiling --task pert --name norman --split $s > ~/cad_out/auto_rev3/norman_${s}_ceiling_v3.txt 2>&1
  python -m src.repr_ceiling --task cond --name pbmc --split $s > ~/cad_out/auto_rev3/pbmc_${s}_ceiling_v3.txt 2>&1
done
cp ~/cad_out/auto_cell/*_repr_ceiling_v3.json ~/cad_out/auto_rev3/ 2>/dev/null
echo DONE > ~/r3_e3.done
exit 0
