#!/usr/bin/env bash
# Re-score every baseline with matched control cells (unit-seeded draws), add real-cell resampling references,
# native scVI/scANVI sampling on test, and GEARS trained with training control cells only. Baselines have no tuning.
cd ~/cad_repo && export CAD_OUT=~/cad_out
st() { echo "$1" >> ~/qb.status; }
source ~/cad/bin/activate
for s in val test; do
  python -m src.run_cell_baselines --task pert --name norman --split $s > ~/qb_cb_norman_$s.log 2>&1 && st "done cb norman $s" || st "FAILED cb norman $s"
  python -m src.run_resample_baseline --task pert --name norman --split $s > ~/qb_rs_norman_$s.log 2>&1 && st "done rs norman $s" || st "FAILED rs norman $s"
  python -m src.run_resample_baseline --task cond --name pbmc --split $s > ~/qb_rs_pbmc_$s.log 2>&1 && st "done rs pbmc $s" || st "FAILED rs pbmc $s"
done
source ~/scvi/bin/activate
for s in val test; do
  python -m src.run_scvi --task pert --name norman --split $s > ~/qb_scgen_$s.log 2>&1 && st "done scgen $s" || st "FAILED scgen $s"
done
python -m src.run_scvi_native --name pbmc --split test --seed 0 > ~/qb_scvi_native_test.log 2>&1 && st "done scvi native test" || st "FAILED scvi native test"
~/gears/bin/python -m src.run_gears --epochs 15 --train-ctrl-only --force > ~/qb_gears_trctrl.log 2>&1 && st "done gears trctrl" || st "FAILED gears trctrl"
st ALL_DONE
exit 0
