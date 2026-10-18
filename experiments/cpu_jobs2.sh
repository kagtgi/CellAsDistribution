#!/usr/bin/env bash
# CPU: re-run the frequency-space baselines on val so that qbar_val.npz exists for the NB evaluation. Waits for cpu_jobs.sh.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
until [ -f ~/rc_done.txt ]; do sleep 20; done
python -m src.run_cell_baselines --task cond --name pbmc --split val > ~/cbv_cond.log 2>&1
python -m src.run_cell_baselines --task pert --name norman --split val > ~/cbv_pert.log 2>&1
echo DONE > ~/cbv_done.txt
exit 0
