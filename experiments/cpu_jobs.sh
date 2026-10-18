#!/usr/bin/env bash
# CPU-only jobs (run next to the GPU queue): representation ceiling on both tasks.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
python -m src.repr_ceiling --task pert --name norman --split val > ~/rc_norman.log 2>&1
python -m src.repr_ceiling --task cond --name pbmc --split val > ~/rc_pbmc.log 2>&1
echo DONE > ~/rc_done.txt
exit 0
