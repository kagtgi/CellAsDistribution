#!/usr/bin/env bash
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
python -m src.c2s_gemma train --tag smoke_c2sg --max-train-cells 64 --bs 16 --micro 4 > ~/smoke_c2sg.log 2>&1
python -m src.c2s_gemma gen --tag smoke_c2sg --split val --n-units 2 --M 10 >> ~/smoke_c2sg.log 2>&1
python -m src.c2s_gemma score --tag smoke_c2sg --split val --M 10 --eps 0.1 >> ~/smoke_c2sg.log 2>&1
echo SMOKE_DONE >> ~/smoke_c2sg.log
exit 0
