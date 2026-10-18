#!/usr/bin/env bash
# CPU: rebuild the two paper UMAP figures at print width (6.5 in) so fonts match the schematic figures. Uses the saved test generations.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
mkdir -p ~/cad_out/figs_v2
python figures/umap_compare.py --task cond --name pbmc --split test --width 6.5 --height 2.6 \
  --tags "CAD (ours)=t_g_cond_s0" "Cell-type mean=cb_type_mean" --out ~/cad_out/figs_v2/umap_cond > ~/umap_cond.log 2>&1
python figures/umap_compare.py --task pert --name norman --split test --width 6.5 --height 2.6 \
  --tags "CAD (ours)=t_g_ctrl_s0" "Perturbed mean=cb_perturbed_mean" "Additive=cb_additive" --out ~/cad_out/figs_v2/umap_pert > ~/umap_pert.log 2>&1
echo DONE > ~/umap_done.txt
exit 0
