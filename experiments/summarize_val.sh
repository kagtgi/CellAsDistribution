#!/usr/bin/env bash
# Validation summary after queue_val.sh: paired tables (cell_results) and NB scores for all runs that saved qbar_val.npz.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_rev; mkdir -p $O
python -m src.cell_results --task cond --name pbmc --split val --out $O --stem val_cond --ref "Cell-type mean=cb_type_mean" \
  --runs "Global mean=cb_global_mean" "Cell-type mean=cb_type_mean" "scVI=cb_scVI" "scANVI=cb_scANVI" "CAD Gemma=g_cond_freq" "CAD Qwen s1=q_cond_freq_s1" > /dev/null 2>&1
python -m src.cell_results --task pert --name norman --split val --out $O --stem val_pert --ref "Perturbed mean=cb_perturbed_mean" \
  --runs "Perturbed mean=cb_perturbed_mean" "Additive=cb_additive" "Ridge=cb_ridge" "scGen-style=cb_scGen_style" "CAD Gemma no ctrl=g_pert_noctrl" "CAD Gemma ctrl mass=g_pert_ctrl" "CAD Gemma ctrl rank=g_pert_rank_ctrl" "CAD Qwen no ctrl s1=q_pert_noctrl_s1" > /dev/null 2>&1
python -m src.cell_results --task pert --name norman --split val --out $O --stem val_pert_rank_vs_mass --ref "mass text=g_pert_ctrl" \
  --runs "mass text=g_pert_ctrl" "rank text=g_pert_rank_ctrl" > /dev/null 2>&1
python -m src.cell_results --task pert --name norman --split val --out $O --stem val_pert_transform --ref "frequency=g_pert_noctrl" \
  --runs "frequency=g_pert_noctrl" "log1p=g_pert_noctrl_log1p" "sqrt=g_pert_noctrl_sqrt" > /dev/null 2>&1
python -m src.nb_eval --task cond --name pbmc --split val --ref cb_type_mean --tags cb_global_mean cb_type_mean cb_scVI cb_scANVI g_cond_freq_rr q_cond_freq_s1 > $O/nb_cond.txt 2>&1
python -m src.nb_eval --task pert --name norman --split val --ref cb_perturbed_mean --tags cb_additive cb_ridge cb_scGen_style g_pert_noctrl g_pert_ctrl_rr g_pert_rank_ctrl q_pert_noctrl_s1 >  $O/nb_pert.txt 2>&1
grep -a -E "NB=|d\(NB\)|missing" $O/nb_cond.txt $O/nb_pert.txt | cut -c1-210
for s in val_cond val_pert val_pert_rank_vs_mass val_pert_transform; do echo "== $s"; grep -a -E "^(\\\\rowcolor\{ours\})?[A-Za-z]" $O/${s}_table.tex | grep -a -v "caption\|begin\|end\|toprule\|midrule\|bottomrule\|centering\|scriptsize\|setlength\|resizebox\|label" | cut -c1-200; done
exit 0
