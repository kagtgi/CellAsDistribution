#!/usr/bin/env bash
# After queue_test.sh: test tables (paired bootstrap), NB scores and a seed summary. Writes ~/cad_out/auto_test/.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_test; mkdir -p $O
python -m src.cell_results --task cond --name pbmc --split test --out $O --stem pbmc_test_cond --ref "Cell-type mean=cb_type_mean" \
  --runs "Global mean=cb_global_mean" "Cell-type mean=cb_type_mean" "scVI=cb_scVI" "scANVI=cb_scANVI" "CAD (Gemma)=t_g_cond_s0" "CAD (Qwen)=t_q_cond_s0" --ours "CAD" > /dev/null 2>&1
python -m src.cell_results --task pert --name norman --split test --out $O --stem norman_test_pert --ref "Perturbed mean=cb_perturbed_mean" \
  --runs "Control mean=cb_control_mean" "Perturbed mean=cb_perturbed_mean" "Additive=cb_additive" "Ridge=cb_ridge" "Control-state tilt=cb_control-state_tilt_(alpha_fit)" "scGen-style=cb_scGen_style" \
         "CAD (Gemma), no control cell=t_g_noctrl_s0" "CAD (Gemma), control residual=t_g_ctrl_s0" "CAD (Qwen), no control cell=t_q_noctrl_s0" "CAD (Qwen), control residual=t_q_ctrl_s0" --ours "CAD" > /dev/null 2>&1
python -m src.cell_results --task pert --name norman --split test --out $O --stem norman_test_pert_vs_additive --ref "Additive=cb_additive" \
  --runs "Additive=cb_additive" "scGen-style=cb_scGen_style" "CAD (Gemma), no control cell=t_g_noctrl_s0" "CAD (Gemma), control residual=t_g_ctrl_s0" "CAD (Qwen), no control cell=t_q_noctrl_s0" --ours "CAD" > /dev/null 2>&1
python -m src.cell_results --task pert --name norman --split test --out $O --stem norman_test_pert_vs_scgen --ref "scGen-style=cb_scGen_style" \
  --runs "scGen-style=cb_scGen_style" "CAD (Gemma), no control cell=t_g_noctrl_s0" "CAD (Gemma), control residual=t_g_ctrl_s0" --ours "CAD" > /dev/null 2>&1
python -m src.nb_eval --task cond --name pbmc --split test --ref cb_type_mean --tags cb_global_mean cb_type_mean cb_scVI cb_scANVI t_g_cond_s0 t_q_cond_s0 > $O/nb_cond.txt 2>&1
python -m src.nb_eval --task pert --name norman --split test --ref cb_perturbed_mean --tags cb_additive cb_ridge cb_scGen_style t_g_noctrl_s0 t_g_ctrl_s0 t_q_noctrl_s0 t_q_ctrl_s0 > $O/nb_pert.txt 2>&1
grep -a -E "nll_pooled" ~/queue_test.log | cut -c1-120 > $O/seed_runs.txt
for f in pbmc_test_cond norman_test_pert norman_test_pert_vs_additive norman_test_pert_vs_scgen; do echo "== $f"; grep -a -E "^(\\\\rowcolor\{ours\})?[A-Za-z]" $O/${f}_table.tex | grep -a -v "caption\|begin\|end\|toprule\|midrule\|bottomrule\|centering\|scriptsize\|setlength\|resizebox\|label" | cut -c1-230; done
grep -a -E "NB=|d\(NB\)|missing" $O/nb_cond.txt $O/nb_pert.txt | cut -c1-210
cat $O/seed_runs.txt
exit 0
