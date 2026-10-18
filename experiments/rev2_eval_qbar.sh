#!/usr/bin/env bash
# Pseudobulk (no sampling) delta/DE check of the stage-1 result, plus strata on it.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_rev2
T='Perturbed mean=cb_perturbed_mean'
R=("Control mean=cb_control_mean" "Additive=cb_additive" "Ridge=cb_ridge" "scGen-style=cb_scGen_style" "GEARS=cb_GEARS"
   "CAD (Gemma), no control cell=t_g_noctrl_s0" "CAD (Gemma), control residual=t_g_ctrl_s0"
   "CAD (Qwen), no control cell=t_q_noctrl_s0" "CAD (Qwen), control residual=t_q_ctrl_s0")
python -m src.de_eval --source qbar --name norman --split test --out $O --stem norman_test_de_qbar --runs "$T" "${R[@]}" > $O/norman_test_de_qbar.txt 2>&1
python -m src.strata_results --name norman --split test --out $O --stem norman_test_strata_qbar_vs_additive --ref "Additive=cb_additive" --runs "${R[@]:3}" \
  --de-json $O/norman_test_de_qbar.json --metrics nll_pooled pearson_delta_ctrl pearson_delta_pert pearson_de20 sign_de20 prec_de20 > $O/norman_test_strata_qbar_vs_additive.txt 2>&1
echo DONE > $O/rev2_eval_qbar.done
exit 0
