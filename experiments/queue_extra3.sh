#!/usr/bin/env bash
# Corrected non-LM residual baseline (b_bar), then one combined test strata/DE file vs additive for the main table.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_rev2
while ! grep -q ALL_DONE ~/x2.status 2>/dev/null; do sleep 30; done
sudo shutdown -c 2>/dev/null
for s in val test; do python -m src.run_residual_baseline --split $s > ~/x3_res_$s.log 2>&1; done
T=(cb_control_mean cb_perturbed_mean cb_additive cb_ridge cb_scGen_style cb_GEARS_trctrl c2sg_s0_eps0.9 cb_additive_residual
   t2_g_noctrl_s0 t2_g_ctrl_s0 t2_g_ctrl_notext_s0 t2_g_contrast_s0 t2_g_text_nobase_s0 t2_g_noctrl_nb_s0 t2_g_noctrl_lowrank_s0
   t2_g_desc_opaque_s0 t2_q_noctrl_s0 t2_q_ctrl_s0)
A=(); for t in "${T[@]}"; do A+=("$t=$t"); done
python -m src.de_eval --source qbar --name norman --split test --out $O --stem test_all_de_qbar --runs "${A[@]}" > $O/test_all_de_qbar.txt 2>&1
R=(); for t in "${T[@]}"; do [ "$t" != cb_additive ] && R+=("$t=$t"); done
M="nll_pooled nll_mix mmd_dm auc_dm mmd_mult auc_mult pearson_delta_pert pearson_de20 sign_de20"
python -m src.strata_results --name norman --split test --out $O --stem test_all_vs_additive --ref "cb_additive=cb_additive" --runs "${R[@]}" \
  --de-json $O/test_all_de_qbar.json --metrics $M > $O/test_all_vs_additive.txt 2>&1
python -m src.strata_results --name norman --split test --out $O --stem test_residual_vs_lmresidual --ref "cb_additive_residual=cb_additive_residual" \
  --runs "t2_g_ctrl_notext_s0=t2_g_ctrl_notext_s0" "t2_g_ctrl_s0=t2_g_ctrl_s0" --de-json $O/test_all_de_qbar.json --metrics $M > $O/test_residual_vs_lmresidual.txt 2>&1
cd ~ && tar czf /tmp/rev2_results3.tgz cad_out/auto_rev2 cad_out/auto_rev2_complete x2.status 2>/dev/null
echo DONE > ~/x3.done
sudo shutdown -h +30
exit 0
