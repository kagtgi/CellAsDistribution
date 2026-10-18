#!/usr/bin/env bash
# After rerun_q_ctrl.sh: rebuild the combined test comparison (needed t2_q_ctrl) and the residual comparison, pack, stop.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_rev2
while [ ! -f ~/qc.done ]; do sleep 60; done
sudo shutdown -c 2>/dev/null; sleep 20; sudo shutdown -c 2>/dev/null
T=(cb_control_mean cb_perturbed_mean cb_additive cb_ridge cb_scGen_style cb_GEARS_trctrl c2sg_s0_eps0.9 cb_additive_residual
   t2_g_noctrl_s0 t2_g_ctrl_s0 t2_g_ctrl_notext_s0 t2_g_contrast_s0 t2_g_text_nobase_s0 t2_g_noctrl_nb_s0 t2_g_noctrl_lowrank_s0
   t2_g_desc_opaque_s0 t2_q_noctrl_s0 t2_q_ctrl_s0)
A=(); for t in "${T[@]}"; do [ -f ~/cad_out/norman/runs/$t/qbar_test.npz ] && A+=("$t=$t"); done
python -m src.de_eval --source qbar --name norman --split test --out $O --stem test_all_de_qbar --runs "${A[@]}" > $O/test_all_de_qbar.txt 2>&1
R=(); for r in "${A[@]}"; do [ "${r%%=*}" != cb_additive ] && R+=("$r"); done
M="nll_pooled nll_mix mmd_dm auc_dm mmd_mult auc_mult pearson_delta_pert pearson_de20 sign_de20"
python -m src.strata_results --name norman --split test --out $O --stem test_all_vs_additive --ref "cb_additive=cb_additive" --runs "${R[@]}" \
  --de-json $O/test_all_de_qbar.json --metrics $M > $O/test_all_vs_additive.txt 2>&1
python -m src.strata_results --name norman --split test --out $O --stem test_residual_vs_lmresidual --ref "cb_additive_residual=cb_additive_residual" \
  --runs "t2_g_ctrl_notext_s0=t2_g_ctrl_notext_s0" "t2_g_ctrl_s0=t2_g_ctrl_s0" --de-json $O/test_all_de_qbar.json --metrics $M > $O/test_residual_vs_lmresidual.txt 2>&1
python -m src.strata_results --name norman --split test --out $O --stem test_qwen_vs_gemma_ctrl --ref "t2_g_ctrl_s0=t2_g_ctrl_s0" \
  --runs "t2_q_ctrl_s0=t2_q_ctrl_s0" --metrics nll_pooled nll_mix > $O/test_qwen_vs_gemma_ctrl.txt 2>&1
cd ~ && tar czf /tmp/rev2_results5.tgz cad_out/auto_rev2 cad_out/auto_rev2_complete cad_out/figs_v2 2>/dev/null
echo DONE > ~/final_tables.done
sudo shutdown -h +40
exit 0
