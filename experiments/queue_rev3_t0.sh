#!/usr/bin/env bash
# rev3 Tier 0 (post hoc, CPU): additive + target-induction baseline, then target-excluded downstream metrics vs additive.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_rev3; mkdir -p $O
for s in val test; do python -m src.run_ti_baseline --split $s > ~/r3_ti_$s.log 2>&1; done
T=(cb_perturbed_mean cb_additive cb_additive_ti cb_ridge cb_scGen_style c2sg_s0_eps0.9 cb_additive_residual
   t2_g_noctrl_s0 t2_g_ctrl_notext_s0 t2_g_noctrl_nb_s0 t2_g_noctrl_lowrank_s0 t2_g_desc_opaque_s0)
A=(); for t in "${T[@]}"; do A+=("$t=$t"); done
python -m src.de_eval --source qbar --exclude-targets --name norman --split test --out $O --stem test_downstream_qbar --runs "${A[@]}" > $O/test_downstream_qbar.txt 2>&1
# sanity: without exclusion must reproduce auto_rev2/test_all_de_qbar for shared runs
python -m src.de_eval --source qbar --name norman --split test --out $O --stem test_all_de_qbar_check --runs "cb_additive=cb_additive" "t2_g_noctrl_s0=t2_g_noctrl_s0" > $O/test_all_de_qbar_check.txt 2>&1
R=(); for t in "${T[@]}"; do [ "$t" != cb_additive ] && R+=("$t=$t"); done
M="nll_pooled nll_mix mmd_dm auc_dm pearson_delta_pert pearson_de20 sign_de20 target_abs_err target_sign"
python -m src.strata_results --name norman --split test --out $O --stem test_downstream_vs_additive --ref "cb_additive=cb_additive" --runs "${R[@]}" \
  --de-json $O/test_downstream_qbar.json --metrics $M > $O/test_downstream_vs_additive.txt 2>&1
python -m src.strata_results --name norman --split test --out $O --stem test_downstream_vs_ti --ref "cb_additive_ti=cb_additive_ti" \
  --runs "t2_g_noctrl_s0=t2_g_noctrl_s0" "t2_g_noctrl_nb_s0=t2_g_noctrl_nb_s0" "t2_g_ctrl_notext_s0=t2_g_ctrl_notext_s0" \
  --de-json $O/test_downstream_qbar.json --metrics $M > $O/test_downstream_vs_ti.txt 2>&1
echo DONE > ~/r3_t0.done
exit 0
