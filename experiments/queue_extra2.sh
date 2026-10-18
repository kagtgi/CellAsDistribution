#!/usr/bin/env bash
# Advisor-requested checks before writing: zero-shot name-overlap and CL-parent baselines, the non-LM additive residual,
# and a clean decoding-speed benchmark (GPU alone). Then pack results and stop the VM.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_rev2; S=~/x2.status
while [ ! -f ~/zs_rerun.done ]; do sleep 60; done
sudo shutdown -c 2>/dev/null; sleep 30; sudo shutdown -c 2>/dev/null
for s in val test; do python -m src.zs_baselines --split $s > ~/x2_zsb_$s.log 2>&1 && echo "done zsb $s" >> $S || echo "FAILED zsb $s" >> $S; done
python -m src.zs_results --split val --out $O > $O/pbmc_zs_val_all.txt 2>&1
python -m src.zs_results --split test --prefix t2_ --out $O > $O/pbmc_zs_test.txt 2>&1
for s in val test; do python -m src.run_residual_baseline --split $s > ~/x2_res_$s.log 2>&1 && echo "done residual $s" >> $S || echo "FAILED residual $s" >> $S; done
R=(cb_perturbed_mean cb_additive cb_additive_residual t2_g_ctrl_s0 t2_g_ctrl_notext_s0 cb_scGen_style)
A=(); for t in "${R[@]}"; do A+=("$t=$t"); done
python -m src.de_eval --source qbar --name norman --split test --out $O --stem test_residual_de_qbar --runs "${A[@]}" > $O/test_residual_de_qbar.txt 2>&1
python -m src.strata_results --name norman --split test --out $O --stem test_strata_residual --ref "cb_additive_residual=cb_additive_residual" \
  --runs "t2_g_ctrl_notext_s0=t2_g_ctrl_notext_s0" "t2_g_ctrl_s0=t2_g_ctrl_s0" "cb_scGen_style=cb_scGen_style" "cb_additive=cb_additive" \
  --de-json $O/test_residual_de_qbar.json --metrics nll_pooled nll_mix mmd_dm auc_dm mmd_mult auc_mult pearson_delta_pert pearson_de20 > $O/test_strata_residual.txt 2>&1
python -m src.speed_bench --out $O/speed_bench.json > $O/speed_bench.txt 2>&1 && echo "done speed" >> $S || echo "FAILED speed" >> $S
cd ~ && tar czf /tmp/rev2_results2.tgz cad_out/auto_rev2 cad_out/auto_rev2_complete x2.status 2>/dev/null
echo ALL_DONE >> $S
sudo shutdown -h +45
exit 0
