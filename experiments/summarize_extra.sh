#!/usr/bin/env bash
# After queue_extra.sh: tables for the extra runs and the GEARS comparison. Writes ~/cad_out/auto_test/.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_test; mkdir -p $O
R=~/cad_out/norman/runs
have() { [ -f $R/$1/cell_$2.json ]; }
have t_g_noctrl_log1p test && python -m src.cell_results --task pert --name norman --split test --out $O --stem norman_test_transform --ref "frequency=t_g_noctrl_s0" \
  --runs "frequency=t_g_noctrl_s0" "log1p=t_g_noctrl_log1p" "sqrt=t_g_noctrl_sqrt" --ours "frequency" > /dev/null 2>&1
have t_g_rank_ctrl test && python -m src.cell_results --task pert --name norman --split test --out $O --stem norman_test_rank_vs_mass --ref "mass text=t_g_ctrl_s0" \
  --runs "mass text=t_g_ctrl_s0" "rank text=t_g_rank_ctrl" --ours "rank" > /dev/null 2>&1
have v_g_noctrl_r8 val && python -m src.cell_results --task pert --name norman --split val --out $O --stem val_pert_lora --ref "rank 16=g_pert_noctrl" \
  --runs "rank 16=g_pert_noctrl" "rank 8=v_g_noctrl_r8" "rank 64=v_g_noctrl_r64" --ours "rank" > /dev/null 2>&1
GOK=0; grep -q "GEARS test:" ~/qx_gears.log 2>/dev/null && [ ~/cad_out/norman/runs/cb_GEARS/cell_test.json -nt ~/gears_smoke.log ] && GOK=1   # only the full 15-epoch run, never the 1-epoch smoke
if [ $GOK = 1 ] && have cb_GEARS val; then
  python -m src.cell_results --task pert --name norman --split val --out $O --stem val_pert_gears --ref "Perturbed mean=cb_perturbed_mean" \
    --runs "Perturbed mean=cb_perturbed_mean" "Additive=cb_additive" "scGen-style=cb_scGen_style" "GEARS=cb_GEARS" "CAD (Gemma), no control cell=g_pert_noctrl" --ours "CAD" > /dev/null 2>&1
fi
if [ $GOK = 1 ] && have cb_GEARS test; then
  python -m src.cell_results --task pert --name norman --split test --out $O --stem norman_test_gears --ref "Perturbed mean=cb_perturbed_mean" \
    --runs "Perturbed mean=cb_perturbed_mean" "Additive=cb_additive" "scGen-style=cb_scGen_style" "GEARS=cb_GEARS" "CAD (Gemma), no control cell=t_g_noctrl_s0" --ours "CAD" > /dev/null 2>&1
  python -m src.nb_eval --task pert --name norman --split test --ref cb_perturbed_mean --tags cb_additive cb_ridge cb_scGen_style cb_GEARS t_g_noctrl_s0 > $O/nb_pert_gears.txt 2>&1
fi
for f in norman_test_transform norman_test_rank_vs_mass val_pert_lora val_pert_gears norman_test_gears; do
  [ -f $O/${f}_table.tex ] && { echo "== $f"; grep -a -E "^(\\\\rowcolor\{ours\})?[A-Za-z]" $O/${f}_table.tex | grep -a -v "caption\|begin\|end\|toprule\|midrule\|bottomrule\|centering\|scriptsize\|setlength\|resizebox\|label" | cut -c1-230; }
done
grep -a -E "NB=|d\(NB\)" $O/nb_pert_gears.txt 2>/dev/null | cut -c1-200
grep -a -E "^  (frequency|rank|bins|levels)|^(frequency|rank|bins|levels)" ~/rc_norman_test.log ~/rc_pbmc_test.log 2>/dev/null | cut -c1-200
grep -a "GEARS" ~/qx_gears.log | cut -c1-200
bash ~/cad_repo/experiments/umap_rebuild.sh > ~/umap_final.log 2>&1
exit 0
