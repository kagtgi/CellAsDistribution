#!/usr/bin/env bash
# Stage 2 decisions on VALIDATION (pre-registered comparisons, experiments/PREREGISTRATION_rev2.md).
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_rev2; mkdir -p $O
have() { [ -f ~/cad_out/$1/runs/$2/cell_val.json ]; }
N=()
for t in v2_q_ctrl v2_q_noctrl v2_q_ctrl_notext v2_q_text_nobase v2_q_contrast v2_q_mass128 v2_q_noctrl_umi v2_q_noctrl_nb \
         v2_q_ctrl_lowrank v2_q_noctrl_lowrank v2_q_desc_real v2_q_desc_other v2_q_desc_opaque; do have norman $t && N+=("$t=$t"); done
python -m src.de_eval --source qbar --name norman --split val --out $O --stem norman_val_de_qbar \
  --runs "Perturbed mean=cb_perturbed_mean" "Additive=cb_additive" "${N[@]}" > $O/norman_val_de_qbar.txt 2>&1
M="nll_pooled nll_mix mmd_dm auc_dm mmd_mult auc_mult pearson_delta_pert pearson_de20 nll_nb_own"
cmp() { ref=$1; stem=$2; shift 2; R=(); for t in "$@"; do have norman $t && R+=("$t=$t"); done
        [ ${#R[@]} -gt 0 ] && have norman $ref && python -m src.strata_results --name norman --split val --out $O --stem $stem \
          --ref "$ref=$ref" --runs "${R[@]}" --de-json $O/norman_val_de_qbar.json --metrics $M > $O/$stem.txt 2>&1; }
cmp v2_q_ctrl   norman_val_vs_ctrl   v2_q_ctrl_notext v2_q_text_nobase v2_q_contrast v2_q_mass128 v2_q_ctrl_lowrank
cmp v2_q_noctrl norman_val_vs_noctrl v2_q_noctrl_umi v2_q_noctrl_nb v2_q_noctrl_lowrank v2_q_desc_real v2_q_desc_other v2_q_desc_opaque
cmp v2_q_desc_opaque norman_val_desc_vs_opaque v2_q_desc_real v2_q_desc_other
cmp cb_additive norman_val_vs_additive v2_q_noctrl v2_q_ctrl
P=(); for t in v2_q_cond_umi v2_q_cond_nb v2_q_cond_lowrank; do have pbmc $t && P+=("$t=$t"); done
[ ${#P[@]} -gt 0 ] && python -m src.cell_results --task cond --name pbmc --split val --out $O --stem pbmc_val_rev2 \
  --ref "v2_q_cond=v2_q_cond" --runs "Cell-type mean=cb_type_mean" "v2_q_cond=v2_q_cond" "${P[@]}" --ours v2 > $O/pbmc_val_rev2.txt 2>&1
echo DONE > $O/summarize_rev2_val.done
exit 0
