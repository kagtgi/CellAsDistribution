#!/usr/bin/env bash
# rev4: one combined LM-control comparison against CAD (no control cell), per seed, including the GenePT and co-expression
# encoders; run after queue_rev4.sh. Writes ~/cad_out/auto_rev4/test_r4_e1_vs_cad_s{0,1,2}.json.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_rev4
T=(cb_additive cb_additive_ti)
for s in 0 1 2; do T+=(t2_g_noctrl_s$s t3_nolm_s$s t4_nolm_genept_s$s t4_nolm_coexpr_s$s t3_g_frozen_mlp_s$s); done
A=(); for t in "${T[@]}"; do A+=("$t=$t"); done
python -m src.de_eval --source qbar --exclude-targets --name norman --split test --out $O --stem test_r4_e1_downstream --runs "${A[@]}" > $O/test_r4_e1_downstream.txt 2>&1
M="nll_pooled pearson_de20 sign_de20"
for s in 0 1 2; do
  python -m src.strata_results --name norman --split test --out $O --stem test_r4_e1_vs_cad_s$s --ref "t2_g_noctrl_s$s=t2_g_noctrl_s$s" \
    --runs "t3_nolm_s$s=t3_nolm_s$s" "t4_nolm_genept_s$s=t4_nolm_genept_s$s" "t4_nolm_coexpr_s$s=t4_nolm_coexpr_s$s" \
           "t3_g_frozen_mlp_s$s=t3_g_frozen_mlp_s$s" "cb_additive=cb_additive" "cb_additive_ti=cb_additive_ti" \
    --de-json $O/test_r4_e1_downstream.json --metrics $M > $O/test_r4_e1_vs_cad_s$s.txt 2>&1
done
python - > $O/r4_params.txt 2>&1 <<'PY'
import json, os
for t in ["t3_nolm_s0", "t4_nolm_genept_s0", "t4_nolm_coexpr_s0", "t4_tp_nolm_s0", "t4_tp_nolm_genept_s0", "t4_tp_g_noctrl_s0"]:
    for nm in ("norman", "norman_tp"):
        f = os.path.expanduser(f"~/cad_out/{nm}/runs/{t}/log.json")
        if os.path.exists(f):
            L = json.load(open(f)); print(nm, t, {k: L.get(k) for k in ("trainable_params", "train_seconds", "gene_emb")})
PY
cd ~/cad_out && tar czf /tmp/r4.tgz auto_rev4 && echo EXTRA_DONE > ~/r4x.done
exit 0
