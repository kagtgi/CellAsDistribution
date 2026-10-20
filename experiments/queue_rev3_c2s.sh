#!/usr/bin/env bash
# rev3 (post hoc): C2S-Gemma sentences re-decoded with the learned rank decoder; eps chosen on validation, test once.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_rev3; S=~/r3_c2s.status
for s in val test; do python -m src.c2s_gemma redecode --tag c2sg_s0 --split $s > ~/r3_c2s_redecode_$s.log 2>&1; done
for e in 0.01 0.03 0.1 0.2 0.3 0.5 0.7 0.9; do python -m src.c2s_gemma score --tag c2sg_s0_learned --split val --eps $e > ~/r3_c2s_val_$e.log 2>&1; done
EPS=$(python3 - <<'PY'
import json, os, statistics as st
best = None
for e in ["0.01", "0.03", "0.1", "0.2", "0.3", "0.5", "0.7", "0.9"]:
    f = os.path.expanduser(f"~/cad_out/norman/runs/c2sg_s0_learned_eps{e}/cell_val.json")
    if os.path.exists(f):
        v = st.mean(u["nll_pooled"] for u in json.load(open(f))["per_unit"].values())
        if best is None or v < best[1]:
            best = (e, v)
print(best[0])
PY
)
echo "eps chosen on val: $EPS" > $S
python -m src.c2s_gemma score --tag c2sg_s0_learned --split test --eps $EPS > ~/r3_c2s_test.log 2>&1
python -m src.de_eval --source qbar --exclude-targets --name norman --split test --out $O --stem test_c2s_learned_de \
  --runs "c2sg_s0_eps0.9=c2sg_s0_eps0.9" "c2sg_s0_learned_eps$EPS=c2sg_s0_learned_eps$EPS" "t2_g_noctrl_s0=t2_g_noctrl_s0" "cb_additive=cb_additive" "cb_perturbed_mean=cb_perturbed_mean" > $O/test_c2s_learned_de.txt 2>&1
python -m src.strata_results --name norman --split test --out $O --stem test_c2s_learned_vs_cad --ref "t2_g_noctrl_s0=t2_g_noctrl_s0" \
  --runs "c2sg_s0_eps0.9=c2sg_s0_eps0.9" "c2sg_s0_learned_eps$EPS=c2sg_s0_learned_eps$EPS" "cb_perturbed_mean=cb_perturbed_mean" \
  --de-json $O/test_c2s_learned_de.json --metrics nll_pooled nll_mix mmd_dm auc_dm pearson_delta_pert pearson_de20 sign_de20 > $O/test_c2s_learned_vs_cad.txt 2>&1
echo DONE >> $S
exit 0
