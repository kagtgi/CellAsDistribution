#!/usr/bin/env bash
# rev4 (experiments/PREREGISTRATION_rev4.md): zero-shot with Gemma, no-LM with GenePT / co-expression embeddings,
# add-on Norman panel with every perturbed gene, CAD + target prior. Ends by summarizing and shutting the VM down.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
S=~/r4.status; O=~/cad_out/auto_rev4; mkdir -p $O
log() { echo "$(date +%H:%M) $*" >> $S; }
G="--model vandijklab/C2S-Scale-Gemma-2-2B --attn eager --grad-ckpt"
B="--epochs 1 --max-train-cells 20000 --eval-split test"
tc() { tag=$1; shift; nm=$1; shift
  [ -f ~/cad_out/$nm/runs/$tag/cell_test.json ] && { log "skip $tag"; return; }
  python -m src.train_cell --tag $tag "$@" $B > ~/r4_$tag.log 2>&1 && log "done $tag $(grep -a 'nll_pooled=' ~/r4_$tag.log | tail -1 | cut -c1-110)" || log "FAILED $tag"; }
nl() { tag=$1; shift; nm=$1; shift
  [ -f ~/cad_out/$nm/runs/$tag/cell_test.json ] && { log "skip $tag"; return; }
  python -m src.train_nolm --name $nm --tag $tag "$@" --eval-split test > ~/r4_$tag.log 2>&1 && log "done $tag $(grep -a 'nll_pooled=' ~/r4_$tag.log | tail -1 | cut -c1-90)" || log "FAILED $tag"; }
log start

# ---- GenePT download (Zenodo 10833191, 574 MB) ----
if [ ! -f ~/genept/GenePT_gene_embedding_ada_text.pickle ]; then
  mkdir -p ~/genept && cd ~/genept && curl -sL -o genept.zip "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1" \
    && echo "3f6ce4317e3a0091978ae5cb8fbf05a3  genept.zip" | md5sum -c - >> $S 2>&1 && unzip -o -j -q genept.zip "*GenePT_gene_embedding_ada_text.pickle" && log "genept ok"
  cd ~/cad_repo
fi

# ---- R4-4: CAD + target prior (post hoc, CPU) ----
python -m src.run_ti_baseline --name norman --split test --base-run t2_g_noctrl_s0 --mode all --tag t2_g_noctrl_s0_tprior > ~/r4_tprior.log 2>&1 && log "done tprior"

# ---- R4-2: no-LM with GenePT / co-expression, main panel, test seeds 0-2 ----
for s in 0 1 2; do nl t4_nolm_genept_s$s norman --gene-emb genept --seed $s; nl t4_nolm_coexpr_s$s norman --gene-emb coexpr --seed $s; done

# ---- R4-3: add-on panel with all perturbed genes ----
if [ ! -f ~/cad_out/norman_tp/panel.json ]; then
  python -m src.preprocess --h5ad $CAD_DATA/NormanWeissman2019_filtered.h5ad --name norman_tp --pert-col perturbation --force-all-targets > ~/r4_prep.log 2>&1 && log "prep ok"
fi
python - >> $S 2>&1 <<'PY'
import pandas as pd, os, json
a = pd.read_csv(os.path.expanduser("~/cad_out/norman/obs.csv"), index_col=0); b = pd.read_csv(os.path.expanduser("~/cad_out/norman_tp/obs.csv"), index_col=0)
same = a.index.equals(b.index) and (a["split"].values == b["split"].values).all()
p = json.load(open(os.path.expanduser("~/cad_out/norman_tp/panel.json")))
print("splits identical:", same, "| targets in panel:", p["n_targets_in_panel"])
if not same: raise SystemExit("SPLIT MISMATCH")
PY
grep -q "splits identical: True" $S || { log "STOP: split mismatch"; sudo shutdown -h +5; exit 0; }
python -m src.run_cell_baselines --task pert --name norman_tp --split test > ~/r4_tp_base.log 2>&1 && log "done tp baselines"
python -m src.run_ti_baseline --name norman_tp --split test > ~/r4_tp_ti.log 2>&1 && log "done tp ti"
nl t4_tp_nolm_s0 norman_tp --seed 0
nl t4_tp_nolm_genept_s0 norman_tp --gene-emb genept --seed 0
nl t4_tp_nolm_coexpr_s0 norman_tp --gene-emb coexpr --seed 0
tc t4_tp_g_noctrl_s0 norman_tp --task pert --name norman_tp $G --base mean --no-ctrl --seed 0

# ---- R4-1: zero-shot cell types with Gemma (test) ----
for k in 0 1 2 3 4; do for tt in def name opaque; do tc t2_zs_g_${tt}_f$k pbmc --task cond --name pbmc $G --zs-fold $k --type-text $tt --seed 0; done; done

# ---- summaries ----
M="nll_pooled nll_mix mmd_dm auc_dm pearson_delta_pert pearson_de20 sign_de20 target_abs_err target_sign"
T=(cb_additive cb_additive_ti t2_g_noctrl_s0 t2_g_noctrl_s0_tprior t2_g_noctrl_nb_s0 t3_nolm_s0 t4_nolm_genept_s0 t4_nolm_coexpr_s0)
A=(); for t in "${T[@]}"; do A+=("$t=$t"); done
python -m src.de_eval --source qbar --exclude-targets --name norman --split test --out $O --stem test_r4_downstream --runs "${A[@]}" > $O/test_r4_downstream.txt 2>&1
R=(); for t in "${T[@]}"; do [ "$t" != cb_additive ] && R+=("$t=$t"); done
python -m src.strata_results --name norman --split test --out $O --stem test_r4_vs_additive --ref "cb_additive=cb_additive" --runs "${R[@]}" --de-json $O/test_r4_downstream.json --metrics $M > $O/test_r4_vs_additive.txt 2>&1
for s in 0 1 2; do
  python -m src.strata_results --name norman --split test --out $O --stem test_r4_emb_vs_nolm_s$s --ref "t3_nolm_s$s=t3_nolm_s$s" \
    --runs "t4_nolm_genept_s$s=t4_nolm_genept_s$s" "t4_nolm_coexpr_s$s=t4_nolm_coexpr_s$s" "t2_g_noctrl_s$s=t2_g_noctrl_s$s" --metrics nll_pooled > $O/test_r4_emb_vs_nolm_s$s.txt 2>&1
done
TP=(cb_additive cb_perturbed_mean cb_additive_ti t4_tp_nolm_s0 t4_tp_nolm_genept_s0 t4_tp_nolm_coexpr_s0 t4_tp_g_noctrl_s0)
A=(); for t in "${TP[@]}"; do A+=("$t=$t"); done
python -m src.de_eval --source qbar --exclude-targets --name norman_tp --split test --out $O --stem tp_downstream --runs "${A[@]}" > $O/tp_downstream.txt 2>&1
R=(); for t in "${TP[@]}"; do [ "$t" != cb_additive ] && R+=("$t=$t"); done
python -m src.strata_results --name norman_tp --split test --out $O --stem tp_vs_additive --ref "cb_additive=cb_additive" --runs "${R[@]}" --de-json $O/tp_downstream.json --metrics $M > $O/tp_vs_additive.txt 2>&1
python -m src.zs_results --split test --prefix t2_ --out $O > $O/pbmc_zs_test_r4.txt 2>&1
log ALL_DONE
sudo shutdown -h +10
exit 0
