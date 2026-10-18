#!/usr/bin/env bash
# Stage 4 summaries on the TEST split (matched controls). Writes ~/cad_out/auto_rev2/test_* and the demo figure.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_rev2
EPS=$(cat $O/c2sg_eps_chosen.txt 2>/dev/null)
have() { [ -f ~/cad_out/$1/runs/$2/cell_test.json ]; }
add() { local nm=$1; shift; for r in "$@"; do have $nm "${r#*=}" && echo "$r"; done; }

# ---- PBMC ----
mapfile -t P < <(add pbmc "Global mean=cb_global_mean" "Cell-type mean=cb_type_mean" "scVI=cb_scVI" "scANVI=cb_scANVI" \
  "scVI, own NB sampling=cb_scVI_native_s0" "scANVI, own NB sampling=cb_scANVI_native_s0" "Training-cell resampling=cb_train_cell_resampling" \
  "CAD (Gemma)=t2_g_cond_s0" "CAD (Qwen)=t2_q_cond_s0" "CAD (Gemma), low-rank head=t2_g_cond_lowrank_s0")
python -m src.cell_results --task cond --name pbmc --split test --out $O --stem test_pbmc --ref "Cell-type mean=cb_type_mean" --runs "${P[@]}" > $O/test_pbmc.txt 2>&1

# ---- Norman ----
mapfile -t R < <(add norman "Control mean=cb_control_mean" "Perturbed mean=cb_perturbed_mean" "Additive=cb_additive" "Ridge=cb_ridge" \
  "Control-state tilt=cb_control-state_tilt_(alpha_fit)" "scGen-style=cb_scGen_style" "GEARS=cb_GEARS_trctrl" \
  "Training-cell resampling=cb_train_cell_resampling" "C2S (Gemma, autoregressive)=c2sg_s0_eps$EPS" \
  "CAD (Gemma), no control cell=t2_g_noctrl_s0" "CAD (Gemma), control residual, mass text=t2_g_ctrl_s0" \
  "CAD (Gemma), control residual, no text=t2_g_ctrl_notext_s0" "CAD (Gemma), control residual, contrast text=t2_g_contrast_s0" \
  "CAD (Gemma), control text, no residual=t2_g_text_nobase_s0" "CAD (Gemma), NB head=t2_g_noctrl_nb_s0" \
  "CAD (Gemma), low-rank head=t2_g_noctrl_lowrank_s0" "CAD (Gemma), opaque gene ids=t2_g_desc_opaque_s0" \
  "CAD (Qwen), no control cell=t2_q_noctrl_s0" "CAD (Qwen), control residual=t2_q_ctrl_s0")
python -m src.cell_results --task pert --name norman --split test --out $O --stem test_norman --ref "Perturbed mean=cb_perturbed_mean" --runs "${R[@]}" > $O/test_norman.txt 2>&1
TAGS=(); for r in "${R[@]}"; do TAGS+=("${r#*=}=${r#*=}"); done
python -m src.de_eval --source qbar --name norman --split test --out $O --stem test_norman_de_qbar --runs "${TAGS[@]}" > $O/test_norman_de_qbar.txt 2>&1
M="nll_pooled nll_mix mmd_dm auc_dm mmd_mult auc_mult pearson_delta_pert pearson_de20 sign_de20 nll_nb_own"
cmp() { ref=$1; stem=$2; shift 2; L=(); for t in "$@"; do have norman $t && L+=("$t=$t"); done
        [ ${#L[@]} -gt 0 ] && have norman $ref && python -m src.strata_results --name norman --split test --out $O --stem $stem \
          --ref "$ref=$ref" --runs "${L[@]}" --de-json $O/test_norman_de_qbar.json --metrics $M > $O/$stem.txt 2>&1; }
cmp cb_perturbed_mean test_strata_vs_pertmean cb_additive cb_scGen_style cb_GEARS_trctrl t2_g_noctrl_s0 t2_g_ctrl_s0 t2_g_ctrl_notext_s0 c2sg_s0_eps$EPS
cmp cb_additive test_strata_vs_additive cb_scGen_style t2_g_noctrl_s0 t2_g_ctrl_s0 t2_g_ctrl_notext_s0 t2_g_contrast_s0 t2_g_noctrl_nb_s0 t2_g_noctrl_lowrank_s0 c2sg_s0_eps$EPS t2_q_noctrl_s0
cmp cb_scGen_style test_strata_vs_scgen t2_g_noctrl_s0 t2_g_ctrl_notext_s0 t2_g_noctrl_lowrank_s0
cmp t2_g_ctrl_s0 test_strata_vs_ctrl t2_g_ctrl_notext_s0 t2_g_text_nobase_s0 t2_g_contrast_s0
cmp t2_g_noctrl_s0 test_strata_vs_noctrl t2_g_noctrl_nb_s0 t2_g_noctrl_lowrank_s0 t2_g_desc_opaque_s0 c2sg_s0_eps$EPS

# ---- seeds: per configuration, the seed-0..2 means and the paired difference to additive / cell-type mean per seed ----
python3 - > $O/test_seeds.txt <<'PY'
import json, os, statistics as st
R = os.path.expanduser("~/cad_out")
def per(nm, tag):
    f = f"{R}/{nm}/runs/{tag}/cell_test.json"
    return json.load(open(f))["per_unit"] if os.path.exists(f) else None
ref = {"norman": per("norman", "cb_additive"), "pbmc": per("pbmc", "cb_type_mean")}
for nm, stem in [("pbmc", "t2_g_cond"), ("pbmc", "t2_q_cond"), ("pbmc", "t2_g_cond_lowrank"), ("norman", "t2_g_noctrl"),
                 ("norman", "t2_q_noctrl"), ("norman", "t2_g_ctrl"), ("norman", "t2_q_ctrl"), ("norman", "t2_g_ctrl_notext"),
                 ("norman", "t2_g_contrast"), ("norman", "t2_g_noctrl_nb"), ("norman", "t2_g_noctrl_lowrank"), ("norman", "t2_g_desc_opaque")]:
    rows = []
    for s in (0, 1, 2):
        p = per(nm, f"{stem}_s{s}")
        if p is None:
            continue
        u = sorted(set(p) & set(ref[nm]))
        rows.append((s, st.mean(p[x]["nll_pooled"] for x in u), st.mean(p[x]["nll_mix"] for x in u),
                     st.mean(p[x]["nll_pooled"] - ref[nm][x]["nll_pooled"] for x in u), st.mean(p[x]["mmd_dm"] for x in u)))
    if rows:
        print(stem, " | ".join(f"s{s}: nll={a:.4f} cell={b:.4f} d_ref={c:+.4f} mmd={d:.4f}" for s, a, b, c, d in rows),
              f"| mean nll={st.mean(r[1] for r in rows):.4f} sd={st.pstdev([r[1] for r in rows]):.4f}")
PY

# ---- zero-shot ----
[ -f ~/cad_out/pbmc/runs/t2_zs_q_def_f0/cell_test.json ] && python -m src.zs_results --split test --prefix t2_ --out $O > $O/pbmc_zs_test.txt 2>&1

# ---- decoder speed (seconds per prompt, from the run logs) and C2S generation speed ----
python3 - > $O/test_speed.txt <<'PY'
import json, os, glob
R = os.path.expanduser("~/cad_out")
for f in sorted(glob.glob(f"{R}/*/runs/t2_*_s0/log.json")):
    d = json.load(open(f))
    if "decode_seconds_per_prompt" in d:
        print(os.path.basename(os.path.dirname(f)), "decode_s_per_prompt", round(d["decode_seconds_per_prompt"], 5), "train_s", round(d["train_seconds"]))
for f in glob.glob(f"{R}/norman/runs/c2sg_s0/gen_log_*.json") + glob.glob(f"{R}/norman/runs/c2sg_s0/train_log.json"):
    d = json.load(open(f)); print(os.path.basename(f), {k: v for k, v in d.items() if k in ("seconds_per_cell", "gen_len_mean", "valid_gene_frac_mean", "train_seconds")})
PY

# ---- demo figure (median unit per case) ----
D=(--case "cond|PBMC, conditional generation|pbmc|t2_g_cond_s0|cb_type_mean|all"
   --case "pert|K562, no control cell, unseen single gene|norman|t2_g_noctrl_s0|cb_additive|s"
   --case "pert|K562, no control cell, pair with one unseen gene|norman|t2_g_noctrl_s0|cb_additive|pu"
   --case "pert|K562, no control cell, pair of two seen genes|norman|t2_g_noctrl_s0|cb_additive|p"
   --case "pert|K562, control-cell residual, pair of two seen genes|norman|t2_g_ctrl_notext_s0|cb_additive|p")
[ -f ~/cad_out/pbmc/runs/t2_zs_q_def_f0/cell_test.json ] && D+=(--case "cond|PBMC, zero-shot cell type (fold 0)|pbmc|t2_zs_q_def_f0|cb_zs_ontology_f0|all")
python figures/demo_outputs.py --out ~/cad_out/figs_v2/demo_outputs --split test "${D[@]}" > $O/demo_outputs.txt 2>&1
exit 0
