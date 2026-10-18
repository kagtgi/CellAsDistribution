#!/usr/bin/env bash
# Stage 4: TEST split, matched control cells, configurations fixed by experiments/DECISIONS_rev2.md. Each configuration once
# per seed. Waits for stage 3 (zero-shot validation) and the C2S-Gemma validation regeneration, applies the two automatic
# rules (C2S eps, zero-shot gate G3), runs four lanes, summarizes, renders the demo figure, then stops the VM.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
O=~/cad_out/auto_rev2; mkdir -p $O
S=~/s4.status
G="--model vandijklab/C2S-Scale-Gemma-2-2B --attn eager --grad-ckpt"
B="--epochs 1 --max-train-cells 20000 --eval-split test"
log() { echo "$(date +%H:%M) $*" >> $S; }

run() {  # run <tag> <task/name> <args...>   (skips finished runs; retries on OOM when lanes share the GPU)
  tag=$1; nm=$2; shift 2
  [ -f ~/cad_out/$nm/runs/$tag/cell_test.json ] && { log "skip $tag"; return; }
  for try in 1 2 3 4; do
    python -m src.train_cell --tag $tag "$@" $B > ~/s4_$tag.log 2>&1 && break
    grep -aq OutOfMemoryError ~/s4_$tag.log || { log "FAILED $tag"; return; }
    log "OOM retry $tag"; sleep 300
  done
  log "done $tag $(grep -a 'nll_pooled=' ~/s4_$tag.log | tail -1 | cut -c1-120)"
}

# ---- wait for stage 3 and the C2S regeneration, then the automatic decisions --------------------------------------
while ! { grep -q ALL_DONE ~/qz.status 2>/dev/null && grep -q DONE ~/c2sg.status 2>/dev/null; }; do sleep 120; done
log "stage 4 start"
EPS=$(python3 - <<'PY'
import json, os, statistics as st
best = None
for e in ["0.01", "0.03", "0.1", "0.2", "0.3", "0.5", "0.7", "0.9"]:
    f = os.path.expanduser(f"~/cad_out/norman/runs/c2sg_s0_eps{e}/cell_val.json")
    if os.path.exists(f):
        v = st.mean(u["nll_pooled"] for u in json.load(open(f))["per_unit"].values())
        if best is None or v < best[1]:
            best = (e, v)
print(best[0])
PY
)
log "C2S eps chosen on validation: $EPS"
echo "$EPS" > $O/c2sg_eps_chosen.txt
python -m src.zs_baselines --split val > ~/s4_zsb_val.log 2>&1
python -m src.zs_results --split val --out $O --gate > $O/pbmc_zs_val.txt 2>&1
ZS=$(python3 -c "import json;print(json.load(open('$O/zs_gate.json'))['pass'])")
log "zero-shot gate G3: $ZS"

N="--task pert --name norman"; C="--task cond --name pbmc"
lane1() {   # control-cell text (long prompts), then C2S-Gemma test generation
  for s in 0 1 2; do run t2_g_ctrl_s$s norman $N $G --base ctrl --seed $s; done
  for s in 0 1 2; do run t2_g_contrast_s$s norman $N $G --base ctrl --ctrl-format contrast --seed $s; done
  run t2_g_text_nobase_s0 norman $N $G --base none --seed 0
  python -m src.c2s_gemma gen --tag c2sg_s0 --split test --seed 0 > ~/s4_c2sg_gen_test.log 2>&1 && log "done c2sg gen test" || log "FAILED c2sg gen test"
  python -m src.c2s_gemma score --tag c2sg_s0 --split test --eps $EPS > ~/s4_c2sg_score_test.log 2>&1 && log "done c2sg score test" || log "FAILED c2sg score test"
  log "lane1 done"
}
lane2() {
  for s in 0 1 2; do run t2_g_cond_s$s pbmc $C $G --seed $s; done
  for s in 0 1 2; do run t2_g_noctrl_s$s norman $N $G --base mean --no-ctrl --seed $s; done
  for s in 0 1 2; do run t2_g_ctrl_notext_s$s norman $N $G --base ctrl --ctrl-text off --seed $s; done
  log "lane2 done"
}
lane3() {
  for s in 0 1 2; do run t2_g_noctrl_nb_s$s norman $N $G --base mean --no-ctrl --head nb --seed $s; done
  for s in 0 1 2; do run t2_g_noctrl_lowrank_s$s norman $N $G --base mean --no-ctrl --head lowrank --seed $s; done
  for s in 0 1 2; do run t2_g_cond_lowrank_s$s pbmc $C $G --head lowrank --seed $s; done
  for s in 0 1 2; do run t2_g_desc_opaque_s$s norman $N $G --base mean --no-ctrl --pert-desc opaque --seed $s; done
  log "lane3 done"
}
lane4() {   # Qwen backbone (backbone comparison), then zero-shot test if the gate passed
  for s in 0 1 2; do
    run t2_q_cond_s$s pbmc $C --seed $s
    run t2_q_noctrl_s$s norman $N --base mean --no-ctrl --seed $s
    run t2_q_ctrl_s$s norman $N --base ctrl --seed $s
  done
  if [ "$ZS" = "True" ]; then
    python -m src.zs_baselines --split test > ~/s4_zsb_test.log 2>&1
    for k in 0 1 2 3 4; do for tt in def name opaque; do
      run t2_zs_q_${tt}_f$k pbmc $C --zs-fold $k --type-text $tt --seed 0
    done; done
  fi
  log "lane4 done"
}
lane1 & sleep 60; lane2 & sleep 60; lane3 & sleep 60; lane4 &
wait
log "all lanes done"

# ---- summaries ---------------------------------------------------------------------------------------------------
bash ~/cad_repo/experiments/summarize_stage4.sh > ~/s4_summarize.log 2>&1 && log "summaries done" || log "FAILED summaries"
echo "stage4 finished $(date)" > ~/s4_finish.txt
sudo shutdown -h +3
exit 0
