#!/usr/bin/env bash
# Stage 2, VALIDATION only (decisions; see experiments/PREREGISTRATION_rev2.md). Qwen3.5-0.8B exploration, seed 0,
# matched control cells. Comparators are re-run here so every paired comparison uses the same control draws.
# usage: bash queue_rev2_val.sh norman|pbmc   (run the two serially when another GPU job is running)
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
B="--epochs 1 --max-train-cells 20000 --eval-split val --seed 0"
run() { tag=$1; shift; [ -f ~/cad_out/$NAME/runs/$tag/cell_val.json ] && { echo "skip $tag" >> ~/q2_$NAME.status; return; }
        [ -n "$ONLY" ] && [[ " $ONLY " != *" $tag "* ]] && return     # optional subset: ONLY="tag1 tag2"
        for try in 1 2 3; do                       # retry when the GPU is shared and a run hits OOM
          python -m src.train_cell --tag $tag "$@" $B > ~/q2_$tag.log 2>&1 && break
          grep -aq OutOfMemoryError ~/q2_$tag.log || { echo "FAILED $tag" >> ~/q2_$NAME.status; break; }
          echo "OOM retry $tag" >> ~/q2_$NAME.status; sleep 600
        done
        echo "done $tag $(grep -a 'nll_pooled=' ~/q2_$tag.log | tail -1)" >> ~/q2_$NAME.status; }
NAME=$1
if [ "$NAME" = norman ]; then
  P="--task pert --name norman"
  run v2_q_ctrl            $P --base ctrl                                    # comparator (current CAD, control residual)
  run v2_q_noctrl          $P --base mean --no-ctrl                          # comparator (current CAD, no control cell)
  run v2_q_ctrl_notext     $P --base ctrl --ctrl-text off                    # B1a
  run v2_q_text_nobase     $P --base none                                    # B1b
  run v2_q_contrast        $P --base ctrl --ctrl-format contrast             # B2
  run v2_q_mass128         $P --base ctrl --topk 128                         # B2 (more genes)
  run v2_q_noctrl_umi      $P --base mean --no-ctrl --weight umi             # B5
  run v2_q_noctrl_nb       $P --base mean --no-ctrl --head nb                # B6
  run v2_q_ctrl_lowrank    $P --base ctrl --head lowrank                     # B7
  run v2_q_noctrl_lowrank  $P --base mean --no-ctrl --head lowrank           # B7
  run v2_q_desc_real       $P --base mean --no-ctrl --pert-desc real         # B3
  run v2_q_desc_other      $P --base mean --no-ctrl --pert-desc other        # B3
  run v2_q_desc_opaque     $P --base mean --no-ctrl --pert-desc opaque       # B3
else
  P="--task cond --name pbmc"
  run v2_q_cond            $P                                                # comparator
  run v2_q_cond_umi        $P --weight umi                                   # B5
  run v2_q_cond_nb         $P --head nb                                      # B6
  run v2_q_cond_lowrank    $P --head lowrank                                 # B7
fi
echo ALL_DONE >> ~/q2_$NAME.status
exit 0
