#!/usr/bin/env bash
# Stage 2+3 scheduler (replaces the val/fill/zs chain after OOMs next to C2S generation).
# Lane A now: short prompts (fit next to C2S-Gemma generation). Lane B after the C2S queue: control-text prompts.
# Then summarize, signal the zero-shot queue (which waits for ~/q2_fill.done).
Q=~/cad_repo/experiments/queue_rev2_val.sh
ONLY="v2_q_noctrl v2_q_noctrl_umi v2_q_noctrl_nb v2_q_noctrl_lowrank v2_q_desc_real v2_q_desc_other v2_q_desc_opaque" bash $Q norman
bash $Q pbmc
while ! grep -q DONE ~/c2sg.status 2>/dev/null; do sleep 120; done
ONLY="v2_q_ctrl v2_q_text_nobase v2_q_contrast v2_q_mass128 v2_q_ctrl_lowrank v2_q_ctrl_notext" bash $Q norman
bash $Q norman          # anything still missing
bash $Q pbmc
bash ~/cad_repo/experiments/summarize_rev2_val.sh
touch ~/q2_fill.done
exit 0
