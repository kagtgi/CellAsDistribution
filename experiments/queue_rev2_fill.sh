#!/usr/bin/env bash
# After the stage-2 validation lanes: re-run any run that failed (e.g. OOM while the GPU was shared); existing runs are
# skipped by queue_rev2_val.sh. Then signal the zero-shot queue.
while ! grep -q ALL_DONE ~/q2_pbmc.status 2>/dev/null; do sleep 120; done
bash ~/cad_repo/experiments/queue_rev2_val.sh norman
bash ~/cad_repo/experiments/queue_rev2_val.sh pbmc
bash ~/cad_repo/experiments/summarize_rev2_val.sh
touch ~/q2_fill.done
exit 0
