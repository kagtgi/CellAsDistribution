#!/usr/bin/env bash
# Wait for every queue, summarize, then stop the VM (billing). Cap: 12 h.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
for i in $(seq 1 360); do
  grep -q ALL_DONE ~/qz.status 2>/dev/null && grep -q DONE ~/c2sg.status 2>/dev/null && grep -q ALL_DONE ~/qb.status 2>/dev/null && [ -f ~/q2_fill.done ] && break
  sleep 120
done
bash ~/cad_repo/experiments/summarize_rev2_val.sh
echo "finished $(date)" > ~/rev2_finish.txt
sudo shutdown -h +2
exit 0
