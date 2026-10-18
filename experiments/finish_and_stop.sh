#!/usr/bin/env bash
# Runs on the VM, independent of any laptop: wait for queue_test.sh, build the test tables/NB/seed summary, then power the VM off
# (a stopped instance stops compute billing; disks and ~/cad_out stay). A 10 h cap stops the VM even if the queue stalls.
cd ~
START=$(date +%s)
while [ ! -f ~/qt_done.txt ]; do
  if [ $(( $(date +%s) - START )) -gt 36000 ]; then echo "TIMEOUT after 10 h" > ~/finish_status.txt; break; fi
  sleep 60
done
echo "queue ended $(date +%T)" >> ~/finish_status.txt
bash ~/cad_repo/experiments/summarize_test.sh > ~/summarize_test.log 2>&1
echo "summary written $(date +%T)" >> ~/finish_status.txt
touch ~/ALL_DONE
sleep 20
sudo shutdown -h now
