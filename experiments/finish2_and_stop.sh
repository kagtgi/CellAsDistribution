#!/usr/bin/env bash
# Replaces finish_and_stop.sh: wait for BOTH queue_test.sh (~/qt_done.txt) and queue_extra.sh (~/qx_done.txt), build all summaries, then power the VM off.
# 14 h cap so a stalled run cannot keep the instance billing.
cd ~
START=$(date +%s)
while [ ! -f ~/qt_done.txt ] || [ ! -f ~/qx_done.txt ]; do
  if [ $(( $(date +%s) - START )) -gt 50400 ]; then echo "TIMEOUT after 14 h" > ~/finish_status.txt; break; fi
  sleep 60
done
echo "queues ended $(date +%T)" >> ~/finish_status.txt
bash ~/cad_repo/experiments/summarize_test.sh > ~/summarize_test.log 2>&1
bash ~/cad_repo/experiments/summarize_extra.sh > ~/summarize_extra.log 2>&1
echo "summaries written $(date +%T)" >> ~/finish_status.txt
touch ~/ALL_DONE
sleep 20
sudo shutdown -h now
