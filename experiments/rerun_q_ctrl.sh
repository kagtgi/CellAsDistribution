#!/usr/bin/env bash
# The three Qwen control-residual test runs exhausted their OOM retries during stage 4 (logged "done" without results).
# Re-run them alone on the GPU (same configuration, matched controls), then refresh the seed summary and stop the VM.
cd ~/cad_repo && source ~/cad/bin/activate && export CAD_OUT=~/cad_out
sudo shutdown -c 2>/dev/null
for s in 0 1 2; do
  python -m src.train_cell --tag t2_q_ctrl_s$s --task pert --name norman --base ctrl --seed $s --epochs 1 --max-train-cells 20000 --eval-split test > ~/s4_t2_q_ctrl_s${s}_rerun.log 2>&1 || echo "FAILED t2_q_ctrl_s$s" >> ~/qc.status
done
bash ~/cad_repo/experiments/summarize_stage4.sh > ~/s4_summarize2.log 2>&1
cd ~ && tar czf /tmp/rev2_results4.tgz cad_out/auto_rev2 cad_out/auto_rev2_complete cad_out/figs_v2 2>/dev/null
echo DONE > ~/qc.done
sudo shutdown -h +40
exit 0
