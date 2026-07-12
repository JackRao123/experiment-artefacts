#!/usr/bin/env bash
# 1 Hz nvidia-smi sampler on all 4 nodes via srun --overlap from the leader
# (the U1-rerun pattern). Run ON the leader; CSVs land on the shared FS at
# glm_prof/poll/<hostname>.csv via poll_mem.sh.
srun --overlap --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  bash /root/.cache/user_artifacts/glm_prof/scripts/poll_mem.sh
