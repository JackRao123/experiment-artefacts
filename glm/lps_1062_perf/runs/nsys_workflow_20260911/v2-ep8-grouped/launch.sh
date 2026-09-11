#!/usr/bin/env bash
set -euo pipefail
source /root/.cache/user_artifacts/devboxes/32vj99q/env.sh
export NSYS_SESSION_NAME=glm53-v2-ep8-grouped-0911
export BT_TRAINER_CONFIG_PATH=/root/glm53-fsdp-nsys-131k-20260911/v2-ep8-grouped/trainer-config.json
export BT_TRAINER_SERVER_CONFIG_PATH=/root/glm53-main-262k-20260909/trainer-server-config.json
export BT_EXPERIMENTAL_FSDP=1 BT_FSDP_GROUPED_MM=1
export BT_FSDP_PREFETCH=1 BT_FSDP_PERSISTENT_BUFFERS=1 BT_FSDP_FAST_LOAD=1
export BT_MEMORY_EFFICIENT_LM_HEAD=0 BT_LM_HEAD_SEQ_CHUNK=4096
export BT_LEADER_ADDR=127.0.0.1 BT_NODE_RANK=0 BT_GROUP_SIZE=1 BT_NUM_GPUS=8
export NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.95
export BT_PROFILE_OUTPUT_DIR=/root/glm53-fsdp-nsys-131k-20260911/v2-ep8-grouped/profiles
unset BT_ROUTING_COUNTS_DIR
active_gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$active_gpu_pids" ]; then
  echo "Refusing to benchmark with existing GPU processes: $active_gpu_pids" >&2
  exit 1
fi
bash /root/glm53-fsdp-nsys-131k-20260911/v2-ep8-grouped/.devbox_up/start_trainer.sh
