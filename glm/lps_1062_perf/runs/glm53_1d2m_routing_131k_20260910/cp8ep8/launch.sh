#!/usr/bin/env bash
set -euo pipefail
source /root/.cache/user_artifacts/devboxes/32vj99q/env.sh
export BT_TRAINER_CONFIG_PATH=/root/glm53-1d2m-routing-131k-20260910/cp8ep8/trainer-config.json
export BT_TRAINER_SERVER_CONFIG_PATH=/root/glm53-main-262k-20260909/trainer-server-config.json
export BT_ROUTING_COUNTS_DIR=/root/glm53-1d2m-routing-131k-20260910/cp8ep8/counts
export BT_EXPERIMENTAL_FSDP=0 BT_FSDP_GROUPED_MM=0 BT_MEMORY_EFFICIENT_LM_HEAD=0
export BT_LEADER_ADDR=127.0.0.1 BT_NODE_RANK=0 BT_GROUP_SIZE=1 BT_NUM_GPUS=8
export NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.95
export BT_PROFILE_OUTPUT_DIR=/root/glm53-1d2m-routing-131k-20260910/cp8ep8/profiles
bash /root/glm53-1d2m-routing-131k-20260910/cp8ep8/.devbox_up/start_trainer.sh
