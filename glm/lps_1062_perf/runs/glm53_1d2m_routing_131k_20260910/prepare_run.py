"""Prepare configs and copies of generated lifecycle scripts, without source patches."""
import json
from pathlib import Path

root = Path(__file__).parent
remote = "/root/glm53-1d2m-routing-131k-20260910"
previous = root.parent / "glm53_pr1355_branch_repro_20260910"
old_remote = "/root/glm53-pr1355-repro-20260910"
base = json.loads((previous / "debug-cp8/trainer-config.json").read_text())
for ep in (8, 1):
    name = f"cp8ep{ep}"
    case = root / name
    lifecycle = case / ".devbox_up"
    lifecycle.mkdir(parents=True, exist_ok=True)
    config = {**base, "expert_parallel_size": ep, "checkpoint_dir": f"{remote}/{name}/checkpoints"}
    (case / "trainer-config.json").write_text(json.dumps(config, indent=2))
    (case / "collect_forward.py").write_text((root / "collect_forward.py").read_text())
    for source in (previous / "lifecycle").glob("*.sh"):
        contents = source.read_text().replace(old_remote, f"{remote}/{name}")
        # Keep the already-committed trainer checkout, independent of output paths.
        contents = contents.replace(f"SRC={remote}/{name}/trainers", f"SRC={old_remote}/trainers")
        (lifecycle / source.name).write_text(contents)
    launch = f'''#!/usr/bin/env bash
set -euo pipefail
source /root/.cache/user_artifacts/devboxes/32vj99q/env.sh
export BT_TRAINER_CONFIG_PATH={remote}/{name}/trainer-config.json
export BT_TRAINER_SERVER_CONFIG_PATH=/root/glm53-main-262k-20260909/trainer-server-config.json
export BT_ROUTING_COUNTS_DIR={remote}/{name}/counts
export BT_EXPERIMENTAL_FSDP=0 BT_FSDP_GROUPED_MM=0 BT_MEMORY_EFFICIENT_LM_HEAD=0
export BT_LEADER_ADDR=127.0.0.1 BT_NODE_RANK=0 BT_GROUP_SIZE=1 BT_NUM_GPUS=8
export NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.95
export BT_PROFILE_OUTPUT_DIR={remote}/{name}/profiles
bash {remote}/{name}/.devbox_up/start_trainer.sh
'''
    (case / "launch.sh").write_text(launch)
