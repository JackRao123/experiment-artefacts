"""Generate run configs and mechanically adapt devbox-up lifecycle copies."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = "/root/glm53-fsdp-nsys-131k-20260911"
OLD = "/root/glm53-pr1355-repro-20260910"
BASE = json.loads((ROOT / "base_config.json").read_text())
for name, ep, debug, grouped in (
    ("debug-ep8", 8, True, False),
    ("ep1-te", 1, False, False),
    ("ep8-te", 8, False, False),
    ("ep1-grouped", 1, False, True),
    ("ep8-grouped", 8, False, True),
):
    folder = ROOT / name
    lifecycle = folder / ".devbox_up"
    lifecycle.mkdir(parents=True, exist_ok=True)
    config = {**BASE, "expert_parallel_size": ep,
              "checkpoint_dir": f"{REMOTE}/{name}/checkpoints",
              "nsight_profiling": {"record_nvtx_ranges": True}}
    if debug:
        config["base_model"] = "/root/glm53-1d2m-262k-20260909/model"
    (folder / "trainer-config.json").write_text(json.dumps(config, indent=2))
    for source in (ROOT / "lifecycle").glob("*.sh"):
        contents = source.read_text().replace(OLD, f"{REMOTE}/{name}")
        contents = contents.replace(f"SRC={REMOTE}/{name}/trainers", f"SRC={OLD}/trainers")
        if source.name == "run_trainer_node.sh":
            contents = contents.replace(
                "exec bash scripts/launch.sh --backend megatron_bridge",
                'exec nsys launch --session-new="$NSYS_SESSION_NAME" '
                '--trace=cuda,nvtx --cuda-event-trace=true --show-output=true --wait=all '
                'bash scripts/launch.sh --backend megatron_bridge')
        (lifecycle / source.name).write_text(contents)
    (folder / "launch.sh").write_text(f'''#!/usr/bin/env bash
set -euo pipefail
source /root/.cache/user_artifacts/devboxes/32vj99q/env.sh
export NSYS_SESSION_NAME=glm53-{name}-0911
export BT_TRAINER_CONFIG_PATH={REMOTE}/{name}/trainer-config.json
export BT_TRAINER_SERVER_CONFIG_PATH=/root/glm53-main-262k-20260909/trainer-server-config.json
export BT_EXPERIMENTAL_FSDP=1 BT_FSDP_GROUPED_MM={int(grouped)}
export BT_FSDP_PREFETCH=1 BT_FSDP_PERSISTENT_BUFFERS=1 BT_FSDP_FAST_LOAD=1
export BT_MEMORY_EFFICIENT_LM_HEAD=0 BT_LM_HEAD_SEQ_CHUNK=4096
export BT_LEADER_ADDR=127.0.0.1 BT_NODE_RANK=0 BT_GROUP_SIZE=1 BT_NUM_GPUS=8
export NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.95
export BT_PROFILE_OUTPUT_DIR={REMOTE}/{name}/profiles
unset BT_ROUTING_COUNTS_DIR
bash {REMOTE}/{name}/.devbox_up/start_trainer.sh
''')
