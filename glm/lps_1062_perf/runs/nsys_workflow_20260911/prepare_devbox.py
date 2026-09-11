"""Prepare experiments from tj-q9exk9w's generated lifecycle, using the old venv/code."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = "/root/glm53-fsdp-nsys-131k-20260911"
BOX = "/root/.cache/user_artifacts/devboxes/q9exk9w"
SOURCE = "/root/glm53-pr1355-repro-20260910/trainers"
base = json.loads((ROOT / "base_config.json").read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", action="append", choices=("devbox-debug", "devbox-ep1-te", "devbox-ep8-te", "devbox-ep1-grouped", "devbox-ep8-grouped"))
parser.add_argument("--base-model", help="Exact restored snapshot path; does not alter completed cases")
args = parser.parse_args()
for name, ep, grouped, debug in (
    ("devbox-debug", 8, True, True),
    ("devbox-ep1-te", 1, False, False),
    ("devbox-ep8-te", 8, False, False),
    ("devbox-ep1-grouped", 1, True, False),
    ("devbox-ep8-grouped", 8, True, False),
):
    if args.case and name not in args.case:
        continue
    case = ROOT / name
    if (case / "benchmark.json").exists():
        print(f"Preserving existing benchmark configuration: {name}")
        continue
    lifecycle = case / ".devbox_up"
    lifecycle.mkdir(parents=True, exist_ok=True)
    config = {**base, "expert_parallel_size": ep, "checkpoint_dir": f"{REMOTE}/{name}/checkpoints",
              "nsight_profiling": {"record_nvtx_ranges": True}}
    if debug:
        config["base_model"] = "/root/glm53-1d2m-262k-20260909/model"
    elif args.base_model:
        config["base_model"] = args.base_model
    (case / "trainer-config.json").write_text(json.dumps(config, indent=2))
    options = {"fsdp": True, "prefetch": True, "persistent_buffers": True,
               "grouped_mm": grouped, "cuda_graphs": False, "lm_head_chunk": 4096,
               "memory_efficient_lm_head": False,
               "capture": "software CUDA/NVTX; event tracing disabled; no CPU/callstack sampling"}
    (case / "run_options.json").write_text(json.dumps(options, indent=2))
    for filename in ("start_trainer.sh", "wait_trainer_health.sh", "run_trainer_node.sh"):
        text = (ROOT / "lifecycle-q9exk9w" / filename).read_text().replace(BOX, f"{REMOTE}/{name}")
        if filename == "run_trainer_node.sh":
            text = text.replace(f"source {REMOTE}/{name}/env.sh", f"source {BOX}/env.sh")
            text = text.replace(f"SRC={REMOTE}/{name}/trainers", f"SRC={SOURCE}")
            text = text.replace(f"SRC={SOURCE}\n", f'''SRC={SOURCE}
export CUDA_DEVICE_MAX_CONNECTIONS=32 BT_MULTI_ADAPTER_ENABLED=0 PYTHONNOUSERSITE=1
export PYTHONPATH="$SRC/models/src:$SRC/server-interface/src:$SRC/server-main/src:$SRC/server-megatron-bridge/src:$SRC/baseten-weight-sync:$SRC/server-megatron-bridge/vendor/megatron-bridge/src:$SRC/server-megatron-bridge/vendor/megatron-bridge/3rdparty/Megatron-LM"
''')
            text = text.replace("exec bash scripts/launch.sh --backend megatron_bridge",
                                'exec nsys launch --session-new="$NSYS_SESSION_NAME" --trace=cuda,nvtx '
                                '--cuda-event-trace=false --show-output=true --wait=all '
                                'bash scripts/launch.sh --backend megatron_bridge')
        (lifecycle / filename).write_text(text)
    # Preserve devbox-up stop semantics with the run-ownership/nsys hardening.
    stop = (ROOT / "lifecycle/stop_trainer.sh").read_text().replace(
        "/root/glm53-pr1355-repro-20260910", f"{REMOTE}/{name}")
    (lifecycle / "stop_trainer.sh").write_text(stop)
    (case / "launch.sh").write_text(f'''#!/usr/bin/env bash
set -euo pipefail
source {BOX}/env.sh
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
active_gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$active_gpu_pids" ]; then
  echo "Refusing to benchmark with existing GPU processes: $active_gpu_pids" >&2
  exit 1
fi
bash {REMOTE}/{name}/.devbox_up/start_trainer.sh
''')
