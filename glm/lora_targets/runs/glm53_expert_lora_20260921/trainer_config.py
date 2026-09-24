"""Generate the matched full-model training configuration used in this experiment."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--layout",
        choices=["baseline", "shared_adapter", "shared_outer"],
        required=True,
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = {
        "base_model": args.model,
        "checkpoint_dir": args.checkpoint_dir,
        "max_seq_len": 131072,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 8,
        "context_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
        "trust_remote_code": True,
        "attention_backend": "flash",
        "lora_rank": 32,
        "lora_alpha": 32,
        "moe_lora_config": None if args.layout == "baseline" else args.layout,
        "moe_flex_dispatcher_backend": "hybridep",
        "expert_weight_storage": "native_fp8",
        "recompute": {"granularity": "full", "method": "uniform", "num_layers": 1},
        "weight_sync": {"type": "local", "path": args.export_dir},
    }
    args.output.write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
