#!/usr/bin/env python3
"""Build the cut-down GLM-5.2 random-init snapshot for the 1-node shim smoke.

LPS-1062 W2-prep (fermi-ordered, hatch (a)): a GLM-5.2 that fits ONE 8xB200
node so the contract shim + the PreProcessNode grad-root landmine fix +
the DSA x combined-executor interaction get hardware evidence on bohr's
B200, instead of burning a B300 mission window to find a boot failure.

Design (per the feasibility math):
- PRESERVED per-layer dims: hidden 6144, MLA/DSA shapes (kv_lora_rank 512,
  kv-lora output 576 = 512+64 rope, q_lora_rank 2048, qk_nope 192 /
  qk_rope 64 / v 256 per head, 64 heads), indexer (index_head_dim 128,
  index_n_heads 32, index_topk 2048, topk freq 4 / skip-offset 3),
  expert hidden 2048, dense MLP 12288, vocab 154880. Activation shapes per
  layer therefore match the real model's — which is what makes the
  BT_DIAL_MEM_PROBE per-layer S_eager readings transferable.
- CUT: num_hidden_layers 78 -> 12 (the smallest N that fields 4 VPP chunks
  with DSA-legal starts: chunk starts must be <=3 or ==3 (mod 4); N=8 only
  offers layers 3,7), and n_routed_experts 256 -> 32 (snapshot stays ~22 GiB
  on disk and builds in minutes; per-token MoE activation volume is
  topk=8-driven and unaffected by the expert count).
- Weights are RANDOM-INIT bf16 (no quantization_config — the bridge's FP8
  dequant path does not fire). Random init is deliberate: the smoke proves
  boot + executor contract + landmine fix + DSA x executor, NOT numerics.

The resulting snapshot is a local HF dir (config.json + safetensors +
tokenizer). Point the trainer at it with base_model=<dir>; glm52_dsa's
_is_glm52_local_path detects it via architectures/model_type.

Usage:
  python3 build_shim_smoke_snapshot.py --output-dir /path/to/glm52_smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

DEFAULT_SOURCE = (
    "zai-org/GLM-5.2-FP8"  # resolved from the HF cache; only config+tokenizer read
)
TOKENIZER_FILES = ("chat_template.jinja", "tokenizer_config.json", "tokenizer.json")


def build(output_dir: Path, source: str, num_layers: int, num_experts: int) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import cached_file

    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    config.num_hidden_layers = num_layers
    config.n_routed_experts = num_experts
    # Plain bf16 weights: drop the FP8 quantization block so the bridge loads
    # the tensors as-is (maybe_dequantize_fp8_blockwise needs scale_inv, which
    # random init does not carry).
    config.quantization_config = None
    config.dtype = "bfloat16"

    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.config.use_cache = False

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    for name in TOKENIZER_FILES:
        try:
            src = Path(cached_file(source, name))
        except OSError:
            continue  # chat_template is optional
        shutil.copy(src, output_dir / name)

    cfg = json.loads((output_dir / "config.json").read_text())
    print(
        f"snapshot written to {output_dir}: layers={cfg['num_hidden_layers']} "
        f"experts={cfg['n_routed_experts']} hidden={cfg['hidden_size']} "
        f"index_topk={cfg['index_topk']} kv_lora_rank={cfg['kv_lora_rank']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--num-layers", type=int, default=12)
    ap.add_argument("--num-experts", type=int, default=32)
    args = ap.parse_args()
    build(args.output_dir, args.source, args.num_layers, args.num_experts)


if __name__ == "__main__":
    main()
