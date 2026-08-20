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

The model is constructed on the META device (zero storage) and each shard is
materialized and written incrementally, so peak host RAM is ~one shard
(~4 GiB), not the whole model — the build is safe on a contended Mac or a
box login node.

The resulting snapshot is a local HF dir (config.json + sharded safetensors +
tokenizer). Point the trainer at it with base_model=<dir>; glm52_dsa's
_is_glm52_local_path detects it via architectures/model_type.

Usage:
  python3 build_smoke_snapshot.py --output-dir /path/to/glm52_smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

DEFAULT_SOURCE = (
    "zai-org/GLM-5.2-FP8"  # resolved from the HF cache; only config+tokenizer read
)
TOKENIZER_FILES = ("chat_template.jinja", "tokenizer_config.json", "tokenizer.json")
_SHARD_BYTES = 4 * 1024**3  # ~4 GiB per shard


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
    # transformers' default init for this arch is normal_(std=initializer_range);
    # match it so activations are in a sane range for the random-init canary.
    std = float(getattr(config, "initializer_range", 0.02))

    # Meta-device construction: the full state-dict structure (names, shapes,
    # dtypes) with zero storage.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    meta_state = model.state_dict()
    total = sum(t.numel() for t in meta_state.values())
    print(f"model: {total / 1e9:.2f}B params, {len(meta_state)} tensors (meta)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Materialize + write incrementally, sharded by size.
    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_idx = 0

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        shard_idx += 1
        name = f"model-{shard_idx:05d}.safetensors"
        save_file(shard, str(output_dir / name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        print(f"  wrote {name}: {shard_bytes / 1024**3:.2f} GiB, {len(shard)} tensors")
        shard = {}
        shard_bytes = 0

    for key, meta_t in meta_state.items():
        t = torch.empty(meta_t.shape, dtype=torch.bfloat16).normal_(0.0, std)
        nbytes = t.numel() * t.element_size()
        if shard_bytes + nbytes > _SHARD_BYTES:
            flush()
        shard[key] = t
        shard_bytes += nbytes
    flush()

    n = shard_idx
    index = {
        "metadata": {"total_size": total * 2},  # bf16 = 2 bytes
        "weight_map": weight_map,
    }
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    # config.json from the (meta) model, then tokenizer files.
    model.config.use_cache = False
    model.config.save_pretrained(output_dir)

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
        f"index_topk={cfg['index_topk']} kv_lora_rank={cfg['kv_lora_rank']} "
        f"shards={n}"
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
