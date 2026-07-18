"""Build a tiny random-weight NemotronH (hybrid Mamba+attention+MoE) checkpoint.

Shrinks the Nemotron-3-Ultra config to a few layers that still cover every
hybrid block type (mamba, attention, moe), instantiates it on CPU with random
weights, and saves it with the Ultra tokenizer. Used as the small THD+CP
forward/backward probe model before touching the 550B checkpoint.

Run inside the trainers server venv on the devbox:

    uv run --no-sync python make_tiny_nemotron_h.py \
        --source /root/.cache/user_artifacts/nemotron3-ultra-550b-nvfp4-dequant-bf16 \
        --out /root/.cache/user_artifacts/ultra_cp4/tiny_nemotron_h
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


TINY_OVERRIDES = {
    # 6 decoder layers covering all three hybrid block types, attention placed
    # away from the edges like the real pattern.
    "layers_block_type": ["mamba", "moe", "attention", "moe", "mamba", "moe"],
    "hidden_size": 256,
    "intermediate_size": 512,
    # Mamba: d_inner = expand * hidden = 512 = 8 heads x 64 head_dim.
    # n_groups=8 keeps ngroups_local_tp divisible by cp for TP<=4, CP<=2.
    "mamba_num_heads": 8,
    "mamba_head_dim": 64,
    "n_groups": 8,
    "ssm_state_size": 128,
    # Attention: 8 heads / 4 KV groups so TP4 divides both. head_dim stays
    # explicit like Ultra (decoupled from hidden/heads).
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 32,
    # MoE: 16 routed experts top-4 so EP2 gives 8 experts per rank.
    "n_routed_experts": 16,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 128,
    "moe_shared_expert_intermediate_size": 256,
    "moe_latent_size": 64,
    # No MTP in the probe; the 550B parity run covers the real stack.
    "num_nextn_predict_layers": 0,
    "mtp_layers_block_type": [],
    "max_position_embeddings": 8192,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Ultra BF16 checkpoint dir")
    parser.add_argument("--out", required=True, help="tiny checkpoint output dir")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((source / "config.json").read_text())
    cfg.update(TINY_OVERRIDES)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    config = AutoConfig.from_pretrained(out, trust_remote_code=True)

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_config(
        config, dtype=torch.bfloat16, trust_remote_code=True
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"tiny NemotronH: {n_params / 1e6:.1f}M params")
    model.save_pretrained(out, safe_serialization=True)

    # transformers' native NemotronH writes ``backbone.embedding.weight``; the
    # Ultra checkpoints (and NemotronHBridge's mapping) use
    # ``backbone.embeddings.weight``. Left unrenamed, the bridge silently maps
    # nothing onto the megatron embedding and the whole residual stream is
    # zeros — CE collapses to exactly ln(padded_vocab) and every LoRA grad is
    # zero (the adapter input is the zero hidden state).
    import glob as _glob

    from safetensors.torch import load_file, save_file

    for shard_path in _glob.glob(str(out / "*.safetensors")):
        tensors = load_file(shard_path)
        if "backbone.embedding.weight" in tensors:
            tensors["backbone.embeddings.weight"] = tensors.pop(
                "backbone.embedding.weight"
            )
            save_file(tensors, shard_path, metadata={"format": "pt"})
    index_path = out / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        if "backbone.embedding.weight" in weight_map:
            weight_map["backbone.embeddings.weight"] = weight_map.pop(
                "backbone.embedding.weight"
            )
        index_path.write_text(json.dumps(index, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    tokenizer.save_pretrained(out)
    for aux in ("chat_template.jinja",):
        if (source / aux).is_file():
            shutil.copy2(source / aux, out / aux)
    # save_pretrained rewrites config.json from the config object; make sure
    # the block-type lists survived round-tripping.
    saved = json.loads((out / "config.json").read_text())
    assert saved["layers_block_type"] == TINY_OVERRIDES["layers_block_type"], saved.get(
        "layers_block_type"
    )
    print(f"saved tiny checkpoint to {out}")


if __name__ == "__main__":
    main()
