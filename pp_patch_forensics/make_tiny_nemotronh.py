"""Generate a random-weight NemotronH hybrid checkpoint for PP-export forensics.

Recreates (and optionally scales) the tiny 8-layer mamba/attention/moe hybrid
used for the PP>1 adapter-export investigations. The --size big variant scales
hidden to 8192 so LoRA adapter tensors (rank 64 x 8192 = 1 MiB bf16) cross
typical NCCL eager/inline-buffer thresholds, testing whether the stock export
path's PP broadcast is size-dependent (H2 size escalation).

Run on the devbox with the server venv:
    server/.venv/bin/python make_tiny_nemotronh.py \
        --template /root/.cache/user_artifacts/tiny-nemotronh-hybrid \
        --out /root/.cache/user_artifacts/tiny-nemotronh-hybrid-big --size big
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# hidden 8192 with matched mamba/attention/moe dims. vocab is cut to 32768 to
# keep the checkpoint small; the export drivers send raw token ids < 32768 and
# never tokenize, so the (copied) tokenizer's larger vocab is harmless.
BIG_OVERRIDES = {
    "hidden_size": 8192,
    "intermediate_size": 16384,
    "head_dim": 64,
    "num_attention_heads": 128,
    "num_key_value_heads": 8,
    # expand=2 -> mamba intermediate = 16384 = mamba_num_heads * mamba_head_dim
    "mamba_num_heads": 256,
    "mamba_head_dim": 64,
    "moe_intermediate_size": 1024,
    "moe_latent_size": 512,
    "moe_shared_expert_intermediate_size": 2048,
    "vocab_size": 32768,
}

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        required=True,
        help="Existing tiny checkpoint dir to take config + tokenizer from.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", choices=["small", "big"], default="big")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.template)
    if args.size == "big":
        for key, value in BIG_OVERRIDES.items():
            assert hasattr(config, key), f"config has no field {key!r}"
            setattr(config, key, value)

    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"instantiated {type(model).__name__} with {n_params / 1e6:.1f}M params")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    for name in TOKENIZER_FILES:
        src = Path(args.template) / name
        if src.exists():
            shutil.copy(src, out / name)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
