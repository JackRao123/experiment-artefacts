"""Qwen3 LoRA SFT matmul FLOPs, audited against checkpoint tensor headers.

FMA = 2 FLOPs. Useful FLOPs exclude activation/kernel recomputation.
Fused QKV and gate/up adapters match this run's Megatron Bridge LoRA.
No scalar-op, optimizer, communication, padding, or kernel-counter claims.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INVENTORIES = {
    "0.6b": ROOT / "inventories/qwen3-06b-inventory.json",
    "30b-a3b": ROOT / "inventories/qwen3-30b-a3b-inventory.json",
}
B300_BF16_DENSE_PEAK = 2.25e15


def estimate(inventory: dict, seq_len: int, rank: int) -> dict:
    if seq_len < 1 or rank < 1:
        raise ValueError("seq_len and rank must be positive")
    c = inventory["config"]
    if c["model_type"] not in ("qwen3", "qwen3_moe") or c["use_sliding_window"]:
        raise ValueError("Only full-attention Qwen3/Qwen3-MoE supported")
    h, n = c["hidden_size"], c["num_hidden_layers"]
    q = c["num_attention_heads"] * c["head_dim"]
    kv = c["num_key_value_heads"] * c["head_dim"]
    v = c["vocab_size"]
    moe = c["model_type"] == "qwen3_moe"
    k = c["num_experts_per_tok"] if moe else 1
    experts = c["num_experts"] if moe else 1
    intermediate = c["moe_intermediate_size"] if moe else c["intermediate_size"]
    if moe and (c["decoder_sparse_step"] != 1 or c["mlp_only_layers"]):
        raise ValueError("This estimate expects every layer to be MoE")

    # Verify every layer/expert and reject unknown tensors, not just layer 0.
    expected = {"model.embed_tokens.weight": [v, h], "model.norm.weight": [h]}
    if "lm_head.weight" in inventory["tensors"]:
        expected["lm_head.weight"] = [v, h]
    for layer in range(n):
        p = f"model.layers.{layer}."
        for name, shape in {
            "self_attn.q_proj.weight": [q, h],
            "self_attn.k_proj.weight": [kv, h],
            "self_attn.v_proj.weight": [kv, h],
            "self_attn.o_proj.weight": [h, q],
            "self_attn.q_norm.weight": [c["head_dim"]],
            "self_attn.k_norm.weight": [c["head_dim"]],
            "input_layernorm.weight": [h],
            "post_attention_layernorm.weight": [h],
        }.items():
            expected[p + name] = shape
        for expert in range(experts):
            prefix = p + (f"mlp.experts.{expert}." if moe else "mlp.")
            for name, shape in {"gate_proj": [intermediate, h],
                                "up_proj": [intermediate, h],
                                "down_proj": [h, intermediate]}.items():
                expected[prefix + name + ".weight"] = shape
        if moe:
            expected[p + "mlp.gate.weight"] = [experts, h]
    actual = {name: value["shape"] for name, value in inventory["tensors"].items()}
    if actual != expected:
        raise ValueError("Checkpoint tensor names/shapes do not match architecture")
    checkpoint_parameters = sum(math.prod(shape) for shape in actual.values())
    if checkpoint_parameters != inventory["checkpoint_parameters"]:
        raise ValueError("Inventory parameter total mismatch")
    unique_parameters = checkpoint_parameters
    if c["tie_word_embeddings"] and "lm_head.weight" in actual:
        unique_parameters -= h * v  # serialized duplicate; tied at runtime

    # Base projection/MLP/router GEMMs per input token. Embedding lookup is not a GEMM.
    attention_projection = 2 * n * h * (2 * q + 2 * kv)
    mlp = 2 * n * k * 3 * h * intermediate
    router = 2 * n * h * experts if moe else 0
    head = 2 * h * v  # count once, including tied output heads
    block_base = attention_projection + mlp + router

    # QK^T + PV over causal pairs S(S+1)/2; query-head count governs both.
    attention = 2 * n * q * (seq_len + 1)
    # Megatron uses one rank-r adapter for fused QKV and one for fused gate/up.
    attention_adapter_io = (h + q + 2 * kv) + (q + h)
    mlp_adapter_io = (h + 2 * intermediate) + (intermediate + h)
    block_lora = 2 * rank * n * (attention_adapter_io + k * mlp_adapter_io)
    head_lora = 0 if c["tie_word_embeddings"] else 2 * rank * (h + v)
    # Shared expert adapters still execute on k dispatched token copies.
    # Their parameter count is independent of k on this EP=1 run.
    adapter_parameters = rank * n * (attention_adapter_io + mlp_adapter_io)
    if not c["tie_word_embeddings"]:
        adapter_parameters += rank * (h + v)

    forward = block_base + head + attention + block_lora + head_lora
    useful = 2 * (block_base + head) + 3 * attention + 3 * (block_lora + head_lora)
    # Full decoder-block checkpointing; output head is outside that checkpoint.
    block_recompute = block_base + attention + block_lora
    # Flash/fused attention backward usually reconstructs scores QK^T once.
    # This is kernel recompute, distinct from the full-block checkpoint forward.
    flash_score_recompute = attention / 2
    return {
        "model": c["model_type"], "seq_len": seq_len, "lora_rank": rank,
        "checkpoint_parameters": checkpoint_parameters,
        "unique_base_parameters": unique_parameters,
        "adapter_parameters_ep1_shared_experts": adapter_parameters,
        "active_base_gemm_weights_per_token": (block_base + head) // 2,
        "forward_components_flops_per_token": {
            "attention_projections": attention_projection, "mlp": mlp,
            "router": router, "output_head": head, "causal_attention": attention,
            "block_lora": block_lora, "head_lora": head_lora,
        },
        "forward_flops_per_token": forward,
        "useful_lora_sft_flops_per_token": useful,
        "full_block_recompute_flops_per_token": block_recompute,
        "executed_pass_model_flops_per_token": useful + block_recompute,
        "executed_flash_model_flops_per_token": useful + block_recompute + flash_score_recompute,
        "useful_flops_per_sequence": useful * seq_len,
    }


def benchmark_statistics(run: dict, flops: dict, peak: float) -> dict:
    if peak <= 0:
        raise ValueError("GPU peak must be positive")
    if run["seq_len"] != flops["seq_len"]:
        raise ValueError("Benchmark and FLOP-estimate sequence lengths differ")
    controls = [w for w in run["windows"] if w["phase"] == "control"]
    fb = [w["fb_elapsed_s"] for w in controls]
    opt = [w["optim_elapsed_s"] for w in controls]
    tokens = run["tokens_per_step"]
    gpu_count = run["num_gpus"]
    tps = tokens / statistics.mean(fb) / gpu_count
    step_tps = tokens / (statistics.mean(fb) + statistics.mean(opt)) / gpu_count
    useful = flops["useful_lora_sft_flops_per_token"]
    return {
        "control_count": len(controls), "num_gpus": gpu_count,
        "control_fb_seconds": fb, "control_fb_seconds_mean": statistics.mean(fb),
        "control_fb_seconds_stdev": statistics.stdev(fb) if len(fb) > 1 else 0,
        "control_optim_seconds_mean": statistics.mean(opt),
        "control_fb_tps_per_gpu": tps, "control_step_tps_per_gpu": step_tps,
        "peak_bf16_dense_flops_per_gpu": peak,
        "mfu_fb": tps * useful / peak, "mfu_including_optim": step_tps * useful / peak,
        "hfu_pass_model_estimate": tps * flops["executed_pass_model_flops_per_token"] / peak,
        "hfu_flash_model_estimate": tps * flops["executed_flash_model_flops_per_token"] / peak,
        "control_losses": [w["loss"] for w in controls],
        "control_grad_norms": [w["grad_norm"] for w in controls],
        "peak_allocated_gib": max(w["peak_allocated_bytes"] for w in controls) / 2**30,
        "peak_reserved_gib": max(w["peak_reserved_bytes"] for w in controls) / 2**30,
        "runtime_profile_fb_seconds": next(
            (w["fb_elapsed_s"] for w in run["windows"] if w["phase"] == "runtime_profile"), None
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=INVENTORIES, required=True)
    parser.add_argument("--seq-len", type=int, default=40960)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--peak-tflops", type=float, default=2250)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = estimate(json.loads(INVENTORIES[args.model].read_text()), args.seq_len, args.lora_rank)
    if args.benchmark:
        result["benchmark"] = benchmark_statistics(
            json.loads(args.benchmark.read_text()), result, args.peak_tflops * 1e12
        )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
