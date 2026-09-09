import time

import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels


def main() -> None:
    device = torch.device("cuda:0")
    sq = 32_768
    skv = 262_144
    heads = 64
    qk_dim = 576
    value_dim = 512
    topk = 2_048

    query = torch.zeros(
        (sq, 1, heads, qk_dim), device=device, dtype=torch.bfloat16, requires_grad=True
    )
    key = torch.zeros(
        (skv, 1, 1, qk_dim), device=device, dtype=torch.bfloat16, requires_grad=True
    )
    torch.manual_seed(0xB300)
    indices = torch.randint(
        0, skv, (1, sq, topk), device=device, dtype=torch.int32
    ).sort(dim=-1).values
    lengths = torch.full((1, sq), topk, device=device, dtype=torch.int32)

    output = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
        query=query,
        key=key,
        topk_indices=indices,
        softmax_scale=qk_dim**-0.5,
        v_channels=value_dim,
        topk_length=lengths,
        all_rows_nonempty=True,
    )
    if output is None:
        raise RuntimeError("fused DSA rejected the production geometry")

    grad = torch.ones_like(output)
    torch.cuda.synchronize()
    started = time.perf_counter()
    output.backward(grad)
    torch.cuda.synchronize()
    print(f"backward_ms={(time.perf_counter() - started) * 1e3:.3f}")


if __name__ == "__main__":
    main()
