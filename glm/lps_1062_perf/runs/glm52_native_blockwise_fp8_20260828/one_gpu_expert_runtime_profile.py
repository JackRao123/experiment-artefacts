"""Capture a kernel trace and allocator snapshot for one benchmark arm."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import one_gpu_expert_runtime_benchmark as benchmark


class ProfiledExpertBlock(benchmark.ExpertBlock):
    def __call__(self, inp: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        if self.arm == "custom":
            assert self.compiled_dequantize is not None
            assert self.native_projections is not None
            with record_function("custom_fc1_materialize_bf16"):
                fc1_weights = self.compiled_dequantize(*self.native_projections[0])
            with record_function("custom_fc1_te_grouped_bf16"):
                fc1_output = benchmark._external_grouped_bf16(
                    self.fc1, inp, self.m_splits, fc1_weights
                )
        else:
            with record_function(f"{self.arm}_fc1_te_grouped_bf16"):
                fc1_output = self.fc1(inp, self.m_splits)
        with record_function("swiglu"):
            gate, up = fc1_output.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
        if self.arm == "custom":
            with record_function("custom_fc2_materialize_bf16"):
                fc2_weights = self.compiled_dequantize(*self.native_projections[1])
            with record_function("custom_fc2_te_grouped_bf16"):
                return benchmark._external_grouped_bf16(
                    self.fc2, hidden, self.m_splits, fc2_weights
                )
        with record_function(f"{self.arm}_fc2_te_grouped_bf16"):
            return self.fc2(hidden, self.m_splits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("bf16", "custom", "generic"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1062)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    gc.collect()
    torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    m_splits = [args.m] * args.experts
    block = ProfiledExpertBlock(
        arm=args.arm,
        checkpoint=args.checkpoint,
        num_experts=args.experts,
        m_splits=m_splits,
    )
    block.compile(
        torch.empty(1, benchmark.HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    )
    torch.manual_seed(args.seed + 1)
    inp = torch.randn(
        sum(m_splits),
        benchmark.HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    torch.manual_seed(args.seed + 2)
    grad_output = torch.randn_like(inp)

    warm_output, _ = benchmark._run_step(block, inp, grad_output)
    del warm_output
    inp.grad = None
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    trace_path = args.output_dir / f"{args.arm}.pt.trace.json"
    memory_path = args.output_dir / f"{args.arm}.memory.pickle"
    summary_path = args.output_dir / f"{args.arm}.profile.json"
    torch.cuda.memory._record_memory_history(
        max_entries=100_000,
        stacks="python",
        context="alloc",
    )
    torch.cuda.reset_peak_memory_stats()
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            with record_function(f"profiled_{args.arm}_full_recompute_step"):
                output, dgrad = benchmark._run_step(block, inp, grad_output)
        torch.cuda.synchronize()
        profiler.export_chrome_trace(str(trace_path))
        torch.cuda.memory._dump_snapshot(str(memory_path))
        summary = {
            "arm": args.arm,
            "trace": str(trace_path),
            "memory_snapshot": str(memory_path),
            "memory_allocated_bytes": torch.cuda.memory_allocated(),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "output_dtype": str(output.dtype),
            "dgrad_dtype": str(dgrad.dtype),
            "storage_after": benchmark._storage_state(block),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
