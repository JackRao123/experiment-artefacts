#!/usr/bin/env python3
"""Post-parse one bench cell: vLLM log + nvidia-smi poller CSV + RESULT_JSON.

The load-bearing numbers vLLM only ever logs (never exposes via API):
  - "Model loading took X GiB and Y seconds"      (per-rank weights + load time)
  - "GPU KV cache size: N tokens"                 (authoritative pool size)
  - "Maximum concurrency for L tokens per request: C.x"
  - periodic "Running: N reqs ... GPU KV cache usage: P%"
  - preemption warnings
Plus the two proofs the production recipe was actually in effect:
  - "Applied Nemotron-3-Ultra NVFP4+LoRA vLLM patches"
  - CUTLASS NVFP4 MoE backend selection.

Pure stdlib; runs with any python3 on the box.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

MIB = 1024.0


def parse_log(text: str) -> dict:
    out: dict = {}

    m = re.search(
        r"Model loading took ([\d.]+) GiB(?: memory)? and ([\d.]+) seconds", text
    )
    if m:
        out["weights_gib_per_gpu"] = float(m.group(1))
        out["model_loading_s"] = float(m.group(2))
    m = re.search(r"Loading weights took ([\d.]+) seconds", text)
    if m:
        out["loading_weights_s"] = float(m.group(1))
    m = re.search(r"Available KV cache memory: ([\d.]+) GiB", text)
    if m:
        out["available_kv_gib_per_gpu"] = float(m.group(1))
    m = re.search(r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)", text)
    if m:
        out["cudagraph_pool_gib"] = float(m.group(1))

    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if m:
        out["kv_pool_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(
        r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text
    )
    if m:
        out["max_concurrency_ctx_tokens"] = int(m.group(1).replace(",", ""))
        out["max_concurrency"] = float(m.group(2))

    out.update(_phase_stats(text))
    out["buckets"] = _bucket_stats(text)
    out["nemotron_patches_applied"] = (
        "Applied Nemotron-3-Ultra NVFP4+LoRA vLLM patches" in text
    )
    out["cutlass_moe_selected"] = bool(re.search(r"VLLM_CUTLASS|CutlassExpertsFp4", text))
    out["oom"] = bool(re.search(r"OutOfMemoryError|out of memory", text))
    return out


def _phase_stats(text: str) -> dict:
    """Peaks over a log region: scheduler state and per-phase rates from the
    periodic engine lines. Peak-gen is the decode-phase capability;
    continuous batching makes the wall-clock aggregate the headline number."""
    out: dict = {}
    running = [int(x) for x in re.findall(r"Running: (\d+) reqs", text)]
    if running:
        out["peak_running_reqs"] = max(running)
    kv_usage = [float(x) for x in re.findall(r"GPU KV cache usage: ([\d.]+)%", text)]
    if kv_usage:
        out["peak_kv_usage_pct"] = max(kv_usage)
    gen_rates = [
        float(x)
        for x in re.findall(r"Avg generation throughput: ([\d.]+) tokens/s", text)
    ]
    nz = [r for r in gen_rates if r > 0]
    if nz:
        out["peak_gen_tok_per_s_logged"] = max(nz)
    prompt_rates = [
        float(x) for x in re.findall(r"Avg prompt throughput: ([\d.]+) tokens/s", text)
    ]
    nz = [r for r in prompt_rates if r > 0]
    if nz:
        out["peak_prompt_tok_per_s_logged"] = max(nz)
    out["preemption_lines"] = len(re.findall(r"preempt", text, re.IGNORECASE))
    return out


def _bucket_stats(text: str) -> list[dict]:
    """Split the log at BUCKET_START/BUCKET_END markers and attribute the
    periodic engine stats to each data-length bucket."""
    buckets: list[dict] = []
    for m in re.finditer(
        r"BUCKET_START ctx=(\d+) n=(\d+).*?(?=BUCKET_END ctx=\1|\Z)",
        text,
        re.DOTALL,
    ):
        b = {"ctx": int(m.group(1)), "num_requests": int(m.group(2))}
        b.update(_phase_stats(m.group(0)))
        buckets.append(b)
    return buckets


def parse_smi(path: Path) -> dict:
    """CSV lines: index, memory.used [MiB]. Max per GPU over the run."""
    peak: dict[int, float] = defaultdict(float)
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            idx, used = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        peak[idx] = max(peak[idx], used)
    gib = {i: round(v / MIB, 1) for i, v in sorted(peak.items())}
    return {
        "per_gpu_peak_gib": gib,
        "hottest_gpu_gib": max(gib.values()) if gib else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--smi", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary: dict = {}
    log_path = Path(args.log)
    if log_path.exists():
        summary["log"] = parse_log(log_path.read_text(errors="replace"))
    smi_path = Path(args.smi)
    if smi_path.exists():
        summary["smi"] = parse_smi(smi_path)
    res_path = Path(args.result)
    if res_path.exists():
        summary["result"] = json.loads(res_path.read_text())

    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
