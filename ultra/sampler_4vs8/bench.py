#!/usr/bin/env python3
"""TRN-1488: NVFP4 Ultra sampler bench, one TP cell per run.

Boots vLLM offline ONCE at the golden max_model_len (262144 unless the pool
can't hold it) with the production NemotronH-NVFP4 recipe (mirrors
``vllm_server.py`` + ``sampler_configs.py`` NEMOTRON_3_ULTRA B200/S256K), then
drives one saturating synthetic generate bucket per --data-lens entry through
the same engine (ascending, so first-call warmup lands in the cheapest bucket):

  prompt = ctx - max_new_tokens - 16 random tokens,
  N requests ~ 1.05x (KV-pool tokens / ctx) (capped at max_num_seqs / --cap),
  max_new_tokens generated with ignore_eos.

Run via run_cell.sh, which sets CUDA_VISIBLE_DEVICES for the TP size, exports
the patch-gate env (_BASETEN_SERVED_MODEL, MODEL_PATH), runs the nvidia-smi
poller, and post-parses the log (vLLM's own "GPU KV cache size" / "Maximum
concurrency" / "Model loading took" lines are authoritative and only exist in
the log stream). BUCKET_START/BUCKET_END marker lines let parse_cell.py
attribute the periodic engine-log stats to buckets.

Emits a single ``RESULT_JSON: {...}`` line and writes the same object to
--out. Fields that could not be measured are null — the doc keeps NaN there.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s bench %(message)s")
logger = logging.getLogger("bench")

# Production recipe constants (sampler_configs.py NEMOTRON_3_ULTRA B200/S256K
# + types.py DEFAULT_CUDAGRAPH_CAPTURE_SIZES + vllm_server NemotronH-NVFP4
# defaulting). Keep in sync manually — this artefact must not import the repo.
MAX_LORA_RANK = 64
MAX_LORAS = 4
MAX_NUM_SEQS = 1000
CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 640, 768, 896, 1000]


def _engine_kwargs(args) -> dict:
    """Production-recipe kwargs, filtered to what this vLLM build accepts.

    Required knobs raise if unsupported; nice-to-have ones are dropped with a
    log line so the doc can record the delta.
    """
    from vllm.engine.arg_utils import EngineArgs

    fields = {f.name for f in dataclasses.fields(EngineArgs)}

    # Everything here is either a stock vLLM kwarg or spike-proven on this
    # exact venv (nvfp4_spike/run_rebase_nvfp4.sh passed disable_custom_all_reduce,
    # enable_flashinfer_autotune, moe_backend straight into LLM() on 0.22).
    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=True,
    )
    if args.recipe == "nemotron-nvfp4":
        # The Ultra NVFP4 serving recipe (vllm_server auto-defaults + golden
        # config): fp8 KV scheme, CUTLASS NvFp4 MoE (Marlin broken), autotune
        # segfault workaround, custom-AR capture crash workaround.
        kwargs.update(
            kv_cache_dtype="fp8",
            disable_custom_all_reduce=True,
            enable_flashinfer_autotune=False,
            moe_backend="cutlass",
        )
    # nemotron-bf16 (Super): golden leaf sets none of those four — bf16 KV,
    # vLLM-default MoE backend and all-reduce.

    # Production knobs whose kwarg spelling varies by build; probe and drop.
    optional = dict(load_format="fastsafetensors")
    for name in ("cudagraph_capture_sizes", "cuda_graph_sizes"):
        if name in fields:
            optional[name] = CAPTURE_SIZES
            break
    else:
        logger.info("no cudagraph capture-size EngineArgs field; using vLLM default")

    for k, v in optional.items():
        if k in fields:
            kwargs[k] = v
        else:
            logger.info("dropping unsupported engine kwarg %s", k)

    if args.enable_lora:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = MAX_LORA_RANK
        kwargs["max_loras"] = MAX_LORAS

    return kwargs


def _capacity_info(llm) -> dict:
    """Best-effort introspection (same probes as nvfp4_compare/score.py).
    The vLLM boot-log lines are authoritative; parse_cell.py reads those."""
    info: dict = {}
    try:
        engine = getattr(llm, "llm_engine", None)
        cache_cfg = getattr(getattr(engine, "cache_config", None), "__dict__", {})
        for key in ("num_gpu_blocks", "num_gpu_blocks_override", "block_size"):
            if key in cache_cfg:
                info[key] = cache_cfg[key]
        model_cfg = getattr(engine, "model_config", None)
        if model_cfg is not None:
            info["max_model_len"] = getattr(model_cfg, "max_model_len", None)
        sched_cfg = getattr(engine, "scheduler_config", None)
        if sched_cfg is not None:
            info["max_num_seqs"] = getattr(sched_cfg, "max_num_seqs", None)
            info["max_num_batched_tokens"] = getattr(
                sched_cfg, "max_num_batched_tokens", None
            )
    except Exception:
        logger.exception("capacity introspection failed (non-fatal)")
    blocks = info.get("num_gpu_blocks")
    bs = info.get("block_size")
    if blocks and bs:
        info["kv_pool_tokens_introspected"] = int(blocks) * int(bs)
    return info


def _log_file_capacity(log_file: str) -> dict:
    """The engine logs the authoritative pool/concurrency during init; by the
    time boot returns, run_cell.sh's tee has flushed those lines to disk."""
    import re

    info: dict = {}
    try:
        text = open(log_file, errors="replace").read()
    except OSError:
        return info
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if m:
        info["kv_pool_tokens_logged"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", text)
    if m:
        info["max_concurrency_logged"] = float(m.group(1))
    return info


def _pick_num_requests(args, capacity: dict, ctx: int) -> int:
    if args.num_requests > 0:
        return args.num_requests
    pool = capacity.get("kv_pool_tokens_logged") or capacity.get(
        "kv_pool_tokens_introspected"
    )
    if not pool:
        logger.warning("no KV pool signal anywhere; falling back to N=64")
        return 64
    conc = pool / ctx
    n = max(4, math.ceil(conc * 1.05) + 1)
    return min(n, MAX_NUM_SEQS, args.cap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, required=True, choices=(2, 4, 8))
    ap.add_argument(
        "--recipe",
        default="nemotron-nvfp4",
        choices=("nemotron-nvfp4", "nemotron-bf16"),
        help="engine-arg recipe: Ultra NVFP4 leaf vs Super BF16 leaf",
    )
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--model", required=True, help="local NVFP4 snapshot dir")
    ap.add_argument("--adapter", default="", help="PEFT dir; empty = no LoRA req")
    ap.add_argument("--enable-lora", type=int, default=1)
    ap.add_argument(
        "--data-lens",
        default="8192,32768,131072,262144",
        help="comma-separated actual context buckets driven through one engine",
    )
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--num-requests", type=int, default=0, help="0 = auto-size")
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--vocab-size", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot-only", action="store_true")
    ap.add_argument("--log-file", default="", help="tee'd log path for capacity readback")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logger.info(
        "cell tp=%d ctx=%d CUDA_VISIBLE_DEVICES=%s",
        args.tp,
        args.max_model_len,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kwargs = _engine_kwargs(args)
    logger.info("engine kwargs: %s", kwargs)

    t0 = time.monotonic()
    llm = LLM(**kwargs)
    boot_wall_s = time.monotonic() - t0
    logger.info("BOOT_WALL_S %.1f", boot_wall_s)

    capacity = _capacity_info(llm)
    if args.log_file:
        capacity.update(_log_file_capacity(args.log_file))
    logger.info("capacity: %s", capacity)

    result: dict = {
        "tp": args.tp,
        "max_model_len": args.max_model_len,
        "adapter": args.adapter or None,
        "boot_wall_s": round(boot_wall_s, 1),
        "capacity": capacity,
        "engine_kwargs": {k: str(v) for k, v in kwargs.items()},
    }

    if not args.boot_only:
        lora_req = None
        if args.adapter:
            from vllm.lora.request import LoRARequest

            lora_req = LoRARequest("bench_adapter", 1, args.adapter)

        sp = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=args.max_new_tokens,
            ignore_eos=True,
        )

        buckets = []
        rng = random.Random(args.seed)
        for ctx in sorted(int(x) for x in args.data_lens.split(",") if x.strip()):
            if ctx > args.max_model_len:
                logger.warning("skipping bucket ctx=%d > max_model_len", ctx)
                continue
            n = _pick_num_requests(args, capacity, ctx)
            prompt_len = ctx - args.max_new_tokens - 16
            prompts = [
                TokensPrompt(
                    prompt_token_ids=[
                        rng.randrange(100, args.vocab_size - 100)
                        for _ in range(prompt_len)
                    ]
                )
                for _ in range(n)
            ]
            logger.info(
                "BUCKET_START ctx=%d n=%d prompt_len=%d gen=%d",
                ctx, n, prompt_len, args.max_new_tokens,
            )
            t1 = time.monotonic()
            outputs = llm.generate(prompts, sp, lora_request=lora_req)
            gen_wall_s = time.monotonic() - t1

            prompt_toks = sum(len(p["prompt_token_ids"]) for p in prompts)
            gen_toks = sum(len(o.outputs[0].token_ids) for o in outputs)
            # Continuous batching interleaves prefill/decode, so per-phase
            # splits from wall time alone are approximations; the periodic
            # engine log lines carry instantaneous rates (parse_cell.py).
            bucket = {
                "ctx": ctx,
                "num_requests": n,
                "prompt_len": prompt_len,
                "max_new_tokens": args.max_new_tokens,
                "prompt_tokens": prompt_toks,
                "gen_tokens": gen_toks,
                "gen_wall_s": round(gen_wall_s, 1),
                "total_tok_per_s": round((prompt_toks + gen_toks) / gen_wall_s, 1),
                "gen_tok_per_s": round(gen_toks / gen_wall_s, 1),
                "total_tok_per_s_per_gpu": round(
                    (prompt_toks + gen_toks) / gen_wall_s / args.tp, 1
                ),
                "finished": sum(1 for o in outputs if o.finished),
            }
            buckets.append(bucket)
            logger.info("BUCKET_END ctx=%d result=%s", ctx, json.dumps(bucket))
        result["buckets"] = buckets

    line = json.dumps(result, sort_keys=True)
    with open(args.out, "w") as f:
        f.write(line + "\n")
    print(f"RESULT_JSON: {line}", flush=True)


if __name__ == "__main__":
    main()
