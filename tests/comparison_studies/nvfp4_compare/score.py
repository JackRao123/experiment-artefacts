#!/usr/bin/env python3
"""NVFP4-vs-bf16 spike: GPU scoring stage (vLLM offline).

This is the GPU half of the Nemotron-3-Ultra NVFP4 go/no-go spike. It does
exactly one of three jobs per invocation so each can run on the node set that
fits the checkpoint it needs (NVFP4 base fits one B200 node; bf16 base does
not):

  generate  Run the *behavior* policy: prompts -> completions, capturing the
            sampled token ids and their per-token logprobs. Run this against
            the NVFP4 checkpoint + adapter, because in RL the NVFP4 sampler is
            what produces rollouts.

  rescore   Run the *target* policy: given token sequences emitted by a prior
            `generate` run, compute this checkpoint's per-token logprob for the
            exact same tokens (teacher-forced, prompt_logprobs). Run this
            against the bf16 checkpoint + the same adapter. Optionally run it
            again against the NVFP4 checkpoint itself as a sanity floor
            (generate vs rescore on the same model should be ~0 divergence).

  boot      Initialize the engine and dump KV/Mamba cache capacity, then exit.
            Run against NVFP4 and bf16 on the same node count to read the
            concurrency headroom NVFP4 buys.

The CPU divergence/ESS/clip metrics live in compare.py so they can run
anywhere without a GPU, matching the swift_compare / tinker_compare split.

IMPORTANT (read before running): vLLM's offline API surface (TokensPrompt,
prompt_logprobs shape, LoRA kwargs, capacity introspection) shifts between
releases. This spike is also how we pin which vLLM serves NemotronH NVFP4 +
unmerged LoRA at all, so expect to adjust the small clearly-marked
``# vLLM-API`` spots for whatever build the spike settles on. Failures here
ARE findings -- e.g. "LoRA on FP8 Mamba in_proj raises X" is the answer we
came for.

Usage:
    python score.py generate \
        --model /weights/nemotron3-ultra-nvfp4 \
        --adapter /weights/step-199 \
        --prompts prompts.jsonl \
        --tp 8 --max-model-len 131072 --max-new-tokens 512 \
        --out gen_nvfp4.json

    python score.py rescore \
        --model /weights/nemotron3-ultra-bf16 \
        --adapter /weights/step-199 \
        --sequences gen_nvfp4.json \
        --tp 32 --pp 1 --max-model-len 131072 \
        --out rescore_bf16.json

    python score.py boot --model /weights/nemotron3-ultra-nvfp4 --tp 8 \
        --max-model-len 131072 --out boot_nvfp4.json

prompts.jsonl: one JSON object per line with either {"prompt": "<text>"} or
{"prompt_token_ids": [int, ...]}. An optional "id" field is preserved; absent
ids default to the line index.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("nvfp4_spike")


# ── prompt / sequence IO ────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj.setdefault("id", i)
            rows.append(obj)
    return rows


def _build_llm(args, *, enable_lora: bool):
    """Construct a vLLM offline engine. The kwargs here are the load-bearing
    ones for this model family; everything else stays on vLLM defaults so the
    spike measures the stock serving path."""
    from vllm import LLM  # vLLM-API: import location stable across recent vLLM

    kwargs: dict[str, Any] = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        trust_remote_code=True,  # NemotronH ships custom config/modeling code
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    if args.pp and args.pp > 1:
        # bf16 Ultra spans nodes; expert-parallel matches the trainer topology.
        kwargs["enable_expert_parallel"] = True
    if args.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype  # NVFP4 recipe uses fp8
    if enable_lora:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = args.max_lora_rank
    if args.extra_engine_kwargs:
        kwargs.update(json.loads(args.extra_engine_kwargs))
    logger.info("Building vLLM engine: %s", {k: v for k, v in kwargs.items()})
    return LLM(**kwargs)


def _lora_request(args):
    if not args.adapter:
        return None
    from vllm.lora.request import LoRARequest  # vLLM-API

    return LoRARequest("spike_adapter", 1, args.adapter)


# ── capacity introspection ──────────────────────────────────────────────


def _capacity_info(llm) -> dict[str, Any]:
    """Best-effort read of KV/Mamba cache capacity. vLLM exposes this
    inconsistently across versions, so probe several spots and record whatever
    is present rather than asserting one shape. The trainers sampler already
    captures the same 'max_concurrency * max_model_len' figure in
    sampler/_vllm_patches.py; this mirrors that intent for the spike."""
    info: dict[str, Any] = {}
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
    info["note"] = (
        "Cross-check against vLLM's own boot log lines 'GPU KV cache size' and "
        "'Maximum concurrency' -- those are the authoritative numbers for "
        "hybrid Mamba+attention layouts."
    )
    return info


# ── modes ───────────────────────────────────────────────────────────────


def mode_generate(args) -> dict[str, Any]:
    from vllm import SamplingParams  # vLLM-API

    rows = _read_jsonl(Path(args.prompts))
    llm = _build_llm(args, enable_lora=bool(args.adapter))
    lora_req = _lora_request(args)

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        # logprobs>=1 guarantees the sampled token's own logprob is returned
        # even when it is not the argmax. We read it back by token id below.
        logprobs=1,
        seed=args.seed,
    )

    prompts_arg, used_token_ids = _prompts_for_generate(rows)
    outputs = llm.generate(prompts_arg, sp, lora_request=lora_req)

    sequences = []
    for row, out in zip(rows, outputs):
        comp = out.outputs[0]
        comp_token_ids = list(comp.token_ids)
        comp_logprobs = _sampled_token_logprobs(comp_token_ids, comp.logprobs)
        sequences.append(
            {
                "id": row["id"],
                "prompt_token_ids": (
                    list(out.prompt_token_ids)
                    if out.prompt_token_ids is not None
                    else row.get("prompt_token_ids")
                ),
                "completion_token_ids": comp_token_ids,
                "completion_logprobs": comp_logprobs,
            }
        )

    return {
        "mode": "generate",
        "model": args.model,
        "adapter": args.adapter,
        "used_token_id_prompts": used_token_ids,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "capacity": _capacity_info(llm),
        "sequences": sequences,
    }


def mode_rescore(args) -> dict[str, Any]:
    from vllm import SamplingParams  # vLLM-API

    prior = json.loads(Path(args.sequences).read_text())
    src_seqs = prior["sequences"]
    llm = _build_llm(args, enable_lora=bool(args.adapter))
    lora_req = _lora_request(args)

    # Teacher-force the exact prompt+completion tokens and read prompt_logprobs.
    full_token_ids = [
        (s["prompt_token_ids"] or []) + s["completion_token_ids"] for s in src_seqs
    ]
    prompts_arg = _token_prompts(full_token_ids)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outputs = llm.generate(prompts_arg, sp, lora_request=lora_req)

    sequences = []
    for s, out in zip(src_seqs, outputs):
        prompt_len = len(s["prompt_token_ids"] or [])
        comp_ids = s["completion_token_ids"]
        comp_logprobs = _prompt_logprobs_for_completion(
            full_ids=(s["prompt_token_ids"] or []) + comp_ids,
            prompt_len=prompt_len,
            prompt_logprobs=out.prompt_logprobs,
        )
        sequences.append(
            {
                "id": s["id"],
                "prompt_token_ids": s["prompt_token_ids"],
                "completion_token_ids": comp_ids,
                "completion_logprobs": comp_logprobs,
            }
        )

    return {
        "mode": "rescore",
        "model": args.model,
        "adapter": args.adapter,
        "rescored_from": str(args.sequences),
        "capacity": _capacity_info(llm),
        "sequences": sequences,
    }


def mode_boot(args) -> dict[str, Any]:
    llm = _build_llm(args, enable_lora=bool(args.adapter))
    return {
        "mode": "boot",
        "model": args.model,
        "adapter": args.adapter,
        "capacity": _capacity_info(llm),
    }


# ── vLLM-API shims (the spots most likely to need a version tweak) ──────


def _prompts_for_generate(rows: list[dict[str, Any]]):
    """Return (prompts_arg, used_token_ids). Prefer text prompts; fall back to
    pre-tokenized prompts if that's what the file carries."""
    if all("prompt" in r for r in rows):
        return [r["prompt"] for r in rows], False
    if all(r.get("prompt_token_ids") for r in rows):
        return _token_prompts([r["prompt_token_ids"] for r in rows]), True
    raise SystemExit(
        "prompts.jsonl rows must each have 'prompt' (text) or 'prompt_token_ids'"
    )


def _token_prompts(token_id_lists: list[list[int]]):
    """Wrap pre-tokenized prompts in whatever vLLM expects this version.
    Newer vLLM wants TokensPrompt objects; older accepts a prompt_token_ids
    list. Try the modern path first."""
    try:
        from vllm.inputs import TokensPrompt  # vLLM-API

        return [TokensPrompt(prompt_token_ids=ids) for ids in token_id_lists]
    except Exception:
        # vLLM-API: very old fallback -- caller passes prompt_token_ids kwarg.
        return token_id_lists


def _sampled_token_logprobs(token_ids, step_logprobs) -> list[float]:
    """Extract the logprob of each *sampled* token from generation output.
    ``step_logprobs[t]`` is a dict {token_id: Logprob} for decode step t."""
    out: list[float] = []
    for tid, lp_dict in zip(token_ids, step_logprobs or []):
        entry = lp_dict.get(tid) if lp_dict else None
        out.append(float(entry.logprob) if entry is not None else float("nan"))
    return out


def _prompt_logprobs_for_completion(
    full_ids: list[int], prompt_len: int, prompt_logprobs
) -> list[float]:
    """Pull per-token logprobs for the completion region out of
    ``prompt_logprobs``. ``prompt_logprobs[i]`` is the model's logprob dict for
    full_ids[i] given full_ids[:i]; index 0 is None. We want positions
    [prompt_len, len(full_ids))."""
    out: list[float] = []
    if prompt_logprobs is None:
        return [float("nan")] * (len(full_ids) - prompt_len)
    for i in range(prompt_len, len(full_ids)):
        lp_dict = prompt_logprobs[i] if i < len(prompt_logprobs) else None
        tid = full_ids[i]
        entry = lp_dict.get(tid) if lp_dict else None
        out.append(float(entry.logprob) if entry is not None else float("nan"))
    return out


# ── cli ─────────────────────────────────────────────────────────────────


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HF path/id of the base checkpoint")
    p.add_argument(
        "--adapter", default=None, help="LoRA adapter dir (PEFT safetensors)"
    )
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--kv-cache-dtype", default=None, help="e.g. fp8 (NVFP4 recipe)")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument(
        "--extra-engine-kwargs",
        default=None,
        help="JSON dict merged into the LLM() kwargs for version-specific flags",
    )
    p.add_argument("--out", type=Path, required=True)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser(
        "generate", help="behavior policy: prompts -> completions+logprobs"
    )
    _add_common(g)
    g.add_argument("--prompts", required=True)
    g.add_argument("--max-new-tokens", type=int, default=512)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("rescore", help="target policy: teacher-force prior tokens")
    _add_common(r)
    r.add_argument("--sequences", required=True, help="a prior generate output JSON")

    b = sub.add_parser("boot", help="init engine, dump cache capacity, exit")
    _add_common(b)

    args = ap.parse_args()
    handler = {"generate": mode_generate, "rescore": mode_rescore, "boot": mode_boot}[
        args.cmd
    ]
    report = handler(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
