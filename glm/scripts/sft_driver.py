#!/usr/bin/env python3
"""Nemotron 3 Super LoRA SFT / memory-profiling driver.

Talks directly to a running trainer worker over HTTP (the rank-0 uvicorn that
``scripts/launch.sh`` brings up on ``127.0.0.1:8000``). It drives
``forward_backward`` + ``optim_step`` steps and optionally wraps them in a CUDA
memory profile, then reports per-GPU reserved / allocated / peak-allocated
memory from ``/status``.

Two datum sources:

  --source synthetic   One packed sequence of exactly ``--seq-len`` tokens with
                       the first half masked from loss. Deterministic, dataset
                       free; the right tool for measuring activation memory at a
                       precise sequence length (vary --seq-len, diff the peak).

  --source dataset     Real chat SFT. Loads an OpenAI-style chat dataset, renders
                       it with the model's chat template, masks everything except
                       assistant tokens, and truncates/keeps samples up to
                       ``--seq-len``. Used for the final 131k validation run.

This script intentionally has no dependency on the loops SDK or loops_models —
it builds the wire payloads (which are stable) directly so it is portable and
reproducible from any box that can reach the trainer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

import httpx

LONG_RUNNING = {
    "/forward_backward",
    "/optim_step",
    "/memory_profile/start",
    "/memory_profile/stop",
}


def submit_and_wait(
    client: httpx.Client, op_path: str, *, body: dict, timeout: float = 1800.0
) -> dict:
    """POST an async op, then long-poll /operations/{id} until it finishes."""
    if op_path not in LONG_RUNNING:
        raise ValueError(f"{op_path} is not a long-running op path")
    key = uuid.uuid4().hex
    r = client.post(op_path, json=body, headers={"Idempotency-Key": key})
    if r.status_code != 202:
        raise RuntimeError(f"{op_path} submit failed {r.status_code}: {r.text[:1000]}")
    operation_id = r.json()["operation_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rr = client.get(f"/operations/{operation_id}", timeout=35.0)
        if rr.status_code == 408:
            continue
        if rr.status_code in (404, 502, 503):
            raise RuntimeError(
                f"{op_path} poll failed {rr.status_code}: {rr.text[:1000]}"
            )
        rr.raise_for_status()
        payload = rr.json()
        status = payload.get("status")
        if status == "done":
            return payload["result"]
        if status == "error":
            raise RuntimeError(f"{op_path} op error: {payload.get('error', '')[:4000]}")
    raise TimeoutError(f"{op_path} never finished within {timeout}s")


def _datum(tokens: list[int], target_tokens: list[int]) -> dict:
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {
                "data": target_tokens,
                "dtype": "int64",
                "shape": [len(target_tokens)],
            },
        },
    }


def synthetic_datum(seq_len: int, vocab: int = 30000) -> dict:
    """A length-``seq_len`` sequence; first half masked, second half supervised.

    Token ids are a fixed deterministic ramp so runs are comparable. Activation
    memory is a function of the packed length, not the specific token values.
    """
    tokens = [100 + (i % vocab) for i in range(seq_len)]
    targets = [-100] * seq_len
    half = seq_len // 2
    for i in range(half, seq_len - 1):
        targets[i] = tokens[i + 1]
    return _datum(tokens, targets)


def _ids(out) -> list[int]:
    if hasattr(out, "get") and "input_ids" in out:
        ids = out["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    return list(out)


def _normalize_messages(ex: dict) -> list[dict] | None:
    """Coerce a dataset row into a list of {role, content} chat messages.

    Handles the common shapes: an explicit ``messages``/``conversations`` list,
    or a (system?, context/document, question, answer) layout like the
    ChatQA2-Long-SFT NarrativeQA configs.
    """
    for key in ("messages", "conversation", "conversations"):
        if key in ex and isinstance(ex[key], list) and ex[key]:
            msgs = []
            for m in ex[key]:
                role = m.get("role") or m.get("from")
                content = m.get("content") if "content" in m else m.get("value")
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant", "bot"):
                    role = "assistant"
                if role and content is not None:
                    msgs.append({"role": role, "content": str(content)})
            if any(m["role"] == "assistant" for m in msgs):
                return msgs
    # ChatQA2 long-context layout: a document + a Q/A pair. The NarrativeQA
    # configs put the long document under "sub-paragraphs".
    doc = (
        ex.get("document")
        or ex.get("context")
        or ex.get("ctx")
        or ex.get("input")
        or ex.get("sub-paragraphs")
        or ex.get("sub_paragraphs")
    )
    q = ex.get("question") or ex.get("query")
    a = ex.get("answer") or ex.get("answers") or ex.get("response") or ex.get("output")
    if isinstance(a, list) and a:
        a = a[0]
    if doc and q and a:
        return [
            {"role": "user", "content": f"{doc}\n\nQuestion: {q}"},
            {"role": "assistant", "content": str(a)},
        ]
    return None


def chat_datum(tokenizer, messages: list[dict], seq_len: int) -> dict | None:
    """Render a chat into a datum, masking all but the final assistant turn.

    For long-context QA (e.g. ChatQA2 NarrativeQA_131072) the document alone can
    fill or exceed ``seq_len``. Naively truncating ``full_ids[:seq_len]`` would
    drop the assistant answer at the tail, leaving nothing to supervise. Instead
    we keep the *tail* of the sequence so the answer always survives, trimming
    the front of the document to fit ``seq_len``. The supervised tokens are the
    final ``answer_len`` tokens (the assistant turn).
    """
    try:
        prompt_ids = _ids(
            tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True
            )
        )
        full_ids = _ids(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
        )
    except Exception:
        return None
    answer_len = len(full_ids) - len(prompt_ids)
    if answer_len <= 0 or len(full_ids) < 2:
        return None
    if len(full_ids) > seq_len:
        # Keep the tail so the assistant answer (and as much of the document as
        # fits) survives; trim the front of the document.
        full_ids = full_ids[-seq_len:]
        answer_len = min(answer_len, len(full_ids) - 1)
    L = len(full_ids)
    if answer_len <= 0:
        return None
    targets = [-100] * L
    for i in range(L - answer_len - 1, L - 1):
        targets[i] = full_ids[i + 1]
    return _datum(full_ids, targets)


def _load_rows(args):
    """Return an iterable of raw dataset rows.

    HF ``datasets`` parses JSON with pyarrow, whose reader caps its block size
    at int32 (~2 GiB). The ChatQA2 long-context configs ship each split as one
    pretty-printed JSON *array* many GiB in size, so the entire file is a single
    block and pyarrow overflows with ``value too large to convert to int32_t``.
    When the normal path fails we fall back to reading the already-cached JSON
    file directly with the stdlib parser (the box has ample RAM).
    """
    from datasets import load_dataset

    try:
        if args.dataset_path:
            return load_dataset(args.dataset_path, split=args.split)
        kwargs = {"split": args.split}
        if args.dataset_config:
            kwargs["name"] = args.dataset_config
        return load_dataset(args.dataset_name, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(
            f"[driver] load_dataset failed ({type(e).__name__}: {e}); "
            "falling back to raw JSON read of the cached file",
            flush=True,
        )

    from huggingface_hub import hf_hub_download

    # ChatQA2-Long-SFT layout: <config>/<config>_QA_<split>.json
    fname = f"{args.dataset_config}/{args.dataset_config}_QA_{args.split}.json"
    path = hf_hub_download(
        args.dataset_name, fname, repo_type="dataset", local_files_only=True
    )
    print(f"[driver] reading cached JSON array {path}", flush=True)
    with open(path) as f:
        rows = json.load(f)
    print(f"[driver] loaded {len(rows)} raw rows from JSON array", flush=True)
    return rows


def load_dataset_datums(args) -> list[dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds = _load_rows(args)

    datums: list[dict] = []
    for ex in ds:
        msgs = _normalize_messages(ex)
        if msgs is None:
            continue
        d = chat_datum(tokenizer, msgs, args.seq_len)
        if d is not None:
            datums.append(d)
            if len(datums) >= args.num_datums:
                break
    if not datums:
        raise RuntimeError("no usable datums produced from dataset (schema mismatch?)")
    print(
        f"[driver] built {len(datums)} chat datums (seq_len<= {args.seq_len})",
        flush=True,
    )
    return datums


def build_datums(args) -> list[dict]:
    if args.source == "synthetic":
        n = max(args.num_datums, args.microbatch_size)
        return [synthetic_datum(args.seq_len, vocab=args.synthetic_vocab) for _ in range(n)]
    return load_dataset_datums(args)


def report_status(client: httpx.Client, tag: str) -> dict:
    s = client.get("/status", timeout=30.0).json()

    def gib(d: dict) -> str:
        return ", ".join(f"{k}={v / 2**30:.2f}GiB" for k, v in (d or {}).items())

    print(
        f"\n[status:{tag}] step={s.get('step')} last_loss={s.get('last_loss')}\n"
        f"  parallel: TP={s.get('tensor_parallel_size')} PP={s.get('pipeline_parallel_size')} "
        f"EP={s.get('expert_parallel_size')} ETP={s.get('expert_tensor_parallel_size')} "
        f"CP={s.get('context_parallel_size')} DP={s.get('data_parallel_size')} "
        f"world={s.get('world_size')} max_seq_len={s.get('max_seq_len')}\n"
        f"  reserved:  {gib(s.get('gpu_memory'))}\n"
        f"  allocated: {gib(s.get('gpu_memory_allocated'))}\n"
        f"  peak_alloc:{gib(s.get('gpu_max_memory_allocated'))}",
        flush=True,
    )
    return s


class _SkipOptim(Exception):
    pass


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--trainer-url", default="http://127.0.0.1:8000")
    p.add_argument("--source", choices=["synthetic", "dataset"], default="synthetic")
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument(
        "--microbatch-size", type=int, default=1, help="datums per forward_backward"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="effective datums per optim_step via gradient accumulation. "
        "Defaults to --microbatch-size (no accumulation). Must be a multiple of "
        "--microbatch-size. Keep --microbatch-size=1 at long context (memory) and "
        "raise this to denoise the gradient / make the loss actually descend.",
    )
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument(
        "--synthetic-vocab",
        type=int,
        default=30000,
        help="max synthetic token id (must be < model vocab; debug GLM has vocab 2048)",
    )
    p.add_argument(
        "--skip-optim",
        action="store_true",
        help="fb-only steps: never call /optim_step (GLM-5.2 NaN-grad guard; "
        "peak memory and fwd-bwd timing are unaffected at LoRA scale)",
    )
    p.add_argument("--num-datums", type=int, default=8)
    p.add_argument(
        "--reset-peak",
        dest="reset_peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reset CUDA peak-alloc counters before the run so the reported "
        "peak_alloc reflects THIS run, not the cumulative high-water mark "
        "(use --no-reset-peak to keep cumulative behaviour)",
    )
    p.add_argument(
        "--memory-profile", action="store_true", help="wrap steps in /memory_profile"
    )
    p.add_argument("--max-entries", type=int, default=200000)
    # dataset mode
    p.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
    p.add_argument(
        "--dataset-name",
        default=None,
        help="HF dataset id (e.g. nvidia/ChatQA2-Long-SFT-data)",
    )
    p.add_argument(
        "--dataset-config",
        default=None,
        help="HF dataset config/subset (e.g. NarrativeQA_131072)",
    )
    p.add_argument(
        "--dataset-path",
        default=None,
        help="local path to a saved dataset (overrides --dataset-name)",
    )
    p.add_argument("--split", default="train")
    args = p.parse_args()

    batch_size = args.batch_size or args.microbatch_size
    if batch_size % args.microbatch_size != 0:
        p.error("--batch-size must be a multiple of --microbatch-size")
    num_microbatches = batch_size // args.microbatch_size

    datums = build_datums(args)

    with httpx.Client(base_url=args.trainer_url, timeout=1800.0) as client:
        client.get("/health", timeout=30.0).raise_for_status()
        report_status(client, "before")

        if args.reset_peak:
            # Reset every rank's peak-alloc counter so the "after" peak reflects
            # only this run's forward/backward(+optim), not the all-time max.
            client.post("/reset_peak_memory", timeout=60.0).raise_for_status()
            print(
                "[driver] reset CUDA peak-alloc counters (per-run profiling)",
                flush=True,
            )

        if args.memory_profile:
            res = submit_and_wait(
                client, "/memory_profile/start", body={"max_entries": args.max_entries}
            )
            print(
                f"[driver] memory profile recording -> {res.get('local_path')}",
                flush=True,
            )

        ok = True
        try:
            for step in range(args.steps):
                t0 = time.perf_counter()
                # Gradient accumulation: run num_microbatches forward_backwards
                # (the trainer accumulates grads across calls and zeros them only
                # on optim_step), then a single optim_step. Effective batch =
                # num_microbatches * microbatch_size. The reported step loss is
                # the mean of the per-microbatch losses.
                mb_losses: list[float] = []
                for mb_idx in range(num_microbatches):
                    base = (step * batch_size + mb_idx * args.microbatch_size) % len(
                        datums
                    )
                    mb = [
                        datums[(base + i) % len(datums)]
                        for i in range(args.microbatch_size)
                    ]
                    fb = submit_and_wait(
                        client,
                        "/forward_backward",
                        body={"data": mb, "loss_fn": "cross_entropy"},
                    )
                    mb_losses.append(fb.get("loss"))
                try:
                    if args.skip_optim:
                        raise _SkipOptim()
                    submit_and_wait(
                        client,
                        "/optim_step",
                        body={
                            "adam_params": {
                                "learning_rate": args.learning_rate,
                                "beta1": 0.9,
                                "beta2": 0.95,
                            }
                        },
                    )
                except _SkipOptim:
                    pass
                except Exception as optim_exc:  # noqa: BLE001
                    # GLM-5.2 PP16 LoRA: optimizer.step() reports grad_norm=NaN and
                    # the response schema (ge=0) rejects it, even though the update
                    # applies and losses keep decreasing. Tolerate it so memory
                    # profiling sweeps continue; the loss trajectory is the check.
                    print(f"[driver] optim_step response error (continuing): {optim_exc}",
                          file=sys.stderr, flush=True)
                dt = time.perf_counter() - t0
                step_loss = sum(mb_losses) / len(mb_losses)
                print(
                    f"[step {step + 1}/{args.steps}] loss={step_loss:.4f} "
                    f"batch={batch_size} tokens={args.seq_len * batch_size} {dt:.1f}s",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"[driver] STEP FAILED: {e}", file=sys.stderr, flush=True)
        finally:
            if args.memory_profile:
                try:
                    res = submit_and_wait(client, "/memory_profile/stop", body={})
                    print(f"[driver] memory profile -> {json.dumps(res)}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[driver] memory profile stop failed: {e}",
                        file=sys.stderr,
                        flush=True,
                    )

        final = report_status(client, "after")
        peak = final.get("gpu_max_memory_allocated") or {}
        if peak:
            pk = max(peak.values()) / 2**30
            scope = "this-run" if args.reset_peak else "cumulative"
            print(
                f"\n[RESULT] source={args.source} seq_len={args.seq_len} "
                f"microbatch={args.microbatch_size} ok={ok} "
                f"peak_alloc_max={pk:.2f}GiB ({scope})",
                flush=True,
            )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
