#!/usr/bin/env python3
"""256k logprob-mismatch probe client.

One file, five subcommands, shared sampled-token JSON schema:

  wait-health       poll sampler /health until 200
  make-inputs       build the 3-regime prompt set (smoke / long-response / long-prompt)
  generate          NVFP4 sampler rollouts via /v1/completions with a
                    token-controlled continuation loop (behavior policy)
  sampler-rescore   teacher-force the same tokens through /v1/completions
                    prompt_logprobs (target policy, vLLM engine)
  megatron-rescore  teacher-force the same tokens through the trainer's
                    POST /forward + GET /operations/{id} long-poll
  compare           pairwise drift/k3/ESS/clip tables, grouped by regime

All engine outputs are written in the same shape so compare() is trivial:
  {"sequences": [{"id", "regime", "prompt_token_ids",
                  "completion_token_ids", "completion_logprobs", "metadata"}]}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

EXP = Path(os.environ.get("EXP", "/root/.cache/user_artifacts/nvfp4_256k_logprob"))
BASE = os.environ.get("SAMPLER_URL", "http://127.0.0.1:8000")


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


_MODEL_ID_CACHE: str | None = None


def served_model_id() -> str:
    """The sampler registers the model under its snapshot path, not 'default'."""
    global _MODEL_ID_CACHE
    if _MODEL_ID_CACHE is None:
        data = http_json("GET", BASE + "/v1/models", None, timeout=30)
        _MODEL_ID_CACHE = data["data"][0]["id"]
    return _MODEL_ID_CACHE


def wait_health(timeout: float = 3600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(BASE + "/health", timeout=2).read()
            print("SAMPLER_HEALTHY", flush=True)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("sampler health timeout")


# ── token/logprob extraction (vLLM 0.22 /v1/completions shapes) ─────────


def parse_token_id(tok: Any) -> int:
    """With return_tokens_as_token_ids, tokens arrive as 'token_id:<int>'."""
    if isinstance(tok, int):
        return tok
    s = str(tok)
    if s.startswith("token_id:"):
        return int(s.split(":", 1)[1])
    if s.lstrip("-").isdigit():
        return int(s)
    raise ValueError(f"cannot parse token id from {tok!r}")


def extract_completion(choice: dict[str, Any]) -> tuple[list[int], list[float], str]:
    lp = choice.get("logprobs") or {}
    toks = lp.get("tokens")
    token_lps = lp.get("token_logprobs")
    if toks is None:
        raise KeyError(
            f"no tokens in choice: keys={sorted(choice.keys())} "
            f"logprobs_keys={sorted(lp.keys())}"
        )
    ids = [parse_token_id(t) for t in toks]
    if token_lps is None:
        token_lps = [None] * len(ids)
    lps = [float("nan") if v is None else float(v) for v in token_lps[: len(ids)]]
    finish = str(choice.get("finish_reason") or choice.get("stop_reason") or "")
    return ids, lps, finish


def extract_prompt_lps(
    data: dict[str, Any], full_ids: list[int], prompt_len: int
) -> list[float]:
    """prompt_logprobs[i] is {token_id_str: {"logprob": ...}} for full_ids[i];
    index 0 is None. We want the completion region [prompt_len, len)."""
    choice = (data.get("choices") or [{}])[0]
    pl = choice.get("prompt_logprobs") or data.get("prompt_logprobs")
    if not isinstance(pl, list):
        lp_obj = choice.get("logprobs") or {}
        raise KeyError(
            f"no prompt_logprobs: data_keys={sorted(data.keys())} "
            f"choice_keys={sorted(choice.keys())} logprobs_keys={sorted(lp_obj.keys())}"
        )
    out: list[float] = []
    for i in range(prompt_len, len(full_ids)):
        entry = pl[i] if i < len(pl) else None
        val = None
        if isinstance(entry, dict):
            tid = full_ids[i]
            e = entry.get(str(tid), entry.get(tid))
            if isinstance(e, dict):
                val = e.get("logprob")
            elif e is not None:
                val = e
        out.append(float("nan") if val is None else float(val))
    return out


# ── make-inputs ─────────────────────────────────────────────────────────


def make_inputs(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def enc(s: str) -> list[int]:
        return tok.encode(s, add_special_tokens=False)

    seed = (
        "Generate JSONL arithmetic word problems. Output only JSON objects, one per "
        "line, with fields id, question, reasoning, answer. Continue with the next "
        "id each line.\n"
    )
    chunk = (
        "Document section: This is synthetic long-context filler about arithmetic "
        "reasoning, units, rates, proportions, and verification. The facts are "
        "intentionally repetitive but varied enough to exercise long-position "
        "attention and state-space cache behavior.\n"
    )
    chunk_ids = enc(chunk)
    long_target = args.long_prompt_tokens
    reps = (long_target // max(1, len(chunk_ids))) + 2
    long_ids = (chunk_ids * reps)[:long_target]
    ask = enc(
        "\nQuestion: Based on the preceding document, give a concise checklist "
        "for verifying arithmetic word-problem answers.\nAnswer:"
    )
    long_prompt = (long_ids + ask)[:long_target]
    smoke_reps = (args.smoke_prompt_tokens // max(1, len(chunk_ids))) + 2
    smoke_ids = (chunk_ids * smoke_reps)[: args.smoke_prompt_tokens]

    rows = [
        {
            "id": "smoke_8k",
            "regime": "8k_smoke",
            "prompt_token_ids": smoke_ids,
            "target_new_tokens": args.smoke_new_tokens,
            "segment_tokens": args.smoke_new_tokens,
            "temperature": 0.0,
        },
        {
            "id": "short_prompt_long_response",
            "regime": "short_prompt_long_response",
            "prompt_token_ids": enc(seed),
            "target_new_tokens": args.long_response_tokens,
            "segment_tokens": args.segment_tokens,
            "temperature": 0.8,
        },
        {
            "id": "long_prompt_short_response",
            "regime": "long_prompt_short_response",
            "prompt_token_ids": long_prompt,
            "target_new_tokens": args.long_prompt_new_tokens,
            "segment_tokens": args.long_prompt_new_tokens,
            "temperature": 0.0,
        },
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tokenizer": args.tokenizer, "rows": rows}, indent=2))
    print(f"WROTE_INPUTS {out}", flush=True)


# ── generate (continuation loop) ────────────────────────────────────────


def sample_once(
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
    temperature: float,
    ignore_eos: bool = False,
) -> tuple[list[int], list[float], str]:
    body = {
        "model": served_model_id(),
        "prompt": prompt_ids,
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "seed": seed,
        "logprobs": 0,
        "return_tokens_as_token_ids": True,
    }
    if ignore_eos:
        body["ignore_eos"] = True
    data = http_json("POST", BASE + "/v1/completions", body, timeout=3600)
    return extract_completion(data["choices"][0])


def generate(args: argparse.Namespace) -> None:
    spec = json.loads(Path(args.inputs).read_text())
    resume_by_id: dict[str, dict[str, Any]] = {}
    if args.resume_from:
        prior = json.loads(Path(args.resume_from).read_text())
        resume_by_id = {s["id"]: s for s in prior["sequences"]}
    sequences = []
    for row in spec["rows"]:
        prompt_ids = list(row["prompt_token_ids"])
        target = int(row["target_new_tokens"])
        seg = int(row["segment_tokens"])
        temp = float(row.get("temperature", 0.8))
        generated: list[int] = []
        logprobs: list[float] = []
        segments: list[dict[str, Any]] = []
        if row["id"] in resume_by_id:
            prev = resume_by_id[row["id"]]
            generated = list(prev["completion_token_ids"])
            logprobs = list(prev["completion_logprobs"])
            segments = list(prev.get("metadata", {}).get("segments", []))
        short_stops = 0
        while (
            len(generated) < target
            and len(prompt_ids) + len(generated) < args.max_total_tokens
            and len(segments) < args.max_segments
        ):
            want = min(
                seg,
                target - len(generated),
                args.max_total_tokens - len(prompt_ids) - len(generated),
            )
            if want <= 0:
                break
            ids, lps, finish = sample_once(
                prompt_ids + generated,
                want,
                args.seed + len(segments),
                temp,
                ignore_eos=args.ignore_eos,
            )
            segments.append(
                {"requested": want, "returned": len(ids), "finish_reason": finish}
            )
            if not ids:
                break
            generated.extend(ids)
            logprobs.extend(lps)
            # Repeated near-empty EOS stops => the model refuses to continue
            # even with EOS stripped; keep what we have.
            if finish != "length" and len(ids) < max(8, want // 16):
                short_stops += 1
                if short_stops >= 3:
                    break
            else:
                short_stops = 0
        sequences.append(
            {
                "id": row["id"],
                "regime": row["regime"],
                "prompt_token_ids": prompt_ids,
                "completion_token_ids": generated,
                "completion_logprobs": logprobs,
                "metadata": {
                    "segments": segments,
                    "target_new_tokens": target,
                    "generated_tokens": len(generated),
                    "prompt_tokens": len(prompt_ids),
                },
            }
        )
        print(
            f"GENERATED {row['id']} prompt={len(prompt_ids)} "
            f"completion={len(generated)} segments={len(segments)}",
            flush=True,
        )
    report = {
        "mode": "generate",
        "source": "sampler_server",
        "base_url": BASE,
        "model": served_model_id(),
        "sequences": sequences,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"NVFP4_GENERATE_DONE {out}", flush=True)


# ── sampler-rescore ─────────────────────────────────────────────────────


def sampler_rescore(args: argparse.Namespace) -> None:
    prior = json.loads(Path(args.sequences).read_text())
    outseq = []
    for s in prior["sequences"]:
        full = s["prompt_token_ids"] + s["completion_token_ids"]
        body = {
            "model": served_model_id(),
            "prompt": full,
            "n": 1,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
            "return_tokens_as_token_ids": True,
        }
        data = http_json("POST", BASE + "/v1/completions", body, timeout=7200)
        comp_lps = extract_prompt_lps(data, full, len(s["prompt_token_ids"]))
        outseq.append(
            {
                "id": s["id"],
                "regime": s.get("regime"),
                "prompt_token_ids": s["prompt_token_ids"],
                "completion_token_ids": s["completion_token_ids"],
                "completion_logprobs": comp_lps,
                "metadata": s.get("metadata", {}),
            }
        )
        print(f"RESCORED_SAMPLER {s['id']} tokens={len(comp_lps)}", flush=True)
    report = {
        "mode": "rescore",
        "source": "sampler_server",
        "model": served_model_id(),
        "rescored_from": args.sequences,
        "sequences": outseq,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"SAMPLER_RESCORE_DONE {out}", flush=True)


# ── megatron-rescore (POST /forward + operations long-poll) ─────────────


def trainer_url() -> str:
    return os.environ.get("TRAINER_URL", "http://127.0.0.1:8000")


def poll_operation(op_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data = http_json(
                "GET", f"{trainer_url()}/operations/{op_id}", None, timeout=40
            )
        except urllib.error.HTTPError as e:
            if e.code == 408:  # pending; server holds ~30s per poll
                continue
            raise
        if data.get("status") == "done":
            return data["result"]
        if data.get("status") == "error":
            raise RuntimeError(f"operation {op_id} failed: {data}")
    raise TimeoutError(f"operation {op_id} did not finish in {timeout}s")


def megatron_rescore(args: argparse.Namespace) -> None:
    prior = json.loads(Path(args.sequences).read_text())
    outseq = []
    for s in prior["sequences"]:
        full = s["prompt_token_ids"] + s["completion_token_ids"]
        payload = {
            "data": [
                {
                    "model_input": {
                        "chunks": [{"type": "encoded_text", "tokens": full}]
                    },
                    "loss_fn_inputs": {},
                }
            ],
            "loss_fn": "cross_entropy",
        }
        req = urllib.request.Request(
            trainer_url() + "/forward",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": f"nvfp4-probe-{s['id']}-{args.run_tag}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            accepted = json.loads(r.read().decode())
        result = poll_operation(accepted["operation_id"], args.timeout)
        row = result["loss_fn_outputs"][0]["logprobs"]["data"]
        plen = len(s["prompt_token_ids"])
        clen = len(s["completion_token_ids"])
        # Wire format: wire[k] = logprob(input[k+1] | input[0..k]) with a 0.0
        # sentinel at the final position (see logprobs_to_loss_fn_outputs).
        # The completion token at absolute index i is scored at wire[i-1], so
        # the completion region is [plen-1, plen-1+clen).
        comp = row[plen - 1 : plen - 1 + clen]
        comp_lps = [float("nan") if v is None else float(v) for v in comp]
        outseq.append(
            {
                "id": s["id"],
                "regime": s.get("regime"),
                "prompt_token_ids": s["prompt_token_ids"],
                "completion_token_ids": s["completion_token_ids"],
                "completion_logprobs": comp_lps,
                "metadata": s.get("metadata", {}),
            }
        )
        print(f"RESCORED_MEGATRON {s['id']} tokens={len(comp_lps)}", flush=True)
    report = {
        "mode": "rescore",
        "source": "bf16_megatron_forward",
        "rescored_from": args.sequences,
        "sequences": outseq,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"MEGATRON_RESCORE_DONE {out}", flush=True)


# ── compare ─────────────────────────────────────────────────────────────


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _collect(
    a: dict[str, Any],
    b: dict[str, Any],
    regime: str | None,
    max_tokens: int | None = None,
):
    by_id = {s["id"]: s for s in b["sequences"]}
    xs: list[float] = []
    ys: list[float] = []
    diag = {"tokens": 0, "dropped_nan": 0, "ids": []}
    for sa in a["sequences"]:
        if regime and sa.get("regime") != regime:
            continue
        sb = by_id.get(sa["id"])
        if sb is None:
            continue
        if sa["completion_token_ids"] != sb["completion_token_ids"]:
            diag.setdefault("misaligned", []).append(sa["id"])
            continue
        diag["ids"].append(sa["id"])
        lp_a = sa["completion_logprobs"][:max_tokens] if max_tokens else sa["completion_logprobs"]
        lp_b = sb["completion_logprobs"][:max_tokens] if max_tokens else sb["completion_logprobs"]
        for x, y in zip(lp_a, lp_b):
            if (
                x is None
                or y is None
                or math.isnan(float(x))
                or math.isnan(float(y))
            ):
                diag["dropped_nan"] += 1
                continue
            xs.append(float(x))
            ys.append(float(y))
    diag["tokens"] = len(xs)
    return xs, ys, diag


def _metrics(x: list[float], y: list[float], clip_low: float, clip_high: float):
    n = len(x)
    if n == 0:
        return {"error": "no tokens"}
    r = [b - a for a, b in zip(x, y)]  # log(target/behavior)
    w = [math.exp(max(-50.0, min(50.0, ri))) for ri in r]
    absd = [abs(ri) for ri in r]
    sw = sum(w)
    sw2 = sum(v * v for v in w)
    return {
        "tokens": n,
        "mean_abs": sum(absd) / n,
        "rms": math.sqrt(sum(d * d for d in absd) / n),
        "max_abs": max(absd),
        "k3": sum(math.exp(-ri) + ri - 1 for ri in r) / n,
        "ess_over_n": ((sw * sw) / sw2) / n if sw2 else 0.0,
        "clip_fraction": sum(1 for v in w if v < clip_low or v > clip_high) / n,
        "mean_log_ratio": sum(r) / n,
        "sum_log_ratio": sum(r),
        "weight_min": min(w),
        "weight_max": max(w),
    }


_PAIRS = [
    ("1_vs_2_floor", "gen_nvfp4", "rescore_nvfp4"),
    ("2_vs_4_quant_vllm", "rescore_nvfp4", "rescore_bf16_vllm"),
    ("4_vs_3_engine_bf16", "rescore_bf16_vllm", "rescore_bf16_megatron"),
    ("1_vs_3_production", "gen_nvfp4", "rescore_bf16_megatron"),
]


def compare(args: argparse.Namespace) -> None:
    files = {name: _load(path) for name, path in (x.split("=", 1) for x in args.inputs)}
    regimes = sorted(
        {
            s.get("regime")
            for f in files.values()
            for s in f["sequences"]
            if s.get("regime")
        }
    )
    report: dict[str, Any] = {"regimes": {}}
    for reg in regimes:
        table = {}
        for label, a, b in _PAIRS:
            if a not in files or b not in files:
                continue
            xs, ys, diag = _collect(files[a], files[b], reg)
            table[label] = {
                "diagnostics": diag,
                "metrics": _metrics(xs, ys, args.clip_low, args.clip_high),
            }
            # Prefix growth curve: metrics on the first N completion tokens.
            # Valid because teacher-forced logprobs at position i depend only
            # on the prefix, so a length-N prefix of a longer trace is exactly
            # what a length-N run would have measured.
            for plen in args.prefix_lengths or []:
                xs_p, ys_p, _ = _collect(files[a], files[b], reg, max_tokens=plen)
                if xs_p and len(xs_p) < len(xs):
                    table[label].setdefault("prefix_metrics", {})[str(plen)] = _metrics(
                        xs_p, ys_p, args.clip_low, args.clip_high
                    )
        report["regimes"][reg] = table
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"COMPARE_DONE {out}", flush=True)


# ── cli ─────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("wait-health")
    s.add_argument("--timeout", type=float, default=3600)

    s = sub.add_parser("make-inputs")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--out", default=str(EXP / "inputs.json"))
    s.add_argument("--long-response-tokens", type=int, default=16384)
    s.add_argument("--segment-tokens", type=int, default=4096)
    s.add_argument("--long-prompt-tokens", type=int, default=255000)
    s.add_argument("--long-prompt-new-tokens", type=int, default=512)
    s.add_argument("--smoke-prompt-tokens", type=int, default=7680)
    s.add_argument("--smoke-new-tokens", type=int, default=256)

    s = sub.add_parser("generate")
    s.add_argument("--inputs", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=1234)
    s.add_argument("--max-total-tokens", type=int, default=262000)
    s.add_argument("--max-segments", type=int, default=32)
    s.add_argument("--resume-from", default=None, help="prior generate output; continue its completions")
    s.add_argument("--ignore-eos", action="store_true", help="keep sampling past EOS (vLLM ignore_eos)")

    s = sub.add_parser("sampler-rescore")
    s.add_argument("--sequences", required=True)
    s.add_argument("--out", required=True)

    s = sub.add_parser("megatron-rescore")
    s.add_argument("--sequences", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--timeout", type=float, default=7200)
    s.add_argument("--run-tag", default="r1")

    s = sub.add_parser("compare")
    s.add_argument("--inputs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--clip-low", type=float, default=0.8)
    s.add_argument("--clip-high", type=float, default=1.2)
    s.add_argument("--prefix-lengths", type=int, nargs="*", default=None)

    a = p.parse_args()
    if a.cmd == "wait-health":
        wait_health(a.timeout)
    elif a.cmd == "make-inputs":
        make_inputs(a)
    elif a.cmd == "generate":
        generate(a)
    elif a.cmd == "sampler-rescore":
        sampler_rescore(a)
    elif a.cmd == "megatron-rescore":
        megatron_rescore(a)
    elif a.cmd == "compare":
        compare(a)


if __name__ == "__main__":
    main()
