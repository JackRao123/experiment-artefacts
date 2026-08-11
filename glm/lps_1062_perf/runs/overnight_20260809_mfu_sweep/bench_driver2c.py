#!/usr/bin/env python3
"""Parameterized bench driver for LPS-1062 overnight parallelism experiments.

Same protocol as bench_driver.py (synthetic random tokens, rng seed 0xB300,
1 warmup window then N main windows, no profilers), but parameterized for the
parallelism sweep: sequence length, GPU count, datums per window.

MFU: length-dependent FLOPs/token via mfu.py (same folder). Reports mfu3x
(3x fwd = useful FLOPs, the notebook convention) and hfu (hw passes actually
run; 4 under full recompute).

Usage (on the node where the trainer HTTP is up, port 8001):
    python3 bench_driver2.py --label expA1-anchor-131k --seq-len 131072 \
        --num-gpus 16 --datums 2 --repeats 3

Writes /root/.cache/user_artifacts/lps1062_bench/<label>.json + SUMMARY line.
With DP>1, send one datum per DP group per window (e.g. DP2 -> --datums 2).

Canary: pass --canary-json <baseline result json> to print per-window
loss/grad_norm drift vs that baseline (anchor run at the same seq len).
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path

import httpx

from mfu import (
    PEAK_FLOPS_GPU,
    executed_flops_per_token,
    fwd_flops_per_token,
    hfu,
    mfu3x,
    useful_flops_per_token,
)

BASE_URL = "http://127.0.0.1:8001"
VOCAB_SIZE = 154_880
FB_TIMEOUT_S = 3600.0
OUT_DIR = Path("/root/.cache/user_artifacts/lps1062_bench")


def make_datum(rng: random.Random, seq_len: int) -> dict:
    tokens = [rng.randrange(VOCAB_SIZE) for _ in range(seq_len)]
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {},
    }


def custmix_lengths(rng: random.Random, target: int = 524_288) -> list[int]:
    """One step-shape drawn from the customer histogram (LPS-1062 overnight):
    70% U(4k,32k) / 28.5% U(32k,64k) / 1.4% U(64k,131k), rounded to 2048-multiples,
    datums until cumulative tokens reach ~target. Drawn once, reused every window.
    """
    lengths: list[int] = []
    total = 0
    while total < target:
        u = rng.random()
        if u < 0.70:
            L = rng.uniform(4096, 32768)
        elif u < 0.985:
            L = rng.uniform(32768, 65536)
        else:
            L = rng.uniform(65536, 131072)
        L = max(2048, int(round(L / 2048)) * 2048)
        lengths.append(L)
        total += L
    return lengths


def datum_token_count(datum: dict) -> int:
    return sum(len(c["tokens"]) for c in datum["model_input"]["chunks"] if c["type"] == "encoded_text")


def submit_and_wait(client: httpx.Client, path: str, body: dict, timeout: float) -> dict:
    r = client.post(path, json=body, headers={"Idempotency-Key": uuid.uuid4().hex}, timeout=60.0)
    if r.status_code != 202:
        raise RuntimeError(f"{path} submit failed: {r.status_code} {r.text[:2000]}")
    operation_id = r.json()["operation_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rr = client.get(f"/operations/{operation_id}", timeout=60.0)
        if rr.status_code == 408:
            continue
        rr.raise_for_status()
        body = rr.json()
        if body.get("status") == "done":
            return body["result"]
        if body.get("status") == "error":
            raise RuntimeError(f"{path} op {operation_id} errored: {body.get('error', '')[:2000]}")
    raise TimeoutError(f"{path} op {operation_id} did not finish in {timeout}s")


def drive_window(client: httpx.Client, label: str, index: int, datums: list[dict],
                 seq_len: int, num_gpus: int, tag: str, n_tokens: int | None = None) -> dict:
    if n_tokens is None:
        n_tokens = seq_len * len(datums)
    t0 = time.perf_counter()
    fb = submit_and_wait(client, "/forward_backward", {"data": datums}, FB_TIMEOUT_S)
    fb_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    opt = submit_and_wait(client, "/optim_step", {"adam_params": {"learning_rate": 1e-5}}, FB_TIMEOUT_S)
    opt_s = time.perf_counter() - t0
    metrics = (opt or {}).get("metrics") or {}
    rec = {
        "label": label,
        "window_index": index,
        "phase": tag,
        "num_tokens": n_tokens,
        "fb_elapsed_s": fb_s,
        "fb_tps_per_gpu": n_tokens / fb_s / num_gpus,
        "optim_elapsed_s": opt_s,
        "loss": (fb or {}).get("loss"),
        "grad_norm": metrics.get("grad_norm"),
    }
    print(
        f"[{label} {tag}{index}] fb={fb_s:.1f}s ({rec['fb_tps_per_gpu']:.0f} tok/s/GPU) "
        f"optim={opt_s:.1f}s loss={rec['loss']} gn={rec['grad_norm']}",
        flush=True,
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seq-len", type=int, default=262_144)
    ap.add_argument("--num-gpus", type=int, default=16)
    ap.add_argument("--datums", type=int, default=2, help="datums per main window")
    ap.add_argument("--warmup-datums", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2, help="main windows")
    ap.add_argument("--lora-rank", type=int, default=32,
                    help="LoRA rank of the run (mfu.py: adapter FLOPs scale with rank)")
    ap.add_argument("--canary-json", default=None,
                    help="baseline result json; print loss/grad_norm drift vs its windows")
    ap.add_argument("--custmix", action="store_true",
                    help="customer-histogram mixed lengths (70% 4-32k / 28.5% 32-64k / 1.4% 64-131k, "
                         "2048-rounded, ~524288 tok/step); one shape drawn once, reused every window")
    args = ap.parse_args()

    rng = random.Random(0xB300)
    out: dict = {
        "label": args.label,
        "seq_len": args.seq_len,
        "num_gpus": args.num_gpus,
        "datums_per_window": args.datums,
        "started": time.strftime("%F %T"),
    }

    custmix_datums: list[dict] | None = None
    custmix_ntokens = 0
    if args.custmix:
        lengths = custmix_lengths(rng)
        custmix_ntokens = sum(lengths)
        out["custmix_lengths"] = lengths
        out["datums_per_window"] = len(lengths)
        print(f"[custmix] {len(lengths)} datums, {custmix_ntokens} tokens/step, "
              f"min {min(lengths)} max {max(lengths)}", flush=True)
        custmix_datums = [make_datum(rng, L) for L in lengths]

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        status = client.get("/status").json()
        out["initial_status"] = status
        print(f"[status] world_size={status.get('world_size')}", flush=True)

        if custmix_datums is not None:
            windows = [drive_window(client, args.label, 0, custmix_datums,
                                    args.seq_len, args.num_gpus, "warmup",
                                    n_tokens=custmix_ntokens)]
            for i in range(args.repeats):
                windows.append(drive_window(client, args.label, i, custmix_datums,
                                            args.seq_len, args.num_gpus, "main",
                                            n_tokens=custmix_ntokens))
        else:
            windows = [drive_window(client, args.label, 0,
                                    [make_datum(rng, args.seq_len) for _ in range(args.warmup_datums)],
                                    args.seq_len, args.num_gpus, "warmup")]
            for i in range(args.repeats):
                windows.append(drive_window(client, args.label, i,
                                            [make_datum(rng, args.seq_len) for _ in range(args.datums)],
                                            args.seq_len, args.num_gpus, "main"))

        final_status = client.get("/status", timeout=600.0).json()
        out["final_status"] = final_status

    mains = windows[1:]
    fb = [w["fb_elapsed_s"] for w in mains]
    toks = sum(w["num_tokens"] for w in mains)
    tps = toks / sum(fb)
    tps_per_gpu = tps / args.num_gpus
    if custmix_datums is not None:
        tokens_per_step = custmix_ntokens
        # Token-weighted per-doc FLOPs/token across the mixed lengths (the
        # indexer term is per-doc causal, so FLOPs must be averaged per doc;
        # useful/executed are averaged separately because their pass
        # multipliers differ per FLOP component — see mfu.py header).
        eff_fwd = sum(L * fwd_flops_per_token(L, args.lora_rank)
                      for L in out["custmix_lengths"]) / custmix_ntokens
        eff_useful = sum(L * useful_flops_per_token(L, args.lora_rank)
                         for L in out["custmix_lengths"]) / custmix_ntokens
        eff_executed = sum(L * executed_flops_per_token(L, args.lora_rank)
                           for L in out["custmix_lengths"]) / custmix_ntokens
        agg = {
            "step_s_mean": sum(fb) / len(fb),
            "tokens_per_step": tokens_per_step,
            "tps": tps,
            "tps_per_gpu": tps_per_gpu,
            "mfu3x": tps_per_gpu * eff_useful / PEAK_FLOPS_GPU,
            "hfu": tps_per_gpu * eff_executed / PEAK_FLOPS_GPU,
            "eff_fwd_flops_per_token": eff_fwd,
            "eff_useful_flops_per_token": eff_useful,
            "eff_executed_flops_per_token": eff_executed,
            "optim_s_mean": sum(w["optim_elapsed_s"] for w in mains) / len(mains),
            "peak_gpu_memory_bytes": max((final_status.get("gpu_memory") or {}).values(), default=0),
        }
    else:
        agg = {
            "step_s_mean": sum(fb) / len(fb),
            "tokens_per_step": args.seq_len * args.datums,
            "tps": tps,
            "tps_per_gpu": tps_per_gpu,
            "mfu3x": mfu3x(tps_per_gpu, args.seq_len, args.lora_rank),
            "hfu": hfu(tps_per_gpu, args.seq_len, args.lora_rank),
            "optim_s_mean": sum(w["optim_elapsed_s"] for w in mains) / len(mains),
            "peak_gpu_memory_bytes": max((final_status.get("gpu_memory") or {}).values(), default=0),
        }
    out["windows"] = windows
    out["aggregates"] = agg

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[done] -> {path}", flush=True)
    print(
        f"SUMMARY {args.label}: {tps_per_gpu:.0f} tok/s/GPU | "
        f"step {agg['step_s_mean']:.1f}s ({agg['tokens_per_step']} tok) | "
        f"mfu3x {100 * agg['mfu3x']:.1f}% | hfu {100 * agg['hfu']:.1f}% | "
        f"peak_mem {agg['peak_gpu_memory_bytes'] / 2**30:.0f} GiB",
        flush=True,
    )

    if args.canary_json:
        base = json.loads(Path(args.canary_json).read_text())
        for w, bw in zip(windows, base.get("windows", [])):
            if w["loss"] is not None and bw.get("loss") is not None:
                print(
                    f"CANARY {w['phase']}{w['window_index']}: "
                    f"dloss={w['loss'] - bw['loss']:+.4f} "
                    f"dgn={(w['grad_norm'] or 0) - (bw.get('grad_norm') or 0):+.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
