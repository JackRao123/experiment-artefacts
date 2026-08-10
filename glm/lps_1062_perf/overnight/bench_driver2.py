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

from mfu import hfu, mfu3x

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
                 seq_len: int, num_gpus: int, tag: str) -> dict:
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
    args = ap.parse_args()

    rng = random.Random(0xB300)
    out: dict = {
        "label": args.label,
        "seq_len": args.seq_len,
        "num_gpus": args.num_gpus,
        "datums_per_window": args.datums,
        "warmup_datums": args.warmup_datums,
        "started": time.strftime("%F %T"),
    }

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        status = client.get("/status").json()
        out["initial_status"] = status
        print(f"[status] world_size={status.get('world_size')}", flush=True)

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
