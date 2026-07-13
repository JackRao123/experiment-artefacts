#!/usr/bin/env python3
"""CP32 131k smoke for logprob emission + RL losses at full scale.

One synthetic ~131k-token datum. For each requested loss_fn: forward_backward
(+ optim_step unless --skip-optim), then validate the returned per-token
logprob row: correct length, finite at supervised positions, 0.0 at masked
ones. Reports step time + peak memory from /status.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid

import httpx

VOCAB = 30000


def build_datum(seq_len: int) -> tuple[dict, int]:
    tokens = [100 + (i % VOCAB) for i in range(seq_len)]
    targets = tokens[1:] + [-100]
    half = seq_len // 2
    weights = [0.0] * half + [1.0] * (seq_len - half)
    logprobs = [-0.05 - (i % 37) * 0.07 for i in range(seq_len)]
    advantages = [0.0] * half + [1.0 if i % 5 else -0.5 for i in range(seq_len - half)]
    advantages[-1] = 0.0

    def td(data, dtype):
        return {"data": data, "dtype": dtype, "shape": [len(data)]}

    datum = {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {
            "target_tokens": td(targets, "int64"),
            "weights": td(weights, "float32"),
            "logprobs": td(logprobs, "float32"),
            "advantages": td(advantages, "float32"),
        },
    }
    return datum, half


def submit_and_wait(client, op_path: str, *, body: dict, timeout: float = 3600.0):
    key = uuid.uuid4().hex
    r = client.post(op_path, json=body, headers={"Idempotency-Key": key})
    if r.status_code != 202:
        raise RuntimeError(f"{op_path} submit failed {r.status_code}: {r.text[:800]}")
    operation_id = r.json()["operation_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rr = client.get(f"/operations/{operation_id}", timeout=35.0)
        if rr.status_code == 408:
            continue
        rr.raise_for_status()
        payload = rr.json()
        if payload.get("status") == "done":
            return payload["result"]
        if payload.get("status") == "error":
            raise RuntimeError(f"{op_path} op error: {payload.get('error', '')[:4000]}")
    raise TimeoutError(op_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trainer-url", default="http://127.0.0.1:8000")
    p.add_argument("--seq-len", type=int, default=131072)
    p.add_argument("--loss-fns", default="cross_entropy,ppo,cispo")
    p.add_argument("--skip-optim", action="store_true")
    args = p.parse_args()

    datum, half = build_datum(args.seq_len)
    ok = True
    with httpx.Client(base_url=args.trainer_url, timeout=3600.0) as client:
        client.get("/health", timeout=30.0).raise_for_status()
        client.post("/reset_peak_memory", timeout=60.0).raise_for_status()
        for loss_fn in args.loss_fns.split(","):
            t0 = time.perf_counter()
            fb = submit_and_wait(
                client, "/forward_backward", body={"data": [datum], "loss_fn": loss_fn}
            )
            dt = time.perf_counter() - t0
            rows = fb.get("loss_fn_outputs") or []
            row = (rows[0].get("logprobs", {}) or {}).get("data") if rows else None
            checks = {
                "rows": len(rows) == 1,
                "row_len": row is not None and len(row) == args.seq_len,
            }
            if row is not None:
                sup = row[half : args.seq_len - 1]
                masked = row[: half - 1]
                checks["supervised_finite"] = all(math.isfinite(v) for v in sup)
                # The sharpest wire check: the row must reproduce the scalar
                # loss the trainer reported. (Exact-0.0 logprobs are legit —
                # ultra-confident tokens round to 0 in fp32; don't count them.)
                if loss_fn == "cross_entropy":
                    row_mean_nll = -sum(sup) / max(len(sup), 1)
                    checks["row_reproduces_loss"] = (
                        abs(row_mean_nll - fb.get("loss")) / max(abs(fb.get("loss")), 1e-9)
                        < 1e-4
                    )
                checks["masked_zero"] = all(v == 0.0 for v in masked)
                checks["last_pos_zero"] = row[-1] == 0.0
            gn = None
            if not args.skip_optim:
                opt = submit_and_wait(
                    client,
                    "/optim_step",
                    body={
                        "adam_params": {
                            "learning_rate": 1e-10,
                            "beta1": 0.9,
                            "beta2": 0.95,
                        }
                    },
                )
                gn = (opt.get("metrics") or {}).get("grad_norm")
            s = client.get("/status", timeout=60.0).json()
            peak = max((s.get("gpu_max_memory_allocated") or {"": 0}).values())
            all_ok = all(checks.values())
            ok &= all_ok
            print(
                f"[131k] {loss_fn}: loss={fb.get('loss'):.6f} grad_norm={gn} "
                f"fb={dt:.1f}s peak={peak / 2**30:.1f}GiB checks={checks} "
                f"{'OK' if all_ok else 'FAIL'}",
                flush=True,
            )
    print("[131k] RESULT:", "PASS" if ok else "FAIL", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
