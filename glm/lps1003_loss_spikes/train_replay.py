#!/usr/bin/env python3
"""Training replay for LPS-1003: run REAL training steps (forward_backward +
optim_step) over the exact prod batches (train_bundle_0_31.jsonl.gz), mirroring
loops_sft.py's recipe: lr 5e-4 cosine over 1528 total steps, betas 0.9/0.95,
eps 1e-8, batch 32, LoRA rank 32 (trainer-side config).

Run from a FRESHLY BOOTED trainer (step 0, fresh LoRA init). Repeat the whole
run from another fresh boot to measure run-to-run divergence at bump steps.

Records per step: op-level loss, client train_mean_nll (mean over datums of
weighted per-datum NLL — the exact W&B metric), per-datum NLL, optim metrics
(grad_norm etc.), and optionally full per-token logprobs (npz per step).
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid

import httpx

from probe_nll import load_bundle, make_datum, per_datum_stats, submit, tensor_data, wait

LR = 5e-4
TOTAL_STEPS = 1528  # 48898 // 32, prod cosine horizon


def lr_at(step: int) -> float:
    return LR * 0.5 * (1 + math.cos(math.pi * min(step, TOTAL_STEPS) / TOTAL_STEPS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainer-url", default="http://127.0.0.1:8001")
    ap.add_argument("--bundle", default="train_bundle_0_31.jsonl.gz")
    ap.add_argument("--out", required=True, help="output JSONL (appended)")
    ap.add_argument("--steps", type=int, default=32, help="replay steps 0..N-1")
    ap.add_argument("--start-step", type=int, default=0,
                    help="resume mid-run (trainer must already be at this step)")
    ap.add_argument("--dump-logprobs", default=None, help="dir for per-step logprob npz")
    ap.add_argument("--tag", required=True, help="arm name, e.g. arm0/arm1")
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
    out = open(args.out, "a", buffering=1)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def emit(rec: dict) -> None:
        rec["ts"] = time.time()
        rec["tag"] = args.tag
        out.write(json.dumps(rec) + "\n")

    with httpx.Client(base_url=args.trainer_url, timeout=2400.0) as client:
        client.get("/health", timeout=30.0).raise_for_status()
        status = client.get("/status", timeout=30.0).json()
        log(f"trainer: step={status.get('step')} world={status.get('world_size')}")
        if args.start_step == 0 and status.get("step") not in (0, None):
            raise SystemExit(f"trainer not fresh (step={status.get('step')}); "
                             f"restart it or pass --start-step")
        emit({"kind": "status", "status": status})

        for step in range(args.start_step, args.steps):
            rows = bundle[step]
            body = {"data": [make_datum(r) for r in rows], "loss_fn": "cross_entropy"}
            t0 = time.time()
            fb_id = submit(client, body, "/forward_backward")
            fb = wait(client, fb_id)
            stats = per_datum_stats(fb, rows)
            mean_nll = sum(s["nll"] for s in stats) / len(stats)

            lr = lr_at(step)
            op = client.post("/optim_step",
                             json={"adam_params": {"learning_rate": lr, "beta1": 0.9,
                                                   "beta2": 0.95, "eps": 1e-8}},
                             headers={"Idempotency-Key": uuid.uuid4().hex})
            if op.status_code != 202:
                raise RuntimeError(f"optim submit failed {op.status_code}: {op.text[:300]}")
            om = wait(client, op.json()["operation_id"])
            wall = time.time() - t0

            rec = {"kind": "train_step", "step": step, "wall_s": round(wall, 1),
                   "lr": lr, "op_loss": fb.get("loss"), "mean_nll": mean_nll,
                   "optim_metrics": om.get("metrics", om) if isinstance(om, dict) else om,
                   "datums": stats}
            emit(rec)
            gn = None
            if isinstance(om, dict):
                gn = (om.get("metrics") or om).get("grad_norm") if isinstance(om.get("metrics", om), dict) else None
            log(f"step {step}: train_mean_nll={mean_nll:.4f} lr={lr:.2e} "
                f"grad_norm={gn} {wall:.0f}s")

            if args.dump_logprobs:
                import numpy as np
                from pathlib import Path
                d = Path(args.dump_logprobs)
                d.mkdir(parents=True, exist_ok=True)
                arrs = {f"d{s['idx']:03d}": np.asarray(
                    tensor_data(fb["loss_fn_outputs"][i]["logprobs"]), dtype=np.float32)
                    for i, s in enumerate(stats)}
                np.savez_compressed(d / f"step{step:03d}.npz", **arrs)

    log("replay complete")


if __name__ == "__main__":
    main()
