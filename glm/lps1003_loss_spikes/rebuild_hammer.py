#!/usr/bin/env python3
"""Rebuild-hammer: repeatedly (train 1 step -> init_trainer_server rebuild ->
probe batch-0 xN) to accumulate statistics on post-rebuild destruction events
(LPS-1003). Any datum NLL > threshold (default 2.0) counts as DESTROYED.

Usage: rebuild_hammer.py --cycles 20 --out hammer.jsonl [--trainer-url ...]
"""
from __future__ import annotations

import argparse
import json
import time
import uuid

import httpx

from probe_nll import load_bundle, make_datum, per_datum_stats, submit, wait

DESTROY_NLL = 2.0


def op(client, path, body):
    r = client.post(path, json=body, headers={"Idempotency-Key": uuid.uuid4().hex})
    if r.status_code != 202:
        raise RuntimeError(f"{path} submit {r.status_code}: {r.text[:300]}")
    return wait(client, r.json()["operation_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-url", default="http://127.0.0.1:8001")
    ap.add_argument("--bundle", default="train_bundle_0_31.jsonl.gz")
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--probes-per-cycle", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="hammer")
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
    out = open(args.out, "a", buffering=1)

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    with httpx.Client(base_url=args.trainer_url, timeout=2400.0) as client:
        client.get("/health", timeout=30.0).raise_for_status()
        events = 0
        for cyc in range(args.cycles):
            # 1. one real training step (batch rotates) so rebuild path is armed
            rows = bundle[cyc % 32]
            fb = op(client, "/forward_backward",
                    {"data": [make_datum(r) for r in rows], "loss_fn": "cross_entropy"})
            om = op(client, "/optim_step",
                    {"adam_params": {"learning_rate": 5e-4, "beta1": 0.9,
                                     "beta2": 0.95, "eps": 1e-8}})
            # 2. rebuild fresh adapter
            ir = op(client, "/init_trainer_server", {"lora_rank": 32})
            # 3. immediate probes on batch 0
            for rep in range(args.probes_per_cycle):
                res = op(client, "/forward",
                         {"data": [make_datum(r) for r in bundle[0]],
                          "loss_fn": "cross_entropy"})
                stats = per_datum_stats(res, bundle[0])
                destroyed = [(s["idx"], round(s["nll"], 2)) for s in stats
                             if s["nll"] > DESTROY_NLL]
                mean = sum(s["nll"] for s in stats) / len(stats)
                rec = {"cycle": cyc, "rep": rep, "mean_nll": mean,
                       "destroyed": destroyed, "tag": args.tag, "ts": time.time(),
                       "datums": [(s["idx"], round(s["nll"], 4)) for s in stats]}
                out.write(json.dumps(rec) + "\n")
                if destroyed:
                    events += 1
                    log(f"cycle {cyc} rep {rep}: DESTRUCTION {destroyed} mean={mean:.3f}")
                else:
                    log(f"cycle {cyc} rep {rep}: clean mean={mean:.3f}")
        log(f"done: {events} destruction probes across {args.cycles} cycles")


if __name__ == "__main__":
    main()
