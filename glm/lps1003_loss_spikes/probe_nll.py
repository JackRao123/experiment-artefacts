#!/usr/bin/env python3
"""Per-batch / per-datum NLL probe for LPS-1003 GLM-5.2-FP8 loss bumps.

Drives a trainer server over HTTP with forward_backward ops built from
probe_bundle.jsonl.gz (exact token ids + prefix_len; no tokenizer needed).
NEVER calls optim_step -> weights are frozen at base state, so identical
submissions measure pure run-to-run numeric wobble.

Wire format mirrors tinker_cookbook.datum_from_model_input_weights(
reduction="mean") exactly, as used by the production client (loops_sft.py):
  model_input tokens = ids[:-1]
  target_tokens      = ids[1:]
  weights            = (1.0 for positions >= prefix_len)[1:], normalized to sum 1
train_mean_nll(batch) = mean over datums of per-datum weighted NLL, matching
the W&B metric of runs ln68q5he / nspvxlhu.

Modes:
  batches  one op per batch (original datum order -> identical packing),
           repeated --repeats times.  Per-datum NLL extracted from
           loss_fn_outputs logprobs.
  datums   one op per datum (--select batch:idx,batch:idx,... or --label),
           repeated --repeats times.

Output: JSONL, one line per completed op, with per-datum NLL + logprob stats.
Optionally dumps full per-token logprobs per (batch,repeat) to .npz.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

# --------------------------------------------------------------------------- #
# Bundle loading + datum construction
# --------------------------------------------------------------------------- #


def load_bundle(path: str) -> dict[int, list[dict]]:
    """batch_id -> rows sorted by idx (original submission order)."""
    batches: dict[int, list[dict]] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            batches.setdefault(r["batch"], []).append(r)
    for rows in batches.values():
        rows.sort(key=lambda r: r["idx"])
    return batches


def make_datum(row: dict) -> dict:
    ids = row["ids"]
    p = row["prefix_len"]
    L = len(ids)
    assert 0 < p < L, f"bad prefix_len {p} for len {L} (batch {row['batch']} idx {row['idx']})"
    targets = ids[1:]
    n_t = len(targets)  # == L - 1
    # raw weights aligned to ids: w[i]=1 for i>=p ; sliced [1 : n_t+1] == [1:L]
    n_sup = L - p  # supervised positions all survive the slice since p >= 1
    wv = 1.0 / n_sup
    weights = [0.0] * (p - 1) + [wv] * n_sup
    assert len(weights) == n_t
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": ids[:-1]}]},
        "loss_fn_inputs": {
            "weights": {"data": weights, "dtype": "float32", "shape": [n_t]},
            "target_tokens": {"data": targets, "dtype": "int64", "shape": [n_t]},
        },
    }


# --------------------------------------------------------------------------- #
# HTTP plumbing (mirrors incident_replay.py)
# --------------------------------------------------------------------------- #


def submit(client: httpx.Client, body: dict, op_path: str = "/forward") -> str:
    r = client.post(op_path, json=body,
                    headers={"Idempotency-Key": uuid.uuid4().hex})
    if r.status_code != 202:
        raise RuntimeError(f"submit failed {r.status_code}: {r.text[:500]}")
    return r.json()["operation_id"]


def wait(client: httpx.Client, op_id: str, timeout: float = 2400.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            rr = client.get(f"/operations/{op_id}", timeout=40.0)
        except httpx.HTTPError as e:
            raise RuntimeError(f"poll transport error (server dead?): {e}") from e
        if rr.status_code == 408:
            continue
        if rr.status_code in (404, 502, 503):
            raise RuntimeError(f"poll failed {rr.status_code}: {rr.text[:500]}")
        payload = rr.json()
        st = payload.get("status")
        if st == "done":
            return payload["result"]
        if st == "error":
            raise RuntimeError(f"op error: {str(payload.get('error'))[:4000]}")
    raise RuntimeError(f"timeout after {timeout}s")


# --------------------------------------------------------------------------- #
# Result extraction
# --------------------------------------------------------------------------- #


def tensor_data(x) -> list[float]:
    """Accept TensorData dict or bare list."""
    if isinstance(x, dict):
        return x["data"]
    return x


def per_datum_stats(result: dict, rows: list[dict]) -> list[dict]:
    outs = result.get("loss_fn_outputs")
    if outs is None:
        raise RuntimeError(f"no loss_fn_outputs in result keys={list(result.keys())}")
    assert len(outs) == len(rows), f"{len(outs)} outputs vs {len(rows)} datums"
    stats = []
    for row, out in zip(rows, outs):
        lp = tensor_data(out["logprobs"])
        p, L = row["prefix_len"], len(row["ids"])
        n_sup = L - p
        sup = lp[p - 1:]  # logprobs aligned with targets=ids[1:]; supervised tail
        assert len(sup) == n_sup, f"sup len {len(sup)} != {n_sup}"
        nll = -sum(sup) / n_sup
        worst = sorted(range(n_sup), key=lambda i: sup[i])[:5]
        stats.append({
            "batch": row["batch"], "idx": row["idx"], "label": row["label"],
            "n_sup": n_sup, "nll": nll,
            "min_lp": min(sup), "max_lp": max(sup),
            "n_below_5": sum(1 for v in sup if v < -5.0),
            "n_below_10": sum(1 for v in sup if v < -10.0),
            "worst_pos": [p - 1 + i for i in worst],       # position in target space
            "worst_lp": [sup[i] for i in worst],
        })
    return stats


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainer-url", default="http://127.0.0.1:8001")
    ap.add_argument("--op-path", default="/forward", choices=["/forward", "/forward_backward"],
                    help="/forward = no_grad, no accumulation (default); "
                         "/forward_backward = full training-fidelity forward")
    ap.add_argument("--bundle", default="probe_bundle.jsonl.gz")
    ap.add_argument("--mode", choices=["batches", "datums"], required=True)
    ap.add_argument("--out", required=True, help="output JSONL (appended)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batches", default=None,
                    help="comma-separated batch ids (default: all in bundle)")
    ap.add_argument("--select", default=None,
                    help="datums mode: comma-separated batch:idx keys")
    ap.add_argument("--label", default=None,
                    help="datums mode: probe every datum with this label")
    ap.add_argument("--dump-logprobs", default=None,
                    help="dir for full per-token logprob .npz dumps (batches mode)")
    ap.add_argument("--tag", default="probe")
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
        emit({"kind": "status", "status": status})

        if args.mode == "batches":
            ids = ([int(x) for x in args.batches.split(",")] if args.batches
                   else sorted(bundle.keys()))
            work = [(bi, bundle[bi]) for bi in ids]
        else:
            if args.select:
                keys = [tuple(int(v) for v in k.split(":")) for k in args.select.split(",")]
                rows = [r for bi, rows in bundle.items() for r in rows
                        if (r["batch"], r["idx"]) in set(keys)]
            elif args.label:
                rows = [r for rows in bundle.values() for r in rows if r["label"] == args.label]
            else:
                ap.error("datums mode needs --select or --label")
            work = [((r["batch"], r["idx"]), [r]) for r in rows]

        for rep in range(args.repeats):
            for key, rows in work:
                body = {"data": [make_datum(r) for r in rows], "loss_fn": "cross_entropy"}
                t0 = time.time()
                op_id = submit(client, body, args.op_path)
                result = wait(client, op_id)
                wall = time.time() - t0
                stats = per_datum_stats(result, rows)
                mean_nll = sum(s["nll"] for s in stats) / len(stats)
                rec = {"kind": args.mode, "key": key, "repeat": rep, "op_id": op_id,
                       "wall_s": round(wall, 1), "op_loss": result.get("loss"),
                       "mean_nll": mean_nll, "datums": stats}
                emit(rec)
                lbl = rows[0]["label"]
                log(f"rep{rep} {args.mode} {key} ({lbl}): mean_nll={mean_nll:.4f} "
                    f"op_loss={result.get('loss')} n={len(rows)} {wall:.0f}s")
                if args.dump_logprobs and args.mode == "batches":
                    import numpy as np
                    d = Path(args.dump_logprobs)
                    d.mkdir(parents=True, exist_ok=True)
                    arrs = {f"d{s['idx']:03d}": np.asarray(
                        tensor_data(result["loss_fn_outputs"][i]["logprobs"]), dtype=np.float32)
                        for i, s in enumerate(stats)}
                    np.savez_compressed(d / f"b{key}_rep{rep}.npz", **arrs)

    log("all work complete")


if __name__ == "__main__":
    main()
