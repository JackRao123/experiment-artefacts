#!/usr/bin/env python3
"""Padded-vs-unpadded parity driver for the PP2/CP8/EP8 bring-up (volta, LPS-1062).

Sends a fixed mixed-length datum set to a trainer server and records the
per-call parity surface: loss, per-datum logprobs, and (when the server
reports it) loss-token counts. Run it against two server launches — one with
partitions unpadded, one with BT_PACK_PAD_TO_MAX=1 (PP1), or PP1-padded vs
PP2-padded (PP2 pads by gate) — then --compare the two JSONs.

Datum set design (seed 0xPA77Y... deterministic across runs):
  - mixed lengths, none 16-aligned, so every partition carries per-doc pads
    AND a tail-fill under padding;
  - total > 2 x 131072 so multiple partitions form (tail-fill exercised on
    every partition, not just the last);
  - explicit per-position `weights` with zero-weight spans, to exercise the
    loss mask beyond the label=-100 sentinel;
  - NO optim_step anywhere: weights never move, so both legs see the same
    model. (forward_backward accumulates grads but does not update.)

Usage (on the node where trainer HTTP is up, port 8001):
    python3 parity_driver.py --label pp1-unpadded [--base-url http://127.0.0.1:8001]
    python3 parity_driver.py --label pp1-padded   [...same...]
    python3 parity_driver.py --compare pp1-unpadded.json pp1-padded.json

Writes /root/.cache/user_artifacts/lps1062_bench/parity_<label>.json.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path

import httpx

VOCAB_SIZE = 154_880  # GLM-5.2 vocab
FB_TIMEOUT_S = 3600.0
OUT_DIR = Path("/root/.cache/user_artifacts/lps1062_bench")

# Parity tolerances (see PACKING_MEMO Part 2):
#   - loss: token-weighted global mean over fp32 all-reduced sums — tight.
#   - per-token logprobs: bf16 forward through a different THD row layout —
#     kernel reduction order differs, so small numeric drift is expected.
LOSS_REL_TOL = 1e-6
LOGPROB_ABS_TOL = 1e-3


def make_datums(permute: str = "identity") -> list[dict]:
    """Fixed mixed-length set: ~262k real tokens over 9 datums -> 3 partitions
    at 131k unpadded; every partition gets a tail-fill when padding is on.
    Lengths are deliberately not multiples of 16 (CP8 pad_multiple).

    ``permute`` reorders the datums (content unchanged — same seed, same
    per-length tokens) for the slot-vs-document permutation discriminator
    (LPS-1062 parity night 2): datum identity is tracked by length (all
    nine lengths are unique)."""
    rng = random.Random(0x9A11)
    lengths = [31_733, 48_201, 12_997, 63_555, 8_003, 27_111, 40_449, 19_231, 10_752]
    datums = []
    for L in lengths:
        tokens = [rng.randrange(VOCAB_SIZE) for _ in range(L)]
        # weights: 1.0 everywhere except a zero-weight span in the middle third
        # (exercises the weights==0 mask on top of the -100 sentinel).
        weights = [1.0] * L
        for i in range(L // 3, L // 3 + min(997, L // 4)):
            weights[i] = 0.0
        datums.append(
            {
                "model_input": {
                    "chunks": [{"type": "encoded_text", "tokens": tokens}]
                },
                "loss_fn_inputs": {
                    "weights": {
                        "data": weights,
                        "dtype": "float32",
                        "shape": [L],
                    }
                },
            }
        )
    if permute == "reverse":
        datums = list(reversed(datums))
    elif permute != "identity":
        raise ValueError(f"unknown --permute value: {permute!r}")
    return datums


def submit_and_wait(client: httpx.Client, path: str, body: dict, timeout: float) -> dict:
    r = client.post(
        path, json=body, headers={"Idempotency-Key": uuid.uuid4().hex}, timeout=60.0
    )
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
            raise RuntimeError(
                f"{path} op {operation_id} errored: {body.get('error', '')[:2000]}"
            )
    raise TimeoutError(f"{path} op {operation_id} did not finish in {timeout}s")


def run_leg(
    base_url: str, label: str, permute: str = "identity", forward_only: bool = False
) -> dict:
    datums = make_datums(permute)
    total = sum(
        len(d["model_input"]["chunks"][0]["tokens"]) for d in datums
    )
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        status = client.get("/status").json()
        print(f"[{label}] server status: mode={status.get('mode')} "
              f"step={status.get('step')} pp={status.get('pipeline_parallel_size')} "
              f"cp={status.get('context_parallel_size')}", flush=True)
        t0 = time.perf_counter()
        # forward_only: same request shape/result schema, no backward —
        # dodges the F3 grad-sync crash on PP1/CP8 boots (LPS-1062).
        path = "/forward" if forward_only else "/forward_backward"
        fb = submit_and_wait(client, path, {"data": datums}, FB_TIMEOUT_S)
        fb_s = time.perf_counter() - t0

    outputs = fb.get("loss_fn_outputs") or []
    per_datum = []
    for d in outputs:
        lp = d.get("logprobs") or {}
        per_datum.append(
            {"data": lp.get("data"), "shape": lp.get("shape"), "dtype": lp.get("dtype")}
        )
    rec = {
        "label": label,
        "loss": fb.get("loss"),
        "loss_fn_output_type": fb.get("loss_fn_output_type"),
        "metrics": fb.get("metrics"),
        "n_datums": len(datums),
        "total_real_tokens": total,
        "fb_elapsed_s": fb_s,
        "per_datum_logprobs": per_datum,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"parity_{label}.json"
    out_path.write_text(json.dumps(rec))
    print(f"[{label}] loss={rec['loss']} tokens={total} fb={fb_s:.1f}s "
          f"outputs={len(outputs)} -> {out_path}", flush=True)
    return rec


def compare(path_a: str, path_b: str) -> int:
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    ok = True

    if a["n_datums"] != b["n_datums"] or a["total_real_tokens"] != b["total_real_tokens"]:
        print(f"FAIL: datum set differs ({a['n_datums']}/{a['total_real_tokens']} vs "
              f"{b['n_datums']}/{b['total_real_tokens']})")
        return 2

    la, lb = a["loss"], b["loss"]
    rel = abs(la - lb) / max(abs(la), 1e-12)
    status = "OK" if rel <= LOSS_REL_TOL else "FAIL"
    ok &= status == "OK"
    print(f"loss: {la} vs {lb}  rel_diff={rel:.3e}  [{status} tol={LOSS_REL_TOL}]")

    oa, ob = a["per_datum_logprobs"], b["per_datum_logprobs"]
    if len(oa) != len(ob):
        print(f"FAIL: loss_fn_outputs count {len(oa)} vs {len(ob)}")
        return 2
    worst = 0.0
    worst_at = None
    for i, (da, db) in enumerate(zip(oa, ob)):
        va, vb = da["data"], db["data"]
        if len(va) != len(vb):
            print(f"FAIL: datum {i} logprob length {len(va)} vs {len(vb)}")
            ok = False
            continue
        for j, (x, y) in enumerate(zip(va, vb)):
            d = abs(x - y)
            if d > worst:
                worst, worst_at = d, (i, j)
    status = "OK" if worst <= LOGPROB_ABS_TOL else "FAIL"
    ok &= status == "OK"
    print(f"logprobs: max_abs_diff={worst:.3e} at {worst_at}  [{status} "
          f"tol={LOGPROB_ABS_TOL}] over {sum(len(d['data']) for d in oa)} tokens")

    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--permute", default="identity", choices=["identity", "reverse"])
    ap.add_argument("--forward-only", action="store_true")
    args = ap.parse_args()
    if args.compare:
        raise SystemExit(compare(*args.compare))
    if not args.label:
        ap.error("--label is required when not comparing")
    run_leg(args.base_url, args.label, args.permute, args.forward_only)


if __name__ == "__main__":
    main()
