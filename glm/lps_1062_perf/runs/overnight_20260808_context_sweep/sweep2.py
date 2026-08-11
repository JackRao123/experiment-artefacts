#!/usr/bin/env python3
"""GLM-5.2 golden-config (EP16+CP16, 2-node) context-length memory sweep.

Same protocol as run 1 (one synthetic datum of length L per forward_backward
+ one optim_step, CE loss), but memory is recorded by per-node nvidia-smi
pollers (mem_poller.sh) instead of an in-process thread: this script records
[t_start, t_end] windows per L; analysis joins windows against the poller
logs from all nodes.

Run on the devbox leader. Trainer must be healthy on 127.0.0.1:8001.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8001"
VOCAB = 154880  # GLM-5.2 vocab_size


def http_post(path, payload, timeout=180):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_op(op_id, timeout_s=7200):
    """Poll /operations/{id}; the endpoint long-polls 30s and 408s while pending."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{BASE}/operations/{op_id}", timeout=75) as r:
                body = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 408:
                continue  # still pending
            return False, {"error": f"http {e.code}", "body": e.read()[:400].decode("utf-8", "replace")}
        except Exception as e:  # trainer died (OOM kill) -> connection reset
            return False, {"error": f"poll exception: {e!r}"}
        if body.get("status") == "done":
            return True, body
        if body.get("status") == "error":
            return False, body
        time.sleep(1)
    return False, {"error": "client-side timeout"}


def run_cycle(L, seed=1234):
    rng = random.Random(seed)
    tokens = [rng.randrange(VOCAB) for _ in range(L)]
    fb = {
        "data": [
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
                "loss_fn_inputs": {},
            }
        ],
        "loss_fn": "cross_entropy",
    }
    resp = http_post("/forward_backward", fb)
    ok, detail = wait_op(resp["operation_id"])
    if not ok:
        return False, detail
    resp = http_post("/optim_step", {"adam_params": {"learning_rate": 1e-4}})
    ok, detail = wait_op(resp["operation_id"])
    return ok, detail


def append_row(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lengths",
        default="16384,32768,49152,65536,81920,98304,114688,131072,"
        "163840,196608,229376,262144,"
        "327680,393216,458752,524288,655360,786432,917504,1048576",
    )
    ap.add_argument("--out", default="/root/.cache/user_artifacts/glm52-prof/results2_windows.jsonl")
    ap.add_argument("--warmup", type=int, default=4096)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    with urllib.request.urlopen(f"{BASE}/health", timeout=15) as r:
        assert r.status == 200, "trainer not healthy"

    # Idle marker row: analysis reads the poller logs around this timestamp.
    append_row(args.out, {"L": 0, "ok": True, "t0": time.time(),
                          "note": "idle marker (trainer ready, no steps yet)"})
    time.sleep(3)

    if args.warmup:
        print(f"[warmup] L={args.warmup}", flush=True)
        t0 = time.time()
        ok, detail = run_cycle(args.warmup)
        print(f"[warmup] ok={ok} wall={time.time()-t0:.1f}s", flush=True)
        if not ok:
            print(json.dumps(detail)[:2000], flush=True)
            sys.exit("warmup failed — trainer not usable")
        time.sleep(2)

    for L in lengths:
        t0 = time.time()
        ok, detail = run_cycle(L)
        t1 = time.time()
        row = {"L": L, "ok": ok, "t0": t0, "t1": t1, "wall_s": round(t1 - t0, 2)}
        if not ok:
            row["error"] = json.dumps(detail)[:1500]
        append_row(args.out, row)
        time.sleep(2)  # let reserved high-water mark land in poller samples
        if not ok:
            print(f"[sweep] stopping at L={L} (first failure)", flush=True)
            sys.exit(0)


if __name__ == "__main__":
    main()
