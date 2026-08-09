#!/usr/bin/env python3
"""GLM-5.2 EP8+CP8 context-length memory sweep against a devbox trainer.

Sends ONE synthetic datum of length L per forward_backward (CE loss, labels
auto-derived by shifting input_ids), followed by one optim_step, while polling
nvidia-smi for per-GPU memory.used. Appends one JSON row per L to the output
JSONL and stops at the first failed op (OOM).

Run on the devbox leader. Trainer must be healthy on 127.0.0.1:8001.
"""

import argparse
import json
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8001"
VOCAB = 154880  # GLM-5.2 vocab_size


def http_post(path, payload, timeout=120):
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
        except Exception as e:  # connection reset etc. — trainer may have died
            return False, {"error": f"poll exception: {e!r}"}
        if body.get("status") == "done":
            return True, body
        if body.get("status") == "error":
            return False, body
        time.sleep(1)
    return False, {"error": "client-side timeout"}


class NvmlPoller(threading.Thread):
    def __init__(self, period=0.25):
        super().__init__(daemon=True)
        self.period = period
        self.stop_evt = threading.Event()
        self.samples = []  # (ts, [mib per gpu])

    def run(self):
        while not self.stop_evt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
                if vals:
                    self.samples.append((time.time(), vals))
            except Exception:
                pass
            self.stop_evt.wait(self.period)


def run_cycle(L, seed=1234):
    """One forward_backward (1 datum, L tokens) + one optim_step."""
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
    t0 = time.time()
    resp = http_post("/forward_backward", fb)
    ok, detail = wait_op(resp["operation_id"])
    if not ok:
        return False, detail, time.time() - t0
    fb_s = time.time() - t0
    t1 = time.time()
    resp = http_post("/optim_step", {"adam_params": {"learning_rate": 1e-4}})
    ok, detail = wait_op(resp["operation_id"])
    return ok, detail, time.time() - t0  # total wall; fb portion in fb_s


def measure_window(fn):
    """Run fn() under the NVML poller; return (ok, detail, wall_s, peaks_mib)."""
    poller = NvmlPoller()
    poller.start()
    time.sleep(1.0)  # idle baseline samples
    t0 = time.time()
    ok, detail, _ = fn()
    wall = time.time() - t0
    time.sleep(2.0)  # reserved high-water mark persists; let poller catch it
    poller.stop_evt.set()
    poller.join(timeout=5)
    peaks = [0] * 8
    for _, vals in poller.samples:
        for i, v in enumerate(vals):
            if i < 8 and v > peaks[i]:
                peaks[i] = v
    return ok, detail, wall, peaks, len(poller.samples)


def append_row(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lengths",
        default="16384,32768,49152,65536,81920,98304,114688,131072,"
        "147456,163840,180224,196608,212992,229376,245760,262144,"
        "294912,327680",
    )
    ap.add_argument("--out", default="/root/.cache/user_artifacts/glm52-prof/results.jsonl")
    ap.add_argument("--warmup", type=int, default=4096)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    with urllib.request.urlopen(f"{BASE}/health", timeout=15) as r:
        assert r.status == 200, "trainer not healthy"

    # Idle baseline row (L=0): model resident, no steps.
    poller = NvmlPoller()
    poller.start()
    time.sleep(3.0)
    poller.stop_evt.set()
    poller.join(timeout=5)
    idle = [0] * 8
    for _, vals in poller.samples:
        for i, v in enumerate(vals):
            if i < 8 and v > idle[i]:
                idle[i] = v
    append_row(args.out, {"L": 0, "ok": True, "peak_mib_per_gpu": idle,
                          "peak_mib_max": max(idle), "peak_mib_min": min(idle),
                          "note": "idle baseline after trainer ready"})

    if args.warmup:
        print(f"[warmup] L={args.warmup}", flush=True)
        ok, detail, wall = run_cycle(args.warmup)
        print(f"[warmup] ok={ok} wall={wall:.1f}s", flush=True)
        if not ok:
            print(json.dumps(detail)[:2000], flush=True)
            sys.exit("warmup failed — trainer not usable")

    for L in lengths:
        ok, detail, wall, peaks, nsamples = measure_window(lambda: run_cycle(L))
        row = {
            "L": L,
            "ok": ok,
            "wall_s": round(wall, 2),
            "peak_mib_per_gpu": peaks,
            "peak_mib_max": max(peaks),
            "peak_mib_min": min(peaks),
            "samples": nsamples,
        }
        if not ok:
            row["error"] = json.dumps(detail)[:1500]
        append_row(args.out, row)
        if not ok:
            print(f"[sweep] stopping at L={L} (first failure)", flush=True)
            sys.exit(0)


if __name__ == "__main__":
    main()
