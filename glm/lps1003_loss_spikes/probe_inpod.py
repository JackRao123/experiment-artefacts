#!/usr/bin/env python3
"""In-pod probe: POST a prebuilt /forward payload to localhost:8001, poll to
completion, print datum-mean NLL. Stdlib only (urllib) — runs with any python3.

Usage: python3 probe_inpod.py /tmp/probe_batch0_forward.json [repeats]
"""
import json
import sys
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8001"


def req(method, path, body=None, headers=None, timeout=120):
    r = urllib.request.Request(BASE + path, method=method,
                               data=body.encode() if isinstance(body, str) else body,
                               headers=headers or {})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, resp.read()


def main():
    payload_path = sys.argv[1]
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    raw = open(payload_path, "rb").read()
    payload = json.loads(raw)
    weights = [d["loss_fn_inputs"]["weights"]["data"] for d in payload["data"]]

    st, body = req("GET", "/status", timeout=30)
    print("status:", body[:160].decode(), flush=True)

    for rep in range(repeats):
        t0 = time.time()
        st, body = req("POST", "/forward", raw,
                       {"Content-Type": "application/json",
                        "Idempotency-Key": uuid.uuid4().hex}, timeout=300)
        assert st == 202, (st, body[:300])
        oid = json.loads(body)["operation_id"]
        result = None
        for _ in range(600):
            try:
                st, body = req("GET", f"/operations/{oid}", timeout=60)
            except Exception as e:  # noqa: BLE001 — 408 long-poll etc.
                continue
            p = json.loads(body)
            s = p.get("status")
            if s == "done":
                result = p["result"]; break
            if s == "error":
                print("OP ERROR:", str(p)[:1500], flush=True); return
            time.sleep(2)
        outs = result["loss_fn_outputs"]
        nlls = []
        for w, out in zip(weights, outs):
            lp = out["logprobs"]["data"] if isinstance(out["logprobs"], dict) else out["logprobs"]
            nlls.append(-sum(a * b for a, b in zip(lp, w)))
        dm = sum(nlls) / len(nlls)
        print(f"rep{rep}: datum_mean={dm:.4f} op_loss={result.get('loss')} "
              f"n={len(nlls)} wall={time.time()-t0:.0f}s", flush=True)
        print("  per-datum:", " ".join(f"{x:.3f}" for x in nlls), flush=True)


if __name__ == "__main__":
    main()
