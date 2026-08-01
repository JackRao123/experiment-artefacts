#!/usr/bin/env python3
"""LPS-1003 parity probe: POST a prebuilt /forward payload, dump FULL per-token
logprobs per rep. Stdlib only -> runs unmodified in a prod trainer pod
(python3, port 8000) and on the devbox (port 8001).

Per rep prints one summary line:
  REP <k> tag=<tag> datum_mean=<m> destroyed=[idx:nll ...] wall=<s>
and writes <out>/<tag>_rep<k>.json.gz with:
  {tag, rep, ts_start, ts_end, op_id, url, status_before, nlls, destroyed,
   logprobs: [[...] x n_datums]}

Usage:
  python3 probe_lp.py PAYLOAD.json --url http://127.0.0.1:8000 --reps 6 \
      --tag window0 --out /tmp/parity [--wait-health 2400] [--gap 0]
"""
import argparse
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid


def req(url, method, path, body=None, headers=None, timeout=120):
    r = urllib.request.Request(url + path, method=method,
                               data=body if isinstance(body, (bytes, type(None))) else body.encode(),
                               headers=headers or {})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, resp.read()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_health(url, budget_s):
    t0 = time.time()
    last = 0
    while time.time() - t0 < budget_s:
        try:
            st, _ = req(url, "GET", "/health", timeout=5)
            if st == 200:
                log(f"HEALTHY after {time.time()-t0:.0f}s")
                return True
        except Exception:
            pass
        if time.time() - last > 60:
            log(f"waiting for {url}/health ({time.time()-t0:.0f}/{budget_s}s)")
            last = time.time()
        time.sleep(3)
    log(f"HEALTH TIMEOUT after {budget_s}s")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--out", default=".")
    ap.add_argument("--op-path", default="/forward")
    ap.add_argument("--gap", type=float, default=0.0, help="seconds between reps")
    ap.add_argument("--wait-health", type=float, default=0.0)
    ap.add_argument("--max-poll-s", type=float, default=1200.0)
    ap.add_argument("--rep-offset", type=int, default=0)
    args = ap.parse_args()

    raw = open(args.payload, "rb").read()
    payload = json.loads(raw)
    weights = [d["loss_fn_inputs"]["weights"]["data"] for d in payload["data"]]
    os.makedirs(args.out, exist_ok=True)

    if args.wait_health and not wait_health(args.url, args.wait_health):
        sys.exit(3)

    for k in range(args.rep_offset, args.rep_offset + args.reps):
        t0 = time.time()
        try:
            st, body = req(args.url, "GET", "/status", timeout=30)
            status_before = json.loads(body)
        except Exception as e:
            status_before = {"error": str(e)}
        try:
            st, body = req(args.url, "POST", args.op_path, raw,
                           {"Content-Type": "application/json",
                            "Idempotency-Key": uuid.uuid4().hex}, timeout=300)
        except urllib.error.HTTPError as e:
            log(f"REP {k} SUBMIT FAILED {e.code}: {e.read()[:300]}")
            time.sleep(10)
            continue
        if st != 202:
            log(f"REP {k} SUBMIT status {st}: {body[:300]}")
            time.sleep(10)
            continue
        oid = json.loads(body)["operation_id"]
        result = None
        deadline = time.time() + args.max_poll_s
        while time.time() < deadline:
            try:
                st, body = req(args.url, "GET", f"/operations/{oid}", timeout=60)
            except Exception:
                time.sleep(2)
                continue
            p = json.loads(body)
            s = p.get("status")
            if s == "done":
                result = p["result"]
                break
            if s == "error":
                log(f"REP {k} OP ERROR: {str(p)[:1200]}")
                break
            time.sleep(2)
        if result is None:
            log(f"REP {k} no result (timeout or error) after {time.time()-t0:.0f}s")
            continue
        outs = result["loss_fn_outputs"]
        lps, nlls = [], []
        for w, out in zip(weights, outs):
            lp = out["logprobs"]["data"] if isinstance(out["logprobs"], dict) else out["logprobs"]
            lps.append(lp)
            nlls.append(-sum(a * b for a, b in zip(lp, w)))
        dm = sum(nlls) / len(nlls)
        destroyed = {i: round(v, 3) for i, v in enumerate(nlls) if v > 2.0}
        rec = {"tag": args.tag, "rep": k, "ts_start": t0, "ts_end": time.time(),
               "op_id": oid, "url": args.url, "status_before": status_before,
               "op_loss": result.get("loss"), "nlls": nlls,
               "destroyed": destroyed, "logprobs": lps}
        fp = os.path.join(args.out, f"{args.tag}_rep{k}.json.gz")
        with gzip.open(fp, "wt") as fh:
            json.dump(rec, fh)
        log(f"REP {k} tag={args.tag} datum_mean={dm:.4f} "
            f"destroyed={destroyed if destroyed else '{}'} wall={time.time()-t0:.0f}s")
        print("  per-datum:", " ".join(f"{x:.3f}" for x in nlls), flush=True)
        if args.gap:
            time.sleep(args.gap)
    log(f"DONE tag={args.tag}")


if __name__ == "__main__":
    main()
