#!/usr/bin/env python3
"""LPS-1003: open REBUILD windows and probe them. Stdlib only (in-pod safe).

Per cycle: forward_backward(payload) -> optim_step -> init_trainer_server
(REBUILD path: state accumulated) -> probe_lp-style /forward reps with full
logprob dumps. Adapter re-init returns weights to base (B=0), so probes stay
comparable across cycles. HARD CAP 2 cycles per process: the 3rd in-process
rebuild deadlocks the trainer (reproduced on prod 2026-07-30).

Usage: python3 rebuild_windows.py PAYLOAD.json --url http://127.0.0.1:8000 \
         --cycles 2 --reps 4 --out /tmp/parity --tag preb
"""
import argparse
import gzip
import json
import os
import time
import urllib.request
import uuid


def req(url, method, path, body=None, headers=None, timeout=180):
    r = urllib.request.Request(url + path, method=method,
                               data=body if isinstance(body, (bytes, type(None))) else body.encode(),
                               headers=headers or {})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, resp.read()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run_op(url, path, body_bytes, max_poll_s=1800):
    st, body = req(url, "POST", path, body_bytes,
                   {"Content-Type": "application/json",
                    "Idempotency-Key": uuid.uuid4().hex}, timeout=300)
    assert st == 202, (path, st, body[:300])
    oid = json.loads(body)["operation_id"]
    deadline = time.time() + max_poll_s
    while time.time() < deadline:
        try:
            st, body = req(url, "GET", f"/operations/{oid}", timeout=60)
        except Exception:
            time.sleep(2)
            continue
        p = json.loads(body)
        s = p.get("status")
        if s == "done":
            return p["result"]
        if s == "error":
            raise RuntimeError(f"{path} op error: {str(p)[:1000]}")
        time.sleep(2)
    raise RuntimeError(f"{path} op timeout after {max_poll_s}s (oid={oid})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", default=".")
    ap.add_argument("--tag", default="reb")
    args = ap.parse_args()
    assert args.cycles <= 2, "3rd in-process rebuild deadlocks the trainer"

    raw = open(args.payload, "rb").read()
    payload = json.loads(raw)
    weights = [d["loss_fn_inputs"]["weights"]["data"] for d in payload["data"]]
    os.makedirs(args.out, exist_ok=True)
    fb_body = raw  # same body shape works for /forward_backward
    optim_body = json.dumps({"adam_params": {"learning_rate": 5e-4, "beta1": 0.9,
                                             "beta2": 0.95, "eps": 1e-8}}).encode()

    for cyc in range(args.cycles):
        log(f"cycle {cyc}: forward_backward...")
        run_op(args.url, "/forward_backward", fb_body)
        log(f"cycle {cyc}: optim_step...")
        run_op(args.url, "/optim_step", optim_body)
        log(f"cycle {cyc}: init_trainer_server (REBUILD)...")
        t_reb = time.time()
        run_op(args.url, "/init_trainer_server",
               json.dumps({"lora_rank": 32}).encode())
        log(f"cycle {cyc}: rebuild done in {time.time()-t_reb:.0f}s; probing...")
        for rep in range(args.reps):
            t0 = time.time()
            result = run_op(args.url, "/forward", raw)
            outs = result["loss_fn_outputs"]
            lps, nlls = [], []
            for w, out in zip(weights, outs):
                lp = out["logprobs"]["data"] if isinstance(out["logprobs"], dict) else out["logprobs"]
                lps.append(lp)
                nlls.append(-sum(a * b for a, b in zip(lp, w)))
            dm = sum(nlls) / len(nlls)
            destroyed = {i: round(v, 3) for i, v in enumerate(nlls) if v > 2.0}
            tag = f"{args.tag}{cyc}"
            rec = {"tag": tag, "rep": rep, "ts_start": t0, "ts_end": time.time(),
                   "op_loss": result.get("loss"), "nlls": nlls,
                   "destroyed": destroyed, "logprobs": lps}
            with gzip.open(os.path.join(args.out, f"{tag}_rep{rep}.json.gz"), "wt") as fh:
                json.dump(rec, fh)
            log(f"REP {rep} tag={tag} datum_mean={dm:.4f} "
                f"destroyed={destroyed if destroyed else '{}'} wall={time.time()-t0:.0f}s")
            print("  per-datum:", " ".join(f"{x:.3f}" for x in nlls), flush=True)
    log("DONE rebuild windows")


if __name__ == "__main__":
    main()
