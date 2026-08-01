#!/usr/bin/env python3
"""LPS-1003 steady-state soak: repeat the same /forward payload until killed.

Per rep: append per-datum NLLs to soak.jsonl + print one line. Full
logprob dumps are written ONLY on anomaly (any datum NLL deviating more
than THRESH nats from its baseline median): the anomalous rep and the
previous (in-RAM) rep are both saved for positional diffing.
Usage: soak.py PAYLOAD URL OUTDIR [REPS]
"""
import gzip
import json
import statistics
import sys
import time
import urllib.request
import uuid

PAYLOAD, URL, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 10_000
THRESH = 0.5
MAX_DUMP_PAIRS = 6
BASE_N = 5

body = open(PAYLOAD, "rb").read()
log = open(f"{OUTDIR}/soak.jsonl", "a", buffering=1)


def req(method, path, data=None, timeout=120):
    r = urllib.request.Request(URL + path, method=method, data=data,
                               headers={"Content-Type": "application/json",
                                        "Idempotency-Key": uuid.uuid4().hex})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, resp.read()


def one_rep():
    st, raw = req("POST", "/forward", body)
    assert st == 202, f"submit {st}: {raw[:300]}"
    op = json.loads(raw)["operation_id"]
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            st, raw = req("GET", f"/operations/{op}", timeout=60)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] poll error: {e}", flush=True)
            time.sleep(5)
            continue
        p = json.loads(raw)
        s = p.get("status")
        if s == "done":
            return p["result"]
        if s == "error":
            raise RuntimeError(f"op error: {str(p.get('error'))[:2000]}")
        time.sleep(1.0)
    raise RuntimeError("op timeout 900s")


def nlls_of(result):
    out = []
    lps = []
    for o in result["loss_fn_outputs"]:
        lp = o["logprobs"]["data"] if isinstance(o["logprobs"], dict) else o["logprobs"]
        lps.append(lp)
        out.append(-sum(lp) / len(lp))
    return out, lps


base = None
dumped = 0
prev = None  # (rep, nlls, lps) of previous rep, in RAM only
hist = []
for k in range(REPS):
    t0 = time.time()
    try:
        result = one_rep()
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] REP {k} ERROR: {e}", flush=True)
        log.write(json.dumps({"rep": k, "error": str(e)[:500], "ts": time.time()}) + "\n")
        time.sleep(10)
        continue
    wall = time.time() - t0
    nlls, lps = nlls_of(result)
    hist.append(nlls)
    if base is None and len(hist) >= BASE_N:
        base = [statistics.median(h[i] for h in hist) for i in range(len(nlls))]
        print(f"[{time.strftime('%H:%M:%S')}] baseline set: "
              + " ".join(f"{v:.3f}" for v in base), flush=True)
    dev = ([nlls[i] - base[i] for i in range(len(nlls))] if base else [0.0] * len(nlls))
    flag = base is not None and any(abs(d) > THRESH for d in dev)
    rec = {"rep": k, "ts": time.time(), "wall": round(wall, 1),
           "nlls": [round(v, 5) for v in nlls], "anomaly": flag}
    log.write(json.dumps(rec) + "\n")
    line = (f"[{time.strftime('%H:%M:%S')}] REP {k} wall={wall:.0f}s nlls="
            + " ".join(f"{v:.3f}" for v in nlls))
    if flag:
        line += "  *** ANOMALY dev=" + " ".join(f"{d:+.2f}" for d in dev)
    if flag and dumped < MAX_DUMP_PAIRS:
        dumped += 1
        with gzip.open(f"{OUTDIR}/anomaly_rep{k}.json.gz", "wt") as fh:
            json.dump({"rep": k, "nlls": nlls, "logprobs": lps}, fh)
        if prev is not None:
            with gzip.open(f"{OUTDIR}/anomaly_prevclean_rep{prev[0]}.json.gz", "wt") as fh:
                json.dump({"rep": prev[0], "nlls": prev[1], "logprobs": prev[2]}, fh)
    print(line, flush=True)
    prev = None if flag else (k, nlls, lps)

print(f"[{time.strftime('%H:%M:%S')}] SOAK DONE ({REPS} reps)", flush=True)
