#!/usr/bin/env python3
"""Adjudication campaign: install triple-exec adjudicator, then cycle
{empty_cache lever -> destroyed rep -> heal rep} N times.

Usage: adjud_campaign.py PAYLOAD URL OUTDIR LEVER_FILE [CYCLES]
"""
import json
import sys
import time
import urllib.request
import uuid

PAYLOAD, URL, OUTDIR, LEVER_FILE = sys.argv[1:5]
CYCLES = int(sys.argv[5]) if len(sys.argv) > 5 else 10
ADJ = "/root/.cache/user_artifacts/lps1003/rearm/adjudicator.py"

body = open(PAYLOAD, "rb").read()
log = open(f"{OUTDIR}/adjud_campaign.jsonl", "a", buffering=1)
seq = int(time.time())


def req(method, path, data=None, timeout=180):
    r = urllib.request.Request(URL + path, method=method, data=data,
                               headers={"Content-Type": "application/json",
                                        "Idempotency-Key": uuid.uuid4().hex})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, resp.read()


def one_rep():
    st, raw = req("POST", "/forward", body)
    assert st == 202, f"submit {st}: {raw[:300]}"
    op = json.loads(raw)["operation_id"]
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        try:
            st, raw = req("GET", f"/operations/{op}", timeout=60)
        except Exception as e:
            print(f"  poll err: {e}", flush=True)
            time.sleep(5)
            continue
        p = json.loads(raw)
        if p.get("status") == "done":
            return p["result"]
        if p.get("status") == "error":
            raise RuntimeError(f"op error: {str(p.get('error'))[:2000]}")
        time.sleep(1.0)
    raise RuntimeError("op timeout")


def nlls_of(result):
    out = []
    for o in result["loss_fn_outputs"]:
        lp = o["logprobs"]["data"] if isinstance(o["logprobs"], dict) else o["logprobs"]
        out.append(-sum(lp) / len(lp))
    return out


def set_lever(actions, pycode=None):
    global seq
    seq += 1
    cfg = {"seq": seq, "mode": "oneshot", "actions": actions}
    if pycode:
        cfg["pycode"] = pycode
    with open(LEVER_FILE, "w") as fh:
        json.dump(cfg, fh)
    return seq


def rep(tag, cyc, i):
    t0 = time.time()
    nlls = nlls_of(one_rep())
    wall = time.time() - t0
    destroyed = any(v > 4.4 for v in nlls[4:])  # docs 4-6 destroyed signature
    rec = {"cycle": cyc, "rep": i, "tag": tag, "wall": round(wall, 1),
           "nlls": [round(v, 4) for v in nlls], "destroyed": destroyed,
           "ts": time.time()}
    log.write(json.dumps(rec) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] cyc{cyc} {tag} wall={wall:.0f}s "
          + " ".join(f"{v:.3f}" for v in nlls)
          + ("  *** DESTROYED" if destroyed else ""), flush=True)


# 1. install adjudicator (idempotent), verify with one clean rep
set_lever([], pycode=f"exec(open('{ADJ}').read())")
rep("install", -1, 0)

# 2. cycles: empty_cache -> treated rep (expect destroyed) -> heal rep
for c in range(CYCLES):
    set_lever(["empty_cache"])
    rep("treated", c, 0)
    rep("heal", c, 1)

print("CAMPAIGN DONE", flush=True)
