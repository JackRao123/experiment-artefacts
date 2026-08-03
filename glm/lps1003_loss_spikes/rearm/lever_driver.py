#!/usr/bin/env python3
"""LPS-1003 lever ladder driver: re-arm cold-start ingredients WITHOUT restart.

For each trial: write levers.json (seq bump + actions), then run REPS_PER
/forward reps. Rep index 0 of the trial is the "treated" rep (the lever hook
applies the actions on every rank right before it). Destroyed = any doc NLL
rising > +1.0 nats above its running median (rep0 signature is UP; ambient
steady-state flips are DOWN, so direction disambiguates).

Usage: lever_driver.py PAYLOAD URL OUTDIR LEVER_FILE
Edit LADDER below to change the campaign.
"""
import json
import statistics
import sys
import time
import urllib.request
import uuid

PAYLOAD, URL, OUTDIR, LEVER_FILE = sys.argv[1:5]
REPS_PER = 3
TRIALS_PER_LEVER = 4

LADDER = [
    {"name": "control", "actions": []},
    {"name": "empty_cache", "actions": ["empty_cache"]},
    {"name": "clear_cute_caches", "actions": ["clear_cute_caches"]},
    {"name": "both", "actions": ["empty_cache", "clear_cute_caches"]},
    {"name": "cublas_ws", "actions": ["clear_cublas_ws"]},
]

body = open(PAYLOAD, "rb").read()
log = open(f"{OUTDIR}/levers.jsonl", "a", buffering=1)
seq = int(time.time())  # monotonic-enough starting seq, > any prior


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


meds = None  # per-doc running median from control trials
ctrl_hist = []

for trial in range(TRIALS_PER_LEVER):
    for lever in LADDER:
        seq += 1
        with open(LEVER_FILE, "w") as fh:
            json.dump({"seq": seq, "mode": "oneshot", "actions": lever["actions"]}, fh)
        for rep in range(REPS_PER):
            t0 = time.time()
            try:
                nlls = nlls_of(one_rep())
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] {lever['name']} t{trial} r{rep} "
                      f"ERROR {e}", flush=True)
                log.write(json.dumps({"lever": lever["name"], "trial": trial,
                                      "rep": rep, "error": str(e)[:500]}) + "\n")
                time.sleep(10)
                continue
            wall = time.time() - t0
            if lever["name"] == "control":
                ctrl_hist.append(nlls)
                if len(ctrl_hist) >= 3:
                    meds = [statistics.median(h[i] for h in ctrl_hist)
                            for i in range(len(nlls))]
            dev = ([nlls[i] - meds[i] for i in range(len(nlls))] if meds
                   else [0.0] * len(nlls))
            fired = any(d > 1.0 for d in dev)
            low = any(d < -0.5 for d in dev)
            rec = {"lever": lever["name"], "trial": trial, "rep": rep, "seq": seq,
                   "wall": round(wall, 1), "nlls": [round(v, 4) for v in nlls],
                   "fired": fired, "lowstate": low, "ts": time.time()}
            log.write(json.dumps(rec) + "\n")
            tag = " *** FIRED" if fired else (" (lowstate)" if low else "")
            print(f"[{time.strftime('%H:%M:%S')}] {lever['name']:>18} t{trial} r{rep} "
                  f"wall={wall:.0f}s nlls=" + " ".join(f"{v:.3f}" for v in nlls) + tag,
                  flush=True)

print("LADDER DONE", flush=True)
