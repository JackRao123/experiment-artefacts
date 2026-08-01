#!/usr/bin/env python3
"""min32k payload: [random-tile 30k synthetic, longest Mudith datum <=30k].

Mudith datum comes from train_bundle_0_31.jsonl.gz (exact prod token ids,
fields: batch/idx/label/ids/prefix_len). Wire format = probe_nll.make_datum.
"""
import gzip
import json
import random

N_TOK, PREFIX = 30_000, 32
LPS = "/root/.cache/user_artifacts/lps1003"


def make_datum(ids, prefix):
    L = len(ids)
    n_t = L - 1
    n_sup = L - prefix
    weights = [0.0] * (prefix - 1) + [1.0 / n_sup] * n_sup
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": ids[:-1]}]},
        "loss_fn_inputs": {
            "weights": {"data": weights, "dtype": "float32", "shape": [n_t]},
            "target_tokens": {"data": ids[1:], "dtype": "int64", "shape": [n_t]},
        },
    }


# datum 0: random 256-token chunk tiled to 30k
rng = random.Random(1003_100)
chunk = [rng.randrange(1000, 140_000) for _ in range(256)]
tile = (chunk * ((N_TOK // 256) + 1))[:N_TOK]

# datum 1: longest bundle row <= 30k tokens
best = None
with gzip.open(f"{LPS}/train_bundle_0_31.jsonl.gz", "rt") as fh:
    for line in fh:
        r = json.loads(line)
        n = len(r["ids"])
        if n <= N_TOK and (best is None or n > len(best["ids"])):
            best = r
print("mudith pick: batch=%s idx=%s label=%s len=%d prefix=%d"
      % (best["batch"], best["idx"], best.get("label"),
         len(best["ids"]), best["prefix_len"]))

body = {
    "data": [make_datum(tile, PREFIX), make_datum(best["ids"], best["prefix_len"])],
    "loss_fn": "cross_entropy",
}
json.dump(body, open(f"{LPS}/min32k/payload_min32k.json", "w"))
meta = {
    "datum0": {"kind": "random-tile", "seed": 1003_100, "len": N_TOK, "prefix": PREFIX},
    "datum1": {"kind": "mudith-bundle", "batch": best["batch"], "idx": best["idx"],
               "label": best.get("label"), "len": len(best["ids"]),
               "prefix": best["prefix_len"]},
}
json.dump(meta, open(f"{LPS}/min32k/payload_min32k.meta.json", "w"), indent=1)
print("payload written")
