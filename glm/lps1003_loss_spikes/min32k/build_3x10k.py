#!/usr/bin/env python3
"""3x10k payload, one 32k multi-doc row: [palindrome, random-tile, mudith-slice].

Every datum: uniform LM supervision over ALL target positions (weight
1/(L-1) everywhere, no prefix masking of any kind).
Mudith doc = train_bundle batch 0 idx 0 token ids, naively truncated to the
first 10,000 tokens. Tokens only — no prefix_len/supervision carried over.
"""
import glob
import gzip
import json
import random

N = 10_000
LPS = "/root/.cache/user_artifacts/lps1003"


def make_datum(ids):
    n_t = len(ids) - 1
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": ids[:-1]}]},
        "loss_fn_inputs": {
            "weights": {"data": [1.0 / n_t] * n_t, "dtype": "float32", "shape": [n_t]},
            "target_tokens": {"data": ids[1:], "dtype": "int64", "shape": [n_t]},
        },
    }


# doc 1: palindrome text, tokenized, first 10k tokens
snap = glob.glob(
    "/root/.cache/team_artifacts/huggingface/hub/"
    "models--zai-org--GLM-5.2-FP8/snapshots/*"
)[0]
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
pal_ids = tok("ABCDEFEDCBA" * (N * 4), add_special_tokens=False)["input_ids"][:N]
assert len(pal_ids) == N

# doc 2: random 256-token chunk tiled to 10k
rng = random.Random(1003_200)
chunk = [rng.randrange(1000, 140_000) for _ in range(256)]
tile_ids = (chunk * ((N // 256) + 1))[:N]

# doc 3: mudith batch 0 idx 0, first 10k tokens
mud = None
with gzip.open(f"{LPS}/train_bundle_0_31.jsonl.gz", "rt") as fh:
    for line in fh:
        r = json.loads(line)
        if r["batch"] == 0 and r["idx"] == 0:
            mud = r
            break
assert mud is not None, "batch 0 idx 0 not found"
mud_ids = mud["ids"][:N]
print("mudith b0 i0: orig len %d -> sliced %d, label=%s"
      % (len(mud["ids"]), len(mud_ids), mud.get("label")))
assert len(mud_ids) == N, f"doc shorter than {N}: {len(mud_ids)}"

body = {
    "data": [make_datum(pal_ids), make_datum(tile_ids), make_datum(mud_ids)],
    "loss_fn": "cross_entropy",
}
json.dump(body, open(f"{LPS}/min32k/payload_3x10k.json", "w"))
meta = {
    "supervision": "uniform over all positions, no prefix masking",
    "datum0": {"kind": "palindrome-ABCDEFEDCBA", "len": N},
    "datum1": {"kind": "random-tile", "seed": 1003_200, "chunk": 256, "len": N},
    "datum2": {"kind": "mudith-b0-i0-first10k", "orig_len": len(mud["ids"]),
               "label": mud.get("label"), "len": N},
}
json.dump(meta, open(f"{LPS}/min32k/payload_3x10k.meta.json", "w"), indent=1)
print("wrote payload_3x10k.json (3 docs x %d tokens, 30k total, uniform LM loss)" % N)
