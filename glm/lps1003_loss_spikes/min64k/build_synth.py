#!/usr/bin/env python3
"""Build a synthetic /forward payload for the LPS-1003 min-config repro.

16 docs x ~50k tokens, prefix_len=32, mean-reduction weights (mirrors
probe_nll.make_datum wire format exactly):
  docs 0-7  : "ABCDEFEDCBA"-style palindrome text repeated, tokenized
              (alphabet rotated per doc so the docs differ)
  docs 8-15 : a random 256-token chunk (fixed seed per doc) tiled to 50k ids
              (position-sensitive: predicting repeats requires attending to
              the right earlier occurrence)
"""
import glob
import json
import random
import string
import sys

OUT = sys.argv[1]
N_TOK = 50_000
PREFIX = 32

snap = glob.glob(
    "/root/.cache/team_artifacts/huggingface/hub/"
    "models--zai-org--GLM-5.2-FP8/snapshots/*"
)[0]
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
vocab = tok.vocab_size

def make_datum(ids):
    p, L = PREFIX, len(ids)
    n_t = L - 1
    n_sup = L - p
    wv = 1.0 / n_sup
    weights = [0.0] * (p - 1) + [wv] * n_sup
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": ids[:-1]}]},
        "loss_fn_inputs": {
            "weights": {"data": weights, "dtype": "float32", "shape": [n_t]},
            "target_tokens": {"data": ids[1:], "dtype": "int64", "shape": [n_t]},
        },
    }

data = []
# docs 0-7: rotated-palindrome text
for d in range(8):
    az = string.ascii_uppercase[d:] + string.ascii_uppercase[:d]
    unit = az[:6] + az[4::-1]  # e.g. ABCDEFEDCBA for d=0
    text = unit * (N_TOK * 4)  # overshoot; truncate in token space
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= N_TOK, f"doc {d}: only {len(ids)} tokens"
    data.append(make_datum(ids[:N_TOK]))
    print(f"doc {d} palindrome unit={unit} ids={N_TOK}", flush=True)

# docs 8-15: tiled random chunk
for d in range(8):
    rng = random.Random(1003_000 + d)
    chunk = [rng.randrange(1000, min(vocab, 140_000)) for _ in range(256)]
    ids = (chunk * ((N_TOK // 256) + 1))[:N_TOK]
    data.append(make_datum(ids))
    print(f"doc {8+d} random-tile seed={1003_000+d}", flush=True)

body = {"data": data, "loss_fn": "cross_entropy"}
with open(OUT, "w") as fh:
    json.dump(body, fh)
print(f"wrote {OUT}: {len(data)} datums x {N_TOK} tokens (prefix {PREFIX})")
