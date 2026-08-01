# Reconstruct the exact batch composition of the GLM runs (shuffle seed 0) and
# profile the content of each batch, to find what distinguishes the spike batches.
# Mirrors loops_sft.py's pipeline exactly: index -> shuffle(0) -> render(thinking off)
# -> encode -> filter (>131072 or trivial) -> consecutive groups of 32 survivors.
import json, random, re, sys
from transformers import AutoTokenizer

DATA = "./data/conversations.train.jsonl"
MAXLEN = 131072
BATCH = 32
N_BATCHES = 75  # covers client steps 0..35

tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.2-FP8", trust_remote_code=True)
TPL = {"enable_thinking": False}

B64_RE = re.compile(r"[A-Za-z0-9+/=]{200,}")
WS_RE = re.compile(r"[ \t]{40,}")
NL_RE = re.compile(r"\n{20,}")


def feats(text, ids_full, p):
    b64 = max((len(m.group()) for m in B64_RE.finditer(text)), default=0)
    ws = max((len(m.group()) for m in WS_RE.finditer(text)), default=0)
    nls = max((len(m.group()) for m in NL_RE.finditer(text)), default=0)
    n = len(text)
    nonascii = sum(1 for c in text[:200000] if ord(c) > 127) / min(n, 200000)
    return dict(chars=n, tokens=len(ids_full), prompt_tokens=p, loss_tokens=len(ids_full) - p,
                max_b64_run=b64, max_ws_run=ws, max_nl_run=nls, nonascii_frac=round(nonascii, 4),
                chars_per_tok=round(n / max(len(ids_full), 1), 2))


offsets = []
with open(DATA, "rb") as f:
    while True:
        o = f.tell(); line = f.readline()
        if not line: break
        if line.strip(): offsets.append(o)
print(f"{len(offsets)} offsets", flush=True)
random.Random(0).shuffle(offsets)

batches, buf = [], []
fh = open(DATA, "rb")
for idx, o in enumerate(offsets):
    fh.seek(o)
    try:
        msgs = json.loads(fh.readline())["messages"]
    except Exception:
        continue
    s_full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **TPL)
    s_prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, **TPL)
    ids_full = tok(s_full, add_special_tokens=False)["input_ids"]
    ids_prefix = tok(s_prefix, add_special_tokens=False)["input_ids"]
    if len(ids_full) > MAXLEN:
        continue
    q = 0
    for a, b in zip(ids_full, ids_prefix):
        if a == b: q += 1
        else: break
    if q >= len(ids_full):
        continue
    d = feats(s_full, ids_full, q)
    d["offset"] = o
    buf.append(d)
    if len(buf) == BATCH:
        batches.append(buf); buf = []
        print(f"batch {len(batches)-1} done ({idx+1} lines consumed)", flush=True)
        if len(batches) >= N_BATCHES:
            break

json.dump(batches, open("batches_profile.json", "w"))

BUMP = {12, 13, 14, 16, 17, 23, 26, 27, 29}
print(f"\n{'step':>4} {'max_tok':>8} {'>32k':>5} {'>100k':>6} {'sum_tok':>9} {'loss_tok':>9} "
      f"{'maxb64':>7} {'maxws':>6} {'maxnl':>6} {'minc/t':>7}  flag")
for i, b in enumerate(batches):
    print(f"{i:>4} {max(d['tokens'] for d in b):>8} "
          f"{sum(1 for d in b if d['tokens'] > 32768):>5} "
          f"{sum(1 for d in b if d['tokens'] > 100000):>6} "
          f"{sum(d['tokens'] for d in b):>9} "
          f"{sum(d['loss_tokens'] for d in b):>9} "
          f"{max(d['max_b64_run'] for d in b):>7} "
          f"{max(d['max_ws_run'] for d in b):>6} "
          f"{max(d['max_nl_run'] for d in b):>6} "
          f"{min(d['chars_per_tok'] for d in b):>7}  {'<-- BUMP' if i in BUMP else ''}")
