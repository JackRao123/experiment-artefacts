# Build the TRAINING-replay bundle: exact token ids + mask boundary for every
# datum in batches 0..31 (seed-0 shuffle order), so the devbox can replay real
# training steps 0..31 (forward_backward + optim_step) with byte-exact prod
# batches. Same row format as probe_bundle: {batch, label, idx, offset,
# prefix_len, ids}. Label = train_replay for all.
import json, gzip, sys
from transformers import AutoTokenizer

BASE = "/private/tmp/claude-501/-Users-jackrao-Documents-trainers/07b21585-67f2-42e0-9d4c-9a2f526651de/scratchpad/lps1003/repro/repro/glm_r1"
DATA = f"{BASE}/data/conversations.train.jsonl"
PROFILE = f"{BASE}/batches_profile.json"
BUNDLE = sys.argv[1] if len(sys.argv) > 1 else "train_bundle_0_31.jsonl.gz"
N_BATCHES = 32

batches = json.load(open(PROFILE))
tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.2-FP8", trust_remote_code=True)
TPL = {"enable_thinking": False}
fh = open(DATA, "rb")

n = 0
with gzip.open(BUNDLE, "wt") as out:
    for bi in range(N_BATCHES):
        for di, d in enumerate(batches[bi]):
            fh.seek(d["offset"])
            msgs = json.loads(fh.readline())["messages"]
            s_full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **TPL)
            s_prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, **TPL)
            ids_full = tok(s_full, add_special_tokens=False)["input_ids"]
            ids_prefix = tok(s_prefix, add_special_tokens=False)["input_ids"]
            p = 0
            for a, b in zip(ids_full, ids_prefix):
                if a == b: p += 1
                else: break
            assert p == len(ids_prefix), f"masking invariant broken batch {bi} idx {di}"
            out.write(json.dumps({"batch": bi, "label": "train_replay", "idx": di,
                                  "offset": d["offset"], "prefix_len": p,
                                  "ids": ids_full}) + "\n")
            n += 1
        print(f"batch {bi} bundled", flush=True)
print(f"DONE: {n} datums -> {BUNDLE}", flush=True)
