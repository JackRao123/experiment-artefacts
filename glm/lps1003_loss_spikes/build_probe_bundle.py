# Build a self-contained probe bundle for the devbox: exact token ids + mask
# boundary for every datum in the target batches (spike batches from both GLM
# runs + quiet controls). The devbox then needs no dataset or tokenizer.
import json, gzip
from transformers import AutoTokenizer

DATA = "./data/conversations.train.jsonl"
BUNDLE = "probe_bundle.jsonl.gz"
TARGET = {
    # bumps seen in patched and/or broken run (steps 0-35 window)
    12: "bump_patched", 13: "bump_patched", 14: "bump_patched",
    16: "bump_both", 17: "bump_both", 23: "bump_both",
    26: "bump_both", 27: "bump_both", 29: "bump_both",
    # broken-run spike steps in the 36-74 window (patched-run labels TBD)
    48: "spike_broken", 55: "spike_broken", 62: "spike_broken",
    70: "spike_broken", 71: "spike_broken",
    # quiet controls
    15: "quiet", 18: "quiet", 24: "quiet", 25: "quiet", 28: "quiet",
    31: "quiet", 50: "quiet", 60: "quiet",
}

batches = json.load(open("batches_profile.json"))
tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.2-FP8", trust_remote_code=True)
TPL = {"enable_thinking": False}
fh = open(DATA, "rb")

n = 0
with gzip.open(BUNDLE, "wt") as out:
    for bi, label in sorted(TARGET.items()):
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
            out.write(json.dumps({"batch": bi, "label": label, "idx": di,
                                  "offset": d["offset"], "prefix_len": p,
                                  "ids": ids_full}) + "\n")
            n += 1
        print(f"batch {bi} ({label}) bundled", flush=True)
print(f"DONE: {n} datums -> {BUNDLE}", flush=True)
