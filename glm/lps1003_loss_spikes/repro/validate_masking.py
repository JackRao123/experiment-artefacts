# Validate the prefix-diff masking for a given model BEFORE spending a training run on it.
#
# Our masking works by rendering the conversation twice - the full thing, and the prompt with
# add_generation_prompt=True - then weighting only the tokens past the common prefix. That is
# correct ONLY IF the prompt rendering is an exact token prefix of the full rendering. Chat
# templates differ per model, and some of them break that assumption:
#
#   - a thinking-mode template can leave a block open, so the first TRAINED token is a control
#     tag like `</think>` rather than the answer (GLM-5.2 does exactly this)
#   - a template can emit a trailing newline in the generation prompt that the full rendering
#     merges into the next token, so the prompt is NOT a token prefix and the boundary lands
#     mid-content, silently training on part of the prompt or dropping the first target token
#   - a template can re-order or inject system content only in one of the two renderings
#
# This checks all of that on real examples and prints exactly what would be trained.
#
# Usage: python validate_masking.py <hf_model_id> [--data path] [--n 8] [--thinking-off]
import sys, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--data", default="data/spike_batches/batches.jsonl")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--thinking-off", action="store_true",
                help="render with enable_thinking=False (only if the template accepts it)")
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

TPL = {"enable_thinking": False} if a.thinking_off else {}


def render(msgs):
    try:
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **TPL)
        pre = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, **TPL)
    except TypeError as e:
        raise SystemExit(f"template rejected kwargs {TPL}: {e}")
    return full, pre


rows = []
for line in open(a.data):
    if len(rows) >= a.n: break
    rows.append(json.loads(line)["messages"])

print(f"model: {a.model}")
print(f"kwargs: {TPL or '(defaults)'}\n")

# does the template even accept enable_thinking?
try:
    tok.apply_chat_template(rows[0][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    accepts = True
except Exception:
    accepts = False
print(f"template accepts enable_thinking: {accepts}")
t = tok.chat_template or ""
for probe in ("<think>", "</think>", "reasoning", "enable_thinking"):
    if probe in t:
        print(f"  template mentions {probe!r}: {t.count(probe)}x")
print()

ok = bad = 0
first_tokens = {}
for i, msgs in enumerate(rows):
    full, pre = render(msgs)
    A = tok(full, add_special_tokens=False)["input_ids"]
    B = tok(pre, add_special_tokens=False)["input_ids"]
    p = 0
    for x, y in zip(A, B):
        if x == y: p += 1
        else: break

    exact = (p == len(B))                      # prompt is a TRUE token prefix
    nontrivial = 0 < p < len(A)
    ft = tok.decode([A[p]]) if nontrivial else None
    first_tokens[ft] = first_tokens.get(ft, 0) + 1
    good = exact and nontrivial
    ok, bad = ok + good, bad + (not good)

    flag = "" if good else "  <-- PROBLEM"
    print(f"  ex{i}: total {len(A):>7,}  prompt {len(B):>7,}  common {p:>7,}  "
          f"trained {len(A)-p:>5}  first={ft!r}{flag}")
    if not exact:
        # show where they diverge - this is the silent-corruption case
        j = p
        print(f"        prompt has {len(B)-p} token(s) BEYOND the common prefix:")
        print(f"          prompt tail : {tok.decode(B[max(0,j-6):])!r}")
        print(f"          full   here : {tok.decode(A[max(0,j-6):j+6])!r}")

print(f"\n  exact-prefix and non-trivial: {ok}/{len(rows)}")
print(f"  first trained token histogram: {first_tokens}")
if bad:
    print("\n  *** MASKING IS NOT SAFE FOR THIS MODEL AS-IS ***")
    print("  The prompt is not an exact token prefix, so the boundary is wrong.")
    print("  Do not train until the rendering is fixed.")
    sys.exit(1)
ctrl = [t for t in first_tokens if t and (t.startswith("<") or t.strip() in ("</think>", "<think>"))]
if ctrl:
    print(f"\n  NOTE: the first trained token is a control tag {ctrl} - the model will be taught")
    print("  to emit it. Either render so it sits in the prompt, or make sure serving matches.")
else:
    print("\n  Masking is safe: prompt is an exact token prefix and training starts at content.")
