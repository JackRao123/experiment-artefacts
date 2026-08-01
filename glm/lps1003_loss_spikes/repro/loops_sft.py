# Generic Loops LoRA SFT driver — one script, any supported base model.
#
# Supersedes the per-model copies now that we train more than one base. Identical recipe and
# masking to glm_full.py/glm_r2.py; the only per-model input is MODEL (+ optional replicas).
#
# RENDERING: every model is rendered with enable_thinking=False. This is NOT cosmetic — under
# the default templates the generation prompt is not an exact token prefix of the full
# rendering for either Nemotron-3-Super or Qwen3.5-122B:
#
#   Nemotron  prompt ends '<think>\n'  full has '<think></think>'          -> 1 token adrift
#   Qwen      prompt ends '<think>\n'  full has '<think>\n\n</think>\n\n'  -> 1 token adrift
#
# which puts the mask boundary a token early and makes the first "trained" token a control tag
# or bare whitespace. Training would run and the loss would look plausible while every example
# was misaligned. Validate any NEW model with scripts/train/validate_masking.py before use.
#
# SERVING MUST MATCH: a model trained thinking-off must be served thinking-off.
#
# Usage:
#   MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 REPLICAS=2 python loops_sft.py
#   MODEL=Qwen/Qwen3.5-122B-A10B REPLICAS=2 python loops_sft.py
import json, os, sys, time, math, random, torch, tinker, wandb
import tinker.types as _tt
if not hasattr(_tt, "ImageAssetPointerChunk"):
    class ImageAssetPointerChunk:  # noqa: N801
        pass
    _tt.ImageAssetPointerChunk = ImageAssetPointerChunk
from transformers import AutoTokenizer
from tinker_cookbook.supervised.data import datum_from_model_input_weights
from tinker_cookbook.supervised.common import compute_mean_nll
from baseten.loops import ServiceClient as BLServiceClient, Datum as BLDatum

M = os.environ["MODEL"]
DATA = os.environ.get("DATA", "/root/oe/data/sft_prepared_final/conversations.train.jsonl")
# Hyperparameters held IDENTICAL across all base models so the runs are comparable.
BATCH = int(os.environ.get("BATCH", "32"))            # GLOBAL batch, sharded across replicas
MAXLEN = int(os.environ.get("MAXLEN", "131072"))
LR = float(os.environ.get("LR", "5e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
RANK = int(os.environ.get("RANK", "32"))
SHUFFLE_SEED = int(os.environ.get("SHUFFLE_SEED", "0"))
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine")
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "100"))
REPLICAS = int(os.environ.get("REPLICAS", "1"))
READY_TIMEOUT = float(os.environ.get("READY_TIMEOUT", "7200"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "oe-grader-sft")
SHORT = M.split("/")[-1]
RUN_NAME = os.environ.get("RUN_NAME", f"{SHORT}_r{REPLICAS}_b{BATCH}_rank{RANK}")
TPL = {"enable_thinking": False}

tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)


def _enc(s): return tok(s, add_special_tokens=False)["input_ids"]


def build_datum(msgs):
    s_full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **TPL)
    s_prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, **TPL)
    ids_full = _enc(s_full); ids_prefix = _enc(s_prefix)
    if len(ids_full) > MAXLEN: return None
    p = 0
    for a, b in zip(ids_full, ids_prefix):
        if a == b: p += 1
        else: break
    if p >= len(ids_full): return None
    # Guard the invariant the masking depends on. If the prompt is not an exact token prefix
    # the boundary is wrong; fail loudly rather than train a misaligned example.
    if p != len(ids_prefix):
        raise SystemExit(f"MASKING BROKEN for {M}: prompt has {len(ids_prefix)-p} token(s) past "
                         f"the common prefix. Run validate_masking.py.")
    w = torch.zeros(len(ids_full), dtype=torch.float32); w[p:] = 1.0
    dt = datum_from_model_input_weights(tinker.ModelInput.from_ints(ids_full), w,
                                        max_length=MAXLEN, reduction="mean")
    return BLDatum.model_validate(dt.model_dump())


def index_offsets(path):
    offs = []
    with open(path, "rb") as f:
        while True:
            o = f.tell(); line = f.readline()
            if not line: break
            if line.strip(): offs.append(o)
    return offs


def stream_batches(path, offsets, batch):
    buf = []
    with open(path, "rb") as f:
        for o in offsets:
            f.seek(o)
            try: msgs = json.loads(f.readline())["messages"]
            except Exception: continue
            d = build_datum(msgs)
            if d is None: continue
            buf.append(d)
            if len(buf) == batch:
                yield buf; buf = []
    if buf: yield buf


def lr_mult(schedule, step, total):
    if total <= 0: return 1.0
    if schedule == "cosine": return 0.5 * (1 + math.cos(math.pi * min(step, total) / total))
    if schedule == "linear": return max(0.0, 1 - step / total)
    return 1.0


print(f"model={M}  replicas={REPLICAS}  rank={RANK}  batch={BATCH}  lr={LR} {LR_SCHEDULE}", flush=True)
print(f"indexing + shuffling (seed={SHUFFLE_SEED})...", flush=True)
offsets = index_offsets(DATA)
random.Random(SHUFFLE_SEED).shuffle(offsets)
total_steps = (len(offsets) // BATCH) * EPOCHS
print(f"{len(offsets)} examples -> ~{total_steps} steps", flush=True)

wandb.init(project=WANDB_PROJECT, name=RUN_NAME,
           config=dict(model=M, batch=BATCH, maxlen=MAXLEN, lr=LR, epochs=EPOCHS, rank=RANK,
                       shuffle_seed=SHUFFLE_SEED, lr_schedule=LR_SCHEDULE,
                       total_steps=total_steps, replicas=REPLICAS, enable_thinking=False))

sc = BLServiceClient()
print(f"provisioning {SHORT} (replicas={REPLICAS}), ready_timeout={READY_TIMEOUT}...", flush=True)
tc = sc.create_lora_training_client(base_model=M, rank=RANK, replicas=REPLICAS,
                                    ready_timeout=READY_TIMEOUT)
print(f"[OK] trainer ready; session={sc.session_id}; streaming...", flush=True)


def run_step(batch, adam):
    for attempt in range(5):
        try:
            fb = tc.forward_backward(batch, loss_fn="cross_entropy")
            op = tc.optim_step(adam)
            fbr = fb.result(); opr = op.result()
            om = getattr(opr, "metrics", None) or {}
            if hasattr(om, "model_dump"): om = om.model_dump()
            return fbr, om
        except Exception as e:
            print(f"  transient step error ({attempt+1}/5): {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    print("  step FAILED after 5 retries; skipping batch", flush=True)
    return None, {}


def save_ckpt(step, tag):
    for attempt in range(5):
        try:
            tc.save_state(f"{tag}-{step}").result()
            tc.save_weights_for_sampler(f"{tag}-{step}").result()
            print(f"  checkpoint saved (state+sampler): {tag}-{step}", flush=True); return
        except Exception as e:
            print(f"  ckpt retry {attempt+1}/5: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(min(2 ** attempt * 5, 60))
    print(f"  ckpt {tag}-{step} FAILED; CONTINUING", flush=True)


step = 0
for ep in range(EPOCHS):
    for batch in stream_batches(DATA, offsets, BATCH):
        lr = LR * lr_mult(LR_SCHEDULE, step, total_steps)
        adam = tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
        fbr, opt_metrics = run_step(batch, adam)
        if fbr is None:
            step += 1; continue
        nll = None
        try:
            outs = [x["logprobs"] for x in fbr.loss_fn_outputs]
            wts = [d.loss_fn_inputs["weights"] for d in batch]
            n = min(len(outs), len(wts))
            nll = float(compute_mean_nll(outs[:n], wts[:n]))
        except Exception as e:
            print(f"  nll calc skipped: {type(e).__name__}: {str(e)[:100]}", flush=True)
        gnorm = opt_metrics.get("grad_norm", opt_metrics.get("gradient_norm"))
        print(f"step {step} ep {ep} train_mean_nll {nll} lr {lr:.2e} grad_norm {gnorm}", flush=True)
        try: wandb.log({"train_mean_nll": nll, "lr": lr, "train_step": step, "epoch": ep,
                        **{f"opt/{k}": v for k, v in opt_metrics.items() if isinstance(v, (int, float))}})
        except Exception: pass
        if SAVE_EVERY and step > 0 and step % SAVE_EVERY == 0:
            save_ckpt(step, "step")
        step += 1

print(f"DONE {step} steps", flush=True)
save_ckpt(step, "final")
wandb.finish()
