# Repro: GLM-5.2-FP8 LoRA SFT (single replica)

Reproduces the `GLM-5.2-FP8_r1_b32_rank32` run — LoRA rank-32 SFT of `zai-org/GLM-5.2-FP8` on
48,898 Opus-adjudicated evidence-grading examples, via Baseten Loops.

## What's here

| file | purpose |
|---|---|
| `loops_sft.py` | the training driver — the *same* script all four runs used |
| `validate_masking.py` | pre-flight: proves the loss mask is correct for a given model |
| `pyproject.toml` | dependencies, **including the uv override that must not be dropped** |
| `uv.lock` | exact resolved versions from the run |

## What you must supply

1. **Data** — `conversations.train.jsonl`, 48,898 lines, 6.3 GB. One JSON object per line with a
   `messages` list of `[system, user, assistant]`. Not in this bundle (size); copy it to the
   machine and point `DATA=` at it.
2. **`BASETEN_API_KEY`** — the run provisions a Loops trainer over the API.
3. **`WANDB_API_KEY`** — optional; drop the `wandb` calls if you don't want tracking.

## Setup

```bash
uv sync                    # pyproject.toml + uv.lock reproduce the exact environment
```

Verify the environment is sane before spending GPU:

```bash
# MUST print nothing — if a `tinker-0.x.x` directory exists, the overlay was clobbered
ls -d .venv/lib/python3.12/site-packages/tinker-0* 2>/dev/null

# MUST be empty — see the safety note below
echo "${TINKER_API_KEY:-<unset>}"
```

## Run

```bash
env -u TINKER_API_KEY \
  MODEL=zai-org/GLM-5.2-FP8 \
  REPLICAS=1 \
  DATA=/path/to/conversations.train.jsonl \
  RUN_NAME=GLM-5.2-FP8_r1_b32_rank32 \
  WANDB_PROJECT=oe-grader-sft \
  python loops_sft.py
```

Everything else is defaulted to the run's settings: batch 32 (global), lr 5e-4 cosine → 0,
1 epoch, LoRA rank 32, shuffle seed 0, max seq len 131,072, checkpoint every 100 steps
(state **and** sampler weights), `grad_norm` logged per step.

Expect ~1,528 steps. On B300×2 it ran at ~0.74 steps/min (~34 h); the original shipped run
was B200×4.

## Three things that will bite you

**1. Upstream `tinker` silently redirects training to another service.**
`tinker-cookbook` depends on the real upstream `tinker`, which installs *over* Baseten's compat
overlay and wins. `import tinker` then points at `tinker.thinkingmachines.dev`, and if
`TINKER_API_KEY` is set it authenticates and trains there instead — no error. The
`override-dependencies` block in `pyproject.toml` prevents the install; running with
`env -u TINKER_API_KEY` makes any residual misroute fail loudly instead of silently.
Tell-tale signs of a misroute: a 65,536 max_seq_len cap, or `base_model ... is not supported`
for GLM-5.2.

**2. The chat template must be rendered with `enable_thinking=False`.**
`loops_sft.py` does this (`TPL`). Under the default template the generation prompt ends at
`<|assistant|><think>`, so the first *trained* token is `</think>` — the model is taught to emit
a tag the template could have supplied. It also prepends a `<|system|>Reasoning Effort: Max`
line, so the two modes differ at both ends of the prompt.
**Whatever you train with, serve with.** The two are measurably equivalent in accuracy
(within-1 0.8347 off vs 0.8372 on, paired, n=500) but the prompts are not interchangeable.

**3. Validate the mask if you change the base model.**

```bash
python validate_masking.py zai-org/GLM-5.2-FP8 --thinking-off
```

The masking works by rendering twice — full, and prompt-only with `add_generation_prompt=True` —
and training everything past the common prefix. That is correct **only if** the prompt is an
exact token prefix of the full rendering, which is model-specific. Under their *default*
templates neither Nemotron-3-Super nor Qwen3.5-122B satisfies it: the prompt runs one token past
the common prefix, so the boundary lands early and the first "trained" token is a control tag or
bare whitespace. Training proceeds and the loss looks plausible while every example is
misaligned. `loops_sft.py` asserts the invariant at datum-build time and refuses to start if it
fails, but run the validator first — it exits non-zero when a model is unsafe.

## Known behaviour of this run

GLM-5.2 is **FP8** and throws isolated loss spikes — ~2–4× the local median, recovering in one
step, with `grad_norm` spiking 3–45× alongside. They are not a bug in this script: three BF16
models trained on identical data with this same script produced **0 spikes in 745 steps**, versus
26 in 147 FP8 steps (Fisher exact p = 6×10⁻²²). The rate is also hardware-dependent — ~17.7% of
steps on B300 versus ~4.8% on B200 over the same window. Adam absorbs them and the final model is
unaffected. See `results/SPIKE_INVESTIGATION.md` in the main repo.

Also expect occasional `UnknownRequestError: server has no record of operation_id ... TTL-evicted`
during training. `run_step` retries and resubmits; 7 occurred in 213 steps and none lost a step.
