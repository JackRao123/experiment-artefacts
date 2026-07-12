# 256k logprob-mismatch probe — NVFP4 vLLM vs bf16 vLLM vs bf16 Megatron

Goal: measure per-token logprob mismatch between the NVFP4 vLLM sampler and the
bf16 Megatron trainer on Nemotron-3-Ultra-550B at long context, and decompose it
into (a) the sampler's generate-vs-rescore floor, (b) pure NVFP4-vs-bf16
quantization inside vLLM, and (c) the same-quant vLLM↔Megatron engine gap.
All comparisons are **base model, no LoRA**, sampled-token logprobs only
(k3 is a sampled estimator of KL(behavior‖target), not exact full-vocab KL).

**Verdict: GO.** Across two independent long rollouts (16,384 and 134,024
generated tokens) plus a 255k-prompt positional stress row, the
production-relevant gap (NVFP4 vLLM sampled logprobs vs bf16 Megatron rescore)
is k3 ≈ 0.001–0.020 nats/token with ESS/N ≥ 0.994 and clip ≤ 1.5% —
comfortably inside `cispo` territory. Decomposition: the bf16 vLLM↔Megatron
engine gap is negligible (k3 ≤ 2e-4), NVFP4 quantization is small
(k3 ≈ 5e-4–4e-3), and the largest and *noisiest* term is the sampler's own
generate-vs-rescore floor, which varies ~100× between same-length traces
(outlier-token-dominated — see run 2 below).

---

## Environment

- Box: `q4okndw` (4×8 B200, project `jrao123-hyd`), 183,359 MiB/GPU.
- Sampler: production entrypoint `sampler.vllm_server` (vLLM 0.22.0), one node,
  TP=8, `max_model_len=262144`.
  - NVFP4: `--kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16
    --moe-backend cutlass --disable-custom-all-reduce
    --no-enable-flashinfer-autotune` (the spike recipe; the autotune flag must
    be the `--no-` form — `--enable-flashinfer-autotune false` silently parses
    as true).
  - bf16: defaults + `--disable-custom-all-reduce` (bf16 KV).
- Trainer: 4-node Megatron via `trainers_mn` shared-NFS venv,
  `max_seq_len=262144`, TP=8 / PP=4 / **EP=4 / ETP=2** (see OOM note),
  `lora_rank=16` wrapper present but adapters untrained ⇒ zero contribution,
  base-model logprobs. `weight_sync: disabled`, offline HF env.
- Checkpoints: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-{NVFP4,BF16}` from the
  on-box HF cache.
- Harness: `experiment_artefacts/nvfp4_256k_logprob/experiment_client.py`
  (this dir), staged to `/root/.cache/user_artifacts/nvfp4_256k_logprob/` on
  the box. Raw outputs in `out/` there; `final_report.json` copied here.

## Sample set (one trace per regime)

| regime | prompt | completion | how generated |
|---|---|---|---|
| `8k_smoke` | 7,680 tok synthetic doc | 256 tok, T=0 | single call |
| `short_prompt_long_response` | 34 tok JSONL-mathgen seed | **16,384 tok, T=0.8** | 4×4,096-tok continuation segments, all `finish_reason=length` |
| `long_prompt_short_response` | **255,000 tok** synthetic doc | 512 tok, T=0 | single call |

The long-response trace was verified non-degenerate: no repeated 64-grams
across 255 blocks, coherent incrementing JSONL problems (`id: 1` → `id: 235`),
every segment stopped on `length` (never EOS). The low unique-token count
(351) is the structured-JSONL vocabulary, not looping.

Long-response generation used the token-continuation loop
(`prompt_token_ids = seed + generated_so_far` per 4k segment) rather than
trusting a "write a lot" prompt; per-segment stop reasons are recorded in the
trace metadata.

## The four logprob sets

1. `1_gen_nvfp4.json` — NVFP4 vLLM **generation-time** sampled-token logprobs.
2. `2_rescore_nvfp4_vllm.json` — NVFP4 vLLM teacher-forced rescore
   (`prompt_logprobs`) of the same tokens.
3. `3_rescore_bf16_megatron.json` — bf16 Megatron `/forward` rescore
   (CE logprobs from `loss_fn_outputs`).
4. `4_rescore_bf16_vllm.json` — bf16 vLLM teacher-forced rescore.

Convention: `r = lp_target − lp_behavior`, `w = exp(r)`, `k3 = mean(e^{−r}+r−1)`
over sampled tokens ⇒ estimates KL(behavior‖target). Clip band [0.8, 1.2].

## Results

### 8k smoke (256 tokens)

| comparison | mean \|Δlp\| | k3 | ESS/N | clip% | max \|Δ\| |
|---|---|---|---|---|---|
| 1 vs 2 floor (NVFP4 gen ↔ NVFP4 rescore) | 2.4e-6 | ~1e-10 | 1.000 | 0 | 1.4e-4 |
| 2 vs 4 quant (NVFP4 ↔ bf16, both vLLM) | 2e-5 | ~4e-8 | 1.000 | 0 | 4e-4 |
| 4 vs 3 engine (bf16 vLLM ↔ bf16 Megatron) | 2e-5 | ~1e-7 | 1.000 | 0 | 2e-3 |
| **1 vs 3 production** | **3e-5** | **~1e-7** | 1.000 | 0 | 2e-3 |

### Long prompt, short response (255k ctx, 512 tokens) — positional stress

| comparison | mean \|Δlp\| | k3 | ESS/N | clip% | max \|Δ\| |
|---|---|---|---|---|---|
| 1 vs 2 floor | 3.2e-6 | ~1e-10 | 1.000 | 0 | 1.3e-4 |
| 2 vs 4 quant | 1e-5 | ~2e-8 | 1.000 | 0 | 3e-4 |
| 4 vs 3 engine | 4e-6 | ~1e-9 | 1.000 | 0 | 4e-5 |
| **1 vs 3 production** | **1e-5** | **~2e-8** | 1.000 | 0 | 3e-4 |

**256k position depth alone is a non-issue.** Every pairing at 255k context is
at measurement-noise level — including Megatron vs vLLM. RoPE/Mamba-state
handling at extreme positions agrees across engines and quants. (Caveat: these
512 tokens are greedy T=0 continuations of a highly redundant prompt, so the
distributions are peaked; peaked distributions suppress drift.)

### Short prompt, long response (16,384 sampled tokens, T=0.8) — the headline

| comparison | mean \|Δlp\| | k3 | ESS/N | clip% | max \|Δ\| | Σ log-ratio |
|---|---|---|---|---|---|---|
| 1 vs 2 floor (same model!) | 0.0113 | **0.0702** | 0.994 | 1.28% | 6.79 | −86.6 |
| 2 vs 4 quant | 0.0101 | 0.0037 | 0.976 | 1.45% | 2.86 | −15.5 |
| 4 vs 3 engine | 0.0026 | 0.0002 | 0.9996 | 0.25% | 0.60 | −2.4 |
| **1 vs 3 production** | **0.0130** | **0.0203** | **0.994** | **1.52%** | 4.06 | −104.5 |

### Run 2: 134,024-token rollout (extension run, same seed prompt)

A second independent rollout from the same 34-token seed, generated in 34
continuation segments until the model spontaneously EOS'd after emitting
exactly 1000 JSONL problems (~134k tokens). Verified non-degenerate (no
repeated 64-grams over 2,093 blocks, coherent ids 1→1000). Full four-way
rescore, with prefix-sliced metrics (valid because teacher-forced logprobs
depend only on the prefix):

Full 134,024 tokens:

| comparison | mean \|Δlp\| | k3 | ESS/N | clip% | max \|Δ\| | Σ log-ratio |
|---|---|---|---|---|---|---|
| 1 vs 2 floor | 0.00079 | 0.00018 | 0.9997 | 0.12% | 1.41 | −25.7 |
| 2 vs 4 quant | 0.00118 | 0.00053 | 0.9992 | 0.16% | 2.31 | −46.4 |
| 4 vs 3 engine | 0.00034 | 0.00006 | 0.9999 | 0.04% | 1.95 | −2.8 |
| **1 vs 3 production** | **0.00130** | **0.00121** | **0.9994** | **0.17%** | 4.15 | −74.9 |

Production (1 vs 3) prefix curve. "Prefix N" means the metrics are computed
over only the **first N completion tokens** of this same 134k trace — valid
as a proxy for an N-token run because every logprob is conditioned only on
the tokens before it, so the first N tokens of a long rollout are exactly
what a length-N rollout would have produced. k3 is roughly flat in length,
mean |Δlp| *falls* with length (later tokens in the 1000-problem document are
highly predictable, shrinking per-token drift — a content effect, not just an
engine effect):

| prefix | mean \|Δlp\| | k3 | ESS/N | clip% |
|---|---|---|---|---|
| 16,384 | 0.00437 | 0.00124 | 0.9982 | 0.66% |
| 32,768 | 0.00342 | 0.00148 | 0.9986 | 0.50% |
| 65,536 | 0.00238 | 0.00201 | 0.9989 | 0.33% |
| 131,072 | 0.00130 | 0.00116 | 0.9994 | 0.18% |
| 134,024 (full) | 0.00130 | 0.00121 | 0.9994 | 0.17% |

**The striking cross-run fact:** run 1's 16k rollout measured floor
k3 = 0.070; run 2's first 16,384 tokens (same seed prompt, same engine, same
config, different sampled trajectory) measured floor k3 = 0.0007 — **two
orders of magnitude apart at the same length.** The k3/max-abs metrics are
dominated by a handful of outlier tokens whose presence varies per trajectory;
run 1 happened to contain a few catastrophic-disagreement tokens (max 6.8
nats), run 2 did not (max 1.4). Mean |Δlp| is far more stable across runs
(0.011 vs 0.004). Two consequences:

1. Single-trace k3 at long rollout lengths is a high-variance statistic —
   don't gate a launch decision on one trace, and expect run-to-run spread.
2. The floor does **not** grow monotonically with length as run 1 alone
   suggested; run 2 shows flat-to-falling per-token drift out to 134k. The
   right mental model: a small per-token base drift (~1e-3 mean |Δlp|) plus a
   rare heavy tail of decode-vs-prefill disagreement tokens whose count per
   trace is roughly Poisson — sequence length increases the *chance* of
   catching some, but density stays low.

Sequence-level Σ log-ratio still compounds (−75 nats over 134k tokens for
production): token-level `cispo` is fine; unchunked sequence-level ratios
remain unusable at these lengths.

## Interpretation

1. **The production gap (1 vs 3) is healthy by the spike's own thresholds**
   (clip < 5%, ESS/N > 0.8, and k3 = 0.020 is at the boundary of the 0.02
   heuristic). Importance-weight mass is essentially intact.

2. **The dominant term is the sampler's decode-vs-prefill floor, not
   quantization.** Rescoring the NVFP4 sampler's *own* tokens through its
   *own* engine disagrees with the generation-time logprobs at k3 = 0.070 —
   larger by k3 than the production comparison itself (k3 is asymmetric and
   outlier-dominated; by mean |Δlp| the floor is 0.0113 vs production 0.0130).
   RMS (0.117) ≫ mean (0.011): a handful of tokens disagree by up to ~6.8 nats
   while the bulk are fine. This floor was ~0 at 256–512 completion tokens and
   large at 16k, i.e. it grows with rollout length. Likely mechanism: decode
   path (incremental Mamba state updates, token-by-token) vs chunked-prefill
   path (SSD scan) accumulating different state numerics over thousands of
   steps; the earlier 8k-ctx spike measured this floor at k3 ≈ 0.0009 over 900
   tokens, consistent with growth in completion length.

3. **NVFP4 quantization costs k3 ≈ 0.0037 on long rollouts** (2 vs 4, both
   teacher-forced, isolating weights-precision + fp8-KV vs bf16-KV). That is
   ~2× the earlier 900-token spike number (0.0019) — mild growth, not blowup.

4. **The Megatron↔vLLM engine gap at bf16 is tiny: k3 ≈ 0.0002** (4 vs 3).
   The long-feared "trainer scores differently than the inference engine" term
   is the *smallest* piece of the decomposition, 20× smaller than
   quantization and ~100× smaller than the sampler's own generate/rescore
   floor at this length. Sequence-level it contributes only −2.4 nats over
   16k tokens.

5. **Sequence-level sums compound.** Σ log-ratio for production is −104 nats
   over 16,384 tokens (target consistently assigns slightly less mass to the
   sampler's tokens). Token-level `cispo` clipping handles this; anything
   consuming *sequence-level* ratios at 16k+ rollout lengths would saturate
   and must not be used unchunked.

## What this changes vs the earlier 8k spike

- The spike's "RL-relevant" k3 ≈ 0.0019 at 900 tokens was mostly floor + quant
  in unknown proportion. These runs separate them and add the trainer leg the
  spike deferred: **the trainer leg is nearly free; the floor is the largest
  and noisiest term.**
- When monitoring a long-context RL run, elevated sampler-vs-trainer logprob
  divergence can come from the vLLM decode path alone (rare heavy-tail
  disagreement tokens) and is not necessarily evidence of weight-sync or
  quantization problems. Compare against a same-model rescore floor before
  blaming anything else.

## Caveats

- **n=1 trace per regime per run** (two independent runs for the long-response
  regime), single seed schedule. The two long-rollout runs bound the
  run-to-run spread of k3 at ~100× — treat any single-trace k3 as an order-of-
  magnitude estimate, not a point value. Mean |Δlp| is the stabler statistic.
- Long-prompt row is greedy + repetitive-prompt ⇒ peaked distributions; the
  positional all-clear is strong for "nothing catastrophic at 256k" but weak
  as a distributional statement.
- Base model only (no LoRA); the FP8-Mamba-LoRA interaction from the spike is
  a separate question. The Megatron boot used an untrained r=16 LoRA wrapper
  (identity at init, B=0) purely to match the validated 262k boot config.
- The 134k run's model chose its own stopping point (1000 problems + EOS); a
  forced-continuation run past EOS (`ignore_eos`) was started but abandoned as
  unnecessary — 134k natural tokens is more rollout-representative anyway.
- k3 estimates KL(behavior‖target) from sampled tokens; exact full-vocab KL
  was out of scope.

## Ops notes (for reruns)

- Boot timings on this box: NVFP4 sampler ~13 min cold (weights 2.7 min,
  torch.compile 3.5 min, warmup 2.4 min); bf16 sampler ~37 min (weight load
  26 min — 130.96 GiB/GPU); Megatron 4-node ~4.5 min warm. Megatron rescore of
  all three traces: ~4 min total (262k-padded forwards at PP=4).
- KV capacity at 262k (vLLM boot logs): NVFP4+fp8KV **38,311,634 tok
  (146.1×)** vs bf16+bf16KV **4,372,401 tok (16.7×)** — 8.8× pool ratio
  (~2× of it is the KV dtype).
- **OOM trap:** TP8/PP4/EP8/ETP1 full-finetune at 262k OOMs during Megatron
  init on this box (12.9 GiB alloc failure at 166 GiB used, rank 7). The
  validated `jack-nemo3ultra-256k-etp` config — **EP=4 / ETP=2 + lora_rank
  set** — boots at ~35 GiB/GPU init footprint. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` also added.
- **Server patch required for PP>1 `/forward` logprobs:** rank 0 is PP stage 0
  and returns `loss_fn_outputs=[]`; the logprob tensor lives on the last
  stage only. Patched `_run_forward_backward` in the shared-NFS
  `trainers_mn` venv to `broadcast_object_list` the wire-format outputs from
  `pipeline_model_parallel_last_rank` (marker: `nvfp4-probe: broadcast
  logprobs across PP`). Upstreaming this (or a rank-0-gather equivalent) is a
  follow-up — without it every PP>1 `/forward` silently returns no logprobs.
- **Alignment:** trainer wire format is `wire[k] = logπ(tok[k+1] | tok[0..k])`
  (0.0 sentinel at the last slot) ⇒ completion region is
  `wire[plen−1 : plen−1+clen]`. vLLM `prompt_logprobs[i]` scores `tok[i]`
  directly ⇒ region `[plen : plen+clen]`. Off-by-one here silently produces
  garbage comparisons.
- `return_tokens_as_token_ids` changes the token *output* format to
  `token_id:<int>` strings; parse accordingly.
- The sampler serves the model under its **snapshot path**, not `default` —
  fetch the id from `/v1/models` before building requests.

## Artifacts

- `out/final_report.json` — run 1 (16k rollout + smoke + 255k-prompt), all
  four pairings per regime.
- `out/final_report_134k.json` — run 2 (134k rollout), four pairings with
  prefix metrics at 16k/32k/64k/128k.
- `traces/` — human-readable prompt/completion text + metadata for every
  generated trace (`*_134k.*` = run 2).
- Raw token/logprob JSONs live on the box under
  `/root/.cache/user_artifacts/nvfp4_256k_logprob/out/` (`*_256k.json` names
  are run 2 at its original 260k target; actual length 134,024).
- Harness: `experiment_client.py` in this directory (deployed copy at
  `.../nvfp4_256k_logprob/scripts/` on the box).

## Follow-ups

- [ ] Rerun the long-response regime with 3–5 seeds per length to put error
      bars on the heavy-tail outlier rate (the dominant uncertainty).
- [ ] Same probe with the GSM8K step-199 LoRA on both sides (the spike's
      FP8-Mamba-LoRA risk) once base numbers are accepted.
- [ ] Isolate the decode-vs-prefill floor mechanism: rescore in chunked
      prefill of varying chunk sizes; if the floor tracks chunk boundaries
      it's Mamba SSD-vs-step numerics.
- [ ] Upstream the PP logprob broadcast patch properly (gated, tested).
- [ ] bf16 + fp8-KV sampler run to close the spike's decomposition TODO.
