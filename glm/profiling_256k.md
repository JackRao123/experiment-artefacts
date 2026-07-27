# GLM-5.2 256k SFT memory profile

Profiled 2026-07-17 on devbox `w64072q` (4 nodes, 32 B200 GPUs, 183,359 MiB
physical / 179.06 GiB per GPU). The trainer used
TP1/PP1/EP32/CP32/DP1, LoRA rank 16, full activation recompute, and a single
262,144-token synthetic sequence per forward/backward.

Stack:

- trainers `8256f79482a5cc976c74f0f4db707dad07253f26`
- Megatron-Bridge `f7be4c142e7b3dc04e1b6aa0db195d95ff711830`
- Megatron-LM `d3932e757c6360940a793b8c62717d1238dd93b8`

## Result

**256k SFT already fits on the current stack without a memory workaround.**

| run | hottest `memory.used` | physical headroom | step time |
|---|---:|---:|---:|
| Fresh boot, fwd/bwd only | 140.95 GiB | 38.12 GiB | 33.4 s |
| First two full SFT steps, including Adam | 159.48 GiB | 19.58 GiB | 34.0 s, 28.1 s |
| Third full SFT step | 159.48 GiB | 19.58 GiB | 26.2 s |

The first optimizer step materializes persistent optimizer state. This raises
the real steady-state peak by 18.54 GiB relative to a fwd/bwd-only profile.
The third step did not increase the hottest GPU further, so 159.48 GiB is the
observed steady-state high-water mark.

The old ~230k wall extrapolation does not describe this stack. Its 131k point
was 138.16 GiB hottest-used; the current stack runs 256k fwd/bwd at only
140.95 GiB.

## DSA transient attribution

A torch allocator trace was captured at 196,608 tokens after a warmup. The
500,000-event ring retained the final 17.4 seconds and included 102 DSA CP
gather allocations.

| allocation class | max single allocation | peak live from observed allocations |
|---|---:|---:|
| DSA CP full-sequence all-gather | 0.211 GiB | 0.211 GiB |
| Already-chunked DSA indexer top-k | 1.500 GiB | 3.236 GiB |
| MoE dispatch + expert temporaries | 2.422 GiB | 13.822 GiB |

The CP gather scales linearly to about 0.281 GiB at 256k. Even deleting it
entirely would save less than 0.3 GiB/GPU. Streaming or chunking that gather is
therefore not justified for the 256k target. The current indexer scorer is
already chunked (`_indexer_topk_from_score_chunks`); its workspace is larger
than the gathered latent KV.

## NCCL A/B

Each variant was tested from a fresh boot with one direct 256k fwd/bwd.

| variant | hottest used | delta | step time | verdict |
|---|---:|---:|---:|---|
| Default (NVLS on, up to 32 channels) | 140.95 GiB | — | 33.4 s | Keep |
| `NCCL_NVLS_ENABLE=0` | 139.11 GiB | −1.83 GiB | 32.9 s | Optional small safety margin |
| `NCCL_MAX_NCHANNELS=1` | 139.11 GiB | −1.83 GiB | 62.4 s | Reject: 87% slower |

Disabling NVLS is the only tested low-cost knob worth retaining as an optional
fallback. It is not required to fit 256k. A one-channel pin buys no more memory
and nearly doubles latency.

No MoE expert-capacity setting was used or evaluated.

## Measurement

- Four node-local NVML samplers recorded every GPU at 20 ms resolution.
- Driver timestamps delimit each operation window.
- The torch snapshot was attributed by allocation stack, with live bytes
  reconstructed from alloc/free events.
- Raw devbox artifacts remain under
  `/root/.cache/user_artifacts/glm256k/profile/results/`.
- Reusable samplers and analyzers are in `glm/scripts/`.

## Recommendation

Use the current TP1/PP1/EP32/CP32/DP1 configuration at 256k. Keep default NCCL
settings unless an extra ~1.8 GiB of margin is operationally useful, in which
case test `NCCL_NVLS_ENABLE=0` under a longer real-data run. Do not build the
streaming DSA gather and do not set `NCCL_MAX_NCHANNELS=1`.

Before calling the configuration production-golden, run a longer real-data SFT
with checkpoint/export activity. The measured 19.58 GiB steady-state margin is
healthy, but this profile did not overlap weight export or checkpoint writes.
