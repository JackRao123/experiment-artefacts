# DeepEP/flex CONFIG AUDIT — was L5 tuned, and is there a tuned-flex probe worth a box slot?

Author: serre, 2026-08-13 ~04:0x CDT, for lebesgue (cauchy-ordered audit;
PROBE-2 GO/NO-GO is lebesgue's call). Code read on the box's exact pins
(mcore 57efae08b; vendored deep_ep fork 1.2.1+20bf98d read on-box). No GPUs
touched.

## Verdict in one paragraph

**L5 was NOT a tuned comparison** — it ran the flex flip with every knob at
default. But the audit's honest answer is closer to "no material knobs" than
"tune them": exactly ONE performance knob is reachable at all
(`moe_flex_dispatcher_num_sms`, and not from trainer JSON), and it attacks
DeepEP *kernel time* — which L5 measured at **parity** with the NCCL a2a it
replaces (8.40s vs 7-8s/step). The -12.6% d4 regression does not live in the
kernels; it lives in flex's host-side/sync structure, which exposes no knob.
A tuned-flex d4 re-probe is therefore **not justified on the L5 evidence**
(details + the one conditioned probe that IS meaningful below).

## 1. The knob inventory (complete)

**Trainer JSON surface** (models/src/loops_models/control.py:337-342):
`moe_token_dispatcher: alltoall|flex`. That is the ONLY trainer-reachable
flex switch. No num_sms, no buffer sizes, no mode, no chunking.

**Bridge mapping** (megatron_config.py:43-50 → flex_dispatcher_backend.py
:70-72): sets `moe_token_dispatcher_type="flex"`,
`moe_flex_dispatcher_backend="deepep"`, `moe_shared_expert_overlap=False`. No
tuning surface. (The B300 capability name-check that killed L5 boot 1 lives
at :46-56/:77-86 — patched on-box, patch preserved.)

**mcore config fields** (TransformerConfig, consumed in
moe/token_dispatcher.py):
- `moe_flex_dispatcher_num_sms` — the only perf knob. None → **20**
  (token_dispatcher.py:1252-1257, "DeepEP's historical mcore default") →
  `set_deepep_num_sms` (fused_a2a.py:260-262) → `Buffer.set_num_sms`
  (deep_ep buffer.py:152-161; even values only). 20 SMs of the B300's 148.
  Reachable today only via provider setattr (the L1-hunk pattern), not
  trainer JSON.
- `moe_hybridep_num_blocks_permute/num_blocks_unpermute/
  num_sms_preprocessing`, `moe_permute_fusion_into_hybridep` — HybridEP-only;
  not our backend.

**mcore buffer management** (fused_a2a.py:35-68): one global buffer,
auto-sized from DeepEP's own config hints
(`Buffer.get_dispatch_config(group.size()).get_nvl_buffer_size_hint(hidden_bytes, …)`;
hidden_bytes = 6144×2 = 12,288 B/token), grow-only recreation. Auto, not a
knob; cannot under-size (it re-allocs larger). The +11 GiB L5 buffer
footprint is this.

**DeepEP Buffer** (vendored fork, buffer.py):
- `num_sms` — as above (the only knob that reaches us).
- `num_nvl_bytes`/`num_rdma_bytes` — constructor args; mcore auto-sizes them.
- `low_latency_mode` + `low_latency_dispatch/combine` (buffer.py:640/:724) —
  a decode-time RDMA API; mcore's training path (fused_dispatch/fused_combine)
  uses normal mode. **Not applicable to training.**
- `num_qps_per_rank` (24), `allow_nvlink_for_normal_mode` (already True) —
  RDMA-side; irrelevant intra-node.
- Config presets per EP-rank count (buffer.py:228-275): at EP8, dispatch
  `Config(20, 6, 256, 6, 128)`, combine `Config(20, 4, 256, 6, 128)` — the
  chunking constants are hardcoded per rank count with a literal
  `# TODO: automatically tune` in the source. Not exposed through mcore.

**Env**: nothing perf-material for the deepep intranode path in the vendored
fork (only `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` for hybridep, and
CUDA_VISIBLE_DEVICES).

## 2. What L5 ran vs what 131k tuning would suggest

L5 = config-only flip (`moe_token_dispatcher=flex`) → num_sms=20, auto
buffers, normal mode, DeepEP's hardcoded EP8 chunking. Untuned by
construction.

What 131k-packed-payload tuning would suggest, if anything: the dispatch/
combine payloads are ~1.6 GB/rank/layer/direction (16,384 tokens × topk 8 ×
12 KB), NVLink-bandwidth-bound — the one lever that exists for
bandwidth-bound kernels is more SMs (num_sms 20 → 32/40), trading compute
SMs for comm SMs. But L5's own trace decomposition says the kernels are
ALREADY at time-parity with NCCL (8.40s vs 7-8s/step) and the regression is
elsewhere — so even a successful num_sms tuning does not attack the measured
-12.6%. The dominant exposure term (peer-imbalance wait, ~70% per
A2A_EXPOSURE_DECOMPOSITION) is dispatcher-agnostic by construction — no flex
knob touches it.

## 3. Verdict and the probe question

**Plain verdict: on the L5 evidence, there is no material exposed knob that
attacks the measured regression term; "L5 was untuned" is true and also
immaterial to its conclusion.** A tuned-flex d4 re-probe (vs the fixed-wheel
~880 anchor) is NOT justified as a flex lever.

The one conditioned probe that IS meaningful (and is NOT a flex-tuning
probe): **flex-as-transport under the overlap executor**, only after the
executor-contract shim exists (EXECUTOR_CONTRACT_SCOPING.md). Rationale: the
flex dispatcher's async machinery (`async_finish=True`,
`allocate_on_comm_stream=True` — the MoEFlexTokenDispatcher DEFAULTS,
token_dispatcher.py:1785-1786/:1847-1848) engages only under the combined
executor's comm-stream scheduling; the plain 1F1B path L5 ran never uses it
(the manager-level dispatch default is async_finish=False, :1274). Whether
flex's host-side behavior (GPU-side layout, no CPU wait) reduces the
CPU-blocked residual the L3 trace flagged is a real, unmeasured question —
but it belongs to the overlap program's A/B list, not to a standalone flex
re-probe. Recommend: no box slot now; revisit as an overlap-transport
variant once the contract shim lands.

## 4. The overlap connection (addendum summary — full text appended to
EXECUTOR_CONTRACT_SCOPING.md §7)

**Does the overlap executor use the flex async hooks? Is DeepEP the designed
enabler of the 25-28s/step prize?** — **No.** The executor supports the
stock alltoall dispatcher as a first-class citizen
(transformer_config.py:2627-2629 names 'alltoall' AND 'flex'); the hiding
mechanism is schedule-level (dispatch/combine nodes on the comm stream,
paired against the other microbatch's compute) and is dispatcher-agnostic;
the 25-28s hideable ceiling was measured on the alltoall/NCCL stack
(gauss's decomposition of the NCCL alltoallv exposure). Flex's async hooks
DO auto-engage under the executor (the token_dispatch/token_combine
defaults), but they sharpen host behavior around the comm — they do not
change the hideable mass. Historical note: the flag was built around
flex-style async dispatch, but the shipped executor supports alltoall
explicitly, and our stack runs alltoall. Consequence for the program: the
overlap EV path is **contract shim (0.5-2d) + memory path (offload or the
dial)** — no DeepEP dependency anywhere. L5 stays "answered-negative as a
standalone lever," full stop — not "not the lever alone."
