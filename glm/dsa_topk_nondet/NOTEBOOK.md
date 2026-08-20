# GLM-5.2 trainer nondeterminism hunt — DSA top-k ranking (LPS-1068 gate failure)

**Canonical running log. Append-only. Read fully before acting.**

Mission (Jack, 2026-08-19): root-cause the nightly SFT gate `trainer_determinism`
failure for GLM-5.2-FP8 (run 32276973889: 3 identical `/forward` calls over six
tiled-pangram CE datums → max per-token logprob spread **1.00138, identical on
all six datums**). Find a minimal repro and narrow the bug location in code,
systematically. Handed off jackrao-session → **kirchhoff** at 2026-08-19 ~15:10 PDT.

## Status: CLOSED 2026-08-19 (session kirchhoff) — ROOT CAUSE CONFIRMED

**The DSA indexer radix top-k kernel (`cudnn.DSA.indexer_top_k_wrapper`,
TRT-LLM CuTe-DSL radix select) is run-to-run nondeterministic on a FIXED
input**: it swaps keys with bitwise-identical scores at the top-k boundary
(tie-break race) and emits nondeterministic output ordering always. Minimal
repro = `box_scripts/topk_micro_repro.py` (1 GPU, no model, seconds). Full
evidence chain and numbers in `results/RESULTS.md`. Fix candidate validated:
masked `torch.topk` fallback is bitwise deterministic at ~1 ms/call at gate
shapes (~10× the radix kernel); the existing dense tie-break bias is inert
for real content (eps 1e-12 < fp32 ulp) and is NOT a fix.

## Findings so far (chronological, each with evidence)

1. **Prior context** (LPS-1063 NOTEBOOK + memory `lps-1063-repro-state`): GLM-5.2
   never uses TE DotProductAttention (attention = mcore DSA/cuDNN path;
   `experimental_attention_variant/`), so the Nemotron tail-pad fix (31327f49)
   is inert for GLM. GLM had a catalogued "padding under THD+CP → nondeterminism
   + value corruption" observation (spread 0.03 @CP8; loss shift ~0.05),
   mechanism unattributed. **This hunt supersedes part of that reading — see (4).**

2. **Module-level repro** (`/root/dsa_pad_repro.py` on the box; local copy in
   Claude scratchpad): single DSAttention layer, DeepSeek-geometry config,
   TP1/CP4 allgather, packed THD multi-doc, torchrun ×4 on B200:
   - Zero-filled pad rows → **top-k indices nondeterministic run-to-run**, but
     confined to pad-row outputs. Poison-filled (1000.0) pad rows → fully
     bitwise deterministic ⇒ **content/tie-driven, NOT uninitialized memory**.
   - `_use_dense_indexer_topk_tie_break` (dsa_cudnn_kernels.py:440) **returns
     False on CUDA** — the deterministic tie-break is deliberately off on the
     training path ("Real indexer scores are not expected to rely on exact
     finite ties").
   - Top-k legality invariant held (0 illegal entries, 0 pad keys selected by
     real queries, 0 pad→real leakage): the padding *layout* machinery
     (dsa_layout/dsa_masking/`_indexer_topk_multi_packed_cp_thd`) is
     self-consistent at this level.
   - Tiled-content (period 37) real rows did NOT flip at this small scale.

3. **Full-stack repro** on the box — 12-layer random-init GLM-5.2 smoke
   snapshot, real trainer server, TP1/PP1/**CP4/EP4**, R3 on (free-routing,
   same warning path as the nightly), LoRA rank 16, **exact nightly pins**
   (trainers f93eb052 / bridge 20fcf2ea / mcore 57efae08):
   - max_seq_len **16384**, datums 0.5–3k tokens: ALL arms bitwise
     deterministic ×4 (padded-tiled, aligned-tiled, padded-random).
   - max_seq_len **65536**, datums 2–12k tokens, ×8 repeats:
     - `padded-tiled-big` [11993,5993,4993,9993,7993,1993]: max spread 0.09
     - `aligned-tiled-big` [12000,6000,5000,10000,8000,2000] (**ZERO padding**):
       fires with **identical per-datum spreads** to the padded arm
       (0.001851/0.0038/0.003754/0.089998/0.004193/0.0) — reproduces the
       nightly's identical-1.00138-across-datums signature (tiled content →
       flips at content-aligned positions with identical deltas).
     - `padded-random-big`: fires hardest, spreads up to **1.93 nats**
       (datums 0/2/3; datums 1/4 zero — content-probabilistic).
     - Datum of **1993 tokens (< index_topk 2048): spread exactly 0.0 in all
       arms, every time.**

4. **Working theory (strong)**: the nondeterminism is in the **DSA indexer
   top-k RANKING kernel** (TRT-LLM CuTe-DSL radix select via
   `_indexer_top_k_wrapper_chunked` → `_cudnn_dsa.indexer_top_k_wrapper`,
   dsa_cudnn_kernels.py:489/467), which only RUNS when a query's candidate
   count exceeds index_topk=2048 — below that the code takes the
   select-everything arange branch (`segment_topk == max_segment_k`,
   dsa_cudnn_kernels.py:819) with no ranking. This cleanly explains:
   - the 16k-clean vs 65k-firing boundary (datum lengths vs 2048),
   - the always-clean 1993-token datum,
   - firing WITHOUT padding (padding is NOT the trigger — kills the old
     padding attribution for the *determinism* symptom; the parity-night
     VALUE-corruption story may be this same flip mechanism, or separate —
     unresolved),
   - the nightly gate failing on all six 10k–60k datums,
   - module-level: zero rows = giant tie plateaus = guaranteed ranking races;
     real content at scale = occasional near-tie races.
   The flip then propagates through downstream layers/MoE routing to O(0.1–2)
   logprob deltas.

5. **Fingerprint localization** (kirchhoff, 2026-08-19 PM): instrumented run
   (probe `--repeats 3 --arms padded-random-big`, failure fired at max spread
   1.93 nats). Aligning 26 fingerprint records/repeat/rank by call order:
   **`dsa_topk` diverges FIRST on all 4 ranks** (ranks 0/2/3 at its very
   first call of the forward; rank 1 at its second); `dsa_sparse_fwd` and
   `moe_router` diverge only at later call indices = downstream propagation.
   Raw jsonl: `fprints_localization_run/`. Analysis: `box_scripts/analyze_fprints.py`.

6. **Kernel micro-repro — CONFIRMED** (`box_scripts/topk_micro_repro.py`):
   `cudnn.DSA.indexer_top_k_wrapper` called 20× on a bitwise-FIXED fp32
   scores tensor (input verified unmutated) returns **different selection
   SETS run-to-run** (19/19 comparisons differ at sk ∈ {4096, 12288} for
   random-bf16, tie-plateau, and zero content; sk=2049 only in the single
   row with seq_len>2048). Every swap exchanges keys with bitwise-identical
   scores ⇒ tie-break race at the k-th boundary. bf16-quantized "random"
   scores tie generically ⇒ real content fires. Rows with seq_len ≤ 2048
   never diverge (arange branch upstream aside, the kernel itself is clean
   below k). Output ORDER nondeterministic in every config.

7. **Fix landscape** (`box_scripts/topk_fallback_ab.py` + bias probe):
   - Masked `torch.topk` fallback (odd-K path body,
     dsa_cudnn_kernels.py:467): bitwise deterministic 20/20, 0.46 ms
     (sk 4096) / 1.12 ms (sk 12288) per 4096-row call vs 0.045/0.107 ms for
     the radix kernel (~10×; ~1 ms absolute). Negligible for gate runs;
     per-op 10× if ever enabled for training steps.
   - Existing dense tie-break bias (`_add_indexer_topk_tie_break`):
     **inert for real content** — `_TOPK_TIE_BREAK_EPS=1e-12`/sk ≈ 8e-17
     per column < fp32 ulp (~2e-7 at |score|≈2); empirically changes 0.00%
     of elements on bf16-derived scores and SET-diff persists 19/19 (fixes
     only zero plateaus). NOT a fix; do not ship option (c)-as-bias.

8. Instrumentation caveat: `dsa_sparse_fwd.out_flat` hashes errored
   (constant `ERR:TypeError`, bf16→numpy); that site's comparison rode on
   `lse`. `dsa_topk` hashes (indices+length) valid throughout.

## Box & environment (box STOPPED at close; recipe kept for re-runs)

- Box: `ssh tj-w5yd8r3` — job **w5yd8r3**, 1 node × 4 B200, project jrao123-hyd.
  Stop: `truss train stop --remote baseten --job-id w5yd8r3`.
- `/root/trainers` @ **f93eb052** (exact nightly commit; bridge 20fcf2ea,
  mcore 57efae08 — submodules initialized).
- Venv: `/root/.cache/user_artifacts/trainers_main/server/.venv` (July tree's
  venv — compiled deps verified IDENTICAL to the new lock: torch 2.11.0+cu128,
  TE 2.16.0, flash_mla 1.0.0+b7643bd, deep_ep 1.2.1, causal_conv1d 1.6.2.post1).
  New-tree code runs via **PYTHONPATH overlay** (see boot recipe).
- Snapshot: `/root/glm52_smoke` — 12-layer / 32-expert random-init GLM-5.2,
  hidden 6144, 64 heads, kv_lora 512, qk nope 192 + rope 64, index_topk 2048.
  Builder: `/root/build_shim_smoke_snapshot.py` (patched: transformers 5.8.x
  `attribute_map {"head_dim": "qk_rope_head_dim"}` ALIAS-CLOBBERS rope 64→192
  during from_pretrained; never assign `config.head_dim`; force
  `qk_rope_head_dim=64` + refresh `qk_head_dim` before `from_config`).
- Instrumented shadow: `/root/megatron_shadow` (built by
  `/root/make_shadow_instrumented.sh` from /root/trainers mcore). Env-gated by
  `BT_FPRINT_DIR`: per-call sha256 fingerprints → `$BT_FPRINT_DIR/rank{R}.jsonl`
  for sites `dsa_topk` (indices+length), `dsa_sparse_fwd` (out_flat+lse),
  `moe_router` (probs+routing_map). Dormant when env unset.
- Probe driver: `/root/glm_gate_probe.py` — arms:
  padded-tiled / aligned-tiled / padded-random (small, 16k-era) and
  padded-tiled-big / aligned-tiled-big / padded-random-big. `--repeats N`.

### Trainer boot recipe (the one that works)

```bash
ssh tj-w5yd8r3
export PYTHONPATH=/root/megatron_shadow:/root/trainers/server/src:/root/trainers/models/src:/root/trainers/baseten-weight-sync:/root/trainers/server/vendor/megatron-bridge/src:/root/trainers/server/vendor/megatron-bridge/3rdparty/Megatron-LM
export BT_TRAINER_SERVER_CONFIG_PATH=/root/.cache/user_artifacts/server_config_local.json
export BT_FPRINT_DIR=/root/fprints   # only for instrumented runs; unset for clean timing
BT_TRAINER_CONFIG_PATH=/root/.cache/user_artifacts/trainer_glm_smoke_cp4.json \
  bash /root/.cache/user_artifacts/.devbox_up/start_trainer.sh
bash /root/.cache/user_artifacts/.devbox_up/wait_trainer_health.sh   # false-alarms if run <60s after start; re-run
# probe:
/root/.cache/user_artifacts/trainers_main/server/.venv/bin/python /root/glm_gate_probe.py \
  --repeats 3 --arms padded-random-big
```

Config `/root/.cache/user_artifacts/trainer_glm_smoke_cp4.json`: base_model
/root/glm52_smoke, max_seq_len 65536, tp1/pp1/ep4/cp4/etp1, trust_remote_code,
attention_backend "auto", lora_rank 16, router_replay_mode "R3",
weight_sync disabled. Server config: `{"trainer_id":"local-trainer",
"callback_url":"","callback_token_path":"","sentry_dsn":""}`.

## NEXT STEPS (pre-registered) — 1✓ 2✓ 3✓ done (kirchhoff); 4 open on Jack's call; 5✓ done

1. **Localize via fingerprints** (trainer was rebooting with BT_FPRINT_DIR at
   handoff): wait health → `rm -f /root/fprints/*` → run probe
   `--repeats 3 --arms padded-random-big` → analyze
   `/root/fprints/rank*.jsonl`: lines per repeat must divide evenly; align by
   call index within repeat; find FIRST (site, call index) whose sha differs
   across repeats, per rank. Decision rule:
   - `dsa_topk` first → indexer radix top-k kernel confirmed → step 2.
   - `moe_router` first with dsa_topk clean → router/grouped-GEMM story; redo
     bisect around MoE (unexpected — re-plan).
   - `dsa_sparse_fwd` first with dsa_topk clean → flash_mla sparse kernel.
2. **Kernel micro-repro** (if dsa_topk): standalone script calling
   `_cudnn_dsa.indexer_top_k_wrapper(scores, seq_lens, top_k=2048, next_n=1,
   return_val=False)` ×20 on a FIXED scores tensor (b·sq rows × sk cols):
   sk ∈ {2049, 4096, 12288}, content ∈ {random bf16→fp32 scores, scores with
   tie plateaus at the 2048 boundary, all-zero rows}. Bitwise-compare indices
   (sort each row first: selection-set comparison; also compare unsorted for
   ordering nondeterminism). Expected: nondeterministic selection at some
   sk/content combo → THE minimal repro (no model, 1 GPU, seconds).
   Note `_indexer_top_k_one_chunk`'s odd-K fallback (dsa_cudnn_kernels.py:467)
   uses plain masked `torch.topk` — that path is the natural deterministic
   fallback/fix candidate; A/B its perf at gate shapes.
3. **Write up**: bug = (probably) TRT-LLM CuTe-DSL radix top-k
   (`cudnn.DSA.indexer_top_k_wrapper`) run-to-run nondeterministic; ship
   options: (a) deterministic torch.topk fallback env-gated for gate runs,
   (b) upstream report with micro-repro, (c) gate redesign note: determinism
   tolerance for GLM must account for this until kernel fixed. ALSO note:
   nightly gate's identical-1.00138 signature = tiled content, now explained;
   padding NOT required (correct the earlier attribution in
   lps-1063-repro-state memory when closing).
4. Optional at-scale confirmation: prod GLM loops trainer (4×8 B200), aligned
   vs gate lens ×3 forwards each — expect BOTH to fail determinism at CP32
   given trigger is length>2048, unless CP32 changes it. Only if Jack wants
   prod evidence.
5. Cleanup when done: stop trainer, `truss train stop --remote baseten
   --job-id w5yd8r3`; snapshot + scripts live on box only — copy anything
   durable here first.

## Session log

- 2026-08-19 ~15:2x–16:1x PDT (kirchhoff): executed pre-registered steps 1–3.
  Health-verified instrumented trainer → cleared fprints → localization probe
  (fired, 1.93 nats) → analysis: dsa_topk first-divergent on all ranks →
  kernel micro-repro on fixed input: SET-level nondeterminism from bitwise
  score ties, 19/19 → torch.topk fallback proven bitwise-deterministic
  (~10×/~1 ms at gate shapes) → dense tie-break bias proven inert for real
  content (ulp math + empirical 0.00%-changed). Copied all scripts, configs,
  and fprint jsonl into this dir (`box_scripts/`, `fprints_localization_run/`,
  `results/RESULTS.md`). Stopped trainer + box w5yd8r3.
  Ship recommendation for Jack: (a) env-gated deterministic torch.topk
  selection for gate/determinism runs (cheap, proven), (b) upstream report
  to cuDNN-frontend/TRT-LLM with topk_micro_repro.py, (c) gate note: GLM
  determinism tolerance unavoidable until kernel fixed; bias path is NOT an option.

- 2026-08-19 ~12:4x–15:1x PDT (outgoing session): everything above. Gotchas
  burned: transformers 5.8.1 alias clobber (2 snapshot rebuilds); old
  trainers_main tree too old for nightly config schema (solved via /root/trainers
  checkout of f93eb052 + PYTHONPATH overlay + local ServerConfig json);
  wait_trainer_health.sh false-alarms right after dispatch; pgrep-self-match in
  ad-hoc waiters. Module repro + probe driver written and validated. Full-stack
  fire at 65k confirmed ×8 repeats, three arms. Instrumented reboot in flight at
  handoff (background task on jackrao session; kirchhoff should just re-check
  health and proceed).
