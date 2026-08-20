# Morning report — PP2/CP8/EP8 @131k night (2026-08-10 → 08-11, pauli)

## TL;DR

**All engineering for the PP2/CP8/EP8 @131k bring-up is complete, cross-reviewed,
and pushed. Zero GPU-seconds were obtainable all night** — the ali B300 pool
(and vul/hyd B200) failed ~25 provisioning attempts from 20:39 to 08:00 with
platform-side errors also hitting other users. Supervisors are still hunting
for a box as of 08:05; from the moment one lands we are **~45-60 min from
first TPS/MFU numbers** (weight load ~15-20 min + bounded probes + d2/d4
profile windows).

## What's staged (ready to execute unmodified)

| stream | branch @ head | state |
|---|---|---|
| bring-up (gibbs) | `jackrao/lps-1062-pp2cp8ep8` @ 21d0c578 | pushed; unit tests CI-green Mac-side |
| packing (volta) | `jackrao/lps-1062-pp2-packing` @ b34b7aed | pushed; gibbs pre-reviewed LGTM; merges post-first-light |
| export test (dedekind) | `jackrao/lps-1062-pp2-export` @ ef2e8036 | pushed; runs on any 4-8 GPUs |

Box-side runbook: `tools/bringup_pp2cp8ep8.sh` (phases: env → checkout →
weights check → configs → cleanliness → group-dump pre-flight → launch with
5-10 min bounded probes) → `profile_driver_new.py --datums 2` then `--datums 4`
(524k tok/step anchor point; `mfu.py` LoRA-corrected convention).

## Key engineering results (tonight's actual product)

1. **(78,2) PP layout = [38+embedding, 40+loss]** in `glm52_dsa.py`.
   Constraint derived and source-verified: DSA topk shared in groups of 4 with
   offset `first_k_dense_replace=3` ⇒ stage 2 must start on 1-based layer
   3+4k; 39/39 is invalid, 38/40 is the closest valid split (fallback 42/36).
   New table-wide unit test enforces the rule for every layout.
2. **The CP>1+PP>1 config gate** (`_validate_thd_context_parallelism`) was the
   only hard blocker at tip — relaxed via **DSA-only exemption** (non-DSA still
   raises, per Jack's "untested ⇒ you do the testing" semantics; docstring
   evidence-ref update is a tracked TODO once validation numbers exist).
3. **No p2p shape problem exists at tip**: CP>1 already forces
   `variable_seq_lengths=True` (shapes exchanged on the wire). The real PP
   requirement is **equal microbatch COUNT across stages** — guaranteed by op
   broadcast + DP1 + deterministic packer. (volta, `PACKING_MEMO.md`, file:line cites.)
4. **Pad-to-131k packing implemented** (`pack_thd_cp_microbatch(pad_to_length=)`):
   tail-fills the last doc's padded region, cu_seqlens untouched, loss-masked,
   TPS-accounting-excluded; gated on PP>1 | `BT_PACK_PAD_TO_MAX`; 7 new unit
   tests. Side effect (deliberate): boot warmup becomes full-size 131k under
   PP2 → fails fast if the config doesn't fit. Waste estimate on customer
   histogram: ~8-12% typical. NOTE: pads still pay attention+MoE body FLOPs
   (router doesn't exclude them) — real-data efficiency ceiling, not a bug.
5. **Rank mapping proof** (source-verified): order `tp-cp-ep-dp-pp` ⇒ PP groups
   {r, r+8} are the ONLY cross-node dim; CP and EP groups sit intra-node
   {0-7}/{8-15}. This is the perf thesis: EP a2a (26-45 s resident cross-node
   in prior traces) moves onto NVLink. `tools/dump_parallel_groups.py` asserts
   it at boot.
6. **Export path analysis** (dedekind, `export_test/EXPORT_TEST.md`): the
   megatron→HF adapter conversion (`stream_adapter_weights_megatron_to_hf`) is
   PP-aware (broadcast_from_pp_rank) and **CP-agnostic by construction**; risk
   concentrated in boot gates, both relaxed on the test branch only (deliberate
   divergence; gibbs's gate wins mainline). 3 tests staged incl. DCP
   round-trip faithfulness (no seed knob exists ⇒ cross-run value-equality is
   invalid by construction — Level-2 tests export-vs-state instead).
7. **Landmines catalogued for PP>1** (volta): R3 replay-drift check breaks if
   sampler stamps `moe_layer_indices` (inert: router_replay_mode=NONE);
   DSA indexer-loss AutoScaler assumes PP=1 (inert: coeff=0.0 for LoRA);
   per-partition max_seqlen becomes 131072 after tail-fill (verify on-box).

## Infra post-mortem (why there are no numbers)

- 20:39–08:00: ~25 provisioning attempts across ali B300 (2×8), vul B200,
  hyd B200 — **zero reached the ssh/topology step alive**.
- Two failure modes: (a) 60-min queue starvation, jobs reaped by the platform
  at ~62 min ("lottery tickets" self-FAILED minutes after devbox-up gave up);
  (b) job allocates, goes RUNNING, dies 20-30 s later with SSH proxy rejecting
  and `exit_code=74`/no error_message — poison-node/boot-flap signature.
- **Cross-user evidence**: #training-events-internal shows the same FAILED
  stream on WP `ali-apse7-prod-1` for charles@parsed and harry@parsed from
  19:43 onward. Not our config, not our tooling.
- 06:50–08:00 additionally: local Mac DNS outage (api.baseten.co unresolvable)
  — masked the pools during that window; recovered on its own.
- Papercut filed: `pc_74b534ac38b4` (infra/major).
- Full ledger: `NOTEBOOK.md` § Box ledger. Automation left running: detached
  per-pool supervisors (create → adopt → resume-provision → stop-on-abandon),
  persistent monitor waking pauli on transitions.

## Open questions the first box answers

1. Does boot warmup pass PP2+CP8 (tiny-probe path pre-merge)? Lever if not:
   `BT_SKIP_WARMUP=1`, or merge volta's branch (makes warmup full-size).
2. Memory: stage-0 holds 2 in-flight microbatch checkpoint sets — expect
  near-golden (~200 GiB used/GPU on B300) + margin; d1→d2→d4 ramp will show it.
3. Wall clock: does intra-node EP a2a beat golden 645 tok/s/GPU (and the
   745 CP8/DP2 record) at 524k tok/step? PP bubble at d4 = 4/5 efficiency
   ceiling — the trade we're measuring.
4. Adapter export under PP2×CP8 on the real model (dedekind's Qwen3 result
   covers the mechanism; GLM MoE+EP emitter path still needs the real box).

## Suggested morning sequence (auto-executes if a box lands first)

1. Box lands → gibbs runs runbook → first-light → d2/d4 profiles (~1 h).
2. Merge volta → real-data-shaped parity checks (padded vs unpadded).
3. dedekind's export test on any 4 idle GPUs; then real GLM export after
   profiling.
4. PR shaping: gibbs's branch (layout+gate+evidence docstring), volta's
   packing, dedekind's test (test-branch relaxations rebased as fixtures).
