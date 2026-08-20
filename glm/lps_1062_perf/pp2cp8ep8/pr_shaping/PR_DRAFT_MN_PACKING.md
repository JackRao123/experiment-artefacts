# PR DRAFT — M=N schedule fix + pad-to-131k packing (LPS-1062)

**Status: DRAFT for Jack's morning review. Not opened as a PR. Parity gate
PENDING (legs run tonight; see §6).**

Suggested title:
`perf(server): one M=N schedule call per step on the THD CP path + uniform partition padding under PP>1 (LPS-1062)`

Suggested branch: `jackrao/lps-1062-pp2-packing` → `main`
(4 commits, 7 files, +428/−168 vs `origin/main` @ merge-base `df831501`.)

---

## 1. TL;DR

On pipeline-parallel (PP>1) THD context-parallel (CP>1) training runs, the
trainer called Megatron's pipeline schedule once per data partition with
`num_microbatches=1`. That convention silently disabled pipelining: the 1F1B
schedule (1-forward-1-backward, Megatron's standard pipeline schedule) cannot
overlap anything when it only ever sees one microbatch, so the two stages
ran as a pure dependency chain — stage 1 idle while stage 0 computed, and
vice versa, every step. We measured ~27s of pure wait per step on the
stage-1 rank (trace analysis, NOTEBOOK ~19:4x rank-8 decomposition entry).

This PR hands the schedule all of the step's microbatches in **one call**
(`num_microbatches = len(partitions)`, "M=N"), so 1F1B actually pipelines:
stage-1 backward(k) overlaps stage-0 forward(k+1). A second commit pads every
THD partition to a uniform length (`max_seq_len`) under PP>1 so each
microbatch in that call presents the same activation shape.

Measured on the GLM-5.2-FP8 LoRA PP2/CP8/EP8 @131k bring-up box (2×8 B300):

| build | tok/s/GPU @ d4 | vs golden anchor (645) | vs best-ever (691) |
|---|---|---|---|
| pre-fix tip + TF32 head | 589 | 0.91× | 0.85× |
| **this change** | **918** | **1.42×** | **1.33×** |

Scaling with microbatch count on the fixed build: d4 918 → d8 1000 →
**d16 1052 tok/s/GPU** (2M tokens/step, peak 194/275 GiB).

## 2. Problem

Terms, defined once:

- **THD packing**: variable-length documents packed into one flat row per
  partition ("total-here-documents" format); the CP (context-parallel) path
  shards each row zigzag across 8 CP ranks.
- **Partition = microbatch**: the runner greedily bins a step's datums into
  partitions of ≤ `max_seq_len` (131,072 tokens) and packs each into one THD
  row. A 4-partition step ("d4") = 4 microbatches per optimizer step.
- **The convoy**: with PP2 (two pipeline stages), calling the schedule once
  per partition with `num_microbatches=1` means each call is a degenerate
  1F1B of a single microbatch — forward, then backward, with a hard p2p
  dependency between stages and nothing to overlap. Stage-0 and stage-1
  parks (idle waits) measured at 36.2s and 27.1s per step respectively
  before this change; p2p wire time itself was 31ms/step — the wall was the
  schedule convention, not the hardware.

Root-cause elimination that supports this attribution (measured, do not
re-litigate): NVLink transport at 600 GB/s microbench; NCCL env +0.4%;
telemetry allreduce absorbed-idle (A/B 578 vs 589); LM-head syncs ~0.4s/step;
layer-count imbalance 18.8 vs 18.25s compute; p2p wire at line rate
(200MB @ 48GB/s). Full list with numbers in NOTEBOOK.md ~19:0x and
HANDOFF.md.

## 3. Fix mechanics

All in `MegatronTrainingRunner._run_forward_backward`
(`server/src/trainers_server/dp_worker/backends/megatron_bridge/training_runner.py`,
THD branch).

**Before**: loop over partitions; per partition, one
`stack.forward_backward(num_microbatches=1, ...)` call; the grad finalize
hook (`finalize_model_grads_func`) was unwired for the loop's duration and
fired once by hand after the loop.

**After**: one `stack.forward_backward(num_microbatches=len(partitions),
data_iterator=iter(microbatches), ...)` call per optimizer step. Three
load-bearing subtleties, each deliberately identical-in-effect to the old
convention (this is the review contract):

1. **Gradient scale**: Megatron's legacy 2-tuple loss path scales each
   microbatch loss by `cp_size / num_microbatches` (mcore
   `schedules.py:336-339`). At `num_microbatches=N` that would silently turn
   the partition SUM into a MEAN (N× under-scaled grads). The THD branch now
   wraps the forward step with the BSHD path's existing
   `_sum_over_microbatches(forward_step, N)` (`loss.py:1053-1075`), which
   pre-multiplies the loss by N before the schedule divides — net per-
   microbatch scale identical to the old M=1 calls. Accumulated grads equal
   the old sum up to fp round-trip. **Do not drop this wrapper.**
2. **Finalize ownership**: the schedule's own tail calls
   `finalize_model_grads_func` once per call (mcore `schedules.py:2466-2481`
   non-interleaved, `:832-838` no-pipelining) — the same function with the
   same arguments as the old explicit post-loop call. The suppress-then-
   finalize machinery (`_suppress_schedule_finalize`) is deleted. This
   equivalence holds **only because there is exactly one schedule call per
   op**; a WARNING comment in the code says so. Any future change that
   splits the op into multiple schedule calls must re-gate finalize to once
   per op, or the whole accumulated grad buffer gets re-reduced per call
   (double finalize).
3. **Empty DP slice**: a DP rank with zero partitions (legal at DP>1) skips
   the schedule call but still calls `finalize_model_grads_func(model_list,
   None)` explicitly — the DP grad all-reduce inside finalize is the
   rendezvous its peers are waiting on. This preserves the old post-loop
   behavior, now explicit, and is covered by
   `test_thd_zero_data_replica_still_runs_dp_collectives`.

Per-partition accounting is unchanged in content: the schedule's
`forward_data_store` on the last stage is per-microbatch in forward order,
so result `i` maps to partition `i` by position (now length-guarded with a
`ValueError`); the datum-offset walk over each microbatch's
`packed_seq_params.cu_seqlens_q` still runs on every rank. CP logprob
stitching, `_dp_reduce_sum`, and the PP loss broadcast are untouched.

`seq_length=max(per-partition lengths)` is cosmetic: it is consumed only by
`get_tensor_shapes`, which is ignored under `variable_seq_lengths=True`
(set for all CP>1 runs; shapes are exchanged on the wire per p2p op —
PACKING_MEMO Q2).

At PP=1 the no-pipelining schedule interleaves fwd/bwd per microbatch —
same math order, same single in-flight activation set, same finalize timing
as the old per-partition calls. The CP32 golden path is not re-profiled by
this change.

**Memory**: in-flight activation sets per stage go from 1 (serialized) to at
most min(M, PP) = 2 at PP2 (~8–9 GiB extra on stage 0 at 131k/CP8 full
recompute). Measured on-box: d1 170 GiB, d4 175, d8 182, d16 194 of 275 —
flat in M, as predicted (MN_SCHEDULE_MEMO §d).

### Pad-to-131k packing (second commit, same branch)

`pack_thd_cp_microbatch(..., pad_to_length=)` extends the **last document's
padded tail** so `cu_seqlens_padded[-1] == pad_to_length`, gated in the
runner by `_thd_partition_pad_to_length()`: active when PP>1, or when
`BT_PACK_PAD_TO_MAX` is set (the parity-test / overhead-measurement vehicle
at PP=1). Every microbatch in the M=N call then presents the same activation
shape on every stage — uniformity and insurance against static-shape
assumptions in less-traveled paths (per PACKING_MEMO Q2, unequal lengths do
not break p2p at tip; the hard PP requirement is call-count equality, which
the shared datum stream already guarantees).

Mechanics and safety:

- The fill uses the standard pad sentinels (token 0, label −100, position 0,
  weights/advantages/logprobs 0.0, temperatures 1.0, `padding_mask=True`,
  spread `routing_pad_ids` under router replay) — identical to the
  per-document 16-alignment tails the validated CP32 golden config already
  runs.
- `cu_seqlens` (unpadded) is never touched: datum accounting
  (`partition_datums = cu_seqlens.numel() − 1`) and logprob stitching read
  real boundaries only. A synthetic pad *document* would break both — that
  option was considered and rejected (PACKING_MEMO Q6).
- Pads are fully loss-masked and excluded from loss normalization and TPS
  accounting, but they **do** pay attention + MoE body compute (the router
  does not exclude them from top-k dispatch). Waste estimate on the customer
  histogram: ~8–12% typical, low-teens worst case (PACKING_MEMO Part 3);
  the parity legs' forward-backward time ratio is the first direct
  measurement of this term (§6).
- Validation: `pad_to_length` must be a positive multiple of the CP
  `pad_multiple`, ≤ `max_length`, and ≥ the partition's own padded total;
  exact-fit is a no-op. 7 new unit tests
  (`test_cp_thd_slicing.py::test_pad_to_length_*`).

### L1 plumbing commit (separable; Jack's call on inclusion)

`b8d868ff` — `comm_overlap.overlap_dispatch_backward_with_experts_wgrad`
trainer config field (default **false**), the E2-revised-spec L1 lever: mcore
defers each MoE layer's expert wgrad GEMMs to a side stream that waits only
on the expert dgrad event, overlapping the dispatch-backward all-to-all. No
recompute constraint (unlike `overlap_moe_expert_parallel_comm`), composes
with full recompute and with M=N. Plumbing shape: popped from the
`model_dump` splat (no default — a rename fails loudly at boot) and written
onto the provider directly, because the bridge `CommOverlapConfig` dataclass
lacks the field; trainer-side validator mirrors mcore's mutual exclusions
(mcore's own asserts run at provider construction, before the write, so they
never see this flag). Reviewed fresh-eyes by doppler (NOTEBOOK 21:2x:
attribute name, single splat site, provider identity, no bridge-side
overwrite, construction-order claim — all verified from source). 14/14 unit
tests green. *Docstring rationale corrected in `6d8b22da` (poincare,
verified against bridge `transformer_config.py:108,130`): the bridge defers
mcore's `TransformerConfig.__post_init__` validators to
`provider.finalize()`, which re-runs them on current fields — and this
write lands AFTER finalize (at the comm_overlap splat site), so mcore's
asserts genuinely never see the flag. The conclusion is unchanged, the
mechanism is the correction: the trainer-side validator is the ONLY
enforcement point for this flag.*

**Branch mechanics**: `b8d868ff` currently sits on
`jackrao/lps-1062-pp2cp8ep8` atop the validation merge `73c24b00`; the
pushed branch tip is now `6d8b22da` (`b8d868ff` → `f2407a10`
`fine_grained_activation_offloading` plumbing → `6d8b22da` review
follow-ups — both belong to the P2 offload lane, NOT this PR). Verified
tonight: it cherry-picks cleanly onto `jackrao/lps-1062-pp2-packing`
(`git apply --check` passes). Include it here as a fifth commit or land it
as a stacked follow-up — either is mechanically sound; it is default-off, so
neither ordering changes behavior. It has **no on-box A/B result yet** (L1
boot is in tonight's box queue); if it ships in this PR it ships as
plumbing-only with the A/B to follow.

## 4. What this PR is not

The on-box validation build was `73c24b00` = this branch merged into the
PP2 bring-up branch, plus the TF32 LM-head patch. The following validated
components are **not in this diff** and come as sibling PRs (see
REPORT_MORNING_0813.md §7):

- (78,2)=38/40 PP2 layer layout + DSA-only CP>1+PP>1 gate exemption,
  `BT_PROFILE_RANKS`, VPP config field + interleaved iterator fix, telemetry
  gate — gibbs/doppler branch, infra PR (weierstrass shaping).
- TF32 LM-head patch — duplicates open PR #995; referenced, not re-cut.
- Export tests + `BT_SAVE_STATE_SYNC` toggle (`db5d1826`) — export PR.

Landing-order note: the M=N call site and the infra PR's VPP iterator fix
(`8c13ed31`) touch the same lines. The reconciliation is marked in-code by
`8b0ef108` (when VPP stacks, `iter(microbatches)` becomes
`_schedule_data_iterator(microbatches, len(model_list))`); whichever PR
lands second resolves there. Coordinated with the infra PR author.

## 5. Evidence

All numbers from the GLM-5.2-FP8 LoRA PP2/CP8/EP8/TP1/DP1 @131k bring-up on
box w56lorq (2×8 B300), 2026-08-12; full chronology in NOTEBOOK.md.
Throughput is tokens/s/GPU on real (non-pad) tokens. mfu3x is the
LoRA-corrected MFU, comparable to post-08-09 tables only.

**Throughput ladder** (dN = N microbatches per step):

| build | d2 | d4 | d8 | d16 |
|---|---|---|---|---|
| clean tip | 538 | 550 | — | — |
| + ship NCCL env | — | 552 | — | — |
| + TF32 head (pre-fix apples-to-apples) | — | 589 | — | — |
| **this change** | **764 (+42% vs 538†)** | **918 (+56% vs 589)** | **1000** | **1052** |

d4 detail: 925/912 across two reads, step 35.7s, mfu3x 8.4% / hfu 12.1%,
peak 175 GiB. d16: 124.6s/step @ 2M tokens, mfu3x 9.6% / hfu 13.9%, peak
194/275 GiB. Anchors: golden EP16/CP16 = 645, best-ever any config = 691 —
918 is 1.42× / 1.33× respectively; 1052 is 1.63× / 1.52×.

**Mechanism confirmation** (traced d4, ranks 0+8): p2p parks collapsed —
stage-0 36.23→8.78s, stage-1 27.10→2.03s per step; residue is the
irreducible PP2 fill/drain. Landed at 1.67× of the 1.9× park-removal
ceiling; the residual is the now-exposed CPU-blocked serialization (~21s/step
on stage 1, separate follow-up lever) plus fill/drain plus MoE all-to-all.

**Attribution honesty**: the 589→918 comparison was measured across the
merge that activated both M=N and pad-to-131k (padding is PP>1-gated, so it
turned on with the merge). Padding only adds loss-masked compute, so the
schedule fix's isolated contribution is at least the measured delta; the
parity legs' fb-time ratio prices the padding term directly (§6). The d4
delta is the same-environment comparison (both sides TF32 head + ship NCCL
env). †The d2 delta crosses build configs: the pre-fix 538 was the
clean-tip first light (the TF32 run drove d4 only), so 538→764 bundles the
M=N gain with the TF32 (+6.7% at d4) and ship-env (+0.4%) terms — read
d4's +56% as the clean same-env number.

**Canaries** (correctness smoke at each rung): d1 degenerate PASS (582
tok/s/GPU — M=1 is the old path by construction); d2 loss 12.305–12.307, in
the 12.2–12.4 canary band, grad norm 0.40–0.46 comparable to pre-fix;
canaries in band through d16. Memory flat in M (in-flight sets =
min(M,PP)=2, as predicted).

**Unit tests**:

- On-box (w56lorq) at `b34b7aed`: **188 passed / 0 failed** across
  `test_cp_thd_slicing` + `test_cp_thd_dispatch` + `test_ce_loss` +
  `test_router_replay_dp_consensus`, incl. 7/7 `test_pad_to_length_*`
  (dedekind, NOTEBOOK 09:1x).
- Mac stubbed-import harness at `d1c939c3`: **33/33** rewritten dispatch
  suite green (volta, NOTEBOOK 18:4x).
- Mac harness re-run tonight at branch HEAD `8b0ef108` (lebesgue): **142
  passed / 5 failed / 40 deselected** in 16s over the same four files. The 5
  failures are the documented harness artifact (torch.multiprocessing.spawn
  children don't inherit the import stubs; they fail in process bootstrap
  before any test logic, identical on unmodified code — harness docstring);
  the 40 deselected are `@pytest.mark.gpu` (no CUDA on Mac).
- `b8d868ff` tests: 14/14 `test_megatron_config.py` green on Mac (poincare,
  reproduced by doppler).
- Known gap, flagged by volta: on-box `make test-server` has not been
  re-run on the M=N head (`d1c939c3`+). Recommend CI or a box run before
  merge.

## 6. Parity gate — PENDING (completes tonight)

Protocol (PARITY_RUNBOOK.md, driver `tools/parity_driver.py`): fixed 9-datum
mixed-length set (262,032 real tokens), CE loss, no optim_step so weights
never move between legs. Pass bars: loss rel ≤ 1e-6; per-token logprobs abs
≤ 1e-3 (a handful of marginal 1–3e-3 exceedances at partition boundaries is
tolerance-appropriate — padding changes zigzag chunk boundaries); datum
counts/lengths EXACT (hard fail otherwise).

- **Leg A (PP2-padded)**: collected, but on step-18 weights — superseded for
  the PP gate (LoRA B zero-init ⇒ only fresh boots are base-model-equivalent).
- **Fresh PP2 leg (`pp2-padded-fresh`)**: BANKED 2026-08-12 ~21:2x on the
  clean-init merged build — loss 12.54805275637226, 262,032 real tokens →
  3 partitions tail-filled to 131,072 (393,216 processed), 9/9 datum
  outputs, fb 75.6s. Sanity gate passed (fresh base weights sit near-but-
  above the step-18 value, as expected).
- **Reference legs B (padded) + C (unpadded)**: run on the golden
  EP16/CP16/PP1 config (the PP1/CP8 reference topology failed grad-sync
  NCCL — finding F3, unrelated to this diff; a production-hardened
  reference is probative for the topology-invariance claim). Originally
  slated for a second box; box B was killed for capacity ~23:5x, so the
  legs run on box A (w56lorq) in the resequenced single-box queue (after
  the L0 probe; doppler mechanics, poincare drives the measurements).
  B-vs-C is the padding-inertness gate and prices padding waste (expected
  fb ratio ≈ 393,216/262,032 ≈ 1.5× token-work); fresh-A-vs-B is the PP
  gate. Note the CP16 reference widens expected logprob drift on the PP
  gate (loss bar unchanged).

**Do not merge before the three-leg verdict lands.** If the verdict is clean
per the bars above, this section flips to PASS with the compare outputs
attached.

## 7. Risks and watch items

1. **Double-finalize if the one-call-per-op invariant breaks** (the
   load-bearing WARNING): a future change splitting the op into multiple
   schedule calls would re-reduce the whole grad buffer per call. Mitigated
   by the in-code WARNING comment and the dispatch tests asserting exactly
   one call. Reviewers: please challenge this invariant.
2. **`max_seqlen` becomes 131,072 for every partition under padding**
   (currently the max *padded* doc length). DSA uses it as a host-side
   coverage proof; semantics remain correct, but kernel heuristics reading
   it should be watched under real data (PACKING_MEMO Q6 watch item).
3. **Padding waste on real traffic**: ~8–12% typical padded-token overhead
   on the customer histogram, charged at near-full body FLOPs (pads are
   MoE-routed). Direct measurement lands with parity legs B/C. This is the
   price of shape uniformity; if it measures badly, the gate can be narrowed
   (e.g. pad to partition max instead of global max) in a follow-up.
4. **R3 router replay**: FIFO-safe under M=N (verified from
   `router_replay.py:114-117,175`), and remains config-rejected under PP>1
   regardless — no behavior change there. The known R3+PP drift-check bug
   (`loss.py:759-767`, sampler-global vs stage-local layer list) is
   pre-existing, documented, and out of scope.
5. **`BT_PACK_PAD_TO_MAX` is a process env var**, read per call: deliberate
   (parity A/B vehicle), but it is an env gate in production code. Toggling
   needs a relaunch. If the project prefers a config field, that is a small
   follow-up; kept as-env tonight because the parity protocol depends on it.
   Footgun: the check is bare truthiness, so `BT_PACK_PAD_TO_MAX=0` turns
   padding ON (filed as papercut `pc_082cd7ccfff2`).
6. **Residual perf levers are known and queued, not hidden**: CPU-blocked
   serialization ~21s/step on stage 1 (grows with M), MoE all-to-all
   exposure ~25% of d16 wall, PP fill/drain. This PR is the schedule fix;
   overlap levers (L1/L2 in E2_REVISED_SPEC.md) stack on top.

## 8. Reviewer walkthrough (file-by-file vs `origin/main`)

Diff: 7 files, +428/−168. Read in commit order — the packing commit
(`36c3c8f4`) stands alone; the M=N commit (`d1c939c3`) is the payload;
`b34b7aed` is a test-fixture mirror; `8b0ef108` is a docs-only seam marker.

1. **`.../megatron_bridge/thd_cp.py`** (+59): the `pad_to_length` parameter
   on `pack_thd_cp_microbatch` — validation block at the top, tail-fill block
   after the partition size check. Verify the sentinel values against the
   per-doc pad block above it, and that `cu_seqlens` (unpadded) is untouched.
2. **`.../megatron_bridge/packer.py`** (+9): threads `pad_to_length` through
   `pack_thd_cp_microbatches`; docstring records the loss-accounting
   inertness claim.
3. **`.../megatron_bridge/training_runner.py`** (+165/−~150, the core):
   - `_pipeline_parallel_world_size` cached at init (mirrors the existing
     DP/CP world-size caching).
   - `_thd_partition_pad_to_length()` — the PP>1 / `BT_PACK_PAD_TO_MAX` gate.
   - `_run_forward_backward` THD branch restructure: deleted
     `_suppress_schedule_finalize`; empty-slice explicit finalize; the single
     M=N call with the WARNING comment; `_sum_over_microbatches` wrap;
     positional result mapping with the new length guard. Check the review
     contract in §3 against the code.
   - One call-site line passing `pad_to_length=self._thd_partition_pad_to_length()`.
4. **`test_cp_thd_slicing.py`** (+120): the 7 `pad_to_length_*` cases —
   layout/masks/cu_seqlens, exact-fit no-op, None passthrough, three
   validation errors, optional-fields + routed-experts tail.
5. **`test_cp_thd_dispatch.py`** (+239/−~113): the dispatch suite rewritten
   to the M=N contract — one schedule call, per-microbatch result mapping,
   finalize ownership by the schedule tail, no-DP-collective ordering
   (`events == ["schedule", "finalize", "dp_reduce"]`), and the
   zero-data-replica collective test covering the explicit empty-slice
   finalize. These tests ARE the contract; read them as executable spec.
6. **`test_ce_loss.py`** (+1), **`test_router_replay_dp_consensus.py`** (+3):
   fixture mirrors of the new cached `_pipeline_parallel_world_size`
   attribute (the fixtures bypass `__init__`).

If `b8d868ff` is included: **`models/src/loops_models/control.py`** (+27:
config field + mutual-exclusion validator + docstring),
**`.../megatron_config.py`** (+14/−1: pop-then-provider-write in
`_build_config`), **`test_megatron_config.py`** (+68: default-off, parse-on,
both exclusions, splat-compatibility contract).

## 9. References

- Design memos (this folder, repo-side): `MN_SCHEDULE_MEMO.md` (M=N
  feasibility, blast radius, parity argument, memory estimate),
  `PACKING_MEMO.md` (PP+CP microbatch semantics, pad safety, landmines).
- Validation chronicle: `NOTEBOOK.md` 2026-08-12 entries ~18:4x (M=N
  implementation), ~00:3x (918 headline + parks collapse), ~01:0x (sweep),
  ~21:2x (fresh PP2 parity leg + L1 review).
- Trace analyses: `results/TRACE_SKEW_ANALYSIS.md`,
  `results/TRACE_RANK8_ANALYSIS.md`.
- Parity: `PARITY_RUNBOOK.md`, `parity/parity_pp2-padded-fresh.json`.
