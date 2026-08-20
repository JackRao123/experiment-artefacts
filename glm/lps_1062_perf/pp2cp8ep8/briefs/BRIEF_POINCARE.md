# BRIEF — poincare: volta succession (PP+CP packing & M=N schedule workstream)

You are **poincare**, successor to **volta** (context budget), working for
**borel** (orchestrator, this Mac — succeeded maxwell ~21:40). Report via
`~/.agents/scripts/send-message.sh borel "poincare: <msg>"`. Peers: **gibbs**
(box/bring-up owner — he owns the trainer lifecycle; never relaunch without
pinging him), **dedekind** (export tests). Read first:
`experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/NOTEBOOK.md` (full log),
`PACKING_MEMO.md` (PP+CP microbatch semantics), `MN_SCHEDULE_MEMO.md` (M=N
design), `PARITY_RUNBOOK.md` (parity protocol), `REPORT_MORNING.md` (program).

## Branch state (all pushed to origin)

`jackrao/lps-1062-pp2-packing` @ **8b0ef108** (= main df831501 +):
- `36c3c8f4` pad-to-131k packing + 7 unit tests
- `b34b7aed` test-fixture mirror of the runner's new `_pipeline_parallel_world_size`
- `d1c939c3` **M=N schedule fix** (the headline change)
- `8b0ef108` E2b merge-seam comment (no behavior change)

Worktree: `~/Documents/wt-pp2-packing` (branch checked out; submodules
initialized — if you re-create it, `git submodule update --init --recursive
loops server/vendor/megatron-bridge` or the pre-push typecheck fails on ty's
vendor extra-paths).

gibbs's `jackrao/lps-1062-pp2cp8ep8` holds the PP2 layout (78,2)=38/40, the
DSA-only CP+PP gate relaxation, and the VPP bits. The on-box merged build was
73c24b00 (M=N + gibbs's bits + TF32 patch). Merge direction at PR time:
volta's branch into/onto gibbs's per maxwell's call.

## M=N implementation map (what you must understand to touch it)

Root cause: the runner called the Megatron schedule once per THD partition
with `num_microbatches=1` — under PP the 1F1B schedule never receives a second
microbatch to overlap, so PP degenerated to a pure dependency chain (the "a2a
convoy" wall: 589 tok/s/GPU @ d4). Fix (`_run_forward_backward` THD branch,
`server/.../backends/megatron_bridge/training_runner.py`): ONE call per
optimizer step with `num_microbatches=len(partitions)`. Hardware result:
918@d4 / 1000@d8 / 1052@d16 tok/s/GPU, canaries in band (gibbs, w56lorq).

Load-bearing details (the review contract — these are intentionally IDENTICAL
to the old convention and must stay so):
1. **Grad scale**: the schedule's legacy 2-tuple loss path multiplies by
   `cp_size / num_microbatches` (mcore schedules.py:336-339). At M=N that would
   turn the partition SUM into a MEAN (N× under-scaled grads, silent). The
   THD branch applies `_sum_over_microbatches(forward_step, num_microbatches)`
   (loss.py:1053-1075) which pre-multiplies by N — net scale identical to the
   old per-partition calls. Never drop that wrapper.
2. **Finalize**: the schedule's tail calls `finalize_model_grads_func` once per
   call — equivalent to the old suppress-during-loop + explicit-post-loop ONLY
   because there is exactly one call per op. A WARNING comment in the code says
   this; any future multi-call split must re-gate finalize or grads get
   double-reduced. Empty-DP-slice ranks (0 partitions) skip the schedule but
   still call finalize explicitly — that join is the DP all-reduce rendezvous.
3. **Per-partition accounting**: the schedule's `forward_data_store` on the
   last stage is per-microbatch in forward order → result i maps to partition
   i by position (length-guarded). Datum accounting walks each microbatch's
   `packed_seq_params.cu_seqlens_q` on ALL ranks (the `datum_offset` check
   must keep firing on every rank, not just the last stage).
4. **PP1 semantics**: the no-pipelining schedule interleaves fwd/bwd per
   microbatch — same memory profile and math order as the old per-partition
   calls. CP32 golden path is not re-profiled by this change.
5. `seq_length=max(per-partition global lengths)` is cosmetic — ignored under
   `variable_seq_lengths=True` (set for all CP>1 runs).
6. R3 router replay is FIFO-safe under M=N (router_replay.py:114-117,175) but
   stays config-rejected under PP>1 regardless.

Memory: in-flight activation sets go 1 → min(M,PP)=2 at PP2 (~8-9 GiB on
stage 0 at 131k/CP8 full recompute) — measured headroom was ~108 GiB. The
d1/d2 ramp confirmed it on-box.

## Pad-to-131k packing (the other half of the branch)

`pack_thd_cp_microbatch(..., pad_to_length=)` tail-fills the LAST document's
padded THD region so `cu_seqlens_padded[-1] == pad_to_length` (= max_seq_len
when gated). Gated in the runner on PP>1 or env `BT_PACK_PAD_TO_MAX`
(process-level env, read per call — A/B toggling needs a relaunch, not a
live flip). Sentinels identical to per-doc pads (token 0 / label -100 /
position 0 / weights 0.0 / logprobs 0.0 / advantages 0.0 / temperatures 1.0 /
ref_logprobs 0.0 / padding_mask True; `routing_pad_ids` spread under R3).
`cu_seqlens` (unpadded) is never touched — datum accounting and logprob
stitching read it. Do NOT switch to a synthetic pad document: it breaks
`partition_datums = cu_seqlens.numel() - 1` accounting and zero-real-length
cu entries are untested in TE/DSA. Pads are loss-masked and excluded from
TPS accounting, but DO pay attention+MoE body FLOPs (the router doesn't
exclude them) — that's the real-data efficiency cost (~8-12% typical on the
customer histogram, see PACKING_MEMO Part 3). Open watch item: `max_seqlen`
becomes 131072 for every partition under padding (DSA host-side coverage
proof path — believed safe, verify under real data).

## Parity driver + legs state (the immediate task)

Driver: `experiment_artefacts/glm/lps_1062_perf/tools/parity_driver.py`
(artefacts, no code-branch merge needed). Fixed 9-datum mixed-length set
(seed 0x9A11; 262,032 real tokens; lengths 31,733/48,201/12,997/63,555/8,003/
27,111/40,449/19,231/10,752 — none 16-aligned except one; zero-weight spans
in the middle third). Never calls optim_step → weights never move between
legs. Pass bars: loss rel ≤1e-6, per-token logprobs abs ≤1e-3, datum
counts/lengths EXACT.

- **Leg A DONE** (PP2-padded, live merged trainer on w56lorq):
  loss=12.361400908911282, 9/9 outputs, fb=64.8s (393,216 processed tokens =
  3×131072 tail-filled). JSON: box
  `/root/.cache/user_artifacts/lps1062_bench/parity_pp2-padded.json`.
- **Leg B**: CP8/PP1 + `BT_PACK_PAD_TO_MAX=1` — gibbs boots it
  (config staged: box `/root/.cache/user_artifacts/lps1062_pp2/trainer_pp1cp8ep8_131k.json`;
  PP1 on 16 GPUs = DP2 — comparison surface is DP-invariant, argument cleared
  with maxwell). Run: `parity_driver.py --label pp1-padded`.
- **Leg C**: same boot WITHOUT the env (gibbs reboots). `--label pp1-unpadded`.
- Verdict: `parity_driver.py --compare <A.json> <B.json>` etc. B vs C isolates
  padding; A vs B is the PP gate. Report the three-leg verdict to maxwell +
  gibbs.

Box access: `ssh tj-w56lorq` from this Mac; driver staged at box
`/root/.cache/user_artifacts/lps1062_pp2/parity_driver.py`; use python
`/root/.cache/user_artifacts/trainers_main/server/.venv/bin/python` (system
python lacks httpx); trainer HTTP on the leader at `127.0.0.1:8001`.

## PR-shaping notes (headline M=N PR)

- The PR description must carry maxwell's review contract — the three
  intentionally-IDENTICAL behaviors: (1) accumulated grads identical up to fp
  round-trip (the ×N/N cancel); (2) per-partition metrics identical
  (positional mapping); (3) PP1 semantics identical (interleave, memory,
  finalize timing). Commit d1c939c3's message already has the text.
- Suggested split for reviewability (already separate commits): pad-to-131k
  packing (36c3c8f4) then M=N (d1c939c3). gibbs's branch (layout + gate +
  evidence docstring) and dedekind's export test PR separately per
  REPORT_MORNING § morning sequence.
- Evidence to attach: gibbs's sweep numbers (918/1000/1052 vs 589 pre-M=N),
  the dispatch-test diff (encodes the contract), the three-leg parity JSONs,
  and the on-box `make test-server` run (dedekind ran 188/188 green at
  b34b7aed; re-run on the M=N head is a gap to close — Mac-verified only).
- E2b seam: when VPP stacks on top, the M=N call site's `iter(microbatches)`
  becomes `_schedule_data_iterator(microbatches, len(model_list))` — comment
  is already at the exact merge-conflict lines (8b0ef108).

## Verification tooling you inherit

- Mac pytest harness: `tools/run_server_tests_mac.py` (meta-path stubs for
  triton/modelopt/TE + a `--target` pylibs dir; instructions in its docstring).
  5 spawn-based dp_consensus tests fail under it on ANY commit (subprocesses
  don't inherit stubs) — harness artifact, not a regression.
- On-box is the source of truth for anything GPU: coordinate via gibbs.

## Open items after the parity verdict

1. On-box `make test-server` on the M=N head (d1c939c3) — close the gap.
2. `max_seqlen`=131072/partition DSA behavior under real data (watch item).
3. R3 drift-check PP fix (loss.py:759-767 compares sampler-global stamp to
   stage-local routers) — deferred; only matters if R3+PP is ever enabled.
4. E2b: VPP2 + overlap_p2p_comm stacking on M=N (M≥2 satisfies the interleaved
   constraint) — the next perf lever after the recompute experiment.

---

## LIVE STATE ADDENDUM (volta, handoff at the leg-B boundary — 2026-08-12 ~21:5x)

Handoff per borel's order at an atomic boundary: leg B has NOT run; the 1-node
PP1/CP8/EP8/DP1 boot (with BT_PACK_PAD_TO_MAX=1) is still compiling. **Box
ownership is moving gibbs → doppler mid-boot — coordinate READY/reboots with
doppler, not gibbs.**

Exact protocol for the remaining work:

1. **Leg B** (when doppler pings READY, PP1 boot WITH BT_PACK_PAD_TO_MAX=1):
   `ssh tj-w56lorq`, then
   `cd /root/.cache/user_artifacts/lps1062_pp2 && /root/.cache/user_artifacts/trainers_main/server/.venv/bin/python parity_driver.py --label pp1-padded --base-url http://127.0.0.1:8001`
   (~1-2 min expected; leg A took 64.8s on PP2.)
2. **Leg C**: ping doppler to reboot the same 1-node config WITHOUT the env
   var, then same command with `--label pp1-unpadded`.
3. **Compare** (on the box or after scp-ing the JSONs back):
   `parity_driver.py --compare /root/.cache/user_artifacts/lps1062_bench/parity_pp1-unpadded.json /root/.cache/user_artifacts/lps1062_bench/parity_pp1-padded.json`
   — this is the padding-inertness gate (B vs C).
4. **PP gate (IMPORTANT — changed since the brief body)**: the existing leg A
   JSON (`parity_pp2-padded.json`) was collected on the sweep trainer at
   **step 18** — 18 optim steps had moved the LoRA adapter, so its forward is
   NOT the base model and A-vs-B at the tight bars would fail for weight-drift
   reasons unrelated to PP. (Verified: `lora_B_init_method="zero"` in the
   vendored bridge, so only FRESH boots are base-model-equivalent.) The valid
   PP gate needs a fresh PP2 leg: when doppler relaunches PP2/CP8 (planned
   anyway for the save_state probe), run `--label pp2-padded-fresh` and compare
   against pp1-padded. Keep the step-18 JSON as an artifact but do not use it
   for the verdict.

Verdict-interpretation nuance (what I hold that isn't in the runbook):
- The **loss** is the robust signal (global token-weighted mean over fp32
  reduces): rel ≤1e-6 is achievable and required. A loss miss beyond ~1e-5 rel
  is systematic — treat as failure, do not negotiate.
- **Per-token logprobs** are the sensitive signal: the padded leg changes THD
  row layout → CP zigzag chunk boundaries → kernel reduction order, so a
  HANDFUL of marginal exceedances (1-3e-3 abs) concentrated at partition-
  boundary tokens is tolerance-appropriate. Failure signatures: diffs on real
  tokens FAR from boundaries (contamination), diffs clustered at tail-pad
  positions bleeding into real spans, or any datum count/length mismatch
  (datum-accounting break — hard fail; suspect order in PARITY_RUNBOOK.md).
- **fb_elapsed_s byproduct**: B vs C ratio measures the padding waste on this
  datum set — expected ≈ 393216/262032 ≈ 1.5× token-work if compute-bound.
  Report the ratio; it's the first direct padding-overhead measurement.
- If leg B's server /status shows step != 0 or the loss comes back wildly off
  12.36 (the step-18 value) — sanity: a fresh base-model boot should land
  NEAR 12.36 but not on it (adapter zeroed). ~12.3-12.5 neighborhood is sane;
  11.x or 13.x means something structural (wrong weights, wrong config) — stop
  and ping borel before comparing.

Then: three-leg verdict message to borel + gibbs + doppler with the compare
outputs pasted; NOTEBOOK entry; the brief's open-items queue is yours.

LATE NOTE (volta, post-handoff): if the reference leg moves to the golden
EP16/CP16/PP1 config (per the PP1/CP8 NCCL failures — see
BRIEF_DOPPLER_ADDENDUM), the protocol mechanics carry over unchanged, but
expect WIDER per-token logprob drift on the A'-vs-B PP gate: CP16 vs CP8
changes zigzag chunking and reduction order more than PP2-vs-PP1 alone would.
More marginal 1-3e-3 exceedances are acceptable there; the loss rel ≤1e-6 bar
is unchanged (fp32 reduces). Also note pad_multiple becomes 32 at CP16, so
partition boundaries shift slightly vs the CP8 math in PARITY_RUNBOOK.md — the
driver's datum set and bars need no change.
