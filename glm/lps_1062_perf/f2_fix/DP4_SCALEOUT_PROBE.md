# F2 DP4 SCALE-OUT PROBE — plan + pre-registered bars (fermi, 2026-08-10)

**Scope: GO/NO-GO scaling probe, NOT a timed arm.** The question it answers:
does the F2 phantom-partition fix hold at EP16/CP8/DP4 across 4 nodes / 32
GPUs, and what does the aggregate wall look like (informational). It does
NOT produce ship-timing evidence and needs **no anchors on the 4-node box**
— nobody should block on anchor re-baselining for a GO/NO-GO read (the
box-variance rule makes cross-box wall comparisons void anyway; the headline
below is a model comparison, not an anchor delta).

**Precondition:** F2 (b)+(c) verdicts on box 2 (this probe is provisioned
ahead of them to save wall-clock; if either fails, the box tears down
unused — Jack's pre-authorization covers the provisioning, helmholtz owns
the go/no-go on spending it).

## 1. Config: `overnight/configs/expB-ep16cp8dp4.json`

Derived from `expB-ep16cp8dp2.json`. **World-size-derived field audit (the
actual job of this deliverable):** every parallelism field in this config is
explicit and *unchanged* — TP1 / PP1 / EP16 / CP8 / ETP1; DP is not a config
field, it derives from the launch: world 32 / (TP1·PP1·CP8) = **DP4** via
`--num-nodes 4`. `max_seq_len=131072` is the per-replica packing row length
(unchanged); LoRA r32/α64, flash attention, weight_sync disabled — all
unchanged. `checkpoint_dir` moves to a probe-specific `/tmp/checkpoints/
glm52-dp4-probe` (node-local; hygiene only). Note: `expG-ep16cp8dp4-max131k.json`
already exists with byte-equivalent semantics (same fields) — this file
supersedes/aliases it for the probe; keep both (expG is the earlier fleet's
naming), do not proliferate a third.

Topology note (why DP4 is the F2-relevant shape): ranks-per-replica = CP8 =
8; EP16 > 8 ⇒ each EP group spans 2 DP replicas — the same cross-DP EP
topology class that deadlocks without F2. The phantom equalization runs on
the pure-DP group (all 4 ranks), so every EP subgroup sees consistent counts.

## 2. Environment (the launch env — every var verified via
   `/proc/<pid>/environ` on all 4 nodes, never env.sh)

Ship base (every boot): `NCCL_IB_QPS_PER_CONNECTION=8
NCCL_IB_SPLIT_DATA_ON_QPS=1 NCCL_NCHANNELS_PER_NET_PEER=8 BT_TF32_LM_HEAD=1`
+ B/F gates `BT_DSA_CP_LAYOUT_CACHE=1 BT_THD_ROPE_HOST_CACHE=1`.
FIX A stays UNSET. FIX C/C′ stay UNSET (single-variable discipline: this
probe measures F2 at DP4, not the stack).

- `BT_F2_PHANTOM_PARTITIONS`: **unset (= default ON)**. The flag-off WARNING
  must NOT appear in trainer_srun.log.
- **`BT_MOE_PROBS_A2A_COMM` MUST BE UNSET (W1 PROHIBITED on this probe)** —
  two reasons, either sufficient: (a) hazard topology — W1's
  second-communicator creation at world-32 EP16 is exactly the multi-EP-group
  case the PR-#28 review fix guards against (the fix refuses to arm there,
  but the probe predates that fix's box verification); (b) single-variable
  discipline — the probe reads F2-at-DP4, not W1. If the log shows the W1
  ACTIVE line, the run is INVALID (env hygiene), not FAIL — fix and re-boot.
- W2/W3 gates all UNSET (the box-3 vendored tree is the shared trainers_main
  clone — gates default OFF; do not patch it from this lane, per the
  shared-CPFS rule).

## 3. Pre-registered acceptance bars (GO/NO-GO)

**MECHANISM (the GO/NO-GO):**
- **P1 (the (a)-equivalent at DP4):** boot to READY / health-OK with default
  warmup (`BT_SKIP_WARMUP` unset — warmup pass-1 sends exactly 1 datum, so
  dp_ranks 1–3 pack 0 real partitions and phantoms must fire). Pre-fix this
  deadlocks >35 min with the py-spy signature (node in token_dispatcher
  all_gather vs node in finalize_model_grads). Bar: READY within the normal
  boot budget (model CPFS-cached; ≤30 min from launch) AND no NCCL-spin
  signature (GPU util not pegged at 100 % with zero progress).
- **P2 (phantom telemetry sane across 4 DP ranks):** during the P3 run, per
  `forward_backward` op every rank's (real + phantom) partition counts are
  EQUAL, phantoms are a strict trailing suffix, and the phantom count is
  exactly max−real per rank. Observable via the controller's per-op
  accounting / debug endpoint; a count mismatch at any rank = NO-GO.
- **P3 (custmix-class heterogeneous datums run clean):** the B_custmix
  20-datum recipe (10,240–63,488 tok, 530,432 tok/step; driver recipe
  `overnight/results/B_custmix_f2/context_node0.txt`) runs to completion at
  DP4: per-window losses finite (no NaN/inf), grad-norm series sane, GPU
  util returns to idle between windows. Quantitative loss comparison across
  meshes is the (c) gate's job on box 2 — NOT this probe (stated to avoid
  double-gating).

**HEADLINE (informational wall, no bar):** steady aggregate tok/s on an
equal-length 131k×d4 run vs the fleet model **≈23.2K agg tok/s** (2× the
2-node winner; ideal-linear reference: 745 tok/s/GPU × 32 ≈ 23.8K). Report
both per-GPU and aggregate; label INFORMATIONAL (two-tier verdicts; no
anchors on this box).

**MEMORY (recorded, not a probe gate):** peak reserved per GPU (the standard
mem poller) recorded against the ≥10 GiB ship bar. A violation flags the
ship-risk for the DP4 mesh (CP8 at 16k×d32 was the known 3.5 GiB violator;
this probe is 131k-shaped) — it does NOT NO-GO the probe.

**Explicitly out of scope:** the flag-OFF deadlock repro at DP4 (causality
already proven at DP2 by (a)+(a3); DP4 adds no new mechanism — same
cross-DP EP topology class); (d)-perf precision timing and the 1a/1b NaN×0
hardening canaries (ship-gated follow-ups per the F2 PR draft); any W1/W2/W3
arm.

## 4. Launch sequence (grothendieck; isolated-launcher pattern per box-2/3
   practice — own log/state dir, own CPFS output subdir)

0. **Box:** 4×8 B300 (helmholtz provisioning). Verify the shared CPFS is
   visible (sentinel file under `/root/.cache/user_artifacts/`) and the
   GLM-5.2-FP8 model cache is warm (boxes 1/2 already fetched it — if
   present, the boot skips the fetch window).
1. **Timing rule (binding):** never overlap the box-3 BOOT/model-fetch with
   a box-1 or box-2 STEADY TIMED WINDOW (shared-CPFS contention inflates
   timed windows). Coordinate with helmholtz's box schedule (the W3-v3
   canary on box 1 is the likely concurrent arm — boot box 3 after its
   timed windows, or before it starts).
2. **Box-3 lane build:** `/root/.cache/user_artifacts/<box3>/server/src` =
   a copy of the shared trainers_main server/src + `f2.patch` applied
   (`git apply --check` first; clean `.orig` after). Isolated launcher
   `start_trainer_<box3>.sh`: own log/state dir, Slurm job-name
   `devbox_trainer_<box3>` (NEVER bare `devbox_trainer` — that's box 1's),
   `F2_SHADOW=1` (PYTHONPATH shadows the venv's trainers_server .pth with
   the patched copy; megatron_core stays on the shared vendored tree via
   the PEP-660 meta-path finder — probe-verified pattern from box 2).
3. **Config staging:** `expB-ep16cp8dp4.json` + `server-config.json` under
   the box-3 subdir; `BT_TRAINER_CONFIG_PATH` /
   `BT_TRAINER_SERVER_CONFIG_PATH` point there. Output ONLY to
   `lps1062_bench/<box3>/` (per-box isolation; same-name collisions were
   the night-1 near-miss class).
4. **Boot:** `bash start_trainer_<box3>.sh --num-nodes 4` with the §2 env on
   the ssh line. Find the HTTP node by probing `:8001/health` on all four
   nodes (the alphabetical-hostname gotcha — rank-0 HTTP is not necessarily
   the leader).
5. **Read P1** from trainer_srun.log (READY; no F2 flag-off WARNING; no W1
   ACTIVE line). **Then P3** via `run_bench2.sh`/`bench_driver2.py` from the
   HTTP node (the custmix recipe), watching P2's per-rank accounting. Then
   the equal-length 131k×d4 steady run for the headline + the mem poller.
6. **md5 discipline:** any file relayed to the box gets md5-verified from
   the consumer side (the relay-scp NUL gotcha is real).
7. **Teardown:** box 3 is fleet-created (helmholtz, under Jack's
   pre-authorization), so it is teardown-whitelisted when the probe
   completes: `truss train stop --remote baseten --job-id <id>`. Report the
   GO/NO-GO + headline + memory numbers to helmholtz before teardown.

## 5. Bar coordination

P1–P3 are mechanism/sanity bars on existing observables — no numeric frame
needed beyond what curie already froze for W3-v3 (this probe shares nothing
with that frame). If the headline or memory numbers need a formal frame
(e.g. a scaling-efficiency band), that's curie's call on the recorded data —
pre-registered here as INFORMATIONAL, so no frame blocks the probe.
