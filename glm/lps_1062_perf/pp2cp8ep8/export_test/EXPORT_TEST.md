# EXPORT_TEST — LoRA adapter export under PP>1 + CP>1

Owner: dedekind (for pauli). Date: 2026-08-10 night. Branch:
`jackrao/lps-1062-pp2-export` (worktree `~/Documents/wt-pp2-export`).

Mission (Jack constraint 3): prove — or find and fix breakage in — the LoRA
adapter EXPORT path when both PP>1 and CP>1, using a minimal model (not GLM).

---

## 1. The export path (as read from source @ df831501)

```
POST /save_weights_for_sampler                      api/server.py:439
  → SaveWeightsForSamplerOp                         api/ops.py:96
  → broadcast_object_list to EVERY GPU rank         api/dispatcher_worker.py:5
  → controller.execute_save_weights_for_sampler     controller/controller.py:233
  → CheckpointManager.save_weights_for_sampler      checkpoints/manager.py:214
  → _publish_lora_adapter                           checkpoints/manager.py:354
      → backend.gather_lora_adapter()               backends/megatron_bridge/backend.py:464
          → bridge._model_bridge.stream_adapter_weights_megatron_to_hf(
              model_list, cpu=True)                 vendor megatron-bridge peft_bridge.py:843
      → nonzero global rank: skip publish (gather is collective, all ranks run it)
      → rank 0: weight_sync client writes
          {weight_sync.path}/sampler_weights/{name}/adapter_model.safetensors
          {weight_sync.path}/sampler_weights/{name}/adapter_config.json
```

Cross-rank gather mechanics (megatron-bridge `param_mapping.py`):
- **PP**: `broadcast_from_pp_rank` — `all_gather_object` over the PP group to
  find the owning stage, then `torch.distributed.broadcast` over the PP group.
  Every rank ends with every tensor. Handles multi-owner (tied embedding) by
  picking lowest PP stage as source.
- **TP/ETP**: `gather_from_tp_ranks` rejoins shards (pinned by
  `test_lora_adapter_inspect.py` TP2 snapshot).
- **EP**: grouped-expert adapters materialized per expert via broadcast+gather.
- **CP**: ***zero references in the conversion layer*** (`param_mapping.py`,
  `peft_bridge.py`, `model_bridge.py` have no `context_parallel`/`cp_` mention).
  CP shards sequence, not weights; export is CP-agnostic *by construction*.

Known past bug (fixed 896cd400, 2026-07-02): export collectives run on a fresh
thread whose CUDA current-device defaults to cuda:0 → NCCL deadlock on every
PP>1 topology. Fix: `torch.cuda.set_device(LOCAL_RANK)` at
`backend.py:470` inside `gather_lora_adapter`. Regression guard:
`test_pp_adapter_export.py::test_qwen3_06b_pp2_export_completes` (PP2/CP1).

## 2. CP>1 evidence audit (CP1 special-casing)

- No integration test has ever booted CP>1 with an export. `context_parallel_size`
  appears in tests only as `=1` (lora_coverage) and in THD unit tests
  (`test_cp_thd_slicing.py`, `test_loss_report.py`, `test_thd_cp_unshard.py`).
- SFT integration cases (`test_sft.py`) all default `cp=1`.
- Git history: CP work is THD data-path (LPS-1063 etc.), nothing export-side.
- Conclusion: PP>1+CP>1 export is genuinely untested. CP1 is not special-cased
  in the export path — the path simply never sees CP at all.

## 3. Boot gates that block PP>1+CP>1 today (must relax on branch)

Both in `backends/megatron_bridge/megatron_config.py`:

- **G1** `_validate_thd_context_parallelism` (l.82): hard `raise ValueError`
  on `pipeline_parallel_size > 1` with `cp_size > 1`. Blocks ALL PP>1+CP>1
  (gibbs needs this relaxed for GLM too — coordination via pauli; identical
  hunk, cherry-pick my commit to avoid divergent edits).
- **G2** `_apply_thd_cp_provider_overrides` (l.107): CP>1 allowed only for
  DSA (GLM-5.2) and HybridModelProvider (Nemotron); anything else raises.
  Blocks the minimal non-GLM model. Relax: let generic GPT providers through
  with default (p2p) CP transport when `attention_backend=fused` — the same
  attention/CP stack the validated hybrid-Mamba path uses. Test-enabling
  change, clearly commented; NOT the production CP-validation story.

## 4. Failure hypotheses (written BEFORE running, per brief)

Ranked by expected likelihood of breaking the test:

- **H1 (certain, boot)**: G1 raises at config build. Trivial fix, see §3.
- **H2 (certain, boot)**: G2 raises for Qwen3. Trivial fix, see §3.
- **H3 (medium, hang)**: Qwen3 attention under THD CP2 (p2p transport, TE fused)
  is an unvalidated stack — could hang in ring-attention P2P or produce NaN.
  Mitigation: `attention_backend="fused"`, tiny seq, 5-10 min bounded probes.
  Fallback: fabricate tiny GLM-5.2 DSA 8-layer checkpoint (§5).
- **H4 (low-medium, hang)**: export collectives on the export thread under a
  PP2×CP2 rank mesh. Device pinning is topology-independent (LOCAL_RANK), and
  with order tp-cp-ep-dp-pp the PP groups are {0,2}/{1,3} — each rank's PP
  group broadcast is self-contained. Should hold; verified by the 60s
  deadlock-cap assertion.
- **H5 (low, wrong content)**: adapter task construction diverges across CP
  replicas → collective mismatch/hang. CP replicas hold identical model
  chunks, so task lists should be identical. LOW risk.
- **H6 (low, wrong content)**: dropped layer subset (classic PP gather bug).
  Covered by Level-1 exact key-set comparison against the PP1/CP1 reference
  (392 pinned keys for Qwen3-0.6B rank-8, reused from `test_pp_adapter_export`).
- **H7 (low, wrong content)**: CP grad reduction leaves CP replicas diverged,
  rank 0 exports a self-inconsistent mix. Not an export bug per se (training
  correctness is gibbs's concern), but Level-1 non-degeneracy checks catch
  gross cases (zeros/NaN).
- **H8 (nuisance)**: cross-topology LoRA-init RNG differs (no seed knob in
  TrainerControllerConfig) → exact PP1-vs-PP2CP2 value comparison invalid by
  construction. Level 2 designed around this (§6).

## 5. Minimal model choice

**Primary: Qwen3-0.6B** (HF cache on every test box; 28 layers, dense;
lora_rank=8 target list already pinned by two existing tests). Requires G2
relaxation. Boot via existing `mb_cluster` harness (+ new `attention_backend`
kwarg), `nproc=4`, `pipeline_parallel_size=2`, `context_parallel_size=2` →
DP=1. 4 of the box's 8 B200s; leaves headroom.

**Fallback (if H3 kills Qwen3 CP)**: tiny GLM-5.2 DSA — 8-layer local HF
snapshot (random bf16 weights, `GlmMoeDsaForCausalLM`/`glm_moe_dsa`
config.json), which passes G2 natively (`experimental_attention_variant="dsa"`,
`cp_comm_type="allgather"`) and hits the `(8,2)` debug layout in
`glm52_dsa.py`. Datums are raw token ids (`encoded_text` chunks), so no
tokenizer dependence in the fb path; risk is in `AutoBridge.from_hf_pretrained`
expectations (indexer shapes, FP8-vs-bf16 config). Only build if needed.

## 6. Test design

Harness: `server/tests/integration/` + `mb_cluster` (local weight_sync into
tmp dir). All runs: Qwen3-0.6B, lora_rank=8, max_seq_len=256, identical
16-token datum (`make_text_datum(range(100,116))`), 1 fb + 1 optim_step
(lr=1e-4) before export, `attention_backend="fused"` on CP runs.

| run | nproc | PP | CP | purpose |
|-----|-------|----|----|---------|
| R0  | 1     | 1  | 1  | reference export (key set, value stats) |
| R1  | 4     | 2  | 2  | **the test** |
| R2  | 8     | 2  | 4  | stretch: wider CP, full-box mesh (PP groups {0,4},{1,5},{2,6},{3,7}) |

(R2 only if R1 passes cleanly and time allows. PP2/CP1 is already covered by
`test_qwen3_06b_pp2_export_completes`; no need to rerun as a separate leg —
if R1 fails and R2 n/a, run PP2/CP1 same-session to isolate CP as trigger.)

Export: `POST /save_weights_for_sampler {"name": <run>, "trainer_server_id":
"test-trainer"}` with 60s cap (deadlock guard; healthy PP2 export is <1s, the
30s long-poll floor + headroom motivates 60s, same as existing test).

### Level 1 — MUST PASS (per run R1, R2)
1. Export returns 200 within 60s (no hang/crash; all rank logs clean).
2. `adapter_model.safetensors` + `adapter_config.json` exist.
3. Sorted `key: shape` list == `EXPECTED_QWEN3_06B_RANK8_PP2_KEY_SHAPES`
   (392 keys = 28 layers × 14 tensors) — byte-identical to the PP1/CP1 set.
   Catches dropped-layer-subset PP gather bugs (H6).
4. Values non-degenerate: all finite; every `lora_A` nonzero; every `lora_B`
   nonzero after the optim step (B init is 0 — all-zero B after training means
   dropped grads); no tensor all-zero.
5. `adapter_config.json`: `lora_rank=8`, expected target modules,
   `auto_mapping` stamped (build_peft_auto_mapping).

### Level 2 — STRONG (faithfulness, RNG-agnostic)
Cross-run exact-value comparison is invalid (H8: no seed control; LoRA-A init
streams differ across world layouts). Instead:

- **L2a (primary)**: DCP round-trip — `save_state` under PP2/CP2, boot PP1/CP1,
  `load_state`, export. `export(PP2/CP2 direct)` vs `export(same state @
  PP1/CP1)` must be bit-identical. Isolates the export gather from training
  RNG. Depends on DCP reshard of adapter-only saves across PP topologies
  (itself unvalidated — a mismatch localizes to reshard-vs-gather via rank
  logs). Pre-written as
  `test_qwen3_06b_pp2cp2_export_faithful_to_state` in the same test file.
- **L2b (fallback)**: statistical — per-tensor mean/std of R1 export vs R0
  reference within loose tolerance; identical distributions expected since
  same data/LR and (near-)identical loss reduction. Report, don't gate.

### On-box procedure (bounded timeouts per Jack)
1. `ssh tj-<job>`; checkout `jackrao/lps-1062-pp2-export` in shared
   `trainers_main` clone; confirm Qwen3-0.6B in HF cache — if the box is on
   the birch/Weka cluster it inherits `team_artifacts/huggingface` (check
   THERE first, per pauli); `env.sh` sets HF_HOME. NOTE (pauli, 08-11 AM):
   the night's box drought was org-shared CPFS cache ENOSPC (exit-74 boot
   deaths) — post-fix the cache may be PURGED, so an empty cache is
   expected, not alarming. If absent: `huggingface-cli download
   Qwen/Qwen3-0.6B` (fast, ~1.5GB).
2. Run the pytest directly (it boots its own torchrun clusters):
   `pytest server/tests/integration/test_pp_cp_adapter_export.py -x -v`
   with per-probe 5-10 min caps. Boot of 0.6B ~1-2 min; hang signatures are
   unambiguous fast. On hang: kill process group, read ALL rank logs in
   tmp `workers.log`, diagnose, relaunch.
3. Alternatively drive manually via devbox lifecycle scripts + curl if the
   pytest harness fights the box (configs on `/root/.cache/user_artifacts/`,
   `BT_TRAINER_CONFIG_PATH`/`BT_TRAINER_SERVER_CONFIG_PATH`).

### If export is BROKEN
Diagnose root cause from rank logs → fix on `jackrao/lps-1062-pp2-export` →
re-run R1 (+R2) → report. This fix is on the critical path for shipping PP2.

## 7. Verdict — validation scope

Per Jack: the CP>1+PP>1 gate is an intentional "untested — you must do the
testing yourself" guard, not a known-broken marker. This test IS that testing
for the export path. Validated/NOT-validated stated explicitly below for the
mainline exemption (gibbs's 21d0c578, DSA-only) to cite.

**VERDICT: the PP>1+CP>1 LoRA adapter EXPORT path WORKS** (for the validated
scope below). Run 2026-08-12 on w56lorq (2×8 B300, leader node), branch
`jackrao/lps-1062-pp2-export` @ 6d962102. Evidence:
`/root/.cache/user_artifacts/lps1062_pp2/export_test/` (on-box; adapters,
worker logs, py-spy stacks).

Validated (all with on-box evidence):

- Topology: **PP2×CP2×TP1×EP1, DP=1, 4×B300 single node** AND **PP2×CP4,
  8×B300 single node** — both green.
- Model class: Qwen3-0.6B dense GPT (non-MoE, non-DSA, non-hybrid).
- CP transport / attention: mcore default P2P ring + TE fused attention
  (`attention_backend="fused"`).
- Export gather across PP on a CP-strided mesh: **YES** — no hang, no crash;
  export completed in seconds (60s deadlock cap never approached).
- Exported key/shape set == PP1/CP1 reference: **YES** — 392 keys (28 layers
  × 14 tensors), byte-identical sorted `key: shape` list
  (`EXPECTED_QWEN3_06B_RANK8_PP2_KEY_SHAPES`).
- Value non-degeneracy: **YES** — all finite, every lora_A nonzero, every
  post-step lora_B nonzero.
- adapter_config.json: **YES** — r=8, alpha=32, 7 target modules
  (q/k/v/o/gate/up/down_proj), `auto_mapping` stamped
  (Qwen3ForCausalLM / transformers.models.qwen3.modeling_qwen3).
- Faithfulness L2b (statistical, vs fresh PP1/CP1 reference, same datum/LR):
  **YES** — 392/392 key match, worst per-tensor std relative deviation
  **1.3%** (loose gate 50%).
- Faithfulness L2a (DCP round-trip PP2/CP2 → PP1/CP1, bit-identical):
  **NOT RUN — BLOCKED** by Finding F1 below (the DCP save leg hangs under
  CP>1; the vehicle broke, not the export).

Explicitly NOT validated (do not cite this test for these):

- GLM-5.2 DSA itself (MoE + EP>1 + grouped-expert adapters + allgather CP
  transport + 131k THD packing) — the export path's EP/grouped-expert
  branches (`_materialize_grouped_expert_adapter_tensor`, packed/shared-outer
  emitters) are NOT exercised by a dense model.
- TP>1 adapter regather combined with PP>1+CP>1.
- Training/loss NUMERICS of any CP>1 stack (export mechanics only).
- Multi-node PP (PP spanning nodes); multi-node anything.
- Long-run stability (1-step probes).
- **DCP checkpoint save/load under CP>1 — separately tested and BROKEN
  (Finding F1). Export and DCP-save are different subsystems; the export
  verdict does not cover checkpointing.**

### Finding F1 (BLOCKER-leaning): DCP `/save_state` hangs under CP>1

**ESCALATION SUMMARY (paste-able):** CP>1 checkpoint saves wedge the trainer
(LPS-1062 finding, 2026-08-12; **confirmed at production scale** 2026-08-12
~22:4x — see "Production-scale reproduction" below). Any training run on a
context-parallel topology — GLM-5.2-FP8 ships golden at CP32 (4×8 B200, 256k)
and CP16 (2×8 B300) — that calls `save_state` (Tinker/loops checkpoint API)
hangs forever: every rank parks in megatron-core dist_checkpointing's
async-finalize polling loop (`maybe_finalize_async_calls` →
`is_current_async_call_done`) while the forked async-writer processes spin
without completing; the trainer must be killed. Attribution is CP-triggered
and PP-independent (probe matrix: PP2/CP1 PASS, PP1/CP2 HANG, PP2/CP2 HANG).
`async_save=True` is unconditional in the trainer CheckpointConfig (no knob),
so every CP>1 run is exposed; no completed CP>1 save has been observed
anywhere (profiling runs never saved; zero CP>1 save tests exist). Repro:
branch `jackrao/lps-1062-pp2-export` @ db5d1826 (`test_save_state_probe`
matrix) + py-spy stacks (small model: `export_test/evidence/`; big model:
`export_test/evidence/big_trainer_save_probe/`). Workaround CONFIRMED at both
scales: `BT_SAVE_STATE_SYNC=1` forces the sync save path, which completes
under CP2 (branch db5d1826,
`test_qwen3_06b_pp1cp2_sync_save_state_probe` green 2026-08-12) **AND at
production scale** (2026-08-12 ~22:5x, w56lorq: PP2/CP8/EP8 @131k sync
`/save_state` completed in 70.8s, 537MB checkpoint written to
/tmp/checkpoints/pp2cp8-save-probe-sync/iter_0000000). NOTE: LoRA adapter
EXPORT (`save_weights_for_sampler`) is a different subsystem and is VALIDATED
under PP2/CP2 and PP2/CP4 — this escalation concerns DCP checkpoint save
only.

**Production-scale reproduction (2026-08-12, box w56lorq):** the real
GLM-5.2-FP8 trainer at **PP2/CP8/EP8/TP1/DP1, 131k seqlen, 2×8 B300, 16
ranks**, clean-init at step 0, was sent one bounded `/save_state` (10-min
cap, `tools/save_state_probe.py`). The op never completed; the trainer stayed
HTTP-200 alive but wedged and had to be killed. This rules out "small-model
artifact" — the hang reproduces on the exact production topology class.

**Mechanism chain (closed at 16-rank py-spy resolution):** the
nvidia_resiliency_ext async-writer daemon (`SpawnProcess-1`,
`async_ckpt/core.py async_process_target`) **crashes on the first queue
item**: `RuntimeError: received 0 items of ancdata` in torch
`multiprocessing/reduction.py:164 recv_handle`, reached via
`torch/multiprocessing/reductions.py:540 rebuild_storage_fd` — i.e. receiving
a CUDA storage file descriptor from the multiprocessing queue failed.
Downstream, the py-spy sweep shows the wedge topology: worker-node ranks park
in `schedule_async_call` → `queue.join`
(`bridge/training/checkpointing.py:378`) waiting for the dead consumer;
leader-node ranks park at the NCCL `barrier` in `save_checkpoint`
(`checkpointing.py:1581`) waiting for the parked ranks. The small-model
signature (ranks polling `maybe_finalize_async_calls` while writer children
spin) is the same break observed one stage later.

**Hypothesis (clearly labeled — NOT a conclusion):** the `received 0 items
of ancdata` signature belongs to the SCM_RIGHTS fd-passing failure class.
Common causes in this class are fd-limit exhaustion (RLIMIT_NOFILE reached in
the daemon or sender, so the ancillary payload is truncated) or a
dead/closed peer socket. Why CP>1 would trip it is not yet explained (e.g.
CP-sharded state changing the number/size of storages handed to the writer).
This gives upstream a first place to look; it has not been verified by
instrumentation.

**Evidence:** `export_test/evidence/big_trainer_save_probe/` (durable,
Mac-side): py-spy dumps for all trainer processes on both nodes
(`pyspy_b300-1-em6dm6d3-0017_*.txt` leader, `pyspy_b300-1-kytcx5dz-0006_*.txt`
worker), `srun_dump_stdout.txt` (NUL-read-proof capture), `probe_run.log`,
and `trainer_srun_saveprobe_async_hang_0229.log` (full trainer log with the
ancdata traceback, snapshotted before relaunch).

- Repro: PP1/CP2 hangs (240s bound, never finalized); PP2/CP2 hangs (600s
  bound); **PP2/CP1 passes** → CP-triggered, PP-independent.
- Stacks (py-spy, all ranks): main/op threads park in
  `megatron/core/dist_checkpointing` `maybe_finalize_async_calls` →
  `is_current_async_call_done` polling loop; the forked async-writer child
  processes spin (~15s CPU) without completing. Stacks:
  `export_test/pyspy_savestate_pp1cp2_*.txt` on-box.
- Blast radius (maxwell's Q1): `async_save=True` is UNCONDITIONAL
  (`megatron_config.py` CheckpointConfig, no knob); CP>1 IS a shipped golden
  topology (GLM-5.2-FP8 CP32/EP32 4×8 B200, CP16/EP16 2×8 B300); save_state
  is user-driven via the loops SDK. No evidence of any completed CP>1 save
  found (gibbs's profiling runs never saved; no CP>1 save tests exist).
  Caveat: repro is the devbox venv (cu13+dsatopk1 cudnn); prod-image repro
  still needed.
- Mitigation probe — **CONFIRMED 2026-08-12 (w56lorq gap window)**:
  `BT_SAVE_STATE_SYNC=1` PP1/CP2 `/save_state` completed in seconds (113s
  wall incl. boot; `iter_0000001` + train_state written, 60MB). The hang is
  ASYNC-PATH-SPECIFIC under CP>1; the sync save path is a proven one-flag
  workaround. Severity: async-only CP bug with a sync fallback — still
  BLOCKER-leaning for CP>1 production because the default (and only
  config-exposed) path is the async one and the failure mode is a silent
  wedge, not an error.

### Finding F2 (box-env, papercut): libcudart clash hard-fails TE fused
attention for non-DSA models on cu12.8-image + cu13-venv boxes

- `flashinfer.jit` ctypes-loads the SYSTEM `libcudart.so.12` by absolute path
  at import (RTLD_GLOBAL); the cudnn dsatopk1 shim then probes
  `dlopen("libcudart.so.12")` + `dlopen("libcudart.so.13")`, finds both, and
  hard-fails (`Multiple libcudart libraries found`). Kills ANY non-DSA model
  on such boxes — the pre-existing PP2/CP1 Qwen3 export test fails there too
  (not CP-specific). GLM's DSA path is immune (never calls TE fused attn).
- Workaround used for all runs above: `LD_PRELOAD=/tmp/ldmask/cudart_mask.so`
  — a 20-line interposer that redirects absolute-path so.12 loads to the
  venv's cu13 runtime and fails soname probes. Source on-box at
  /tmp/ldmask/cudart_mask.c; documented here for reproduction:
  flashinfer import survives (redirected), shim sees only cu13 (probe fails).
- NOT needed on prod images (cu13-matched) — devbox-only mismatch.

## 8. Reporting

- Timestamped entries to `pp2cp8ep8/NOTEBOOK.md`.
- Message pauli at: plan done; box up + first boot; verdict (works/broken+why);
  fix landed.
- Papercuts: `papercuts add ... --tag lps1062`.
