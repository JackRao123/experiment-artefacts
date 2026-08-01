# LPS-1003 handoff (unified; status block updated 2026-07-31 evening)

> # STATUS 2026-07-31 EVENING — trigger & mask proven; defect NOT yet named. Read parity/NOTEBOOK.md first.
>
> **Issue 2 (medium non-crash loss spikes): the environmental cause of the
> prod-vs-devbox split is proven, a prod mitigation is validated, and the
> defective code has been narrowed to two GPU kernels — but the root defect
> is not yet named and NO root fix has shipped.**
>
> **Proven (bidirectional, 10/10 vs 0/13+ boots):** the bug fires iff
> `CUDA_DEVICE_MAX_CONNECTIONS` is unset (prod default) and is fully masked
> by `=1` (devbox default — the entire week-long prod-vs-devbox mystery).
> conn=1 is a concurrency-semantics MASK, not a fix: correct code must not
> depend on hardware launch-queue serialization.
>
> - **Deterministic repro** (10/10 boots): prod-exact env, conn UNSET,
>   BT_SKIP_FULL_WARMUP=1, batch-0 /forward at /health → rep0 destroys
>   partition-tail datums {4,5,6} at 5-10 nats, heals by rep1 (instrumented
>   boots occasionally hit more partitions and flicker mid-window).
> - **Prod A/B**: conn unset → 6/6 boot windows fired; LWS patch adding
>   conn=1 → 0/16 reps over 2 fresh boots (deployment 5wolkzw — still
>   carrying the mitigated env; boot-window validated only).
> - **Layer bisection** (unchanged, still solid): corruption enters at
>   `decoder.layers.1.self_attention.core_attention` (DSAttention), only in
>   second-zigzag-chunk bins; output = plausible-scale wrong KV content.
>
> **2026-07-31 day session (7 single-variable boots) FALSIFIED the overnight
> "missing stream/event dependency on the consumption path" reading:**
> top-k env-stream launch pin → still fires (and in-situ both streams are 0);
> no expandable_segments → fires; full device sync after EVERY gather →
> fires; radix top-k replaced by torch.topk → fires. Full matrix + evidence:
> NOTEBOOK "DAY SESSION 2026-07-31".
>
> **2026-08-01 minimization probe: cp>1 is part of minimal f.** A cp1
> minimal config (tp1/pp8/ep2 dp2 @64k — pp8 legal on 78 layers via the
> `_glm52_dsa.py` uneven layout) with the proven trigger (prodenv conn
> unset + BT_SKIP_FULL_WARMUP=1) and synthetic 2×50k datums came back
> CLEAN 5/5 (max per-datum |ΔNLL| 0.0056 vs the 5–11-nat signature; no
> rep0-outlier structure — only ambient near-tie wobble). Expected
> structurally: cp1 bypasses packed-THD/zigzag entirely. See
> `min64k/README.md`. **Step 2 (same date): cp16 golden leaf @ 32k with
> 2×30k docs (incl. a real bundle datum with row-tail supervision) is ALSO
> clean 3/3** — single-doc partitions, per-rank local seq ~1.9k. So CP +
> trigger alone ≠ f; the missing ingredient is prod-scale row geometry
> (64–131k multi-doc packed rows) — see `min32k/README.md`. **Step 3:
> cp16@32k MULTI-doc (3×10k, real tail doc) ALSO clean 3/3.** Two side
> findings from step 3 (details INVESTIGATION.md 2026-08-01 later):
> (a) **/forward zero-fills logprobs at weight-0 positions** — all
> masked-payload dumps measure supervised bands only; probe with uniform
> weights; (b) ambient per-token churn on real text is HUGE between clean
> identical forwards (12–18% of tokens >1 nat, max ~19, all rep pairs,
> position/packing-independent) — detectors must key on asymmetric rep0
> per-datum MEAN shifts, never token deltas. **POSITIVE CONTROL (08-01,
> `ctrl/`): batch-0 partition 1 ALONE (docs 0-6, one 254.5k row, uniform
> loss) FIRES 1/1** — rep0 destroys docs 4/5/6 (+1.2 to +4.6 nats), heals
> by rep1; day's negatives validated same-day. Cutoff mapped: chunks 0-12
> clean, 13-16 flicker (reps 1-2 unstable too), 17-31 hard-destroyed (−2
> to −6.2 mean/chunk); hard onset ~row 135k vs midpoint 127.5k. f is now
> one 254.5k row / 7 real docs / single partition / 3 reps. Row-scale
> threshold 30k↔254.5k unbisected — next rung: same design @ ~64k, ~131k.
> **Steady-state soak (146 reps, same boot): the second-half instability
> is PERSISTENT — docs 4–6 coherently visit a lower-NLL state ~8%/rep
> (doc5 3.98 median vs 2.16 min), rate flat, NOT window-decay.** Leading
> (unconfirmed) inversion: the rare LOW state may be the correct
> computation — i.e. ~92% of prod-env steady-state forwards semi-degraded
> on row tails, every step, gradients included. Decisive discriminator
> not yet run: conn=1 boot + same payload → its level names the correct
> state. See `ctrl/README.md` addendum.
>
> **Standing conclusion:** the corruption originates INSIDE the fused DSA
> score/attention kernel path under ambient GPU concurrency. Suspects:
> (1) cuDNN CuTe indexer forward (warp-specialized, CLC dynamic persistent
> tile scheduler — a skipped tile leaves scores at the -inf prefill =>
> silent plausible mis-selection; fits the positional law), (2) FlashMLA
> sparse forward. Next discriminator is BUILT AND READY: arm H double-exec
> (`bash parity/devbox_streamfix2.sh H`) — runs both kernels twice per call,
> bitwise compare; the self-disagreeing kernel is the defect.
> - Side product: the top-k env-stream launch IS a real latent defect
>   (only kernel in the wheel with an unpinned TVM-FFI launch) — fixed on
>   branch `jackrao/dsa-topk-stream-pin` (wheel `+dsatopk3`, tests pass).
>   It is NOT the LPS-1003 defect; do not present it as such.
> - Everything below this block is the earlier investigation record.

> # STATE AS OF 2026-07-31 (pre-overnight) — superseded by the block above.
>
> **Do not treat the "The bug" paragraph below as current.** It is preserved for
> lineage. The mechanism it names has been tested and is not supported.
>
> ## What was tested and is now CLOSED
>
> | Hypothesis | Verdict | Evidence |
> |---|---|---|
> | Top-k reads scores past `seq_lens` | dead twice over | evening source read (4 walls) + runtime: invalid region **100% -inf over 1.6e12 elements** |
> | Top-k `output_indices` under-write (`torch.empty`) | not supported | **0 unwritten slots over 42,336 calls / 190,506,624 rows** in-situ, 91 shapes, allocator-staging control **1.0000** on every one; 0 dups, 0 out-of-window; + 48 local prod geometries clean |
> | Candidate-flood / smem-spill drop edge | safe by construction | `else` spills to gmem when `enable_gmem_store`; capacity == bucketed `max_num_cols` ≥ cols when not. Tested to 98,240 ties vs a 16,384 buffer |
> | **Data** | dead, exhaustively | **all 1024 documents** probed: 0 datums >2.0 nats, worst in corpus **1.153** vs prod 5-11. Label correlation +0.0113 nats (t=+1.25) |
> | **Script / packing / position** | dead, 3 ways | tail-slot effect **-0.0027 nats over 500,679 tokens** (t=-0.39); packing **provably deterministic** (4-periodic launch shapes) so no deterministic input transform can produce heal-on-replay; masking boundary verified (first supervised target is the constant `{"`) |
> | **Tokenizer** | healthy | worst tokens decode to unguessable content at stable positions |
> | Forward-only vs real training | closed | `trainF` replays ran `forward_backward`+`optim_step` from step 0: 0.7642 vs prod step0 1.45-1.53 |
> | Entrypoint diff | closed | `server/scripts/launch.sh` IS the prod entrypoint; devbox uses the identical one |
> | **Memory freshness (cross-process)** | impossible | driver **scrubs** pages between processes (8 GiB sentinel test → all zeros). ⇒ **the spec's E2 whole-GPU poison is futile by construction**, whatever sentinel is chosen |
>
> Devbox is now **0/9** fresh windows vs prod **4/4** — including one boot with
> the warmup mitigation OFF (`BT_SKIP_FULL_WARMUP=1`), which removes the
> confound that my first two boots had it ON.
>
> ## The one devbox experiment still worth running
>
> **Free-list poisoner.** Cross-process poison can't work (driver scrubs), and
> the top-k output buffer was already covered precisely. What is untested is the
> broad within-process case: fill every cached allocator block across size
> classes with an in-range sentinel, free them WITHOUT `empty_cache()`, then run
> the op — so any uninitialised buffer in that forward (top-k scratch
> `buffer_torch`, attention workspace, MoE dispatch, KV) sees garbage. Pair with
> `BT_SKIP_FULL_WARMUP=1` and the batch-0 probe as the first real op. If that is
> clean, the uninit family is closed on the devbox wholesale rather than one
> buffer at a time.
>
> ## PARKED by Jack (2026-07-31)
>
> The **prod-image-vs-devbox-venv** diff. It is a real candidate — `LD_LIBRARY_PATH=""`
> at `run_trainer_node.sh:39` means the devbox resolves CUDA libs to pip wheels
> (verified via `/proc/self/maps`) while `launch.sh` says the image preserves
> `/usr/local/cuda/lib64`, and the node's system CUDA is 12.8 against a cu130
> build — but it is a shot in the dark until something narrows it. Don't spend a
> session on it without a reason.
>
> ## Calibration note
>
> The per-token loss histogram (36% of supervised tokens under 1e-4 nats, 12.9%
> carrying 73% of the loss) is **NOT itself a finding** — heavy-tailed per-token
> loss is normal for any LM and especially for templated JSON SFT. The only
> load-bearing part is the arithmetic: moving a datum 0.76 → 6.0 nats needs
> ~1250 extra nats, i.e. wholesale document corruption, which content or masking
> cannot supply.
>
> ## Where that leaves it
>
> Everything on the input and logic side is exonerated. The residual is the prod
> runtime/build environment, and the only high-information move left is the
> `probe2` harness in **`audit` mode on a fresh prod session** (prod-safe: no
> extra allocation, GPU→CPU syncs only) — where events actually fire. Needs
> Jack's go-ahead to touch a live session. `fill_(-1)` glue hardening remains
> worth shipping as api.py:51-54 contract hygiene, but NOT as "the fix".
>
> Tooling + raw data: `probe2/` (see `probe2/README.md`). Full logs:
> INVESTIGATION.md sections "2026-07-30 night" and "2026-07-31 — BISECTION".

**Mission for the next session: implement and verify the fix. Investigation is
DONE.** ← *stale, see the STATE block above.* Read order: this file →
CODE_AUDIT_TOPK.md (the defect) → VERDICT.md
(evidence chain + addenda) → experiment_handoff.md (verification experiment
spec — REWRITTEN 07-30 evening; its original read-past-scores program was
code-refuted, do not use stale copies) → EVIDENCE_INDEX.md (claim→artifact
map) → INVESTIGATION.md (full chronological log, for depth). Raw data:
live_prod_probes/ (prod destruction events) + devbox_artifacts/ (full devbox
mirror incl. probe tooling).

## The bug (one paragraph)

cuDNN-frontend DSA indexer top-k (vendored wheel
`nvidia_cudnn_frontend-1.26.0+dsatopk1`, reached only via GLM-5.2's cp>1
multi-document packed THD forward — the only prod traffic with shapes big
enough to matter): `output_indices` is allocated `torch.empty`
(decode_varlen.py:684) and never pre-filled, violating the wheel's OWN api.py
L51-54 contract ("initial (-1) state"). Verified 07-30 evening: multi-CTA
dispatch variants are compiled OUT on our path (static one-CTA-per-row grid)
and the short-row branch full-writes incl. -1 padding — the under-write lives
in the long-row RADIX write-out of the `large_occupancy` compiled variant
(num_rows>148 compile key; prod-only shapes, never covered by small tests);
exact failing edge still unproven (dump test/in-range poison are the
provers). Downstream glue
(`_compact_valid_topk_indices`, `_topk_in_bounds` in Megatron
dsa_cudnn_kernels.py) accepts any non-negative in-range int → partition-TAIL
documents intermittently attend garbage key sets → 5-11 nat per-doc NLL,
healing as allocator memory is overwritten. Window opens at every boot and
every init_trainer_server rebuild. Empirically: 4/4 prod windows fired;
byte-exact replays flat; rotation test proved position-anchoring; CP itself
exonerated (Nemotron-3-Ultra CP4: 73 steps, 0 spikes, flattest of 7 series).
Status: confirmed-by-elimination + mechanism-complete; kernel-level catch
(kill test / dump) still pending — part of fix verification below.

## The fix (layered; none written yet)

1. **Glue hardening** (small, first): `output_indices.fill_(-1)` before
   kernel launch in `_indexer_top_k_wrapper_chunked`
   (dsa_cudnn_kernels.py:489-523) or the wheel API layer.
2. **Wheel fix**: make the long-row radix write-out store exactly top_k
   slots per row (or API pre-fills, honoring its documented -1 initial
   state) → rebuild vendored wheel. Suffix `+dsatopk2` is CLAIMED by PR #821
   (open, backward patches) — fold into #821's wheel flow or use `+dsatopk3`.
   Rebuild recipe: server/vendor/wheels/README.md:206-285.
3. **Salvage from closed PR #843**: warmup-after-rebuild (real gap regardless
   — the rebuild path accepts customer ops with zero warmup). #843 closed as
   mitigation-not-fix; see its closing comment.
4. **Upstream report** (Jack has NOT yet approved — confirm first):
   NVIDIA/cudnn-frontend issue framed as api.py contract violation. As of
   07-30: unreported upstream, no release after 1.26.0, upstream PRs #410/#407
   don't touch output init. Framing ammo (07-30 evening): this is NVIDIA's own
   announced GLM-5.2 long-context CP-training recipe
   (github.com/NVIDIA-NeMo/Megatron-Bridge/discussions/4957), decode_varlen is
   the ONLY top-k impl the wheel ships (api.py:18-19 binding), and the
   `large_occupancy` (num_rows>148) compile variant prod runs is structurally
   uncovered by small-shape tests — not a Baseten misuse.

## Verification set (before + after fix)

- Kill test (causal A/B): fresh boot window, force the pure-PyTorch odd-K
  fallback (dsa_cudnn_kernels.py:475-486) → symptom should vanish.
- Local repro: rerun devbox poison (devbox_artifacts/poison_gpus.py) with
  IN-RANGE POSITIVE INTS (int32 in [0, 262144)) — 0xFF failed because it IS
  the -1 sentinel; zeroed pages get filtered.
- Dump test: during an event, log tk_result["indices"] for the last row
  chunk → expect wrong-in-range values / duplicates / prev-layer echoes.
- Post-fix acceptance: N fresh boot + rebuild windows with the batch-0 probe
  (probe_nll.py + train_bundle_0_31.jsonl.gz; also on devbox CPFS
  /root/.cache/user_artifacts/lps1003/) → zero datums with NLL > 2.0.
  Batch-0 partition map: docs [0-6][7-14][15-22][23-30][31]; destroyed=tails.
- Probe kit: probe_nll.py (/forward probes), rebuild_hammer.py (window
  cycling), probe_inpod.py (in-prod-pod, port 8000). Legacy input builders:
  probe_bundle.jsonl.gz (22 labeled batches), batches_profile.json,
  reconstruct_batches.py, build_probe_bundle.py, loss_histories.json,
  plot_loss_compare.py, repro/ (Mudith's client incl. validate_masking.py).

## Tickets & docs to update with the fix

- Dedupe into **LPS-1003** (umbrella, In Progress). No other internal ticket
  covers this; FOLLOWUP_TICKETS.md item 4 was never filed and is stale.
- Still-unfiled separate tickets: (a) init_trainer_server rebuild DEADLOCK
  (3rd in-process rebuild wedges trainer — reproduced on prod); (b) Issue-3
  crash-restart desync is LPS-1013 (Backlog).
- ~~Correct experiment_artefacts/glm/cudnn_audit/README.md §5.3~~ DONE
  2026-07-30 evening: correction note added under §5.3 (packed CP-THD chunked
  top-k IS the prod path; static per-row dispatch; risk = large_occupancy
  radix write-out; fill_(-1) required, not a docstring nicety).

## Live operational state (verify before touching)

- **Devbox tj-3y0gjkq** (2×8 B300 ali) is the LIVE one; tj-3y0n54q and the other
  older boxes are dead. As of 2026-07-31 05:52Z: trainer STOPPED, `squeue`
  empty, both nodes idle, GPUs at 0 MiB — verified. Lifecycle strictly via
  `/root/.cache/user_artifacts/.devbox_up` scripts; ALWAYS verify stop with
  `squeue` (a silent stop failure once cost 7.5h).
  - Probe kit + harness live at `/root/.cache/user_artifacts/lps1003/probe2/`
    (mirrored to `probe2/` in this dir). Boot via
    `bash .../probe2/boot_probe2.sh <audit|stage|stagefix>`, which only exports
    env then calls the standard `start_trainer.sh`.
  - **Two local patches on this box that another agent may overwrite** (a
    devbox-up fix was in flight 07-31): `.devbox_up/run_trainer_node.sh` derives
    `BT_LEADER_ADDR` from `scontrol show hostnames | head -1` resolved to an IP
    (torchrun's static rendezvous hosts its store on the Slurm NODEID-0 node,
    which is NOT the leader pod here, and a pod cannot resolve its own hostname)
    — without it all 16 ranks hang silently in `TCPStore::TCPStore`. And
    `trainers_main/server/.../megatron_controller.py` carries the uncommitted
    full-footprint warmup patch: **it burns the window, so pass
    `BT_SKIP_FULL_WARMUP=1` for any prod-equivalent fresh-window test.**
  - Gotchas: the FastAPI server binds on **global rank 0** = the NODEID-0 node
    (here `tj-3y0gjkq-1`), not the leader — `curl :8001/health` on the leader
    returns nothing while the trainer is healthy. `srun --overlap` fan-outs queue
    behind a running trainer's CPU allocation, so use direct `ssh tj-<job>-<n>`
    for diagnostics mid-run. CPFS readdir staleness hides leader-created dirs
    from workers → `mkdir -p` on the worker before redirecting into a shared path.
- **B200 GLM catch test: armed** (branch-session agent owns it): session
  7qk8l9q; deployment auto-recreates via session reuse when hyd B200 frees.
  RESTORE still owed by it: TrainingAcceleratorWorkloadPlane override
  B200-multinode → back to vul-atl-prod-1. (trainer_accelerator_priority
  already restored to null — verified.)
- **Nemotron-3-Ultra experiment: COMPLETE + torn down** (0 spikes/73 steps;
  report:
  https://wandb.ai/baseten-training/jackrao-lps1003-compare/reports/LPS-1003:-CP-vs-non-CP-loss-spike-comparison-(train_mean_nll)--VmlldzoxNzYyNTA3Mw== ).
  Mirrors in W&B projects jackrao-lps1003-compare (full) + jackrao-lps1003
  (4-run overlay). Debug runs were removed from oe-grader-sft (hygiene);
  mirrors rebuilt from local logs.
- PR #843 CLOSED (mitigation superseded); PR #821 OPEN (bwd patches, the
  wheel-flow vehicle). Local repo main = a02e06a; B300 stack = branch
  trainer-cuda13-sm103 (0e0b65a for GLM fidelity) — B300 devbox venv MUST
  build from that branch, not main (cu128 lock breaks torch import).

## Identifiers & links (salvaged from the original 07-29 brief)

- Linear: https://linear.app/baseten/issue/LPS-1003 (assignee Jack). Slack
  thread: https://basetenlabs.slack.com/archives/C0BE8KE102E/p1785310909859549
- W&B originals: baseten-training/oe-grader-sft — ln68q5he (Mudith GLM,
  broken image), Super r1/r2 + Qwen flat controls. (Debug GLM reruns deleted
  from that project; series preserved in the mirror projects above.)
- Fix lineage: #696 (odd-topk fallback) → #814 (topk OOB-write patch,
  +dsatopk1) → #829 (image pin trainer-cuda13-sm103-0e0b65a) → this bug
  (unwritten-rows, forward, silent) → #821 open (bwd hardening).
- Org/platform: org-99340d71961343c28c5c567d705ab0c0 (BL15KQ0), clusters
  ali-apse7-prod-1 (B300, nodes report as L20D) + hyd-euis1-prod-1 (B200);
  ClickHouse baseten.trainer_deployment_logs (datum_mean = train_mean_nll ×
  num_loss_tokens / n_datums); dataset HF
  baseten/openevidence-grader-sft-opus46-48k (6.28 GB, 48,898 lines).
- Keys on Jack's machine: WANDB_API_KEY + HF_TOKEN in ~/.zshrc;
  BASETEN_K3_API_KEY in env (export BASETEN_API_KEY explicitly for the SDK).
  Kubeconfigs in ~/.kube/ (ali-apse7-prod-1.yaml, hyd-euis1-prod-1.yaml).
- Investigation history incl. the original three-issue framing (Issue 1 =
  #814 crash FIXED; Issue 2 = this bug; Issue 3 = crash-restart desync
  LPS-1013): INVESTIGATION.md.
