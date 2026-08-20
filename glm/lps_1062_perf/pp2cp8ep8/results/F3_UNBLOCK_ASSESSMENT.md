# F3 UNBLOCK ASSESSMENT — can a PP1/CP8 parity leg boot through the first grad sync?

Author: serre, 2026-08-13 ~02:0x CDT, for cauchy. Bounded assessment, no box
actions taken. Builds on `results/F3_PP1_NCCL_DIAGNOSIS.md` (code-verified
structure of the crash).

## Verdict in one paragraph

The env-only discriminator set is a **coin flip (~40-55%)** as an *unblock* —
it routes around the exact transport paths if the fault is NCCL-side, and does
nothing if the fault is an earlier compute kernel (the allreduce-as-messenger
case). **But the parity leg probably doesn't need the grad sync at all**: the
parity measurements (fresh-boot step-0 loss, per-token logprobs) are pure
forward quantities, and the crash lives exclusively in the backward's grad
sync. A **skip-warmup + forward-only** variant dodges F3 structurally — no
diagnosis required — at ~85-90% expected success. Recommend that first; keep
the env ladder for the fb-through-sync version if the leg must prove the train
path.

## 1. Likelihood the env set gets through the sync: ~40-55%

Reasoning (from the diagnosis doc):

- The failing collective is structurally identical to the one that passes at
  PP2 (same 8-rank intra-node DP+CP group, one allreduce, adapters-only
  buffer, ~2× size). The error is `cudaErrorContained` (226) — sticky; the
  allreduce is the messenger. The DP+CP communicator is initialized lazily at
  the crashing call (no `broadcast_params` on this stack).
- **If transport-side** (first-use cuMem shareable-buffer import / NVLS /
  P2P mapping under PP1's 109 GB/rank memory map vs PP2's 55 GB): the toggles
  bypass the exact paths. `NCCL_CUMEM_ENABLE=0` → legacy cudaIpc path;
  `NCCL_NVLS_ENABLE=0` → no multicast; `NCCL_P2P_DISABLE=1` → SHM fallback,
  no NVLink peer access at all. P(env fixes it | transport-side) is high,
  ~0.8.
- **If compute-side** (an earlier kernel's invalid access surfacing at the
  first sync): no NCCL env helps; `CUDA_LAUNCH_BLOCKING=1` only localizes —
  and is far too slow for a 131k leg regardless.
- Prior split ~50/50 (the NVLink-peer-memory error text + first-use-init
  framing vs PP2's healthy comm init on the same box) → overall ~40-55% that
  at least one variant passes.

## 2. RECOMMENDED: the structural dodge (no diagnosis needed)

The parity claim needs **boot + fresh-step-0 forward numerics**. The backward
contributes nothing to loss/logprobs at step 0 (base weights, pre-update).
The crash is in `finalize_model_grads`, which every schedule gates on
`not forward_only` (e.g. schedules.py:828). So:

- `BT_SKIP_WARMUP=1` (backend.py:182-183) — skips the fb startup warmup where
  both crashes happened; proven reachable on this tree (L0 boot 2 hit READY
  with it).
- Drive the leg with the **`/forward` op** instead of `/forward_backward`:
  `ForwardOp` exists on the 73c24b00 tree (api/ops.py:37-43) and runs
  `controller.execute_forward_backward(details, forward_only=True)`
  (worker.py:165) → no backward, **no grad sync, the DP+CP communicator is
  never initialized** — F3 is dodged by construction, under every hypothesis
  (H1 backward-kernel fault included).
- Driver change: one line — parity_driver.py:116 posts to
  `/forward_backward`; point it at `/forward` (same request shape, same
  loss/logprobs in the response; the runner's per-partition logprob path is
  not gated on forward_only). poincare-sized edit.
- Expected cost: first 131k forward JIT-compiles inline (~15-20 min first op;
  warm weight cache). Correctness leg, so wall time is secondary.
- Residual risk (~10-15%): a forward-path fault at the 131k shapes — but
  forward (incl. MoE a2a + CP comms) completed cleanly in both failed boots,
  and the all-78-layers-per-rank forward shape class is what the golden
  EP16/CP16/PP1 production config runs.
- Numerics note: step-0 forward measurements are pre-update, so no transport
  or sync-path choice can touch them; the leg's 1e-6/1e-3 bars are
  unaffected.

**Boot spec (dodge variant), PP1/CP8 unpadded parity leg, 73c24b00+TF32 tree:**

```bash
cd /root/.cache/user_artifacts && source env.sh
export BT_TRAINER_CONFIG_PATH=/root/.cache/user_artifacts/lps1062_pp2/trainer_pp1cp8ep8_131k.json
export BT_TRAINER_SERVER_CONFIG_PATH=/root/.cache/user_artifacts/lps1062_pp2/trainer_server.json
unset BT_PACK_PAD_TO_MAX            # unpadded leg
BT_SKIP_WARMUP=1 bash /root/.cache/user_artifacts/.devbox_up/start_trainer.sh --num-nodes 1
# wait for READY (health :8001), then:
python3 /root/.cache/user_artifacts/lps1062_pp2/parity_driver.py --label pp1cp8-unpadded-fwdonly
#   ...with the driver's submit_and_wait target changed to "/forward"
```

Snapshot rule unchanged: `cp .devbox_up/trainer_srun.log
lps1062_pp2/logs/<label>.log` before any relaunch.

## 3. If the leg must include the backward (fb through the first grad sync)

Then the env ladder from the diagnosis doc, in fix-arm order (max
pass-probability per boot; NCCL_DEBUG kept on so a failure still diagnoses):

```bash
# V1 (keeps NVLink speed; routes around cuMem import + multicast):
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,NVLS,GRAPH \
NCCL_DEBUG_FILE=/root/.cache/user_artifacts/lps1062_pp2/logs/f3fix_nccl_%h_%p.log \
NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 \
  bash /root/.cache/user_artifacts/.devbox_up/start_trainer.sh --num-nodes 1

# V2 (if V1 fails; strongest transport bypass — SHM fallback, slow a2a,
# tolerable for a correctness leg):
... same env ... NCCL_P2P_DISABLE=1
```

- Same config/topology: `trainer_pp1cp8ep8_131k.json`, `--num-nodes 1`,
  `BT_PACK_PAD_TO_MAX` unset, warmup ON (fails fast at seq=64 if it fails).
- Pass bar: READY → driver step-0 fb completes the first grad sync →
  loss/logprobs JSON banked.
- V1 passes ⇒ cuMem/NVLS path implicated, leg runs at full speed. V2 passes ⇒
  NVLink P2P transport definitively implicated. Both fail ⇒ compute-side
  (H1); stop — no cheap env unblock exists on this box.

## 4. Other cheap unblocks, honestly assessed

- **Box-specificity test**: PP1/CP8 on a *different* box would cleanly
  separate config from hardware (leader GPU0 had nonfatal NVLink RX-pipe Xid
  137/145 five minutes before the DP1 boot, and `cudaErrorContained`'s cause
  (b) is literally "certain classes of hardware errors"). **Not available
  tonight** — box B (3m9o7kq) was released on Jack's order ~23:5x. If any
  second box frees, this is a zero-code experiment worth one boot.
- **BT_SKIP_WARMUP alone (with /forward_backward)**: NOT an unblock — the
  warmup is only where the *first* grad sync happens to run; skipping moves
  the same comm-init + allreduce to the first real step's backward.
- **CUDA_LAUNCH_BLOCKING=1**: diagnosis-only (serializes launches; can mask a
  race without fixing it; hours at 131k). Do not spend a parity leg on it.
- No trainer config knob avoids the DP+CP grad allreduce — it *is* the grad
  sync; a training leg cannot skip it. Only forward-only avoids it, which is
  §2.
