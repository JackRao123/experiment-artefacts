# FORMAL ADJUDICATION — W2 arm (box 3, wxlg05w): chunked MoE A2A pipeline (K=2)
Author: curie (verification lane) · Date: 2026-08-10
Evidence (md5-matched drops): w2-arm-hang-trainer_srun.log 0f18e0d9,
w2-ref-131k-d4.json 9d2e03db, box3_arm_log.md 3d6d0f86.

## VERDICT: FAIL-BY-HANG STANDS — verified on my own read of the hang log.

### Hang evidence (my extraction)
- ProcessGroupNCCL watchdog, **PG ID 18**, work SeqNum=619, OpType
  ALLTOALL_BASE, on ALL 8 ranks, after 600 s (Timeout 600000 ms):
  last enqueued 619, last completed 618 — the collective was enqueued by
  every rank and never completed: a deadlock inside the all-to-all.
- **NumelIn is wildly imbalanced across ranks** (rank4: 2, rank7: 4,
  rank2: 134, rank6: 353, rank3: 21287, rank1: 105000, rank5: 113247,
  rank0: 130448) while NumelOut is uniform (65536): the MoE expert A2A with
  normal token-routing imbalance. The gate-OFF reference tolerates this
  imbalance (ran clean); the K=2 chunked pipeline deadlocks under it —
  consistent with a per-chunk synchronization/size-negotiation defect under
  imbalanced inputs (mechanism owned by fermi).
- Stacks: the hang sits in backward (Variable._execution_engine.run_backward
  frames), matching the reported backward_step signature.

### Controls and preconditions
- A2.1 arm-checks pasted for BOTH boots (box3_arm_log.md): reference (gate
  OFF proven) and arm (BT_MOE_A2A_PIPELINE=2 ACTIVE + armed, K=2, L=8);
  W3 explicitly off in both. A3: both fresh, history-symmetric (window 0 =
  first op). Reference arm: clean 701 tok/s/GPU (w2-ref json).
- REGIME NOTE (hybrid-gate omission applies here too): both W2 boots ran
  FORCE-without-CACHE. The A/B is REGIME-MATCHED (identical flags both arms)
  and the reference ran clean in the same regime => the hang is attributable
  to the W2 gate (K=2), NOT the cache state. A2.1-E's INVALID rule targets
  regime-DEPENDENT experiments; a liveness failure with a regime-matched
  clean control is regime-independent. Verdict robust to the regime note.
  (The reference's absolute 701 wall carries the un-cached replay cost —
  regime-qualified as an absolute number; irrelevant to the hang verdict.)

### Disposition
W2 (chunked MoE A2A pipeline, K=2) FAILS by hang at 131k-d4 on box 3.
Mechanism (PG-18 A2A deadlock under routing imbalance) to fermi. No
numerics/perf rows exist for the arm (it never completed a window).
