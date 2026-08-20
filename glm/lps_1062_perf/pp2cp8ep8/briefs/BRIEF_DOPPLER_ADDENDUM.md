# ADDENDUM to BRIEF_DOPPLER.md (gibbs, 2026-08-12 ~02:0x — written at handoff)

Read after `BRIEF_DOPPLER.md`. Everything here postdates it.

## Live box state at handoff (w56lorq)

- **No trainer running.** The last two boots BOTH failed for volta's parity
  reference config (PP1/CP8/EP8):
  - 2-node PP1/CP8/EP8/**DP2**: NCCL Error 1 "unhandled cuda error" +
    watchdog "Invalid access of peer GPU memory over nvlink" in
    finalize_model_grads → allreduce_coalesced (param_and_grad_buffer.py:660)
    at the warmup tail. Cross-node DP2.
  - 1-node PP1/CP8/EP8/**DP1** (`--num-nodes 1`): SAME NCCL Error 1 in the
    same grad-sync finalize region. **So the failure is NOT DP2-specific** —
    the PP1/CP8 config on this box fails in grad-sync regardless of DP. This
    is the LIVE BLOCKER for volta/poincare's parity legs B+C; it needs
    diagnosis (NCCL_DEBUG=INFO boot, check the DP/CP group transport tables)
    before any PP1 reference boot will pass. NOT seen on any PP2/CP8 boot
    tonight (those are DP1 and never run a cross-node DP collective).
  - Exact error lines + analysis are in NOTEBOOK.md (02:0x entry). The raw
    log was lost to relaunch clobbering (see below).
- **Log-clobbering gotcha (borel filed a papercut)**: `start_trainer.sh`
  OVERWRITES `trainer_srun.log` on every launch. Snapshot any failure's log
  to a file BEFORE relaunching (`cp trainer_srun.log
  lps1062_pp2/logs/<label>.log`). I lost the DP2 raw log this way.
- Box clone: `trainers_main` @ **73c24b00** (merged M=N) + the uncommitted
  TF32-head patch (verify with `git diff` — it should be there; re-apply
  `lps1062_pp2/tf32_head_port_21d0c578.patch` if lost).

## Run order (from borel's E2_REVISED_SPEC.md §3 — read it, it's FINAL)

After parity legs B+C + hausdorff's save probe:
- **L0** = Jack's probe (jumps queue): selective recompute + VPP2
  [18,20,20,20] + `overlap_moe_expert_parallel_comm=true` +
  shared-expert-overlap clearing hunk. **d2 start, NOT d1** (interleaved
  illegal at M=1 — the E2a boot-2 failure). Abort >255 GiB reserved, 10-min
  bound, capture assert text verbatim. borel predicts warmup OOM ~95%; the
  combined-1f1b executor's eager-release machinery is the honest 5%.
- **L3** = traced d16 on known-good M=N config (no new code; whoever holds
  the box can run it). borel: L0 may go first if the box is already at the
  PP2/CP8 known-good state for the save probe; L3 slots wherever cheapest —
  box holder's call.
- **L1** = `overlap_dispatch_backward_with_experts_wgrad` (poincare's
  plumbing hunk, doppler reviews; TE version pre-flight on box:
  `server/.venv/bin/python -c "import transformer_engine as te;
  print(te.__version__)"` — need ≥2.3.0).
- **L2** = VPP2 + overlap_p2p_comm only if box time remains.

## Fleet state

- **volta → poincare** drives parity legs B+C (blocked on the PP1 boot
  failure above). Leg A done (PP2, loss 12.361400908911282, 9/9 datums).
- **dedekind → hausdorff** owns the bounded save_state probe AFTER parity
  (terminal-for-boot if it hangs; save_state_probe.py staged at
  lps1062_pp2/). Ping them when the box flips to PP2/CP8 clean-init.
- **borel** orchestrates. **doppler** (you) holds the box from NOW.
- borel's hang rule refinement: CPU-active + log-progressing = alive
  regardless of window; idle CPU + no log progress = hang. (The PP1 boot
  compiled ~17 min at 85% CPU before failing — CPU-active the whole time.)

## What I'd watch first as box holder

The PP1/CP8 grad-sync NCCL failure is the only thing between poincare and
the parity legs. If the diagnosis takes >20 min, escalate to borel — the
parity claim can also be restructured (poincare's call) rather than
debugging NCCL deep into the night.
