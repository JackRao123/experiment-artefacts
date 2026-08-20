# PROBE RESULT — blockK K=25 warm-pool revival probe (BLOCKK_WARMPOOL_PREREG.md)

Driver: curie (box mechanics lane). Authorized: turing GO 2026-08-14 ~23:3x CDT;
hertz rulings (1) tree revert, (2) canonical config, (3) B/F gates UNSET,
(4) env prereg-verbatim. ONE boot, fail-closed. Box wprm693 (2x8 B300,
ali-apse7-prod-1), trainer job 103, bench label blockK25wp-j103-20260814T235004Z.

## Setup verification (all sha256, pre-boot)

- trainers 73c24b00 + applied diff == TF32 reference patch ALONE
  (f503c9bf1baccbd260bfa531525f6f3f4de2c7bd6b114b73fda9a4a06697e801) —
  db5d1826 hunk REVERTED per ruling 1.
- bridge 20fcf2ea; mcore 57efae08b + gate-stack dirty-diff
  e1e46818dfc684566815fdf482574c08bc275451d4471706dc89d0154eee3653 EXACT.
- wheel cudnn-frontend 1.27.0 verified in-venv.
- config: canonical blockK25 (ddc75eed241b79eba85bb8b19ebfd29fafcb56af4083868d5ec63ab47ad9a82f),
  weight_sync=disabled, sha-verified from BOTH nodes (CPFS).
- env: canonical mission block + BT_SKIP_WARMUP=1; BT_PROFILE_RANKS unset;
  B/F gates UNSET (declared delta vs S3-era boot, boot decl file on record).
- Pre-boot clears (declared): stray PENDING bash job 93 scancelled (would have
  stolen the allocation); two 15.5h orphan spawn children (618 MiB each on
  worker GPUs 1/6) killed. Post-teardown sweep: no dp_worker, GPUs 0 MiB.

## Result

**warmup0 (d1, M=1): COMPLETED — reproduction EXACT.**
fb=268.9s, optim=8.5s, loss=12.325664, gn=0.424732 — canary IN-BAND
(12.2–12.4; drift vs the reference 12.3242/0.4185 = +0.0015 loss, +0.006 gn).
Trainer peak line: 256.68 GiB reserved / 253.25 allocated (reference:
256.8/253.2 — within 0.15 GiB). Poller max during the warmup0 window:
worker 264.3 GiB (reference 265.0), leader 217.3 (reference 218.3).
The memory model's warmup0 prediction (~256 torch / ~264–266 smi) CONFIRMED.

**main0 (first d2, M=2, 262,144 tokens): ACTUAL CUDA OOM — trainer self-terminated.**
Leader (stage-0) ranks spiked 217.3 → 267.4 GiB in ~20 s. OOMs on leader
GPUs 3/5/6 (tried to allocate 708 MiB / 1.62 GiB / 2.21 GiB with 259.2–259.4 GiB
already torch-allocated; process totals 266.1–266.8 vs the 267.69 cap).
`CRITICAL: ... op ForwardBackwardOp hit CUDA OOM; terminating trainer process`
→ SIGKILL cascade → driver connection-refused at main0 → no bench JSON
(expected on a kill; fold_mem crash cosmetic). Worker stayed ~264.4.

On the 618 MiB "Process NNNN" entries in the OOM messages: pids
124065/124068/124072 = this boot's OWN multiprocessing spawn_main children
(born 23:39, 618 MiB, same signature as every boot of this stack incl. the
W1b reference and the S3 boot) — intrinsic stack footprint, NOT contamination.
(Materiality note for the record: GPU 3 had 649 MiB free vs 708 MiB requested,
so the child was the marginal straw for THAT alloc; the process was nonetheless
at 99.5% of cap and climbing — the trajectory was at the ceiling regardless.)

## Read of record (for hertz's adjudication)

- warmup0 reproduction gate: CONFIRMED (the model's warmup0 number is exact).
- Plateau read: NEVER REACHED — the first d2 main OOMed; no mains exist.
- Fail-closed clause: actual OOM → the probe ends; one boot consumed; no
  retries. No fit claim is made — but the negative result is sharper than the
  prereg's FAIL band (which assumed a measurable plateau ≈265): there is no
  plateau. The first M=2 window's stage-0 demand is +50 GiB over the M=1
  warmup0 (leader 217.3 → 267.4 poller) and exceeds the hardware ceiling.
- Interpretation: the warm-pool hypothesis — that the 265.0 warmup0 peak was
  cold-pool first-window excess — is REFUTED. The binding term is the M≥2
  pipeline-depth demand on stage 0, not a cold-pool artifact. The W1b
  DOES-NOT-FIT verdict stands and is reinforced: blockK K=25 does not reach a
  d2 steady state at 131k on this stack, warm or cold.
- Secondary d4 perf leg: not run (PASS-conditional).
- Canary: the only completed window (warmup0) was in-band; no drift event.

## Box end state (per turing's GO: IDLE-ARMED for Jack's stop/keep)

Trainer down; all trainer procs cleared incl. the boot's orphaned spawn child
124065; GPUs 0–4 MiB both nodes; squeue empty. Stray jobs cleared during the
session (all declared): job 93 (pre-existing pending bash), jobs 104/107/108
(my own stray sruns — pkill self-match footgun, papercut pc_fa0c434b659a).
Wheel held by curie until Jack's stop/keep lands via turing.

## Evidence index (Mac-side, sha256 box-vs-Mac verified)

`runs/overnight_20260813_overlap_campaign/blockk_warmpool_probe/`:
- probe_boot_trainer_srun.log — final post-mortem pull (18f6a111c797…)
- blockK25wp-j103-20260814T235004Z.driverlog (55c142706804…)
- mem.b300-1-ea7ns7ym-0002.csv (38861d2385f1…), mem.b300-1-smw3h6mp-0001.csv
  (69058cf2a967…) — 2 s poller, per-GPU
- blockK25_warmpool_boot_decl_20260814T233756Z.txt (0830d9187de0…)
- wait_health_probe{1,2,3}.log
- s3_boot_trainer_srun_final.log — pre-teardown S3 snapshot (80a360dd161e…)
