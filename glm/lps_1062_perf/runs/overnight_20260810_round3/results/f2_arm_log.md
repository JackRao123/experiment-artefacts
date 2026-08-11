# F2 arm log — box 2 (wxlgv5w), grothendieck, 2026-08-10 UTC

## (a) fix-ON boot — PASS
- job 8 devbox_trainer_wxlgv5w, dispatched ~08:11, config expB-ep16cp8dp2.json (EP16/CP8/DP2)
- env (/proc-verified): F2_SHADOW=1, BT_F2_PHANTOM_PARTITIONS unset (fix ON), BT_SKIP_WARMUP unset (1-datum warmup = deadlock trigger)
- loaded-path proof: import probe under exact boot env -> wxlgv5w/server/src; packing.py 28 phantom refs
- warmup pass-1 CLEARED (pre-fix deadlock point); pass-2 full-footprint THD (4 docs ~131072 tok); READY — kernels warm 139.0s; /health 200 OK
- dispatch->READY ~13.5 min (normal band; pre-fix >35 min pass-1 hang)
- job 8 scancelled 08:26 UTC by fourier (his (a3) step-1; stick then yielded to grothendieck); grothendieck re-issued scancel 08:28 (no-op on dead job), confirmed 0 procs, GPUs drained <1 GiB at 08:28:59

## (a3) kill-switch control (BT_F2_PHANTOM_PARTITIONS=0) — PASS (hang reproduced)
- job 9, dispatched 08:29:40 UTC, same env as (a) + flag OFF
- log frozen 08:40:13 (112180 B, no warmup pass-2, no READY, HTTP never up); job RUNNING 21:43 at capture
- py-spy 08:51:28 (f2_a3_pyspy_0851.txt): node0 (6hrsfsfh) pid 12442 MainThread ACTIVE all_gather_into_tensor <- gather_from_sequence_parallel_region <- token_dispatcher.py:1150 preprocess (warmup fwd, MoE dispatch) = node0 token_dispatcher all_gather SIGNATURE MATCH
- node1 (y4wkfkc4) pid 9646 MainThread idle in _coalescing_manager <- finish_grad_sync <- finalize_model_grads.py:502 <- run_startup_warmup = node1 finalize_model_grads SIGNATURE MATCH
- nvidia snapshot: f2_a3_nvidia_0851.txt; log snapshot: f2_a3_trainer_srun.log
- kill: scancel ~08:53 UTC (timebox honored: pass-1-clear-equivalent 08:40:40 + 10 min)

## (c) DP1-golden canary run (f2c-canary-131k-d4.json) + (c1) screen data
- Run on job-10 flag-ON boot (/proc environ: F2_SHADOW=1, BT_F2_PHANTOM_PARTITIONS unset=ON, BT_SKIP_WARMUP unset=default; config expB-ep16cp8dp2.json)
- Drift vs r3anchor DP1 golden (warmup-matched 2): w0 -34.5e-3, m0 -48.9e-3, m1 -58.1e-3, m2 -67.8e-3 (monotone divergence)
- FRAMING (helmholtz, frozen): this shape at DP2 packs 2=2 partitions per replica by construction (4x131072 tokens, 1 partition per datum, 2 datums per replica) => ZERO phantom-fired windows in the run; the (c1) split-consistency screen is VACUOUS for fired windows and reduces to config-drift characterization (DP1/CP16 vs DP2/CP8 reduction-topology trajectory divergence). Adjudication: curie.
- Telemetry gap (papercut): per-op partition counts are not logged at INFO on this build; equal-count evidence here is structural (packing arithmetic) + empirical (arm-B flag-OFF completion = counts agree, per the (a3)-proven mechanism).

## (c2) clean gate — equal-count DP2 flag-ON vs flag-OFF A/B (verdict-carrier)
- Arm A: f2c2-flagON-131k-d4.json — job-10 boot (environ above), 131k-d4, warmup-datums 2, repeats 20. WEIGHT-RESET CLAIM REFUTED (helmholtz 10:55): warmup0 was consistent-with but not proof-of reset; the (c2) arm-A invalidation proved the trainer does NOT reset weights between driver runs (8 prior optim steps => 0.07-0.13 trajectory offset). Boot-history symmetry (fresh boot, identical op history before window 0) is now a codified arm precondition.
- Arm B: reboot BT_F2_PHANTOM_PARTITIONS=0 + BT_SKIP_WARMUP=1 (default 1-datum boot warmup IS the unequal-count trigger — (a3) — so flag-OFF must skip it; warmup never optim_steps => weights stay at init => trajectories comparable). Same driver args. House band <=2e-3/>5e-3, per-window 1:1.
- HONEST VERDICT WORDING (pinned): the DP2 verdict covers deadlock-causality (a+a3) + customer-shape completion (b) + fix-inertness-at-equal-count (c2). Phantom-FIRED numerics rest on design argument (exact [0,0] contribution) + (b)-sanity, with fibonacci 1a/1b as the explicit ship gate.

## (c2) VERDICT — PASS (11:27 UTC, grothendieck)
- Arm A (flag-ON, FRESH boot, symmetric history): f2c2-flagON-fresh-131k-d4.json — job 12 boot, /proc environ F2_SHADOW=1, BT_F2_PHANTOM_PARTITIONS unset (ON), default warmup (no optim_step, weights at init into window 0)
- Arm B (flag-OFF): f2c2-flagOFF-131k-d4.json — BT_F2_PHANTOM_PARTITIONS=0 + BT_SKIP_WARMUP=1
- 21 windows 1:1, matched warmup-datums=2: MAX |dloss| = 0.00165 (main0), all windows inside the <=2e-3 house band, none near the 5e-3 stop. dgn tight.
- Reading: at structurally equal partition counts phantoms never fire; the fix is numerically INERT there (one 4-byte all-reduce per step). The invalid first pass (aged-boot arm A) is recorded above with the refuted weight-reset claim.
- F2 DP2 verdict set COMPLETE: (a) causal boot PASS, (a3) kill-switch signature-matched PASS, (b) custmix completion PASS, (c1) vacuous/config-drift (curie), (c2) inertness PASS. (d) perf + fibonacci 1a/1b/2 = ship-gate deferred.
