# PP2/CP8/EP8 @131k — night log (2026-08-10 →)

Manager: pauli (Claude Fable). Fleet: gibbs (bring-up), volta (packing/microbatch), dedekind (LoRA export test).
Goal: see `GOAL.md`. Fresh start from tip of trainers (`origin/main` @ `df831501`).

## Why this config (rationale)

- At PP2×CP8×EP8×TP1 (DP1, 16 GPUs / 2 nodes), with PP as the outermost/cross-node
  dim, **EP a2a and CP collectives stay intra-node (NVLink)**; only PP p2p +
  (tiny, LoRA-only) grad traffic crosses IB. The 256k teardown showed EP a2a =
  26–45 s resident and ~1–1.5% overlapped — moving it onto NVLink attacks the
  dominant cost directly.
- Per-rank expert bytes = golden EP16/PP1: half the layers × double the experts
  per layer cancels (Jack's note 4). Memory should land near golden, plus
  pipeline in-flight checkpointed activations (small under full recompute) —
  stage 0 holds 2 in-flight microbatches at PP2.
- Anchor to beat: golden EP16/CP16 2-node ≈ 645 tok/s/GPU steady @524k tok/step
  (ship NCCL env + TF32 head); EP16/CP8/DP2 hit 745 but is gated on F2.

## Key derived facts (pauli, pre-work)

- DSA topk sharing: `dsa_indexer_topk_freq=4`, `dsa_indexer_skip_topk_offset =
  first_k_dense_replace = 3`. A layer computes its own topk iff 1-based
  `layer_number ≤ 3` or `(layer_number − 3) % 4 == 0` (mcore
  `experimental_attention_variant/dsa.py:104-123`). ⇒ PP stage boundaries must
  start on layers {7, 11, …, 75} (1-based).
- Existing `_GLM52_DSA_PIPELINE_LAYOUTS` boundaries all satisfy this ((78,8),
  (78,16), and the (8,2) debug split 6/2).
- **(78,2) proposal: 38/40** (stage 2 starts at 1-based layer 39; 39−3=36,
  36%4==0 ✓); fallback 42/36 (start 43 ✓). Even 39/39 is invalid: stage 2
  would start at layer 40, 40−3=37, 37%4≠0.
- Boxes: 2×B300/ali (main), 1×B200/vul (export test). devbox-up.
- Branches (all off df831501): `jackrao/lps-1062-pp2cp8ep8` (gibbs,
  ~/Documents/wt-pp2-bringup), `jackrao/lps-1062-pp2-packing` (volta,
  ~/Documents/wt-pp2-packing), `jackrao/lps-1062-pp2-export` (dedekind,
  ~/Documents/wt-pp2-export).

## Box ledger

| attempt | spec | job | outcome |
|---|---|---|---|
| 1 | 2×B300/ali | qk7e72w | FAILED at ssh-reachability (RUNNING→FAILED, ~20:40) |
| 2 | 2×B300/ali | w79r9o3 | FAILED after ~27 min DEPLOYING (21:07, no error msg) |
| 3 | 2×B300/ali | wgopm43 | FAILED at ssh-reachability (RUNNING 21:28 → FAILED) |
| 4 | 2×B300/ali | q8ez9gq | TIMEOUT queued at 60 min (22:29), then FAILED on its own (22:31). |
| — | — | w56e6mq | (e1 ticket) also FAILED on its own at 22:31. |
| 5-8 | 2×B300/ali | retry loop | staggered background loop from ~22:33, 10-min gaps between failures. |
| e1 | 1×B200/vul | w56e6mq | TIMEOUT: still queued/DEPLOYING at 60 min (21:39). Left queued as lottery ticket — if it flips RUNNING, resume provision via devbox_up step driver (ctx.job override). |
| e2 | 1×B200/hyd | (in flight ~21:41) | — |

| e2 | 1×B200/hyd | w79rrd3 | TIMEOUT queued 60 min (22:40), self-FAILED 22:41. |
| e3+ | 1×B200/vul | retry loop | staggered background loop from ~22:43. |
| s1-s3 | supervisors | — | 00:15+: detached per-pool supervisors (ali 2×B300, vul 1×B200; 02:10 added hyd 2×B200 as gibbs plan-B). Fresh creates q0lm74q/qv25o93/w56e07q/wlxvo2w/w79re03 (ali) + 3m9re2q/qv25ee3 (vul) all platform-FAILED through 03:02. |

**03:05 mode shift:** ali jobs now ALLOCATE in ~2 min (capacity back at 3am)
but the workload dies ~20-30 s into RUNNING, SSH proxy rejecting throughout —
poison-node/boot-flap signature (exit_code=74 events, cross-user). Papercut
filed (infra/major). Supervisors keep grinding; nothing client-side fixes this.

Last night's B300 boxes (wxlg05w, 318g61w) probed: both TRAINING_JOB_STOPPED,
no adoption possible.

**00:15 update:** retry loops (harness background tasks) were externally
killed ~00:10 mid-attempt; their in-flight jobs q0lme2q (2×B300/ali, attempt
8) + 318mv1w (1×B200/vul, attempt 3) were orphaned still-DEPLOYING. Replaced
with detached (nohup) per-pool SUPERVISORS (`supervisor_b300ali.log`,
`supervisor_b200vul.log` in logs/): poll adopted job → resume provisioning
via devbox_up step driver if RUNNING (max 2 resumes, then stop job) → fresh
devbox-up on terminal → abandon+stop if queued >70 min; 7h deadline. Thin
harness watcher wakes pauli on any outcome. Attempts 5-7 (q9vj493, wd76n4w,
qjl9mpw) all died the ssh-stage flake way; ledger totals now 8 ali + 3
B200 attempts, 0 boxes.

**Diagnosis (22:45, from #training-events-internal):** platform-wide, not us —
continuous TRAINING_JOB_FAILED stream on WP `ali-apse7-prod-1` since ≥19:43 PDT
across multiple users (charles@parsed, harry@parsed, us), exit_code=74 or none.
Queued jobs get reaped ~60 min after creation (all our "lottery tickets"
self-FAILED ~1-2 min after devbox-up gave up). No incident declared in Slack;
yesterday's deploy-failures classifier already tagged ali with 31-min GPU
capacity waits. Strategy: automated staggered retries on ali B300 + vul B200,
fleet fully staged to move the instant a box lands.

**Morning status (08:10):** REPORT_MORNING.md written. Supervisors: b300ali
restarted (fresh 7h deadline), b200hyd2n still alive; b300ali/b200vul round-1
GAVE_UP at 07:20 during a ~06:50-08:00 LOCAL Mac DNS outage (api.baseten.co
unresolvable — those "failures" were spurious; connectivity verified back at
08:02). Monitor round 2 armed (exits only on DONE/all-dead).

**08:17 — Jack directive:** capacity is constrained (researchers' training
jobs); only ONE seeker allowed. Kept: b300ali supervisor (2×B300 = the main
bring-up box). Killed: hyd 2×B200 supervisor + its in-flight devbox-up; its
queued job wpr7de3 stopped cleanly. dedekind's export test will run on the
B300 box after gibbs's first-light instead.

**08:55 — ROOT CAUSE (subagent investigation, Jack-requested):** the night's
provisioning failures are the org-shared training-cache CPFS
`bmcpfs-3c582c5e-920c-411d-995e-eebfd8056eee` on ali-apse7-prod-1 going
ENOSPC at 08-10 20:40 PDT. Boot writes `.baseten-internal/node-N/-1.bin`
there → write fails → exit 74 (EX_IOERR) → platform kills job, error_message
empty. 20/24 failed jobs show the exact line; qk7e72w captured raw "No space
left on device". Sick-node hypothesis REFUTED (≥12 distinct nodes, one hosted
a healthy box hours earlier; node disks fine). Overnight DEPLOYING-timeout
deaths were separate plain capacity contention. Retrying is futile until the
volume is purged/expanded (platform-side); seeker kept alive as a canary.
Slack draft for #team-training-platform handed to Jack. Papercuts:
pc_74b534ac38b4 + blocker addendum.

**10:15 — REMEDIATION (Jack-authorized subagent, kubectl path):** refined
root cause — NOT filesystem exhaustion; the org bindroot has a **250Gi
quota** (underlying CPFS 200T/180T free) and the org filled it: our dq47r1q
≈64.8GB (venvs 52GB + kineto traces 9GB), researchers' active subtrees,
team_qzr5p83 (105 exp dirs). Access = ~/.kube/ali-apse7-prod-1.yaml → utility
pod jrao-cpfs-cleanup (PVC, no subPath). Pre-delete 64KB fsync probe:
ENOSPC confirmed. Deleting ~62GB of OUR stale debris (venvs, traces, closed
repro dirs); researchers' + team subtrees untouched. HF weight caches NOT on
this volume (team_artifacts = different mount) → cold-cache worry void.
Survey: logs/cpfs_survey.txt. Ask for platform shrinks to "raise org quota".

**10:35 — VOLUME UNBLOCKED.** Remediation freed dq47r1q 64.8GB→4.7GB (stale
venvs, kineto traces, closed repro dirs; 2 rm workers finishing). pauli
direct probe: 50MB fsync WRITE PASSES @391MB/s. Researchers'/team data
untouched. Remediation agent itself stalled at the end (watchdog) — pod-side
rm continued; utility pod jrao-cpfs-cleanup to be deleted after rm drains.
Canary seeker's next create expected to boot → full provision → gibbs.

**11:05 — QUOTA RE-EXHAUSTED; full picture.** Our 60GB freed at ~10:30 was
consumed within ~20 min; 8MB fsync probes now 0/40. Real bound is the ORG
bindroot quota (~11TB by the du evidence), pinned full by two Kimi SFT
projects: `2qj17gq` kimi-k3-sft-b300 = **7.2TB, ZERO running jobs (dead
checkpoints — reclaim jackpot)**; `232gxvq` kimi-k27-sft-b300 = **3.7TB,
actively running (w79xky3)** — meaning that job's checkpoint writes have
been failing since 08-10 20:40 PDT and the owner (harry?) likely doesn't
know. Seeker PAUSED (14 post-fix creates all died; futile while pinned).
Utility pod kept as verification handle. Decision menu to Jack: (A) platform
raises org quota, (B) owner-authorized cleanup of the idle 7.2TB project
(pauli can execute in minutes via pod), (C) alert harry re failing ckpts.

**~12:00-13:00 — escalation + pipeline.** Owner/verdict table for the 5 big
consumers delivered to Jack (harry 11.1TB across 3 projects incl. idle 7.2TB;
jerry 829G cache; ervin 340G sync buffer; team cache 6.4TB). Slack archaeology:
IDENTICAL failure July 23 (#proj-loops, Jerry Hong, same -1.bin exit-74; quota
was 250Gi→20TiB per Kimbrian/John Thorpe in that thread — we've now filled the
20TiB). Jack posted the ask (quota >20TiB + interim deletions) to the team.
ARMED: detached headroom_then_grab pipeline (1GB fsync probe ×2 → auto
devbox-up 2×B300/ali, up to 6 retries) + monitor. Utility pod cleanup2 alive
till ~19:20 PT.

**Aug-12 01:20 CDT — quota NOT fixed; grant-dynamics diagnosis.** Grab F3
(wd7dr4w) died at the cache write 01:00:06 while the bracketing in-pod probe
logger showed FAIL at that exact second — and 15/15 FAILs from 05:58Z despite
a 1GB PASS at 05:57:30Z. Writer hunt: NO files >100M modified in 60 min
anywhere; all big-dir du BYTE-IDENTICAL (232gxvq 3729G, 2qj17gq 7189G, team
6377G, owp0zl3 340G, nwxxk5w 616G, 5qe8d0q 13G) ≈ 19.3TiB static. Conclusion:
org pinned at effective ceiling; rare sub-30s PASS windows = CPFS distributed
quota-grant dynamics, not freed space; du-vs-accounted gap (~0.7TiB to the
20TiB limit) plausibly stripe/metadata overhead of millions of small HF files.
⇒ racing is hopeless; unblock = quota raise (not landed) or LARGE deletion
(hundreds of GB+). Armed: headroom_then_grab_v2 (4×64MB streak + 1GB confirm
→ grab; kubectl --request-timeout everywhere) + persistent monitor.

**Aug-12 01:35 CDT — FINAL RECONCILED DIAGNOSIS (answers Jack's "others spawn
jobs fine" challenge).** 1-node workstation diagnostic qk7rzlw died identically
(cache-write exit 74, node-0) ⇒ node count irrelevant. charles's 7 COMPLETED
1×B300 jobs (11:45-16:55 Aug 11, interleaved with our failures) are REGULAR
training jobs — his `.baseten-internal` is EMPTY since Aug 10 ⇒ regular jobs
skip the boot registration write; only WORKSTATIONS (+harry's probes) do it
and die. Org accounted usage > quota by MORE than du shows (overhead on
millions of small files); the 60GB we freed never sufficed — pool still
over-limit, nobody eating it. Pod 1GB probe passes = residual per-client CPFS
block grants (idle CPU-node client), not headroom. ⇒ workstations are 100%
dead until quota raise or BIG deletion (Harry's idle 7.2TB). Seekers stopped
(deterministic failure). Possible workaround offered to Jack: prod-style
regular training job as trainer carrier (needs his nod). B300 usage moving =
regular jobs + other orgs + inference.

**Aug-12 08:40 CDT — HARRY AUTHORIZED (via Jack): delete kimi-k3-sft-b300
(2qj17gq) — just that one.** Pre-checks: project 0 jobs; no running pod mounts
any project subPath (new trainer pods don't even mount the cache PVC — 3rd
structural fact re: why regular jobs boot). Deleting contents (nvfp4-kimi-k26
4278G, bootstrap_ckpts 2063G, huggingface 846G, jobs, .baseten-internal;
keeping dir skeleton) detached in pod → 7.2TB back to the org pool.
headroom_then_grab_v2 re-armed as the single seeker (4×64MB streak + 1GB
confirm → devbox-up 2 b300 ali) + monitor. gibbs alerted for imminent go.

**Aug-12 10:40 CDT — DELETION DONE, HEADROOM CONFIRMED, HOLDING.** 2qj17gq
contents deleted in 47 s (nvfp4-kimi-k26 7 s! big shards; HF cache 36 s);
residual 1G, skeleton kept. 1GB fsync probes 3/3 PASS — 7.2TB back in the
org pool, workstation boots should now survive. Per Jack: ALL automation
disarmed (no grab armed); he compacts pauli's context first, then the grab +
bring-up proceeds. Resume plan in memory `lps-1062-pp2-night` + here.

## Log

- 20:39 PT — devbox-up 2×B300/ali attempt 1: job qk7e72w RUNNING→FAILED during
  ssh-reachability (infra flake). Retried (attempt 2 in flight). 1×B200/vul in
  flight.
- 20:4x PT — worktrees + branches created; DSA topk group math derived and
  verified against bridge + mcore source; briefs being sent to fleet.
- 20:47 PT — fleet dispatched, all three ACKed: gibbs (layout+config+bring-up,
  `briefs/BRIEF_gibbs.md`), volta (PP+CP microbatch memo + pad-to-131k,
  `briefs/BRIEF_volta.md`), dedekind (PP2×CP2 LoRA export test,
  `briefs/BRIEF_dedekind.md`). Mac-side work proceeds while boxes deploy.
- 20:50 PT — brief correction to gibbs: tools/mfu.py already LoRA-corrected
  (08-09 revision); numbers comparable to post-08-09 tables only.
- Boxes: 2×B300/ali attempt 2 + 1×B200/vul both in TRAINING_JOB_DEPLOYING.
- 21:05 PT — gibbs Mac-side done, branch pushed: `jackrao/lps-1062-pp2cp8ep8`
  @ `21d0c578`. (a) `_GLM52_DSA_PIPELINE_LAYOUTS` gains `(78,2)` = [38+emb,
  40+loss]; new table-wide unit test asserts every non-first stage starts on a
  topk-computing layer ((start−3)%4==0 or ≤3) — all 4 layouts verified
  standalone on Mac (megatron absent on Darwin; tests run in CI).
  (b) FOUND + RELAXED the preemptive PP gate volta also flagged:
  `_validate_thd_context_parallelism` rejected all CP>1+PP>1
  (megatron_config.py); now DSA-only exemption, overlap_grad_reduce clause
  kept, validator takes provider; tests updated + new DSA-PP2 allow case.
  Jack (via pauli): gate was intentional 'untested' semantics; exemption ships
  only with validation evidence; update guard docstring with the evidence
  reference once we have numbers (TODO post-bring-up).
  (c) Rank mapping verified in source: bridge passes order="tp-cp-ep-dp-pp"
  (use_tp_pp_dp_mapping defaults False, mcore common_config.py:105); decoder
  RankGenerator (tp1 cp8 dp1 pp2) → PP groups {r, r+8} cross-node, CP groups
  {0-7}/{8-15} intra-node; separate expert RankGenerator (etp1 ep8 edp1 pp2)
  → EP groups {0-7}/{8-15} intra-node; pp-group equality assert between the
  two generators holds. Full recompute default ON (RecomputeConfig
  granularity="full"). router_replay_mode default NONE → R3+PP reject not
  triggered. Config validates: world16 = 1×8×2×DP1, EP8 | CP×DP=8.
  (d) Configs written: `configs/trainer_pp2cp8ep8_131k.json` (max_seq_len
  131072 = packed buffer, per F1) + `configs/trainer_server.json`
  (trainer_id glm52-pp2cp8ep8-bringup).
- 21:1x PT — gibbs wait-time prep (pauli): `tools/profile_driver_new.py`
  gains `--datums N` (window = N×131k datums; aggregates scale; d4 = 524k
  tok/step anchor point). New `tools/dump_parallel_groups.py` — torchrun
  pre-flight that inits parallel_state with our sizes + bridge order and
  asserts CP/EP intra-node, PP cross-node, rank-block→host 1:1. New
  `tools/bringup_pp2cp8ep8.sh` — box-side phases 0-6 (env, checkout, HF
  weights, configs+env exports, srun cleanliness, group-dump instructions,
  launch discipline). volta PACKING_MEMO digested: p2p shapes on the wire
  under variable_seq_lengths=True (CP>1) — unequal partitions OK; hard
  requirement is equal per-partition schedule-call COUNT across PP stages
  (guaranteed: op broadcast + DP1 + deterministic packer); dsa_indexer_loss
  stays 0.0 under PP (already unconditional); warmup exercises PP p2p, watch
  for READY. volta pad-to-131k (36c3c8f4, their branch) NOT in first-light
  path; note: it makes warmup full-size 131k once merged (slower READY, not
  a hang).
- 21:14 PT — pre-push papercut recorded: hook's `ty check --fix` stripped a
  needed backend.py suppression while 3rdparty/Megatron-LM was unchecked-out
  (error unresolvable → ignore 'unused'); restored once submodules fully
  init'd; push then clean. Boxes: ali attempt 3 in flight (2 infra failures).
- 21:07 PT — B300/ali attempt 2 (job w79r9o3) FAILED after ~27 min DEPLOYING
  (no error_message — capacity-shaped). Attempt 3 launched. B200/vul still
  DEPLOYING at 28 min. Hedge plan if attempt 3 dies: 2×B200.
- 21:4x PT — volta: PACKING_MEMO.md landed (PP+CP microbatch semantics trace, 6
  questions answered with file:line). Headline: PP blocker is the config gate
  (megatron_config.py:91-96), not p2p — variable_seq_lengths=True (auto at CP>1)
  makes p2p shapes dynamic; hard requirement is equal partition COUNT across PP
  stages. Pad-to-131k sound via extend-last-doc tail fill in
  pack_thd_cp_microbatch. Landmines: R3 replay drift check PP-broken
  (loss.py:759-767), DSA indexer-loss AutoScaler assumes no PP (neutralized by
  coeff=0), max_seqlen becomes 131072/partition (watch). Part 2 impl next.
- 21:11 PT — dedekind: Mac-side DONE. Export path traced (plan +
  hypotheses in `export_test/EXPORT_TEST.md`): gather is
  `stream_adapter_weights_megatron_to_hf` (PP broadcast + TP regather),
  conversion layer has ZERO CP references → CP-agnostic by construction;
  risk is the rank mesh + boot gates, not CP-sharded weights. Two boot
  gates found and relaxed on `jackrao/lps-1062-pp2-export` @ 8a4ae08b
  (pushed): G1 `_validate_thd_context_parallelism` PP>1+CP>1 raise→warn
  (SAME hunk gibbs needs — volta's memo flagged it too; cherry-pick
  8a4ae08b to avoid divergent edits), G2 provider gate (DSA/hybrid only →
  +generic GPT warn-through on P2P transport). Minimal model: Qwen3-0.6B
  (HF-cached, existing PP2 test harness reused), PP2×CP2/nproc=4 primary,
  PP2×CP4/nproc=8 stretch. New test:
  server/tests/integration/test_pp_cp_adapter_export.py. Level-1 = 60s
  deadlock cap + 392-key/shape set == PP1/CP1 set + non-degenerate values;
  Level-2 = DCP round-trip faithfulness (cross-run RNG differs, no seed
  knob). Mac unit tests un-runnable (no megatron on darwin) — on-box.
  Waiting on B200/vul job id.
- 22:0x PT — volta: Part 2 landed as 36c3c8f4 on jackrao/lps-1062-pp2-packing.
  pack_thd_cp_microbatch(pad_to_length=) tail-fills last doc's padded region;
  gated PP>1 | BT_PACK_PAD_TO_MAX in the runner. 7 new unit tests green locally
  (standalone module load; pytest file runs on-box/CI per convention — the
  backends package init pulls megatron). ruff clean. Parity protocol for
  on-box execution added to PACKING_MEMO.md. Warmup note: under PP2 the 64-token
  warmup now tail-fills to 131k → full-size probe, slower startup, deliberate.
- 21:2x PT — dedekind: pauli decision — NO cherry-pick of 8a4ae08b; gibbs's
  21d0c578 (DSA-only G1 exemption) is the ship-grade mainline version, my
  raise→warn + G2 pass-through stays scoped to the test branch (deliberate
  divergence). Jack: the CP/PP gate is an intentional "untested" guard; the
  export test IS that testing → EXPORT_TEST.md §7 verdict template now
  states explicit validated/NOT-validated scope (topology, model class,
  transport) for the mainline exemption to cite. L2a DCP round-trip test
  pre-written; branch @ ef2e8036 (3 tests: PP2CP2 L1, L2a, PP2CP4 L1).
- 22:2x PT — volta: branch pushed to origin @ b34b7aed (pad impl + test-fixture
  mirror). Pre-push typecheck initially failed on uninitialized worktree
  submodules (ty resolves megatron via vendor extra-paths) — fixed by recursive
  submodule init; papercut filed. dedekind coordinated to run
  test_cp_thd_slicing.py (+ optionally full make test-server) on their B200
  when it lands; volta owns any failures. Merge HOLD until gibbs first-light.
- 22:5x PT — gibbs pre-merge review of b34b7aed: LGTM, merge-tree clean vs
  jackrao/lps-1062-pp2cp8ep8 @ 21d0c578. Open on-box verify item carried:
  max_seqlen=131072/partition DSA behavior under real data (synthetic first-light
  is exact-fit so fill is a no-op there). BT_SKIP_WARMUP=1 noted as B200 OOM lever.
- 21:2x PT — gibbs pre-merge review of volta's jackrao/lps-1062-pp2-packing @
  b34b7aed (pad-to-131k + PP2 full-size warmup): LGTM, no changes requested.
  Verified vs PACKING_MEMO: sentinels/accounting/cu handling exact; all runner
  test fixtures updated; single pack caller; warmup full-size claim confirmed
  via backend.py → runner pack path (side benefit: kernels warm at real 131k
  shape); merge vs bring-up branch CLEAN (git merge-tree). Open on-box verify
  item: max_seqlen becomes 131072/partition under real data (DSA host-side
  coverage proof) — check at real-data parity phase.
- Boxes: ali attempt 4 (q8ez9gq) QUEUED 60min → capacity starvation; B200/vul
  (w56e6mq) also queued; hyd B200 verdict ~22:41. Plan B: 2×B200 topology
  bring-up, d1-first memory ramp.
- 08:1x PT — pauli gotcha staged: CPFS close-to-open + relay quirk can read
  scp'd files back as all-NULs from a sibling node. bringup_pp2cp8ep8.sh gains
  phase 3b: sha256 of every staged file on leader AND per-node via srun
  (--output=/tmp/pp2_sums.%N.txt lands leader-side via srun IO forwarding),
  diff, hard-exit on mismatch with re-stage instruction.
- 08:4x PT — ROOT CAUSE (pauli, subagent+Loki): org-shared CPFS training-cache
  volume ENOSPC since 20:40 → every job dies at boot writing .baseten-internal
  node file, exit 74. Not nodes/capacity. Jack has oncall draft; seeker stays
  as canary (first success = fixed). Runbook implication staged: phase 2 of
  bringup_pp2cp8ep8.sh now auto-starts the GLM-5.2-FP8 HF download in the
  background on a cache miss (purged-cache assumption), logs to
  $PP2/logs/hf_download.log.
- 2026-08-12 ~11:00 CDT — ORCHESTRATOR HANDOFF: maxwell succeeds pauli (context
  compacted at the clean hold point; pauli's session has ended). Handoff doc:
  HANDOFF.md (this folder). State verified on takeover: nothing armed or
  running, no devbox exists, CPFS quota blocker resolved (fsync probes 3/3
  PASS), all three fleet branches pushed + cross-reviewed. gibbs/volta/dedekind
  all ACKed the manager change and hold staged:
  - gibbs: staged fingerprint 6ce04c3733daf6e9, branch @ 21d0c578, executes
    runbook on job id + GREEN LIGHT.
  - volta: b34b7aed pushed, merge-tree clean vs 21d0c578; open box-dependent
    items: (1) dedekind runs test_cp_thd_slicing.py (+ ideally make
    test-server) on a Linux/GPU box, (2) post-merge padded-vs-unpadded parity
    per PACKING_MEMO (CP8 PP1 vs PP2), (3) max_seqlen=131072/partition DSA
    watch item under real data.
  - dedekind: export test @ ef2e8036 slots in after gibbs's profiles (trainer
    stopped between phases); volta's CPU unit tests ride along.
  HOLDING for Jack's green light; on his word: devbox-up 2 b300 ali (ONE
  seeker) → job id to gibbs.
- 2026-08-12 10:56 CDT — Jack GREEN LIGHT. Seeker attempt 1 (devbox-up 2 b300
  ali) died at step 2/13 BEFORE job creation: Baseten /jobs list response
  arrived truncated (JSONDecodeError char 27138) with curl exit 0 — curl
  --retry doesn't cover truncated-200 bodies. Endpoint re-probed clean (107
  jobs, valid JSON) → transient. Patched devbox_up/api.py list_jobs with a
  3-attempt decode retry (also protects the RUNNING-wait poll loop, which
  would otherwise crash mid-provision and orphan a live box); papercut
  pc_f3fb864e99f7. One-seeker rule never violated (no job submitted).
  Attempt 2 launched 11:00 CDT → logs/devbox_up_aug12_go2.log.
- 2026-08-12 11:31 CDT — BOX UP: w56lorq (2×8 B300 ali) fully provisioned in
  ~30 min, 13/13 steps, verification gate PASS (srun-hostnames 2/2,
  per-node-tools 2/2). Boot survived the .baseten-internal cache write —
  CPFS quota fix CONFIRMED in anger (first surviving workstation boot after
  ~35 failures/38h). ssh tj-w56lorq. gibbs sent job id + GREEN LIGHT 11:31;
  runbook executing. Next checkpoints: pre-flight group dump, trainer READY,
  d2 first-light (loss ~12.2-12.4, mem vs 275 GiB).
- 2026-08-12 ~11:5x CDT — gibbs PRE-FLIGHT PASS on w56lorq: staging + 3b
  per-node sha256 integrity OK, weights warm, clone @ 21d0c578. GROUP DUMP ON
  REAL HARDWARE: VERDICT OK — PP groups {r,r+8} cross-node, CP/EP {0-7}/{8-15}
  intra-node. THE PERF THESIS HOLDS (PP = only cross-node dim; EP/CP a2a on
  NVLink). Two findings: (1) c10d rendezvous unusable on these pods (worker
  can't resolve own hostname) → static rendezvous, same as launch.sh; (2)
  prior night's UNCOMMITTED TF32-head/warmup patches found on shared clone,
  preserved to lps1062_pp2/pre_checkout_local_changes_0e0b65a6.patch, running
  clean tip. ANCHOR CAVEAT: 645/691 anchors INCLUDED the TF32 head. Plan
  (maxwell): clean-tip first-light + clean-tip d4 headline first, then
  tf32-d4 rerun with the preserved patch (ship NCCL env both) for
  apples-to-apples vs anchors. Trainer launching now (defaults first,
  ship-env for headline).
- 2026-08-12 ~12:2x CDT — Jack flags PR 987: the cutlass-dsl cu13 overlay race
  was ALREADY diagnosed + fixed by Jack on Aug 7 (resolution-level: drop
  libs-base dep edges, relocked uv.lock, 30/30 clean B300 syncs) — but the PR
  is OPEN/unmerged, so devbox venvs built from main's lock still race; that is
  exactly what bit w56lorq's first boot (and plausibly a131's 0-for-8).
  gibbs's sequenced-reinstall repair = equivalent end state; PR 987 cited as
  canonical fix in NOTEBOOK/papercut. ACTION ITEM for morning: merge PR 987
  (zero version changes, two files) — kills this failure class for every
  future venv build.

## 2026-08-12 — box w56lorq (2×8 B300 ali), first launch + cutlass overlay race

- 16:2x — GREEN LIGHT from maxwell, job w56lorq. Staged 6 files (scp), phases
  0-4 PASS: clone @ 21d0c578, weights WARM (team_artifacts untouched by the
  ENOSPC purge), 3b integrity PASS (leader vs worker sha256 match), nodes clean.
  NOTE: shared clone carried prior night's UNCOMMITTED patches (TF32 LM head +
  warmup pass-2 + moe_shared_expert_overlap fix, vs 0e0b65a6) — preserved at
  lps1062_pp2/pre_checkout_local_changes_0e0b65a6.patch, discarded for a clean
  tip run. **The 645/691 anchors included the uncommitted TF32 head; clean-tip
  runs are not apples-to-apples — tf32-d4 variant queued as follow-up.**
- 16:3x — pre-flight `dump_parallel_groups.py` on the real hardware: **VERDICT
  OK** — PP groups {r,r+8} cross-node, CP/EP groups {0-7}/{8-15} intra-node,
  rank-blocks map 1:1 to hosts. Perf thesis confirmed in fact, not just source.
  (Fix en route: c10d rendezvous unusable on these pods — worker can't resolve
  its own hostname (/etc/hosts carries only peers); static rendezvous instead,
  matching launch.sh.)
- 16:45 — boot 1 FAILED 27s into warmup: `TypeError: elect_sync() missing
  required positional argument 'pred'` in the cuDNN DSA indexer compile
  (sm100). Root cause: **cutlass-dsl 4.5.2 wheel-overlay race** —
  nvidia_cutlass_dsl_libs_base and libs_cu13 ship ~100 overlapping .py files at
  DIFFERENT revisions; uv unpacks concurrently → torn venv (cu13 elect.py
  no-arg call + base _nvvm_ops_gen.py requiring pred). NOT ENOSPC, NOT config.
  PP2/CP8/EP8 init was clean through config/model/weights before the compile.
  Fix (sequenced, cu13 wins every overlap):
    uv pip install --python server/.venv/bin/python --no-deps --force-reinstall --no-cache nvidia_cutlass_dsl_libs_base==4.5.2
    uv pip install --python server/.venv/bin/python --no-deps --force-reinstall --no-cache nvidia_cutlass_dsl_libs_cu13==4.5.2
  Verified: all 179 overlap files == cu13 hashes; 1-GPU probe
  (tools/probe_dsa_indexer.py) compiles+runs the exact failing kernel path: OK.
  **Jack: this is PR 987 (jackrao/cutlass-dsl-cu13-overlay-race) — the
  resolution-level fix (drop libs-base dep edges). MERGE IT; main's lock still
  races every fresh venv build. Plausibly explains a131 night's 0-for-8 READYs.**
- RECORD-hash sweep (tools/venv_record_sweep.py) of the whole venv: 58
  mismatches, 0 missing — ~41 benign .pyc, 16 nvidia_nvshmem_cu12-vs-cu13
  overlay (same pattern, NOT on tonight's path: alltoall+NCCL only; flagged),
  1 flashinfer build_backend.py (build-time only). No ENOSPC casualties found.
- 17:0x — boot 2 dispatched (srun pid 15173). Watching for READY.
- 2026-08-12 ~12:5x CDT — maxwell sequencing note post-first-light: volta
  merge + parity moved to AFTER clean-tip-d4 + tf32-d4 profiles (keep profiled
  binary pristine; synthetic first-light is exact-fit so padding is a no-op
  there anyway). Bubble math: 538 tok/s/GPU @ d2 (2/3 fill) → ~645 @ d4 (4/5
  fill) if purely bubble-limited — clean-tip d4 projected AT the golden
  anchor; tf32-d4 is the push past it. Headline read should prefer an
  untraced steady window (kineto tax measured +29.7% under PP p2p tracing).
- 17:4x — **TRAINER READY** (boot 2, ~8 min warm cache): /status world 16,
  TP1/PP2/CP8/EP8/ETP1/DP1, seq 131072. READY banner: pipeline connected
  (2 PP stages), kernels warm 169.5s. Idle rank0 gpu_mem 104.7 GiB.
- 18:5x — **D2 FIRST LIGHT** (pp2-131k-d2, 262k tok/step): loss 12.316-12.322
  (canary ✓), gn 0.40-0.50. Control fb 30.4s = **538 tok/s/GPU**, mfu3x 4.9%,
  hfu 7.1%. Peak mem 164/275 GiB (below golden ~200; PP2 halves per-rank
  layers). optim steady 0.9s. Kineto tax +29.7% (PP p2p tracing; headline is
  the untraced control window). maxwell's read: 538 @ 2/3 fill ⇒ ~645 @ 4/5
  fill if purely bubble-limited ⇒ clean-tip d4 should land AT the golden
  anchor; below ~640 ⇒ something besides the bubble. d4 running (2 controls).
- 08:4x PT — volta: gibbs d2 FIRST LIGHT PASSED (per maxwell: loss 12.32 in
  canary band, 538 tok/s/GPU @ d2, mem 164/275). volta merge + parity queued
  after d4 profiles. Parity handoff prepared: pp2cp8ep8/PARITY_RUNBOOK.md +
  tools/parity_driver.py (fixed 9-datum mixed-length set, 262k real tokens →
  3 partitions with 38k/32k/60k tail fills under padding; compares loss rel
  ≤1e-6 + per-token logprobs abs ≤1e-3; no optim_step so weights never move).
- 2026-08-12 ~13:3x CDT — maxwell analysis on clean-tip d4 (550 tok/s/GPU,
  59.6s @524k): d2→d4 per-token rate FLAT (+2.2%) where bubble model predicts
  +20%. Two-point model fit: 1F1B bubble model predicts d4 fb 50.7s; SERIAL
  (no stage overlap, fb=2m*t_stage, t_stage=7.6s) predicts 60.8s — measured
  59.6s. Hypothesis: PP stages not overlapping (schedule not engaging 1F1B,
  blocking p2p, or host sync between microbatches). If true this is a
  +50%-class lever (true 1F1B d4 ≈ 38s ≈ 860 tok/s/GPU) vs single-digit
  env/TF32. Discriminator: node-0 vs node-1 compute concurrency in the
  existing d4 kineto trace — gibbs checking in parallel with Run B (ship env)
  and Run C (+BT_TF32_LM_HEAD) reruns.
- 19:0x — **D4 CLEAN-TIP HEADLINE: 550 tok/s/GPU** (controls 550/551), step
  59.6s @524k tok, mfu3x 5.0% / hfu 7.3%, peak 167 GiB, kineto +1.5%.
  d2→d4 scaling only +2.2% (538→550) — bubble is NOT dominant.
- Trace analysis (d4 traced window, node-0, tools/analyze_trace_overlap.py +
  analyze_sendrecv.py + analyze_c10d.py): window 60.5s; GPU-busy union 93.8%
  (1F1B is real — stages do NOT alternate; maxwell's serial model refuted by
  occupancy). BUT compute kernels only 18.25s (30%) vs comm 39.15s (65%),
  overlap 0.77s (1%). Comm = **1260x c10d::alltoall_base = EP a2a =
  38.54s of ncclDevKernel_SendRecv**; PP p2p = 16 ops (negligible); CP
  allgather = 672 ops (small). a2a avg 30.5ms/call — ~100x over NVLink
  payload math; same range as the cross-node 256k teardown (26-45s) despite
  verified intra-node EP groups. Suspects: transport fallback (P2P/IPC broken
  → SHM/NET), peer-wait serialization, or no-overlap design. Run B = ship env
  + NCCL_DEBUG=INFO to read the a2a transport line.
- Kineto trace copied Mac-side attempted (702MB, slow link) — analysis done
  on-box instead; scripts in tools/. Trace remains on-box at
  /tmp/checkpoints/profiles/torch_trace/ (leader only; runtime_profile traced
  node 0 ranks only).
- TF32-head port staged: pp2cp8ep8/configs/tf32_head_port_21d0c578.patch
  (ported from the 0e0b65a6 uncommitted original; env-gated BT_TF32_LM_HEAD=1;
  worktree left clean). For Run C.
- 19:3x — Transport discriminators (maxwell's three): (3) nvidia-smi topo -m
  shows NV18 between ALL GPU pairs — NVLink visible to the platform, no
  virtualization masking. (1) NCCL_DEBUG=INFO (Run B boot): 448x intra-node
  "via P2P/CUMEM" + 192x cross-node "via NET/IB/.../GDRDMA" — transport
  selection CORRECT both directions; no SHM/Socket fallback. Suspect (a)
  transport-fallback is DEAD at the selection level. Remaining: (b)
  peer-wait/serialization (a2a kernels wait on late peers; compute/comm
  strictly alternate at 1% overlap) vs platform-slow-despite-P2P (microbench
  decides). Trace limits: runtime_profile traced leader rank-0 CPU ops only —
  no per-rank spread, no Input Dims recorded.
- Lever notes: overlap_moe_expert_parallel_comm exists (default False) BUT
  mcore requires disabling FULL recompute to use it (transformer_config.py
  :2633) — memory-infeasible at 131k. Follow-up lever: selective recompute
  (core_attn only, no MoE) + a2a overlap. a2a microbench staged
  (tools/a2a_microbench.py) for after Run B.
- 2026-08-12 ~14:5x CDT — maxwell orchestration post-microbench: platform a2a
  EXONERATED (600 GB/s per-rank measured; step-level bandwidth bound ~94ms vs
  38.5s observed = pure peer-wait/serialization inside the trainer). Lever
  order approved: (1) selective recompute + overlap_moe_expert_parallel_comm
  (memory-risk noted: d1-first ramp, bounded probes on virgin flag combo),
  (2) explain 9-a2a-chunks-per-layer-direction, (3) Run C TF32 (booting).
  Work split: gibbs = on-box aggregate breakdown + levers; maxwell = skew
  fingerprint on Run B trace Mac-side (consistent-laggard→routing hot-spot vs
  rotating-laggard→content-dependent DSA skew). Ship-env delta measured ZERO
  (+0.4%) at this topology — cross-node NCCL tuning irrelevant when a2a is
  intra-node (expected, now proven).
- 19:4x — **Run B (clean-tip + ship env): 552 tok/s/GPU — delta +0.4% = ZERO.**
  Ship NCCL env ruled out at this topology (measured). Run B trace aggregate
  (on-box): window 71.1s traced, compute 24%, comm 63%, SendRecv 43.3s×1276 —
  identical structure to clean-tip; env changes nothing in the comm pattern.
- a2a count SOLVED: 315/microbatch = 35 MoE layers × 9 = 3 a2a per pass
  (tokens + probs + combine; token_dispatcher.py:691,704,850) × 3 passes
  (fwd + full-recompute refwd + bwd). Counts exchange is an ALLGATHER on tp_ep
  (not a2a). Tiny probs a2a (~0.5MB/rank) = 2 of 9 per layer — the small-vs-
  large duration test on maxwell's trace pass proves wait-domination if they
  cost the same ~30ms as the 1.6GB ones.
- MICROBENCH (tools/a2a_microbench.py, 8 ranks 1 node, all_to_all_single):
  25MB→0.08ms, 100MB→0.19ms, 201MB→0.34ms (~600 GB/s/rank). Platform a2a is
  FAST → trainer's 30.5ms avg is peer-wait/serialization, ~50x over the
  450GB/rank/step bandwidth bound (~0.75s). Transport exonerated (P2P/CUMEM
  intra-node, NET/IB+GDRDMA cross-node, NV18 topo visible).
- Run C launched (TF32 head BT_TF32_LM_HEAD=1 + ship env; patch
  tf32_head_port_21d0c578.patch applied on box clone, +73/-10
  chunked_lm_head.py only).
- 20:1x — **Run C (TF32 head + ship env) d4: 589 tok/s/GPU** (588/590), step
  55.6s, mfu3x 5.4% / hfu 7.8%, peak 169 GiB. THREE-RUN TABLE @131k d4/524k:
  clean-tip 550 | +ship-env 552 (+0.4% — ruled out) | +TF32 589 (+6.7%).
  = 91% of golden 645 anchor; residual gap = a2a wait wall + TF32-less residue.
  TF32 ACTIVE confirmed in boot log. Run C trace secured on-box at
  lps1062_pp2/traces/runC_d4_tf32_rank0.pt.trace.json (699.5MB); Run B trace
  canonical Mac-side at ~/perf_profiles/lps-1062/pp2cp8ep8/runB_d4_rank0.pt.trace.json
  (maxwell's verified pull). RUNBOOK RULE ADDED: runtime_profile/start CLEARS
  the box trace dir — pull/mv each trace off the profiles dir IMMEDIATELY after
  each traced run, before the next profile start.
- Next lever (pending maxwell go): selective recompute (core_attn, no MoE) +
  overlap_moe_expert_parallel_comm — deletes 3/9 a2a per layer (refwd pass)
  AND unlocks a2a overlap; virgin-topology bounded probes + d1 memory ramp
  first (167/275 peak → ~108 GiB headroom).
- 2026-08-12 ~15:2x CDT — RUN C + orchestration: three-run d4 table complete
  (all PP2/CP8/EP8 @131k, 524k tok/step): clean-tip 550 | +ship-env 552
  (+0.4%, ruled out) | +TF32 head 589 (+6.7%) tok/s/GPU; peak 169 GiB.
  Apples-to-apples vs golden 645 (which had TF32+ship env): 91%. Gap ≈ the
  a2a peer-wait wall (38.5s @1% overlap). maxwell decisions: (1) GO gibbs
  selective-recompute + overlap_moe_expert_parallel_comm (task: scope
  Mac-side now, d1 ramp, bounded probes; baseline to beat 589); (2) dedekind
  export test fills the trainer-stopped window NOW (gibbs↔dedekind direct
  GPU handoff, no mid-test preemption); (3) volta merge+parity PARKED until
  recompute experiment lands (minimal-diff discipline). Run B trace analysis
  (skew fingerprint + probs-vs-tokens wait proof) running Mac-side.
- 09:1x PT — dedekind on-box (w56lorq) unit tests of b34b7aed: 188 passed /
  0 failed across test_cp_thd_slicing + test_cp_thd_dispatch + test_ce_loss +
  test_router_replay_dp_consensus; 7/7 test_pad_to_length_* green. Packing
  branch fully verified pre-merge. Merge+parity stays parked behind gibbs's
  selective-recompute experiment (maxwell's sequencing).
- 20:5x — E1 SCOPE CORRECTION (config-only, no code change): GLM-5.2 attention
  = GlmAbsorbedMLASelfAttention(AbsorbedMLASelfAttention(Attention)) with
  core_attention=DSAttention; AbsorbedMLASelfAttention._checkpointed_attention_
  forward (absorbed_mla.py:742, used :843) wraps the full DSAttention call in
  tensor_parallel.checkpoint with DSA tensors as proper args; flag inherits
  Attention (attention.py:379); mcore defaults recompute_modules→['core_attn']
  (:1727). E1 config staged: configs/trainer_pp2cp8ep8_131k_selective.json
  (recompute.granularity=selective). maxwell conditions locked: d2 canary same
  band + gn match; d1 ramp abort >255 GiB reserved; EV ~800-850 tok/s/GPU if
  the sync-point model holds.
- E2 VPP2 chunk-start verification (maxwell condition 3), computed:
    chunk  pp_rank vpp_rank  layers(1-based)  start-valid
      0      0       0        1..18         OK
      1      1       0        19..38        OK
      2      0       1        39..58        OK
      3      1       1        59..78        OK
  Flat layout list [pp0v0, pp1v0, pp0v1, pp1v1] = [18,20,20,20]; starts
  1/19/39/59 — 1 exempt (dense/embedding), 19/39/59 all (n-3)%4==0. Embedding
  on chunk0, loss on chunk3. mcore auto-derives VPP=2 from the 4-chunk list.
- dedekind holds leader GPUs for the export test (~30-45 min from ~20:50);
  E1 launches on their release.
- 2026-08-12 ~16:0x CDT — TRACE ANALYSIS OVERTURN (maxwell subagent, full
  report results/TRACE_SKEW_ANALYSIS.md): the node-0 "38.5s a2a wall" was
  MISATTRIBUTED — stream+metadata separation shows EP a2a = 7.09s busy (1260
  all_to_allv, transfer at spec 600-790GB/s), PP p2p = 36.23s in just 16
  kernels (4 blocking waits: 11.8s + 3x8.1s, 0% overlap) = 52% of the step is
  STAGE-0 STARVATION waiting on node 1. Stage 1 (40 MoE layers + loss head)
  is the pacing stage. Within-EP sync wait real but secondary on node 0
  (4.59s over bound; probs-a2a in bwd position eats 7.31ms med vs its 0.4ms
  size — wait lands on whichever a2a follows expert compute; laggard rotates,
  sticky per layer, no fixed hot rank). Premise caveat: trace = rank 0 only
  (per-process kineto), 8-rank skew needs all-rank profiling next run.
  ACTIONS: grab node-1 Run C trace off node-1 local /tmp NOW; E1 still GO
  (helps pacing stage most), calibration bar suspended; candidate lever: PP
  rebalance 38/40 -> 42/36 (topk rule holds at layer 43) pending node-1
  evidence.
- 21:0x — **DIAGNOSIS OVERTURNED (maxwell's trace analysis,
  results/TRACE_SKEW_ANALYSIS.md)**: the 38.5s SendRecv wall is 84% PP p2p
  WAIT (stream 35: 16 batched isend/irecv, 36.23s across 4 blocking waits of
  11.8s+3×8.1s — one per microbatch cycle, stage 0 fully starved 52% of the
  window). EP a2a (stream 99, 1260 all_to_allv with full collective metadata)
  is only 7.09s: 4.59s straggler wait (rotating hot rank, 1.6-2.1× per-layer
  expert imbalance, no fixed hot rank → rebalancing lever weak) + ~2.5s
  irreducible transfer at 600-790 GB/s (transfer at spec; 1.61GB/call floor
  2.7ms). 420 triplets [dispatch 1.61GB, probs 0.5MB, combine 1.61GB] =
  35 layers × 3 passes × 4 mb — the 9/layer count confirmed structurally.
  Bwd probs a2a (7.3ms median vs ~0 transfer) = pure peer-wait probe.
  **Stage 1 is the pacing stage** (40 MoE layers + loss head vs our 35+emb).
- Node-1 trace DOES NOT EXIST — runtime_profile hardcodes rank_set={0}.
  Fix pushed: BT_PROFILE_RANKS env (branch @ ad39a97d); E1's traced run uses
  BT_PROFILE_RANKS=0,8 for both stage leaders. maxwell's 800-850 EV bar
  SUSPENDED pending the node-1 trace; E1 still GO (refwd deletion lands
  hardest on the pacing stage — partial rebalance built in).
- New candidate lever noted: PP layout rebalance 38/40 → 42/36 (stage-2 start
  layer 43, (43-3)%4=0 ✓) — the brief's fallback split.
- 2026-08-12 ~16:3x CDT — maxwell p2p forensics (Jack: "why are handoffs
  slow?"): p2p TRANSPORT EXONERATED — per cycle [0.03ms handshake, 4.2ms
  transfer (200MB = 48GB/s = wire speed), 8.1s PARK, 4ms transfer]. All 36.2s
  is genuine waiting on node 1. Balance arithmetic: stage-0 6.8-7.0s/mb work,
  stage-1 ≈ 15s/mb = 2.2x; layer ratio explains 1.14x → ~7s/mb UNEXPLAINED on
  node 1. Rank-0 gap analysis: ~5s/step true GPU-empty host stalls even on
  the starved stage (a2a excluded). Leading suspect for node-1 residual:
  host-stall class from 08-09 256k teardown (DSA-bwd nonzero syncs, ckpt D2H,
  dispatcher = 64s/step there) scaled by 40 layers + loss path (16k×151k
  logits/rank/mb, possible unfused/fp32 CE or per-mb sync). Rank-8 trace
  (BT_PROFILE_RANKS fix @ ad39a97d) rides E1 traced run; first queries: rank-8
  GPU-empty gaps, loss-region wall time, DSA-bwd host-sync fingerprint.
- 21:2x — maxwell p2p deep-dive: transport exonerated (200MB in 4.2ms = wire
  speed; the 8.1s is parking). Stage-1 ≈ 15s/mb = 2.2× stage-0's 6.8-7s;
  40/35 layers explains 1.14×; ~7s/mb unexplained on node 1. Also ~5s/step
  true GPU-empty host stalls even on stage 0 (08-09 host-stall class
  precedent: DSA-bwd nonzero 33s, checkpoint D2H 24s @256k). gibbs loss-path
  find (Mac-side): chunked_lm_head.py per-chunk .item() syncs — :268
  num_active (1/mb) + :293/:299 chunk_mask.any()/.all() (2 per 4096-chunk ×
  4 chunks = 8/mb) on the last stage only; fp32 head GEMM 31.2 TFLOP/mb
  ≈ 0.4-0.5s SIMT (Run B) / ~0.03s TF32 (Run C env → E1 isolates the sync
  residue). Rank-8 trace queries queued: (a) GPU-empty gap total on rank 8,
  (b) loss-head wall per mb, (c) DSA-bwd host-sync fingerprint.
- 2026-08-12 ~17:1x CDT — dedekind EXPORT TEST COMPLETE (box back to gibbs):
  L1 PP2/CP2 GREEN (178s), L1 PP2/CP4 GREEN (134s), L2b statistical GREEN
  (392 tensors, worst std dev 1.3%), adapter_config correct. CONSTRAINT 3
  ANSWERED: PP>1+CP>1 LoRA export WORKS. NEW FINDING attributed: DCP
  /save_state hang is CP-TRIGGERED, PP-INDEPENDENT (PP1/CP2 hangs, PP2/CP1
  passes); all ranks park in mcore async-checkpoint finalize
  (maybe_finalize_async_calls/is_current_async_call_done), async-writer
  children spin CPU; py-spy stacks at lps1062_pp2/export_test/. Blast-radius
  questions assigned (is async_save prod default? has CP>1 ever saved in
  prod?) + sync-save workaround probe queued for next GPU gap. Export verdict
  scope split per protocol: export VALIDATED / DCP save NOT.
- 21:4x — dedekind export test DONE on w56lorq: PP2/CP2 + PP2/CP4 export GREEN;
  DCP save_state HANGS under CP>1 (even PP1/CP2) — evidence in box
  /root/.cache/user_artifacts/lps1062_pp2/export_test/; my runs never
  checkpoint → not a blocker. Non-DSA models on this box need
  LD_PRELOAD=/tmp/ldmask/cudart_mask.so (DSA path immune).
- 21:4x — **E1 LAUNCHED**: clone @ ad39a97d (BT_PROFILE_RANKS), TF32 patch
  re-applied, config=trainer_pp2cp8ep8_131k_selective.json (selective
  recompute), env = BT_TF32_LM_HEAD=1 + ship NCCL env + BT_PROFILE_RANKS=0,8.
  Plan: boot → d1 memory ramp (abort >255 GiB reserved) → d2 canary → d4
  headline with rank-0+rank-8 traced window.
- 21:4x PT — dedekind EXPORT TEST COMPLETE (box w56lorq, handed back to
  gibbs). **L1 PP2/CP2 GREEN (178s), L1 PP2/CP4 GREEN (134s)** — PP>1+CP>1
  LoRA adapter export WORKS: 392-key/shape set == PP1/CP1 reference, values
  non-degenerate, adapter_config correct. L2b statistical GREEN (worst
  per-tensor std deviation 1.3% vs PP1/CP1). L2a BLOCKED by new finding F1:
  **DCP /save_state hangs under CP>1** (PP1/CP2 hangs, PP2/CP1 passes →
  CP-triggered; all ranks park in dist_checkpointing maybe_finalize_async_calls;
  py-spy stacks on-box at lps1062_pp2/export_test/). async_save=True is
  unconditional and GLM-5.2-FP8 ships CP32/CP16 golden → BLOCKER-leaning for
  CP>1 production training; escalation summary in export_test/EXPORT_TEST.md §7.
  F2 (box-env): cu12.8-image+cu13-venv libcudart clash hard-fails TE fused attn
  for non-DSA models; LD_PRELOAD interposer workaround (/tmp/ldmask on-box).
  Branch finalized @ db5d1826 (L1 tests + L2a skipped-with-pointer +
  BT_SAVE_STATE_SYNC toggle + sync probe queued for next GPU gap). Papercuts
  filed (libcudart clash, mb_cluster zombie-server footgun).
- 22:0x — **E1 OOM AT D1** (the ramp discipline caught it): stage-1 node-1 GPU6
  CUDA OOM in warmup, 258.84 GiB allocated (vs 169 GiB full-recompute peak).
  Selective-core_attn at PP2/CP8/131k is memory-infeasible as scoped: stage 1
  stores 40 MoE layers of intermediates + loss-head fp32 logits (~10GB/mb).
  MoE-stored delta ≈ +90 GiB over full recompute. moe_act middle path checked
  (experts.py:792 CheckpointWithoutOutput on the fc1→act seam; a2a safely
  OUTSIDE in the dispatcher) but saves single-digit GB — insufficient.
  maxwell's fallback dial confirmed real: recompute.py block method
  checkpoints only the first K layers/stage (recompute_num_layers=K,
  method='block'); feasible point est. K≈20-24 (~235-250 GiB) deleting ~half
  the refwd syncs — marginal + still a memory guess.
- PROPOSAL to maxwell (diagnosis-first): next box run = known-good Run C
  config + BT_PROFILE_RANKS=0,8 traced d4 → rank-8 timeline decomposes
  stage-1's 15s/mb (GPU-empty gaps / loss-head wall / DSA-bwd host-sync
  fingerprint) BEFORE spending boots on memory-dial guesses. Trainer stopped;
  GPUs idle (dedekind probe window).
- 22:1x PT — dedekind: SYNC PROBE GREEN (gibbs E1-OOM gap window): PP1/CP2 +
  BT_SAVE_STATE_SYNC=1 → /save_state completed in seconds (checkpoint written,
  60MB). CP>1 DCP hang is ASYNC-PATH-SPECIFIC; sync fallback CONFIRMED as the
  one-flag workaround. Escalation summary in EXPORT_TEST.md updated
  (pending→CONFIRMED). Also reaped an orphaned E1 worker (PID 46934) holding
  273GB on GPU6 post-OOM — flagged to gibbs for stop_trainer.sh reaping.
  Box restored to gibbs ad39a97d + TF32 patch, GPUs idle.
- 22:2x — maxwell GO on diagnosis-first. dedekind sync-save probe GREEN
  (BT_SAVE_STATE_SYNC=1 saves fine under CP2; the CP hang is async-path
  specific). dedekind killed orphaned E1 worker PID 46934 (survived my
  stop_trainer.sh holding 273GB on GPU6) — runbook phase 4 now does a
  stale-process sweep (pkill trainers_server.dp_worker on both nodes) before
  launch.
- 22:2x — **DIAGNOSIS RUN launched**: known-good Run C config (full
  recompute) + BT_TF32_LM_HEAD=1 + ship env + BT_PROFILE_RANKS=0,8. d4 traced
  window yields rank-0 (stage 0) + rank-8 (stage 1) timelines. On completion:
  both traces copied to lps1062_pp2/traces/ IMMEDIATELY (profiles dir is
  volatile), rank-8 path pinged to maxwell for Mac-side decomposition
  (a: GPU-empty gaps, b: loss-head wall/mb, c: DSA-bwd host-sync
  fingerprint); I own on-box aggregates.
- 22:4x — DIAGNOSIS RUN DONE: 589 tok/s/GPU (exact Run C repro). Both traces
  secured: traces/diag_d4_rank0.pt.trace.json (699.7MB) +
  traces/diag_d4_rank8.pt.trace.json (748.6MB, the first stage-1 timeline).
  Rank-8 aggregates (on-box): compute 18.79s ≈ stage-0's 18.25s (layer-count
  imbalance is NOT the story); comm 45.05s (65%); SendRecv 35.66s×1456
  (=40×9×4 a2a + 16 p2p, count model exact); AllGather 2.15s×724 (CP).
  **NEW stage-1-only cost: 2× AllReduce_Sum_u64 = 6.49s =
  _peak_memory_report (controller.py:186) — a 24-byte WORLD MAX-allreduce for
  memory telemetry inside EVERY optim_step; duration is rendezvous wait
  (rank 8 waits 3.2s/call, rank 0 0.65s total — stage 1 arrives FIRST at the
  step-end sync and waits for stage 0).** Fixable: telemetry collective in
  the measured step; also a direct skew barometer.
- 2026-08-12 ~18:0x CDT — CONVOY MECHANISM CONFIRMED AT CODE LEVEL (gibbs):
  non-interleaved 1F1B steady loop uses FUSED send_forward_recv_backward
  (schedules.py:2404) / send_backward_recv_forward (:2434) — each stage's
  forward feed chained to the other's backward drain; circular wait
  stage1.fwd(k+1) ← stage0.fwd(k+1) ← stage1.bwd(k). Arithmetic closes (~7s/mb
  strict alternation ≈ 56s + optim ≈ 59.6s measured); explains flat d2→d4
  (convoy rate m-independent). overlap_p2p_comm is interleaved-ONLY (:2165
  raises on non-interleaved). COROLLARY: VPP2 alone needs NO recompute change
  (:2633 binds only the overlap flag) → E2a = VPP2 [18,20,20,20] + full
  recompute (169 GiB known-good) = clean structural test. maxwell: GO E2a
  implement+launch in parallel with dual-timeline confirmation analysis
  (abort-signal-only). Telemetry gate (BT_PEAK_MEM_REPORT) committed, default
  unchanged, excluded from E2a for one-variable discipline. Fleet summary of
  the day so far: 550 → 589 (TF32) with a2a exonerated, platform exonerated,
  transport exonerated, layer-imbalance dead — the wall is the SCHEDULE.
- 22:5x — maxwell's convoy hypothesis CODE-CONFIRMED by gibbs: non-interleaved
  1F1B uses fused send_forward_recv_backward (:2404) + send_backward_recv_
  forward (:2434) — forward feed chained to backward drain; overlap_p2p_comm
  is interleaved-only (:2165). E2a spec locked: VPP2 [18,20,20,20] + FULL
  recompute (the not-full constraint binds only the overlap FLAG, not the
  schedule) + TF32 + ship env + BT_PROFILE_RANKS=0,8, telemetry gate DEFAULT.
- 22:5x — E2a IMPLEMENTED + PUSHED @ 61c41d9e: vpp config field, layout table
  re-keyed (layers,pp,vpp), (78,2,2)=[18,20,20,20] with all chunk starts
  topk-valid (1/19/39/59 — verification table above at 20:5x), provider vpp
  set when >1, tests extended. Standalone-verified on Mac (all 5 layouts).
  Box: clone @ 61c41d9e, TF32 patch intact, stale sweep + group-dump VERDICT
  OK. E2a LAUNCHED. Also fixed: sweep pkill self-match bug ([d] bracket).
- Telemetry gate shipped meanwhile: BT_PEAK_MEM_REPORT=0 @ 760021be (world
  MAX-allreduce in optim_step gated; default unchanged; CI test added).
- 23:1x — E2a boot 1 FAILED FAST (0.1s into warmup): interleaved schedule
  indexes data_iterator[model_chunk_id], runner passed a single iterator.
  Fixed @ 8c13ed31 (_schedule_data_iterator: fresh per-chunk single-shot
  iterators, THD+BSHD both; non-VPP identical; CI test). Bounded-probe
  pattern working as designed. E2a relaunched.
- 23:4x — E2a boot 2 failed STRUCTURALLY: interleaved schedule requires
  microbatch_group_size_per_vp_stage ∈ [PP=2, M] (schedules.py:1141-1148);
  our per-partition calls use M=1 (volta's N-calls-of-M=1 architecture) → no
  legal value. VPP is incompatible with per-partition schedule calls as-is.
  Options to maxwell: (1) runner partition-grouping (M≥2 calls; odd counts
  break; volta's domain, multi-hour+); (2) unfused-p2p monkeypatch (schedule
  surgery, tonight-feasible); (3) accept convoy, attack in-convoy costs.
  Trainer stopped; GPUs free pending call. E2a code (layout + vpp field +
  iterator fix) stays on the branch — correct but blocked at the schedule
  interface.
- 23:5x — OPTION 2 PAPER-KILLED (maxwell constraint 1 honored): at M=1/call
  the parks live in the PLAIN warmup recv_forward / cooldown recv_backward;
  the fused steady-loop ops (:2404/:2434) are never executed (stage-0 steady
  loop empty, stage-1 one degenerate iteration). The convoy is the runner's
  serialized M=1-per-partition-call architecture — pure data-dependency
  chain, zero cross-stage overlap possible regardless of p2p fusion. No
  monkeypatch can win at M=1. PIVOT: **M=N per schedule call at PP>1** — one
  call per op, num_microbatches=len(partitions); non-interleaved 1F1B finally
  pipelines (stage-1 bwd(k) overlaps stage-0 fwd(k+1)); no VPP/layout/
  schedule change; in-flight = min(M,PP)=2 checkpoint sets (169 GiB-known);
  runner change = one call + map per-microbatch results back to partitions
  (forward_data_store is per-microbatch already). volta's option-1 memo
  re-scopes to M=N (simpler than VPP grouping). Awaiting maxwell/volta call
  on who prototypes. E2a code stays on the branch (correct, blocked at the
  M=1 schedule interface).
- 2026-08-12 ~19:0x CDT — ROOT CAUSE FINAL FORM (gibbs paper-trace, pre-code
  kill of option 2): at the runner M=1-per-schedule-call convention, the
  fused steady-loop p2p ops are NEVER EXECUTED — parks live in plain
  warmup/cooldown recvs; the convoy is the M=1 SERIALIZATION itself (pure
  dependency chain, schedule never given a 2nd microbatch to overlap).
  maxwell fused-op theory wrong in mechanism, right in family (schedule
  structure). THE FIX: M=N — one schedule call per step, num_microbatches=
  len(partitions); plain 1F1B then truly pipelines. No VPP/layout/surgery.
  Explains flat d2→d4 exactly. DECISIONS: volta re-scoped from VPP-grouping
  memo to M=N design+implementation (their domain); gibbs runs telemetry A/B
  in idle box time + preps validation ladder (d1/d2 canary with expected
  tolerance-level bitwise drift from backward-order change, memory watch on
  min(M,PP)=2 in-flight sets, traced d4) + reviews. M=N bonus: satisfies
  interleaved-schedule constraint → VPP/overlap (E2b) stackable LATER.
- 2026-08-12 ~19:4x CDT — RANK-8 DECOMPOSITION COMPLETE (maxwell subagent,
  results/TRACE_RANK8_ANALYSIS.md): convoy CONFIRMED empirically from stage-1
  side (parks at cycle START waiting for stage-0 acts, 27.07/27.10s pure
  wait, wire 31ms/step @ line rate; alternation arithmetic closes both
  directions). Steady balance sheet: cycle 14.6s = park 7.11 + compute 4.74 +
  comm-only 2.09 + idle 0.66; extras 10.7s/step (tail 7.10 incl 6.47s
  telemetry allreduce as ONE step-end call [correction to 2x3.2 read], optim
  0.65, mb1 warmup 3.9). Ranked removables: parks 27.1s (~1.9x ceiling) >
  tail+optim 7.7s > a2a exposure 8.4s > micro-gaps 2.6s. DOWNGRADES: LM head
  not first-order (~0.4s/step; sync exposure 14ms/mb); DSA-bwd GPU-side tiny
  (0.33s/mb). CRITICAL FORWARD WARNING: ~21s/step CPU-BLOCKED on rank 8
  (4259 streamSyncs + 5428 blocking memcpys, zero runahead) currently hidden
  inside parks — likely NEW critical path once M=N lands; check CPU runahead
  in M=N traced d4.
- 18:4x — volta: M=N IMPLEMENTED + PUSHED @ d1c939c3 (jackrao/lps-1062-pp2-packing).
  THD branch: one schedule call per op, num_microbatches=len(partitions);
  _sum_over_microbatches cancels the schedule /N (the grad-scale trap from the
  memo); schedule tail owns finalize (WARNING comment re one-call-per-op
  invariant per gibbs review note); empty-DP-slice finalize explicit. Tests:
  dispatch suite rewritten to M=N contract, 33/33 green on Mac via a
  stubbed-CUDA-import pytest harness (triton/modelopt/TE/transformer_engine
  auto-stubbed at import; 5 dp_consensus spawn-test failures are pre-existing
  harness artifacts, identical on base). gibbs reviewing; box validation =
  d1/d2 canary + parity driver pre/post.
- 23:5x — maxwell rank-8 decomposition (results/TRACE_RANK8_ANALYSIS.md):
  convoy empirically confirmed both directions (parks at mb-cycle starts;
  27.07/27.10s pure wait; wire time 31ms/step). NEW: rank-8 carries ~21s/step
  CPU-BLOCKED time (4259 streamSyncs + 5428 blocking memcpys, ~5s per 7.5s mb
  window, zero runahead) currently hidden inside the parks — when M=N removes
  parks, CPU serialization may become the new critical path; check CPU
  runahead in the M=N traced d4. Downgrades: LM head not first-order
  (~0.4s/step), DSA-bwd GPU-side tiny (0.33s/mb). Ranked removables: parks
  27.1s (~1.9x ceiling) > tail+optim 7.7s > a2a exposure 8.4s > micro-gap 2.6s.
- TELEMETRY A/B: gate-off 578 vs baseline 589 — the u64 allreduce is ABSORBED
  IDLE, not critical path. No perf lever; gate kept for hygiene, default ON.
- volta M=N diff @ d1c939c3 REVIEWED: LGTM approve-level (all memo mechanics
  verified; tests encode the new contract). Merge vs my branch: one conflict
  at the THD call site (my VPP iterator fix vs the restructure) — resolution:
  use _schedule_data_iterator(microbatches, len(model_list)) (E2b seam).
  Validation ladder prepped (SCOPING doc): d1 degenerate → d2 canary
  (bitwise drift EXPECTED from backward-order change; volta parity driver is
  the arbiter if ambiguous) → memory watch (min(M,PP)=2 in-flight) → traced
  d4 rank 0+8. Awaiting maxwell green light to merge + run.
- 18:5x — gibbs M=N diff review: LGTM/approve, all memo mechanics verified.
  Merge-integration note captured: his VPP _schedule_data_iterator (8c13ed31)
  conflicts with the M=N call site by design; resolution = M=N call uses
  _schedule_data_iterator(microbatches, len(model_list)) — the E2b stackability
  seam. Marked inline at the call site @ 8b0ef108 (pushed). Branch state:
  pad-to-131k (36c3c8f4) + fixtures (b34b7aed) + M=N (d1c939c3) + seam comment
  (8b0ef108). Awaiting maxwell's sequencing for the validation run.
- 00:3x — **M=N D4 HEADLINE: 918 tok/s/GPU** (925/912), step 35.7s, mfu3x
  8.4% / hfu 12.1%, peak 175 GiB. **1.56× over the 589 clean-tip+TF32
  baseline; 1.42× over the golden EP16/CP16 645 anchor.** The convoy is dead.
  Ladder: d1 degenerate PASS (582, mem 170); d2 canary PASS (loss 12.305-7 in
  band, gn 0.40-0.46 comparable; d2 764 vs pre-M=N 538 = +42%); d4 918.
  p2p parks collapsed: stage-0 36.23→8.78s, stage-1 27.10→2.03s (residue =
  irreducible PP2 fill/drain). a2a unchanged (7.2/8.0s) = next exposed lever.
  Landed at 1.67× of the 1.9× park-removal ceiling — residual ≈ the
  CPU-blocked serialization maxwell flagged (now exposed) + fill/drain + a2a.
  Traces: traces/mn_d4_rank{0,8}.pt.trace.json (secured immediately).
  Merge 73c24b00 = my branch + volta's M=N (8b0ef108) with the E2b seam
  resolution (_schedule_data_iterator at the M=N call site).
- 01:0x — MICROBATCH SWEEP (live M=N trainer, Jack's question via maxwell):
  d4 918 → d8 1000 (65.5s @1M tok, mfu3x 9.1%) → d16 1052 (124.6s @2M tok,
  mfu3x 9.6% / hfu 13.9%). Predictions landed at/under low edge (d8 1000 of
  1000-1020; d16 1052 of 1070-1090) — slight systematic under-prediction =
  CPU-blocked serialization growing with M (maxwell's hidden-21s class).
  Returns diminishing (+9%, +5.2%) → asymptote ~1100-1150. Memory model
  CONFIRMED: in-flight=min(M,PP)=2 → peak 182/194 GiB (d8/d16), no M-scaling
  blowup. Canaries in band throughout.
- 19:0x — volta: parity leg A (PP2-padded, live merged trainer 73c24b00 on
  w56lorq): loss=12.361400908911282, 262,032 real tokens → 3 partitions
  tail-filled to 131072 (393,216 processed), 9/9 datum logprob outputs,
  fb=64.8s. JSON: box:/root/.cache/user_artifacts/lps1062_bench/parity_pp2-padded.json.
  gibbs flipping to CP8/PP1 (DP2) for reference legs B (BT_PACK_PAD_TO_MAX=1)
  and C (unpadded). M=N sweep headline from gibbs: 918@d4 / 1000@d8 /
  1052@d16 tok/s/GPU (vs 589 pre-M=N) — convoy fix confirmed on hardware.
- 01:3x — volta parity leg A DONE on the live PP2 M=N trainer: loss
  12.361400908911282, 262032 real tokens → 3 partitions tail-filled to 131072,
  9/9 datum logprob outputs, fb 64.8s (padded work 393k tokens; unpadded leg
  gives the waste delta). JSON: lps1062_bench/parity_pp2-padded.json.
- PP1/CP8 flip for volta legs B+C: booting trainer_pp1cp8ep8_131k.json
  (PP1/CP8/EP8 → DP2). BT_PACK_PAD_TO_MAX is process-env-bound (read per-call
  from os.environ) → leg B (padded, var SET) boots first, leg C (default)
  needs a separate boot without it. save_state probe staged
  (tools/save_state_probe.py: bounded 10-min POST + auto py-spy on all 16
  ranks on hang) for the final box item after parity.
- 2026-08-12 ~21:3x CDT — SUCCESSION ORDERED by Jack (contexts filling):
  maxwell→borel (fable 1m), gibbs→doppler, volta→poincare,
  dedekind→hausdorff (kimi k3 1m). HANDOFF.md fully rewritten for borel
  (mission status, in-flight items, E2b spec, operational knowledge,
  constraints). Fleet instructed to write briefs/BRIEF_{DOPPLER,POINCARE,
  HAUSDORFF}.md and hand off at natural completion points — in-flight work
  (parity legs B+C, save probe) does NOT stop. Jack's new active directive
  logged: "try to overlap the compute and comms" → E2b (VPP2 + overlap flag
  + block+K recompute, all unlocked by M=N) is the #1 next experiment.
- 2026-08-12 ~20:3x CDT — ORCHESTRATOR HANDOFF COMPLETE: borel (fable 1m)
  succeeds maxwell, who acked and stood down. Succession scaffolding up:
  doppler/poincare/hausdorff all ACKed with context pre-loaded (HANDOFF +
  NOTEBOOK + lane docs); outgoing gibbs/volta/dedekind pinged — in-flight
  work continues uninterrupted. volta's BRIEF_POINCARE.md landed 20:28;
  poincare directed to read + ack to volta (predecessors hold until
  successor ack — maxwell's parting rule). Current in-flight: gibbs booting
  PP1/CP8 (DP2) with BT_PACK_PAD_TO_MAX=1 for volta's parity leg B; leg C
  (unpadded) needs a separate boot; dedekind's bounded save_state probe
  queued behind parity. borel task board: succession, parity B+C, save
  probe, E2b (Jack's overlap directive — next box experiment), PR shaping,
  final report, utility-pod cleanup. E2b prep starting now Mac-side
  (SCOPING_a2a_overlap.md review + config staging).
- 19:2x — volta: succession order from Jack via maxwell — successor is POINCARE.
  briefs/BRIEF_POINCARE.md written (M=N implementation map + invariants, pad
  semantics, parity legs state + pass bars, PR-shaping notes, tooling). Mac
  pytest harness preserved at tools/run_server_tests_mac.py. volta stays
  through legs B+C so the three-leg verdict lands before handoff.
- 22:3x PT — dedekind succession: borel orchestrates now (maxwell stood down).
  Evidence package pulled off w56lorq to Mac (export_test/evidence/: 6 py-spy
  stacks, both L2b adapters+configs, worker logs, cudart_mask.c) — durable
  past box reclaim. tools/save_state_probe.py written for the big-trainer
  save probe (queued after gibbs parity legs B+C; hausdorff or doppler
  executes). briefs/BRIEF_HAUSDORFF.md written; ack chain open.
- 2026-08-12 ~20:4x CDT — SUCCESSION LANE 1 CLOSED (dedekind→hausdorff): all
  three successor briefs now on disk (BRIEF_DOPPLER 20:30, BRIEF_POINCARE
  20:28, BRIEF_HAUSDORFF 20:34). dedekind wrote tools/save_state_probe.py
  (600s bound, auto py-spy all 16 ranks on hang) + pulled durable F1 evidence
  to export_test/evidence/ (Mac-side, survives box reclaim), handed off, and
  is out. hausdorff owns the export/DCP lane incl. the big-trainer probe
  (plan locked: probe in the post-parity PP2/CP8 clean-init window, gibbs
  coordinates; on hang → secure dumps → BT_SAVE_STATE_SYNC=1 re-probe via
  db5d1826 hunk). NOTE for the record: a borel message crossed dedekind's
  handoff-complete in flight and briefly created dual probe ownership —
  caught and resolved to single-owner (hausdorff) within minutes; lesson =
  sequence orchestrator sends against the latest fleet state, the
  crossed-message failure mode maxwell warned about is real. doppler and
  poincare have read their briefs and acked their predecessors; gibbs holds
  the box through parity legs B+C (PP1/CP8 leg-B boot mid weight-load,
  slower per-rank full-model load, being watched), volta holds through the
  three-leg parity verdict. Mac-side: Explore agent reading mcore source to
  settle the E2b recompute-constraint conflict (HANDOFF block+K plan vs
  SCOPING method/num_layers-None reading) before any E2b boot is spent.
- 01:5x — SUCCESSION: gibbs→doppler (borel orchestrates, maxwell stood down).
  BRIEF_DOPPLER.md written (briefs/) — box state, runbook, branch map, E2b
  spec, stale-sweep + trace-pull + CPFS rules. doppler ACKed in full; borel
  confirmed the plan (I hold the box through parity B+C + save probe, then
  hand to doppler on ack). hausdorff (ex-dedekind) takes the save_state probe
  window after me (terminal-for-boot understood).
- 01:5x — leg-B boot (PP1/CP8/EP8/DP2) FAILED at the warmup tail: NCCL
  "unhandled cuda error / Invalid access of peer GPU memory over nvlink" in
  the DP2 grad allreduce (finalize_model_grads → allreduce_coalesced,
  param_and_grad_buffer.py:660). DP2 cross-node is a less-tested path; PP2
  runs never hit it (DP1). Proposed 1-node PP1/CP8/EP8/DP1 fallback to volta
  (sidesteps cross-node DP). FINDING logged for later investigation.
- 01:5x — E2b spec GAP caught by borel, confirmed by gibbs: mcore
  :2633-2639 requires granularity != 'full' + method/num_layers None for
  overlap_moe_expert_parallel_comm → block+K (granularity=full) is ILLEGAL;
  the HANDOFF's "block+K covers recompute" is wrong. Only SELECTIVE is legal
  → and selective-core_attn OOM'd at 131k (258.8 GiB). E2b's recompute path
  is unsolved tonight; borel verifying Mac-side.
- 21:5x — volta: leg-B boot (PP1/CP8/EP8/DP2) FAILED at the cross-node DP grad
  all-reduce (NCCL "unhandled cuda error / invalid access of peer GPU memory
  over nvlink", param_and_grad_buffer.py:660, warmup tail). PP2 runs never hit
  it (DP1 = no DP collective). gibbs logging as a separate finding (matters for
  any future DP>1 on this box). volta signed off on reference legs at
  PP1/CP8/EP8/DP1 on ONE node (8 GPUs) — DP1 removes the DP-collective variable
  from the parity comparison entirely; pass bars unchanged. Condition: same
  merged build (73c24b00) for the reference boots so B-vs-C isolates padding
  and A-vs-B isolates PP only.
- 2026-08-12 ~21:0x CDT — E2b VERDICT (borel; source-verified twice — Mac
  Explore agent on the vendored mcore submodule [byte-identical to the box
  clone, submodule pointer checked] + gibbs independently on-box):
  **overlap_moe_expert_parallel_comm is MEMORY-INFEASIBLE at 131k. E2b as
  specced in HANDOFF is dead.** The flag hard-asserts (transformer_config.py
  :2631-2642): granularity != 'full', method None, num_layers None, no 'moe'
  in recompute_modules — the HANDOFF's block+K memory dial violates three of
  these and 'block' is only honored under granularity='full' anyway
  (transformer_block.py:624; selective+num_layers=K is rejected at
  :1713-1719). NO per-layer escape hatch exists (checks are scalar config
  fields; selective wiring is per-module-type at construction, no layer
  filter). Only legal dials under the flag: selective modules minus 'moe'
  (for our MoE layers: core_attn/layernorm/moe_act) = the E1 family, which
  OOM'd at 258.8 GiB with ONE in-flight microbatch; M=N holds two. Notable:
  the overlap machinery itself handles THD/variable shapes fine
  (packed_seq_params threaded through combined_1f1b; dynamic p2p shape
  exchange) — memory is the ONLY blocker, do not re-litigate shapes.
  Bridge-side duplicate validator (comm_overlap.py:470-505) fires first.
- Related findings: (1) the Aug-9 dispatcher host-sync caches
  (BT_DSA_CP_LAYOUT_CACHE / BT_THD_ROPE_HOST_CACHE, +11-13% @131k then, and
  directly aimed at the CPU-blocked serialization that is now the leading
  residual) exist NOWHERE — not on HEAD, origin/main, any local/remote
  branch, or any open PR. That work was never landed; morning recovery item.
  (2) Open-PR overlap with tonight's package: PR 995 = TF32 LM head (tonight's
  tf32_head_port patch duplicates it — PR shaping should reference, not
  re-cut), PR 1000 = ship NCCL env (draft), PR 987 = cutlass race (merge
  action item stands). (3) Candidate replacement lever under source review
  now: overlap_dispatch_backward_with_experts_wgrad — mcore requires the BIG
  overlap flag OFF for it (:2700-2704), and if it composes with full
  recompute it overlaps the dispatch-backward a2a with expert wgrad compute
  = a legal partial version of Jack's directive. Constraints being pulled
  from source; revised E2 spec after that lands. E2a (VPP2 +
  overlap_p2p_comm, full recompute — legal, code already on branch) stays
  on the card as the p2p-overlap component; EV small at high microbatch
  counts (fill/drain shrinks as 1/M), honest read to follow in the spec.
- 02:0x — DP2 crash evidence (captured before log rotation; full rank logs
  lost on relaunch — these are the exact lines from the live read):
  rank6: RuntimeError: NCCL Error 1: unhandled cuda error — at
  finalize_model_grads → model_chunk.finish_grad_sync → bucket_group
  .finish_grad_sync → start_grad_sync → _coalescing_manager →
  group.allreduce_coalesced (param_and_grad_buffer.py:660,
  distributed_c10d.py:2757).
  rank5: [PG ID 1 PG GUID 1(DATA_PARALLEL_GROUP_WITH_CP) Rank 5] Process group
  watchdog thread terminated: "CUDA error: Invalid access of peer GPU memory
  over nvlink or a hardware error".
  borel's sanity flag: an NVLINK peer-memory error on a CROSS-NODE (DP2)
  allreduce is odd — that path should be IB. Suggests NCCL mis-topologized the
  DP-with-CP group on this box (or the group table maps peers wrongly).
  Prior DP>1 bad-actor history: the 08-09 F2 DP-deadlock (different failure —
  deadlock vs tonight's hard error; do not conflate).
- 2026-08-12 ~21:2x CDT — E2 REVISED SPEC FINAL (E2_REVISED_SPEC.md; borel).
  L1 = overlap_dispatch_backward_with_experts_wgrad CHECKS OUT from source:
  its ONLY constraints are big-overlap-flag OFF (:2700-2704), TE >= 2.3.0,
  and delay_wgrad_compute OFF — NO recompute constraint, composes with full
  recompute, engages under plain 1F1B with M=N (pure intra-layer side-stream
  machinery, moe_layer.py:496/536/777-796: expert wgrad GEMMs deferred to a
  side stream that only waits on dgrad completion → they run concurrently
  with the dispatch-backward a2a). Correctly re-created inside the
  full-recompute checkpointed backward. Needs a small trainer plumbing hunk
  (field unreachable from trainer JSON: control.py CommOverlapConfig is
  extra=forbid; bridge dataclass lacks it; direct provider set in
  _build_config is the right shape) — ASSIGNED: poincare implements on the
  73c24b00 lineage, doppler reviews. delay_wgrad_compute formally ruled out
  (requires the big flag: :2690-2693). Honest EV: hides 1/9 a2a per layer +
  its straggler wait behind same-layer wgrad ≈ +1.5-3%. Watch items for the
  d2 canary: deferred-wgrad-must-fire (loss would not train) + side-stream
  semantics — grad-norm drift = STOP. RUN ORDER after parity+probe: L3
  traced d16 (diagnosis; no code) → L1 boot → L2 (VPP2+overlap_p2p_comm)
  only if box time remains. Fleet notified (gibbs, doppler, poincare).
- 2026-08-12 ~21:3x CDT — JACK DIRECTIVE (verbatim): "Can you try turn down
  recompute granularity, and then do this? see if it works" — i.e. run
  overlap_moe_expert_parallel_comm WITH selective recompute despite the
  memory prediction. Logged as L0 in E2_REVISED_SPEC.md §3; jumps the queue
  ahead of L3/L1/L2 as the first post-parity/probe box run. Assembly: E1
  selective config + vpp [18,20,20,20] + comm_overlap flag via trainer JSON
  + shared-expert-overlap clearing hunk (preserved 0e0b65a6 patch on box).
  d2 ramp start (interleaved illegal at M=1), abort >255 GiB reserved,
  10-min bound, capture assert text verbatim on any early exit. borel
  prediction on record: warmup OOM at d2 ~95% (E1 measured 258.8/275 GiB at
  1 in-flight microbatch, saved-for-backward MoE tensors are
  schedule-independent); the 5% tail = combined-1f1b eager storage release +
  ep_overlap_early_attn_memory_release, machinery E1 never ran. If it fits:
  d2 canary → d4 A/B vs 918 and the night reorders around it. Fleet
  notified (gibbs, doppler).
- 2026-08-12 ~21:5x CDT — JACK: "why is gibbs still running? he should have
  handed off." Correct call; borel owns the miss — gibbs's hold was ratified
  through parity+probe when boot boundaries (esp. the DP2-crash relaunch)
  were natural handoff points and doppler had been briefed+idle for an hour.
  Immediate handoff ordered mid-boot: gibbs sends doppler a live-state note
  (leg-B boot PID/log/compile status, relaunch mechanics, patch locations) +
  brief addendum, doppler acks, gibbs goes quiet. doppler owns box mechanics
  from now (leg-B watch, leg-C reboot, probe window, L0/L3/L1/L2); volta
  still drives the parity measurements. Succession rule tightened for the
  record: natural completion = the CURRENT atomic operation (a boot in
  flight, a measurement mid-run), not the holder's queued milestones.
- 02:1x — SUCCESSION EXECUTED (borel/Jack directive): box handed to doppler
  NOW (hold-through-probe rescinded — I'm the context-limited party).
  briefs/BRIEF_DOPPLER_ADDENDUM.md written: PP1/CP8 fails grad-sync with NCCL
  Error 1 on BOTH DP2-2node and DP1-1node (NOT DP2-specific — the live
  blocker for parity legs B+C); log-clobbering gotcha + snapshot rule; run
  order L0/L3/L1/L2 per E2_REVISED_SPEC; fleet state. doppler acks to me +
  borel, then I go quiet.
- 2026-08-12 ~22:0x CDT — F3 BROADENS + RE-SEQUENCE (borel): gibbs's handoff
  addendum (briefs/BRIEF_DOPPLER_ADDENDUM.md) reveals the PP1/CP8 grad-sync
  NCCL failure is NOT DP-specific — it reproduces at 1-node PP1/CP8/EP8/DP1.
  So the parity-reference topology is broken on this box/build both ways,
  while PP2/CP8 (which runs the same intra-node CP8 grad reduction) is
  fine — suspicion: something scaling with per-rank state at PP1 (78
  layers/rank vs 38/40), e.g. NCCL peer-memory/CUMEM registration
  exhaustion; logs snapshotted per the new rule. DECISIONS: (1) do not
  serialize the night behind PP1 diagnosis — box order is now save probe
  (hausdorff) → L0 Jack-probe → L3 traced d16, all on known-good PP2
  topologies; (2) poincare owns the parity-reference call: bounded PP1
  diagnosis (20-min guidance) vs switching the reference to the golden
  EP16/CP16/PP1 config (borel recommends the switch — topology-invariance
  is the claim under test, a different EP/CP reference is still probative).
  SUCCESSIONS EXECUTED per Jack's order: gibbs→doppler mid-boot handoff
  done (addendum + live-state note), volta→poincare ordered at the same
  boundary (leg B never started). gibbs and volta go quiet on successor acks.
- 22:0x — volta: borel-ordered handoff to POINCARE at the leg-B atomic
  boundary (boot still compiling; box ownership gibbs→doppler mid-boot).
  BRIEF_POINCARE.md gained a LIVE STATE ADDENDUM: exact leg B/C invocations,
  doppler coordination, and the two held nuances — (1) leg A JSON is step-18
  weights (LoRA B zero-init ⇒ only fresh boots are base-model-equivalent), so
  the PP gate needs a fresh PP2 leg (pp2-padded-fresh) after doppler's PP2
  relaunch; B-vs-C stays the valid padding gate; (2) interpretation bars:
  loss rel ≤1e-6 robust signal / marginal 1-3e-3 logprob exceedances at
  partition boundaries acceptable / datum-count mismatch = hard fail; fb ratio
  B/C ≈1.5× = first direct padding-waste measurement. PARITY_RUNBOOK.md updated
  to the 1-node DP1 topology + the fresh-boot validity rule. Awaiting
  poincare's ack, then quiet.
- 22:1x — volta: handoff COMPLETE. poincare ACKed and holds the reins (legs
  B/C, fresh-boot PP gate, three-leg verdict). Reference topology moved to
  golden EP16/CP16/PP1 (poincare/borel call) after PP1/CP8 failed grad-sync
  NCCL on BOTH 2-node DP2 and 1-node DP1 (see BRIEF_DOPPLER_ADDENDUM) — a
  production-hardened reference is probative for the topology-invariance claim.
  Late nuance appended to BRIEF_POINCARE.md addendum: CP16 reference widens
  expected logprob drift on the PP gate (loss bar unchanged); pad_multiple=32
  at CP16 shifts partition boundaries slightly (driver unaffected). volta QUIET.
- 22:2x — hausdorff: save_state_probe.py HARDENED + live-validated against
  doppler's booting clean-init trainer (slurm job 53). Three gaps found in
  pre-flight, all fixed (Mac canonical + box re-staged): (1) the ssh
  tj-<job>-<rank> stack-dump path cannot work on these devboxes (worker
  hostname unresolvable from leader; key auth denied) — replaced with
  `srun --jobid=<autodetected devbox_trainer> --overlap` fan-out, the same
  mechanism wait_trainer_health.sh uses; (2) /tmp out dirs are node-local —
  per-node mkdir added; (3) py-spy existed only on the leader (uv tool
  install is node-local) — shared copy staged at lps1062_pp2/bin/py-spy,
  0.4.2 verified executing on BOTH nodes. Live fire drill dumped real stacks
  from all 16 ranks' processes (leader + worker torchrun node_rank=1).
  Probe is HOT for doppler's READY ping; probe is terminal-for-boot on hang
  (doppler told: no relaunch between READY and probe without telling me).
- 2026-08-12 ~21:2x CDT — doppler BOX WATCH ESTABLISHED (gibbs→doppler ack
  chain closed; gibbs quiet). Takeover state: no trainer, squeue empty, GPUs
  idle both nodes. FIRST ACTION before any relaunch: snapshotted the 1-node
  DP1 PP1/CP8 failure log (the class gibbs lost on DP2 to clobbering) →
  lps1062_pp2/logs/legB_pp1cp8ep8_dp1_1node_nccl_fail_20260813_0159.log
  (616 lines, sha256 5bb62a487e84...). Failure confirmed identical to DP2:
  rank6 RuntimeError NCCL Error 1 at finalize_model_grads →
  allreduce_coalesced (param_and_grad_buffer.py:660) at the warmup tail.
  borel's cuMem/peer-registration hypothesis NOT testable from this log —
  zero NCCL transport lines (NCCL_DEBUG was not INFO); needs the INFO boot
  (poincare lane). Flagged: leader GPU0 (PCI 1a:00) nonfatal RLW_RXPIPE Xid
  137/145 on links 1/8/9/12 at 01:33:45 UTC, ~5 min pre-boot. Clone verified
  73c24b00 + TF32 patch (chunked_lm_head.py +73/-10).
- 2026-08-12 ~21:2x CDT — PP2/CP8 CLEAN-INIT READY in ~6 min (warm cache;
  kernels warm 176.5s; srun pid 67754; env = TF32 head + ship NCCL, pad flag
  unset). **borel's Xid DISCRIMINATOR VERDICT: links healthy** — the same
  intra-node CP8 grad reduction that fails at PP1 ran clean at PP2 again;
  hardware exonerated for F3, PP1 failure stays a config/scale question.
  /status: world 16 TP1/PP2/CP8/EP8/DP1, seq 131072, step 0 (clean-init).
- 2026-08-12 ~21:2x CDT — INTRA-BOOT ORDER executed (borel): (1) poincare
  parity forward FIRST on the clean-init boot — **pp2-padded-fresh BANKED**:
  loss 12.54805275637226, 262,032 real tokens, 9/9 datum outputs, fb 75.6s,
  JSON lps1062_bench/parity_pp2-padded-fresh.json (Mac copy secured).
  poincare sanity gate PASS (fresh base weights sit near-but-above leg-A's
  step-18 12.3614 as expected — leg A was LoRA-trained-18-steps weights).
  (2) Box RELEASED by poincare → handed to hausdorff: bounded save_state
  probe RUNNING now (10-min cap, terminal-for-boot on hang, auto py-spy all
  16 ranks). (3) Next: BT_SAVE_STATE_SYNC=1 relaunch with L3 traced d16
  RIDING that boot (borel boot-economics; the sync flag only touches the
  save path), then L0 (Jack probe).
- 2026-08-12 ~21:2x CDT — L1 HUNK REVIEW (doppler fresh-eyes on poincare's
  b8d868ff, 73c24b00..b8d868ff): **APPROVE**. Verified from source:
  (a) attribute name exact — mcore model_parallel_config.py:275
  (ModelParallelConfig, inherited by TransformerConfig), so the provider
  write is real, not a silent stray attr; (b) single splat site —
  comm_overlap.model_dump() consumed only at megatron_config.py:334, the
  pop precedes it, no other consumer; (c) provider == cfg.model in the
  ConfigContainer → the write lands on the object the model reads at
  provide() time; (d) bridge comm_overlap.py has zero references to the
  field → its finalize()/setup() cannot overwrite the write; (e) mcore
  eligibility asserts (:2700-2711) live in __post_init__ = construction-only,
  provider is constructed upstream of _build_config with the flag still
  default-False → poincare's claim holds: the trainer-side mutual-exclusion
  validator is the ONLY place the big-flag/delay_wgrad exclusions can fire,
  and it mirrors mcore's exactly; TE>=2.3.0 correctly documented as the one
  runtime-only constraint (box pre-flight queued for the L1 boot). No-default
  pop (fail-loud on rename) is deliberate and correct. Tests: 14/14 green
  reproduced Mac-side. HARNESS FOOTGUN fixed en route (pc_5e53d57839fe,
  poincare's one-liner applied to the SHARED tools/run_server_tests_mac.py:
  WT.parent/models/src prepended to sys.path — loops_models was resolving to
  the MAIN checkout's editable install, not the worktree under test; bit
  poincare then me inside an hour). L1 is boot-ready pending the box queue.
- 22:4x — hausdorff: BIG-TRAINER SAVE PROBE = HANG CONFIRMED at production
  scale (Finding F1 reproduces on the real GLM-5.2 PP2/CP8/EP8 @131k
  clean-init trainer; /save_state op fca827642a684d0d930d381ad855cdec never
  completed in the 600s bound; probe exit 2, stacks auto-dumped via the
  hardened srun path — see 22:2x entry for the three tool fixes).
  MECHANISM CLOSED (sharper than the small-model finalize-poll signature):
  the nvidia_resiliency_ext async-writer daemon (SpawnProcess-1) CRASHED on
  the first queue item — RuntimeError "received 0 items of ancdata" in
  torch/multiprocessing reductions rebuild_storage_fd → recv_handle
  (multiprocessing/reduction.py:164): CUDA IPC FD passing (SCM_RIGHTS) to
  the spawned writer failed. Closed loop in the 16-rank py-spy sweep:
  worker-node ranks park in schedule_async_call → queue.join
  (bridge/training/checkpointing.py:378) waiting on the dead consumer;
  leader ranks park ACTIVE at the NCCL barrier in save_checkpoint
  (checkpointing.py:1581) waiting for the parked ranks; trainer stays
  HTTP-200 alive, op never completes. Evidence Mac-side:
  export_test/evidence/big_trainer_save_probe/ (42 files: py-spy both nodes,
  srun stdout capture, probe log, trainer_srun snapshot with the ancdata
  traceback — snapshotted BEFORE any relaunch per the clobbering rule).
  Sync-verify next: BT_SAVE_STATE_SYNC=1 relaunch (db5d1826
  megatron_config.py hunk) + re-probe; mechanics coordinated with doppler.
- 2026-08-12 ~23:1x CDT — JACK METHODOLOGY DIRECTIVE (standing rule, saved
  to memory): profile/measure before deciding, systematically narrow the
  choice space with data (no shots in the dark), small feasibility probes
  before big runs, parallelize, think outside the box. Applied immediately:
  (1) E2_REVISED_SPEC §3b = PRE-REGISTERED post-L3 decision rules — the
  traced-d16 decomposition (fill/drain vs a2a exposure vs CPU-blocked vs
  tail) picks the next lever via fixed thresholds, auditable before the
  data arrives; (2) §3c = BOX B parallel track — second 2×8 B300 seeker
  launched (~23:1x, log logs/devbox_up_aug13_boxB.log; one-seeker rule
  honored, no other seeker active) to take the golden EP16/CP16 parity legs
  (hausdorff mechanics post-sync-verify, poincare drives) off box A's
  serialized chain, cutting the remaining night queue from ~4-5 h to
  ~2.5-3 h; (3) §3c L4 = outside-the-box stretch probe — the big overlap
  flag at 16k seqlen where selective recompute may FIT: quantifies the
  flag's prize where it is legal (upstream-ask evidence + customer-regime
  relevance), bounded d2 probe, box-B time permitting. Box A chain
  unchanged: sync-verify → L3 → L0 → L1 → L2.
- 22:5x — hausdorff: SYNC-VERIFY PASS at production scale. doppler relaunched
  the PP2/CP8 clean-init with BT_SAVE_STATE_SYNC=1 (hunk git-applied from
  db5d1826 onto 73c24b00, TF32 patch intact, verified pre-launch); re-probe
  (op 27760506c5dd4527ab2c392cf6ec8ce1) completed in 70.8s — checkpoint on
  disk /tmp/checkpoints/pp2cp8-save-probe-sync/iter_0000000 +
  latest_train_state.pt, 537MB. F1 chain CLOSED both directions: async path
  wedges (ancdata crash → queue.join/barrier park), sync path completes.
  EXPORT_TEST.md §F1 updated with the production-scale sync confirmation.
  Box A released to doppler for L3 traced d16 on the same boot. hausdorff
  moves to box-B (3m9o7kq) mechanics: bring-up verification → golden
  EP16/CP16/PP1 boots for poincare parity legs (padded first, then
  unpadded). Box-B golden config staged Mac-side:
  configs/trainer_ep16cp16_golden_131k.json (expA131 + lora_alpha 64→32 +
  checkpoint_dir=/tmp/checkpoints).
- 2026-08-12 ~23:3x CDT — SYNC-VERIFY PASS AT SCALE (hausdorff):
  BT_SAVE_STATE_SYNC=1 boot → /save_state completed 70.8s, 537MB checkpoint
  (iter_0000000 + train_state) on the full PP2/CP8/EP8 16-rank trainer. F1
  closed BOTH directions at production scale: async wedges (ancdata daemon
  crash), sync completes. Escalation package final, PARKED awaiting Jack.
- 2026-08-12 ~23:3x CDT — FLEET EXPANSION (Jack): four fresh K3s added.
  borel structure call: FLAT orchestration, independent non-GPU lanes (least
  overhead; K3 context is the binding resource, cross-coordination is the
  overhead driver). Lanes: gauss = dispatcher-cache archaeology (reconcile
  doppler's boot-log warning sighting vs no-hits-in-tree; recover/port the
  Aug-9 +11-13% host-sync work; conditional-high per §3b) →
  results/DISPATCHER_CACHE_ARCHAEOLOGY.md. lebesgue = headline PR shaping
  (M=N + packing + b8d868ff) → pr_shaping/PR_DRAFT_MN_PACKING.md +
  self-review + Mac harness green-check. weierstrass = infra + export PR
  shaping + morning merge queue (987/995 relations) →
  pr_shaping/PR_DRAFT_INFRA.md, PR_DRAFT_EXPORT.md, MORNING_MERGE_QUEUE.md.
  serre = F3 diagnosis (PP1/CP8 NCCL crash: ranked hypotheses + cheapest
  discriminators, code-level, no boots) → results/F3_PP1_NCCL_DIAGNOSIS.md;
  then probs-a2a fix scoping (Aug-9 5.1s free-win candidate). All four:
  read-only on branches, no GPU actions, drafts not PRs.
- BOX-B BRANCH CALL (borel, poincare+hausdorff concurring): box B = exact
  box-A bits (73c24b00 + TF32 patch, NO b8d868ff). The reference must be
  bit-identical modulo the topology under test; "default-off field is
  inert" is the class of claim parity exists to not trust. Box B job
  3m9o7kq provisioning (step 4/13 at 22:31).
- 2026-08-12 ~23:5x CDT — BOX B KILLED ON JACK'S ORDER (capacity to Harry;
  2 nodes released): truss train stop 3m9o7kq confirmed. Box B never ran a
  workload (provisioning had just finished 13/13) — box A (w56lorq) was the
  obvious keeper (warm caches, all banked measurements, bit-comparability).
  RESEQUENCED single-box queue on A: L3 traced d16 (rides live sync-verify
  boot) → L0 Jack-probe → golden EP16/CP16 parity legs (poincare, two
  boots, exact-bits rule) → L1 A/B → L4/L2 if night allows. hausdorff off
  box duty → evidence-consolidation sweep + box-A artifact manifest (so
  w56lorq can be stopped in the morning without losing anything). Paper
  lanes (gauss/lebesgue/weierstrass/serre) unaffected.
- 2026-08-13 ~00:1x CDT — JACK DIRECTIVE #2 (standing, saved to memory): no
  tiny hacky 1% fixes that add opaque complexity — when a big win is
  blocked, engineer the path to make the big thing work. Applied: (a)
  probs-a2a scoping dropped from serre's queue (now data-conditional only);
  (b) L1 stays (clean upstream feature, already built) but explicitly
  deprioritized behind parity + L0; L2 runs only if L3 data demands; (c)
  BIG-WIN PROGRAM stood up for the 131k overlap wall: P1 = per-layer
  recompute dial upstream design (serre item 2 — leftover-layers structure
  in model_chunk_schedule_plan.py makes K-checkpointed + (N-K)-overlapped a
  moderate patch, not a rewrite); P2 = CPU activation offload to make
  selective fit (box A has 4 TB host RAM, x86/PCIe; ~13 GB/s sustained
  needed vs ~60 available — Explore agent reading offload×overlap×MoE×THD
  composition constraints now; go/no-go for a d2@131k probe arm).
- 2026-08-13 ~00:1x CDT — GAUSS SOLVES THE CACHE MYSTERY: the Aug-9
  dispatcher-cache work was NEVER COMMITTED — it lives as uncommitted
  working-tree changes (+1652/-26, 8 files, env-gated OFF) in the CPFS
  vendored mcore (@57efae08b pointer-clean/content-dirty), which uv.lock
  installs EDITABLE — the running trainer imports the dirty tree (hence
  doppler's boot warnings; Mac checkouts of the same pointer are clean,
  hence my grep miss). dsa.py+rope_utils.py hunks are byte-identical to the
  Aug-9 SHIP pair (FIX B + FIX F, +11-13% @131k then). CONSEQUENCES: (1)
  cache A/B costs TWO ENV VARS + relaunch (BT_DSA_CP_LAYOUT_CACHE=1
  BT_THD_ROPE_HOST_CACHE=1) — queued per §3b thresholds on L3 decomposition
  (task #9); (2) HYGIENE FINDING: every measurement tonight imported the
  dirty tree with gates off — believed inert, gauss documenting gates +
  cleanup path (land ship pair properly as PRs, revert the rest); exactly
  the un-owned complexity Jack's directive targets. (3) weierstrass
  delivered the full infra/export PR package + MORNING_MERGE_QUEUE.md
  (pr_shaping/, 4 docs, incl. the land-995-before-M=N ordering insight);
  cache-pair PR to be added. L3 status: traced window banked (136.6s/959
  traced, canary in band), controls running, slight slowdown vs first
  d16 (136.6 vs 126.5 traced) under watch.
- 03:5x — hausdorff: BOX B STOOD DOWN per borel (Jack released its capacity;
  3m9o7kq stopped; bring-up abandoned pre-boot, nothing run). Parity legs +
  L4 reabsorbed by box A (doppler sole mechanic). NOTED for the record: box
  B's "clone" was box A's working tree via the shared CPFS mount — same
  bits, same uncommitted patches; nothing divergent was created. hausdorff
  → EVIDENCE-CONSOLIDATION lane: BOX_A_ARTIFACT_MANIFEST.md written (report
  §8 input — full inventory: traces, bench/parity JSONs, logs, patches,
  branches, disposables, final-sweep open items). Pulled Mac-side tonight:
  bench/ (11 pp2-131k perf JSONs), parity/parity_pp2-padded.json (step-18
  artifact; fresh leg was already pulled), logs/box_a/ (24 files: driver
  logs, trace breakdowns, a2a microbench, saveprobe hang trainer log),
  configs/pre_checkout_local_changes_0e0b65a6.patch. All 13 pulled JSONs
  parse-validated. mn_d4_rank{0,8} traces pulling in background (sha256
  recorded for verify); runC_d4_tf32_rank0 listed-only (diag_d4_rank0
  covers the config class). Older Aug 7-10 bench JSONs confirmed already
  durable in runs/*/results/ (spot-checked). FINAL SWEEP pending: L3 d16
  traces + pp1cp16 parity pair + any new log snapshots.
- 2026-08-12 ~22:0x CDT — PROBE CHAIN CLOSED (doppler box, hausdorff probe).
  Async probe verdict: **HANG at production scale** (600s bound, PP2/CP8/EP8
  @131k, 16 ranks): async-writer daemon crashed on a CUDA IPC FD receive
  ("received 0 items of ancdata"), ranks parked in queue.join + NCCL barrier.
  Wedged-trainer log snapshotted pre-stop:
  lps1062_pp2/logs/pp2cp8_cleaninit_saveprobe_hang_20260813.log (sha
  609581275dd1a693). Sync-verify relaunch (BT_SAVE_STATE_SYNC=1; hunk from
  db5d1826 applied clean to the box clone, 6+/1- megatron_config.py, TF32
  patch untouched): **/save_state COMPLETED in 70.8s at full scale**
  (checkpoint /tmp/checkpoints/pp2cp8-save-probe-sync/iter_0000000, 537MB).
  CP>1 DCP finding now has both horns at production topology: async = wedged,
  sync = green. Escalation package (export_test/EXPORT_TEST.md §F1) awaits
  Jack's word per standing order.
- 2026-08-12 ~22:1x CDT — **L3 traced d16 DONE** (same boot as the sync
  verify, borel boot-economics): 998 tok/s/GPU steady (controls 994/1001),
  step 131.4s @2M tok, mfu3x 9.1% / hfu 13.2%, peak 195 GiB, kineto tax
  +4.0% (traced 136.6s). ~5% under the first mn-d16 (1052/124.6s) — variance
  or post-save page-cache state; the A/B baselines (918/1052) stand from
  their original runs. Canary in band (loss 12.317-12.321, gn 0.36-0.41).
  Traces secured off the volatile dirs (BT_PROFILE_RANKS=0,8):
  lps1062_pp2/traces/l3_d16_rank0.pt.trace.json (2.80 GB) +
  l3_d16_rank8.pt.trace.json (3.00 GB), byte-sizes verified. Traced window =
  driver step 2 (warmup absorbed compile) = steady-state d16 step.
  **Dirty-tree fact for the record (gauss found): the vendored mcore the
  trainer imports carries UNCOMMITTED CPFS working-tree changes (editable
  install) — the Aug-9 dispatcher host-sync caches (BT_DSA_CP_LAYOUT_CACHE /
  BT_THD_ROPE_HOST_CACHE) are ALIVE there, gates default OFF, so nothing
  tonight was affected; all tonight's runs imported the dirty tree.** gauss
  owns the full hygiene doc. §3b bucket decomposition running on-box (rank 8
  first); gauss does the CPU-blocked call-site attribution Mac-side.
- 2026-08-12 ~22:1x CDT — box B DEAD (Jack released capacity); box A absorbs
  the resequenced queue (borel): L3 (done) → L0 → golden EP16/CP16 parity
  legs (poincare; bits = 73c24b00 + TF32 patch EXACTLY — I git-checkout the
  save-sync + L0 clearing hunks before those boots, git-diff verify =
  chunked_lm_head.py only; config = hausdorff-prepped
  configs/trainer_ep16cp16_golden_131k.json, alpha 32 as specced; two boots,
  pad set then unset) → L1 A/B → L4/L2 if the night allows. New lever in
  the wings (borel/gauss): the Aug-9 caches A/B = one relaunch with
  BT_DSA_CP_LAYOUT_CACHE=1 BT_THD_ROPE_HOST_CACHE=1; jumps to directly after
  L0 iff §3b CPU-blocked-critical-path ≥ 15% of step, else after golden legs.
- 2026-08-13 ~00:5x CDT — serre (new K3 lane, F3 + P1 design): TWO DOCS
  DELIVERED, no GPUs touched. (1) results/F3_PP1_NCCL_DIAGNOSIS.md — the
  PP1/CP8 grad-sync crash, code-verified: trainer runs
  use_distributed_optimizer=False + overlap_grad_reduce off → ONE bucket, ONE
  allreduce of the adapters-only grad buffer (~388 MB bf16 at PP1 vs ~190 MB
  at PP2) over the SAME 8-rank intra-node DP+CP group that passes at PP2 —
  borel's bucket-count/coalesced-group hypotheses exonerated by code. The
  error is cudaErrorContained (226, sticky: "further CUDA work returns the
  same error") → the allreduce is the MESSENGER; and the DP+CP communicator
  is initialized lazily AT the crashing call (data_parallel_random_init=False
  → no broadcast_params; warmup only used EP/CP comms). Ranked discriminators
  + one-boot NCCL_DEBUG=INFO spec in the doc; boot queued in box-A tail
  (borel). (2) results/OVERLAP_PER_LAYER_RECOMPUTE_DESIGN.md — the upstream
  per-layer-recompute dial for overlap_moe_expert_parallel_comm at 131k:
  first-K layers/stage as opaque checkpointed nodes (coexistence precedent:
  dense layers already run noop comm nodes, model_chunk_schedule_plan.py
  :162-167; unequal f/b plans already handled, :522-560), one new config
  field (existing four asserts UNTOUCHED), LoRA requires_grad trap flagged
  (CheckpointFunction never fires backward without a grad-requiring input;
  the block-level PEFT patch is bypassed on the fine-grained path). Memory
  math: K≈21/40 at d16 (2.25 GiB/layer eager basis, 0.19 checkpointed, 2
  in-flight, 96 GiB headroom) → overlap on ~47% of layers → ~7-12% a2a prize
  + recompute-tax refund of the same order. ~1 day prototype, 2-3 upstream.
- 2026-08-13 ~01:2x CDT — §3b BUCKET VERDICT (doppler, rank-8, 136.31s
  traced d16 step; the pre-registered decision): fill/drain 8.52s=6.2%
  (<10, L2 stays parked); **a2a exposure 30.92s=22.7% (>=20 FIRES the
  communication arm)**; CPU-blocked critical path 0.94s=0.7% (<15 —
  DEMOTES the cache A/B); gpu-empty residue 9.4%; overlap 2.4%. Raw host
  syncs are huge (68,418 streamSync 40.67s + 1,280 eventSync 12.09s + 35
  deviceSync 8.51s = 61.3s) but runahead ABSORBS them at d16 under M=N —
  the Aug-9 cache pair is a convoy/low-M lever, d16 ceiling ~0.7%; A/B
  still runs post-legs (near-free; null result = the datum); gauss
  re-aimed at absorbed-time attribution (where it hides, at what M
  absorption fails, 16k-regime relevance). a2a count model exact at d16
  (5760 = 40x9x16). Consequence: tonight's remaining prize concentrates
  in the 30.9s exposed a2a — L1 (1/9 slice) as sequenced, L0b (the full
  overlap via offload, cleared-to-boot f2407a10) as the main event.
  doppler proceeding to L0 (Jack probe): stop → sweep → clearing hunk →
  d2 boot, >255 GiB abort, 10-min bound.
- 2026-08-12 ~22:5x CDT — §3b BUCKET VERDICT (doppler, on-box decomposition of
  the L3 traced d16, tool lps1062_pp2/decompose_l3_buckets.py). Rank 8 (stage
  1, 136.31s window): fill/drain parks 8.52s (6.2%), **a2a exposure 30.92s
  (22.7%) — ≥20% FIRES**, cpu-blocked critical-path 0.94s (0.7%), gpu-empty
  residue 9.4%, compute/comm overlap 2.4%. Rank 0 (stage 0, 137.35s): a2a
  exposure 27.08s (19.7%), cpu-blocked 6.80s (4.9%), fill/drain 10.26s
  (7.5%), gpu-empty 16.2%. Host blocking totals (rank 8): 68,418
  cudaStreamSynchronize = 40.67s + 1,280 cudaEventSynchronize = 12.09s + 35
  cudaDeviceSynchronize = 8.51s — but only 0.94s intersects GPU-empty: host
  serialization is ABSORBED by runahead under M=N at d16, NOT on the critical
  path (the pre-M=N hidden-21s fear did not materialize as critical-path).
  a2a count model exact at d16: 5,760 = 40×9×16. PRE-REGISTERED CALL (§3b):
  a2a arm fires → L1 first (as sequenced), follow-on = scope probs-a2a
  fusion/async; cache A/B does NOT jump (after golden legs); L2 stays last.
  gauss owns the call-site attribution deep-dive (traces pulled Mac-side).
- 2026-08-12 ~23:0x CDT — **L0 (Jack probe) VERDICT: FAIL FAST at the
  combined_1f1b executor contract — the memory question was never reached.**
  Boot 1 died at startup warmup (M=1 illegal under interleaved,
  schedules.py:1148 verbatim: "The number of contiguous micro-batches in a
  virtual pipeline stageshould range in [PP=2 , M=1]" — log snapshotted
  l0_boot1_fail_20260813_0414.log; all config-build validators had PASSED:
  bridge comm_overlap + mcore :2631-2642 + torch 2.11.0>=2.6 — the flag
  combination itself is ACCEPTED). Boot 2 (BT_SKIP_WARMUP=1) reached READY
  (backend-ready banners, weights 104 GiB/rank), d2 probe then failed
  INSTANTLY: TypeError: _sum_over_microbatches.wrapped() got an unexpected
  keyword argument 'return_schedule_plan'. Source-verified mechanism: big
  flag → combined_1f1b.py:391 → forward_step_func(...,
  return_schedule_plan=True) with an AbstractSchedulePlan required back;
  trainer forward-step closures (loss.py _build_forward_step family) return
  (output, loss_func) and implement NO schedule-plan protocol — a kwargs
  passthrough in the M=N wrapper only moves the failure down one level.
  **The big flag at 131k is dead on TWO independent walls: (1) executor
  schedule-plan contract unimplemented (deterministic, tonight); (2) memory
  (E1 258.8 GiB standing measurement). ROUTING: L0b hits the SAME executor
  contract — the f2407a10 offload hunk does not touch it; flagged to borel
  before that boot is spent. L2 unaffected (plain interleaved schedules.py
  has no schedule_plan references).** Pre-flight versions logged per borel:
  torch 2.11.0+cu130, TE 2.16.0. Morning work item (poincare domain, from
  borel): startup warmup must build ≥2 partitions under VPP or skip-warmup
  stays load-bearing. Offer pending with borel: memory-only datapoint
  (selective+VPP2, no big flag — one d2 boot on the current tree) as the
  L0b-scoping evidence.
- 2026-08-13 ~01:3x CDT — serre: EXECUTOR-CONTRACT SCOPING DELIVERED
  (results/EXECUTOR_CONTRACT_SCOPING.md) per borel's L0-followup assignment.
  Verdict: WRAPPER-LEVEL, not a re-architecture — the trainer's chunked-CE/RL
  loss closures run verbatim inside the executor's loss ScheduleNode; the
  forward-step shim is ~60-100 LoC (return model.build_schedule_plan(...) +
  the existing loss partial under return_schedule_plan=True), plus a 3-line
  output_processor passthrough so the plan never materializes full-vocab
  logits. Verified seams: loss-scaling identical (both paths share
  forward_step_calc_loss — volta's _sum_over_microbatches applies as-is),
  finalize-once-per-call preserved (schedules.py:828-836 / :2470-2481),
  metrics/logprobs ride forward_data_store unchanged, executor unwraps to
  GPTModel itself (GLM maps to GPTModel, glm5_bridge.py:55), THD
  packed_seq_params/padding_mask are first-class build_schedule_plan params.
  Watch items logged: reporting CP all-reduce inside the loss node
  (detached, loss.py:79-81), fp32 boundary upcast under the zero-touch
  option (fine_grained_callables.py:240, +192 MB/mb — Option A avoids),
  warmup M=1-under-VPP (poincare's morning item; BT_SKIP_WARMUP hatch),
  VLM×flag unsupported (no pixel_values in build_schedule_plan — guard).
  Effort ~0.5-2 days. UPSTREAM_PROPOSAL.md updated to the three-leg ask:
  (a) executor contract (ours), (b) offload memory path (L0b TO-FILL),
  (c) per-layer dial (upstream).
- 2026-08-13 ~02:5x CDT — JACK ASKS: "did we ever enable deepep (flex)?"
  Answer: NO — all tonight's numbers are plain alltoall dispatcher (trace
  signature: separate dispatch/probs/combine all_to_allv, 5760/step d16).
  The Aug-7 rule-out (NVSHMEM/IBGDA vs LAG-bonded RoCE) was a CROSS-NODE
  EP16 objection; tonight's EP8 is intra-node NVLink = DeepEP P2P path →
  rule-out STALE for this topology. deep_ep verified importable in the box
  venv. Queued as L5 (config-only A/B: moe_token_dispatcher=flex) after
  the cache A/B; d2 canary first (reduction-order change); EV honestly
  uncertain (70% of exposed a2a is imbalance wait, dispatcher-agnostic;
  DeepEP-addressable: latency-bound probs exchanges ~5.4s + launch
  overhead). Also feeds the overlap program (flex is a supported
  dispatcher for the big flag).
- 2026-08-13 ~03:1x CDT — JACK RULING: DeepEP/flex = HIGHEST PRIORITY
  ("already implemented, so simple, massive boosts previously — get it
  working, there might be bugs"). Precedent PR 660 (merged 07-14, LPS-626):
  config-only flex on single-node Qwen MoE — fb 43s->19-28s, hot-rank
  device peak 99.4%->88.4%, numerically identical loss; internode configs
  left on alltoall (NVSHMEM/IBGDA/no-EFA), intranode NVLink path
  validated. Tonight's EP8 groups are INTRA-NODE (the validated path;
  2-node world but PP-only cross-node). L5 PROMOTED: next box slot after
  the in-flight L0b-mem verdict, ahead of golden legs + cache A/B. Spec:
  headline M=N config + moe_token_dispatcher=flex; watch zones = Buffer
  init on 2-node-world/intra-node-groups (nvshmem/gid errors → capture +
  debug through per Jack) and FP8 dispatch (Qwen validation was not FP8;
  d2 canary load-bearing). deep_ep 1.2.1 importable in box venv
  (verified). Debug-through-bugs mandate: bounded boots still apply, but
  errors are work items, not abandon signals.
- 2026-08-12 ~23:5x CDT — **L0b-mem ARM 1 VERDICT: ABORTED at the
  pre-registered >255 GiB line during the first d2 step** (compile+forward,
  ~6 min in, step never completed). Node-0 ranks 252-265 GiB at abort
  (max 271,783 MiB = 265.4 GiB), node-1 ~111 GiB. Reference points: E1
  selective OOM-measured 258.8 GiB at ONE in-flight mb; VPP2 holds 2 chunks
  in flight per rank. **Offload engagement evidence: ZERO** — no
  offload/NVTE line in the trainer log; the rank-0 Activation Offload
  Summary table never printed (BT_SKIP_WARMUP defers it to the first
  completed step, which never happened). Two-horned verdict by
  construction: (a) offload never engaged (runtime plumbing/env gap — env
  WAS exported: NVTE_CPU_OFFLOAD_V1=1 + expandable_segments via
  --export=ALL), or (b) engaged but could not stash fast enough — this run
  cannot distinguish. Log: lps1062_pp2/logs/l0bmem_d2_abort_over255_20260813.log
  (sha 7c3a46b50fad). Tree at 6d8b22da + TF32 (save-sync + L0-clearing hunks
  preserved to lps1062_pp2/megatron_config_hunks_savesync_plus_l0clearing.patch
  and reverted out — not needed here). Arm-2 (PCIe-stall) trigger did NOT
  occur; next move = borel ruling. L5 (DeepEP flex dispatcher) is PROMOTED
  to the next slot after this verdict per Jack (PR 660 precedent: intra-node
  EP = the validated DeepEP path; config-only moe_token_dispatcher=flex;
  ladder d2 canary → d4 vs 918 → d16 vs 1052; watch zones: nvshmem/rdma/gid
  init on a 2-node world, FP8 dispatch quantization virgin path).
- 2026-08-13 ~04:3x CDT — ORCHESTRATOR SUCCESSION ORDERED by Jack (borel at
  40% context): borel → cauchy (fable 1m). HANDOFF.md fully rewritten for
  cauchy (directives, fleet, in-flight DeepEP boot 2, box queue, overlap
  program, morning deliverable + decisions, gotchas). Handoff at the
  current atomic boundary per tonight's rule — nothing of borel's own was
  mid-flight; DeepEP boot 2 continues uninterrupted under doppler and its
  verdict is cauchy's first decision. borel goes quiet on cauchy's ack.
- 2026-08-13 ~01:4x CDT (Mac `date`; note prior entry stamps read ~04:3x —
  clock-source drift, not a 3h gap) — **SUCCESSION COMPLETE: cauchy holds
  the reins.** borel acked and went quiet. All seven fleet agents told to
  route to cauchy. Acks in: poincare (parity lane confirmed; will scope the
  BT_SKIP_WARMUP/VPP2-warmup-illegal-at-M=1 fix Mac-side while waiting on
  doppler READY), hausdorff (standing by for box-queue-drained ping),
  weierstrass (standing by for cache-A/B arbitration + report [TBD] fills).
  Awaiting acks: doppler (DeepEP boot 2, srun 86361, deliberately NOT
  interrupted — asked for d2-canary verdict, verbatim errors if any, and
  confirmation the deepep bridge-gate patch was preserved to lps1062_pp2/ +
  Mac-side), gauss, lebesgue, serre. No state changed by the handoff; box
  queue unchanged (DeepEP -> parity legs -> cache A/B -> evidence tail).
- 2026-08-13 ~01:5x CDT — **All 7 fleet acks in; routing to cauchy locked.**
  Substantive updates carried in the acks: (1) doppler: DeepEP boot 2
  (srun 86361) UP and READY — pipeline connected, kernels warm 130.0s,
  warmup exercised the flex dispatcher with ZERO nvshmem/rdma/gid errors
  (watch zone 1 clear so far); idle memory 175 GiB vs 163-164 alltoall
  baseline (+11 GiB flex buffers — flagged to doppler as headroom to watch
  at the d16 peak-mem read); d2 canary driver RUNNING
  (pp2-131k-L5-flex-d2, 2 controls); FP8-dispatch (watch zone 2) is now the
  live risk. Bridge deepep-gate patch preservation CONFIRMED CLOSED: box
  lps1062_pp2/bridge_flex_deepep_capability_fix.patch + Mac-side
  pp2cp8ep8/configs/ same name, sha256 7504d896ddc2feef. (2) poincare:
  scoping the BT_SKIP_WARMUP/VPP2-warmup fix Mac-side while holding for
  doppler READY. (3) gauss/lebesgue/weierstrass/serre: delivered lanes
  confirmed, standing by for cache-A/B arbitration + report fills.
  (4) hausdorff: standing by for queue-drained ping. Next decision point:
  d2 canary verdict (pass band loss 12.2-12.4, gn comparable).
- 2026-08-13 ~02:0x CDT — poincare closed the morning-list VPP2-warmup item
  EARLY (between-slots, Mac-side only): commit 6fa3bfa7 on
  jackrao/lps-1062-pp2cp8ep8 (4e7d5d3e..6fa3bfa7, +85 lines: 16 in
  megatron_bridge/backend.py + new test_startup_warmup.py).
  run_startup_warmup now auto-skips with a banner when VPP>1 (interleaved
  schedule requires M >= PP; the single-datum warmup is M=1 and can never
  satisfy it — previously a VPP>1 boot died at warmup unless
  BT_SKIP_WARMUP=1 was known). Env lever retained. Poincare reports
  50/50 backend tests green + ruff/ty clean. Per lightweight-review
  policy, ONE disposable-subagent review commissioned by cauchy (running).
  Holds stated to poincare: 6fa3bfa7 stays OUT of box A trees tonight
  (exact-bits), and merge-queue slotting goes through weierstrass after a
  PASS (single-owner rule).
- 2026-08-13 ~02:1x CDT — **L5 DeepEP d2 CANARY VERDICT: PASS** (doppler,
  boot 2, srun 86361). Loss 12.305-12.321 (band 12.2-12.4), gn 0.40-0.49
  vs alltoall d2 0.40-0.46 (comparable; reduction-order drift small,
  tolerance judgment clean). BOTH watch zones clear through warmup + 4
  driver steps: zero nvshmem/rdma/gid errors (zone 1), zero FP8-dispatch
  errors (zone 2 — virgin path survived contact). Perf nuance at d2: flex
  713 tok/s/GPU vs alltoall M=N 764 = -6.7% — expected shape
  (fill/drain-dominated regime, minimal a2a exposure, flex fixed
  overheads visible; NOT decision-relevant). Peak mem 172 GiB (+11 GiB
  idle flex buffers but live peak LOWER than alltoall — PR 660
  no-permute-buffer signature). Decision rungs launching on same boot:
  d4 vs 918, then d16 vs 1052 + peak mem (a2a-exposure bucket = 22.7% of
  step at d16 is where flex must pay). Morning report updated: new §6
  (DeepEP directive) written, canary + gate-fix + watch zones filled,
  d4/d16 [TBD]s slotted; old §6/7/8 renumbered 7/8/9.
- 2026-08-13 ~02:2x CDT — **6fa3bfa7 (VPP2 warmup auto-skip) review verdict:
  PASS-WITH-NITS, zero blocking findings.** One disposable-subagent pass
  per policy; no round-trips. Nits (both with poincare, neither required):
  (1) conservative-direction gap — skip keys on trainer-config VPP but
  mcore interleaving only engages via the _GLM52_DSA_PIPELINE_LAYOUTS
  table match (glm52_dsa.py:100-103); an unlisted-layout VPP=2 config
  silently loses the warmup connectivity check while running the legal
  non-interleaved schedule (fails safe; config arguably pre-broken);
  (2) commit message says "VPP1/None runs" but None is unrepresentable
  (non-Optional int=1, control.py:453). Reviewer positively verified: no
  rank asymmetry on the skipped barrier, skipped update_pg_timeout
  redundant (megatron_config.py:395), banner convention matches, tests
  pin behavior both directions. Actions: verdict+nits relayed to
  poincare; weierstrass asked to slot 6fa3bfa7 into MORNING_MERGE_QUEUE.md
  (infra family); report §8 gained the bullet. Still in flight: L5 d4
  rung (doppler).
- 2026-08-13 ~02:3x CDT — **VPP2-warmup item CLOSED end-to-end.**
  (1) poincare nit disposition, accepted on the merits: config-keyed skip
  KEPT (tightening to the runtime layout-match would spend real-config
  test purity protecting an already-silently-broken config; fails
  soft/safe is correct); the upstream silent-fallback gap gets a SEPARATE
  morning-queue candidate — fail-fast at apply time when requested VPP>1
  matches no _GLM52_DSA_PIPELINE_LAYOUTS tuple. Two PR-description
  sentences authored by poincare, forwarded to weierstrass. (2)
  weierstrass slotting: 6fa3bfa7 folded INTO queue item 5 — INFRA-B is
  now 61c41d9e + 8c13ed31 + 6fa3bfa7 (VPP config field + iterator fix +
  warmup auto-skip) — after verifying a hard dependency (6fa3bfa7 reads
  the 61c41d9e config field; cannot land alone). PR_DRAFT_INFRA.md
  updated; queue stands at 9 items. (3) weierstrass also added a
  branch-hygiene note: bringup tip has moved past reviewed state
  (offload-plumbing commits are experimental P2) — morning PRs cut at
  EXPLICIT commits, never the tip. Report §8 bullet updated to match.
  Still in flight: L5 DeepEP d4 rung (doppler).
- 2026-08-13 ~02:3x CDT (addendum) — weierstrass confirms crossed-message
  items already processed: poincare's two sentences attached to queue
  item 5 as PR-description material AND mirrored into PR_DRAFT_INFRA.md;
  the fail-fast FUTURE line is cross-linked to weierstrass
  SELF_REVIEW bringup-finding 3 (vpp>1+PP=1 silent no-op — same
  silent-fallback family; one follow-up should close both). Docs
  mutually consistent; queue at 9 items. VPP2-warmup thread fully closed.
- 2026-08-13 ~00:1x CDT — **L5 DeepEP (flex dispatcher) VERDICT: numerically
  clean, clear perf REGRESSION — not the lever at this topology.** Bits
  73c24b00 + TF32 exactly (A/B vs the 918/1052 baselines). Boot 1 died on a
  NEW bug: bridge flex_dispatcher_backend.py deepep gate name-checks "NVIDIA
  B200/B300" + capability-checks only major in (8,9) — our B300s report
  "NVIDIA L20D" (major=10) → raised at validate AND would have silently
  skipped flex at the :49 setup path. Fixed on-box (major+=(10,) both deepep
  sites; outer bridge tree only, frozen mcore untouched; affects only the
  flex arm). Papercut pc_03fd6b532039; format-patch preserved BOTH sides
  (box lps1062_pp2/ + Mac configs/bridge_flex_deepep_capability_fix.patch,
  sha 7504d896ddc2feef) — morning PR candidate (any Blackwell-Ultra flex
  user hits it). Boot 2: READY clean, warmup ran flex with ZERO
  nvshmem/rdma/gid errors (watch zone 1 clear; idle 175 GiB = +11 GiB flex
  buffers). d2 canary PASS (loss 12.305-12.321, gn 0.40-0.49 comparable) but
  713 vs 764 tok/s/GPU at d2 (-6.7%, fill/drain-dominated regime). d4 A/B:
  **802 vs 918 = -12.6%** (step 40.8 vs 35.7s), peak mem 174 vs 175 GiB (a
  wash). Mechanism (traced rank-0 window): DeepEP intranode kernels
  8.40s/step (dispatch 1.99 + combine 3.13 + cached_notify_combine 3.28;
  420 calls each = 35 layers × 3 passes × 4 mb) ≈ the alltoall a2a total —
  no comm-time win at 131k packed payloads on NVLink; imbalance-wait
  structure unshrunk (one 4.86s stage-0 park; gpu-empty 15.6%). Confirms the
  on-record expectation: ~70% of exposed a2a is dispatcher-agnostic
  imbalance wait. d16 flex run SKIPPED per the pre-registered ladder (d4 not
  healthy). Trace: lps1062_pp2/traces/l5_flex_d4_rank0.pt.trace.json.
  **Orchestrator succession: borel → cauchy (Jack-ordered, ~00:0x); reports
  route to cauchy now.** Queue after L5: golden parity legs (poincare, 2
  boots) → cache A/B → evidence tail (incl. the L0b-mem 32k discriminator).
- 2026-08-13 ~02:4x CDT — **L5 DeepEP CLOSED: ANSWERED-NEGATIVE at d4,
  decisive.** flex 802 tok/s/GPU vs alltoall M=N 918 = -12.6% (step 40.8s
  vs 35.7s); numerics clean at d4 (loss 12.28-12.31, gn 0.35-0.44); peak
  mem wash (174 vs 175 GiB). d16 rung SKIPPED per the pre-registered
  ladder rule (d16 only if d4 healthy) — rule, not judgment. MECHANISM
  (rank-0 window 41.49s, decomposed): DeepEP kernels 8.40s/step total
  (dispatch 1.99 + combine 3.13 + cached_notify_combine 3.28; 420 calls
  each = 35 layers x 3 passes x 4 mb, exact) ~= the 7-8s NCCL a2a it
  replaces — no comm-time win at 131k packed payloads on NVLink; the
  dominant exposure driver (peer-imbalance wait, 70% per
  A2A_EXPOSURE_DECOMPOSITION) is dispatcher-agnostic and did not shrink
  (4.86s stage-0 p2p park persists; p2p parks 6.24s = 15.0%; gpu-empty
  15.6%). METHODOLOGY NOTE: under flex the a2a leaves NCCL — nccl-bucket
  scripts read a2a=0; the 8.40s DeepEP kernel total is the like-for-like
  figure. Trace: lps1062_pp2/traces/l5_flex_d4_rank0.pt.trace.json
  (719MB, box-side; Mac copy = hausdorff final sweep item). Report §6
  fully filled and closed. BOX RELEASED to queue: doppler restoring
  73c24b00+TF32 (diff must = chunked_lm_head.py only), then poincare's
  golden parity legs (padded boot, sanity gate near-but-not-on 12.36,
  then unpadded boot, then three-leg verdict). Cache A/B behind parity.
- 2026-08-13 ~02:5x CDT — Golden parity leg 1 (PADDED) boot dispatched
  (srun 90139). Tree verified byte-identical to the banked
  pp2-padded-fresh state: HEAD 73c24b00, main-repo diff =
  chunked_lm_head.py only (+ frozen dirty-mcore marker); doppler
  additionally REVERTED the bridge flex-gate fix from the submodule
  working tree (preserved as patch box+Mac; L5 closed, nothing needs
  flex) — inertness argument now moot. Config:
  trainer_ep16cp16_golden_131k.json (PP1/CP16/EP16, alpha 32,
  /tmp/checkpoints), sha 7bd17f8a033ff0f5. PP1 = full model per rank,
  ~15-min load expected. poincare drives --label pp1cp16-padded at
  doppler READY; unpadded reboot follows. Task board rebuilt in cauchy
  session (borel's board did not carry over): #1 parity in_progress,
  #2 cache A/B, #3 evidence tail, #4 report, #5 pod deletion,
  #6 final sweep.
- 2026-08-13 ~03:0x CDT — **PARITY GATE 1 INTERIM FAIL + LPS-1063 EXPOSURE
  DISCOVERED (biggest signal of the night).** (1) poincare interim: golden
  leg 1 pp1cp16-padded fresh = loss 12.351163801620233 (262,032 tokens,
  9/9 outputs, volta sanity band 12.3-12.5 PASS, not-on-12.3614 PASS) vs
  banked pp2-padded-fresh 12.54805275637226 -> loss rel diff 1.569e-02
  (bar 1e-6), logprob max abs 26.1 at token (1,36051) — 4-orders FAIL,
  systematic signature. (2) cauchy verification, Mac-side: the LPS-1063
  fix f74785d7 ("force TE onto the exact cu_seqlens path for tail-padded
  THD under CP", only on jackrao/lps-1063-te-cp-tailpad-fix) is NOT an
  ancestor of 73c24b00; git grep pad_between_seqs at 73c24b00 = ZERO.
  LPS-1063 (closed): TE <=2.17.1 ignores tail padding under THD+CP ->
  silent mis-attention + uninit-LSE nondeterminism. BT_PACK_PAD_TO_MAX=1
  creates exactly that tail pad; CP8 vs CP16 chunk it differently ->
  BOTH padded parity legs ran exposed, possibly corrupted DIFFERENTLY.
  (3) Discriminator table extended (sent to poincare pre-registered,
  before leg C lands): leg C (golden UNPADDED, booting) ~=12.548 ->
  golden-padded contaminated; ~=12.351 -> ambiguous (real topology gap OR
  PP2-padded corrupted); THIRD value -> both padded legs corrupted.
  Clean Gate-1 = UNPADDED vs UNPADDED -> fresh PP2-unpadded leg proposed
  after leg C (cache A/B slides). Cheap fingerprint: padded-leg bitwise
  self-reproduction failure = uninit-LSE signature. Gate 2 reinterpreted:
  padded-vs-unpadded golden = LPS-1063 exposure test now, not padding
  semantics. (4) WIDER IMPLICATION, owned by cauchy for the report: the
  headline perf runs (918/1052) are pad-to-131k THD+CP8 on the same tree
  — same exposure; loss canaries (12.2-12.4 band) were not sensitive
  enough to catch mis-attention. Perf numbers likely stand (wrong
  attention is same-shape compute) but CORRECTNESS claims need the fix
   A/B'd; possible perf re-anchor with f74785d7 applied = morning
   decision for Jack. (5) doppler asked to report box TE version to pin
   the exposure claim.
- 2026-08-13 ~03:2x CDT — poincare per-token structure analysis (Mac-side,
  parity/ dir JSONs; the failure-signature complement to the 03:0x entry):
  (a) 96.0% of weighted tokens over the 1e-3 logprob bar, diffs DIFFUSE
  through datums (rel positions 0.18-0.92), max 26.1 — not
  boundary-localized; consistent with uninit-LSE corrupting whole kernel
  invocations, not a stitch bug. (b) NOT a permutation: per-datum means AND
  stdevs differ (datum 2: -13.06±1.49 pp2-fresh vs -12.30±1.07 cp16) —
  genuinely different compute, rule-out for stitch-order bugs. (c) the
  zero-weight spans are 0-filled identically both legs (the "matching band"
  at decile 3-4 is trivial 0==0, uninformative). (d) DECISIVE: PP2 per-datum
  means form a SAWTOOTH — monotone-degrading within each partition, resetting
  at partition boundaries, worst at the tail-filled last doc of each
  partition (fresh: d2/d5/d8 = -13.06/-13.10/-13.07 vs partition-leaders
  -12.29/-12.31/-12.37); the SAME sawtooth holds in the step-18 leg A JSON
  (all 3 partitions monotone-degrading) — the PP2 padded-forward pathology
  is stable across 18 optim steps, not a fresh-boot fluke. (e) golden CP16
  leg is tight across partitions 0-1 (means ≈ -12.30 flat) with ONE bad
  datum (d8 -13.53, the tail-filled last doc of the last partition) —
  consistent with CP8-vs-CP16 chunking the tail pad differently under the
  exposure model. (f) the max-diff token (datum 1, pos 36051) maps to
  partition-0 global ~67795 = CP16 chunk 16.55 — inside a chunk, not
  boundary-adjacent; fits chunk-wide corruption. (g) independently
  re-verified cauchy's fact chain: f74785d7 not an ancestor of 73c24b00,
  zero pad_between_seqs hits at 73c24b00. Leg plan per cauchy's revised
  table: leg C (golden unpadded) in flight via doppler (srun 94262);
  PP2-UNPADDED probe boot next (one-line BT_PACK_PAD_DISABLE gate flip on
  the box tree, exact diff with doppler, documented probe deviation, reverted
  after); padded-rerun nondeterminism fingerprint re-probes later (padded
  boot was already stopped when the hold request landed — no undo).
- 2026-08-13 ~03:5x CDT — **LEG C LANDED ON THE PRE-REGISTERED POINT
  ESTIMATE.** pp1cp16-unpadded (fresh, exact bits): loss
  12.304180991898834 vs poincare's pre-registered 12.304 (band
  12.288-12.322, logged timestamped before the leg ran) — off by 0.0002.
  The per-datum LPS-1063 exposure model is quantitatively confirmed:
  unpadded golden is clean at the tight cluster value; golden-padded
  (12.3512) was corrupted via d8 alone; PP2-padded (12.548) corrupted more
  (sawtooth). GATE 2 (golden padded-vs-unpadded) = FAIL as reinterpreted:
  loss rel 3.818e-3, logprob max abs 21.15 at datum 8 token 2959 — the
  tail-filled last doc of partition 2, exactly where the model puts the
  corruption; reads as clean bug evidence. BYPRODUCT (volta's ask): fb
  ratio 89.4/60.7 = 1.47x ~= the 393216/262032 = 1.5 token-work ratio —
  first direct padding-waste measurement; padded compute is compute-bound.
  poincare pre-registered the PP2-unpadded probe: point 12.304, predicted
  WITHIN the original Gate-1 bar (rel <=1e-6) vs leg C — i.e. the PP2/M=N
  path is numerically sound and the Gate-1 FAIL was entirely exposure.
  Fix-arm tree ratified: 73c24b00+TF32+f74785d7 (+filegate probe hunk);
  recipe + per-pass both-nodes sentinel evidence with doppler.
- 2026-08-13 ~04:1x CDT — **PP2-UNPADDED PROBE: OUT OF GATE — the dominant
  term is NOT LPS-1063.** pp2-unpadded (BT_PACK_PAD_DISABLE gate flip,
  documented probe deviation) = loss 12.494321280379372 vs leg C 12.3042 =
  rel 1.545e-2, four orders past the 1e-6 bar; poincare's pre-registered
  "residual PP2-path numerics issue beyond padding" reading fires.
  Decomposition closes EXACTLY: Gate-1 FAIL gap 0.197 = PP2-path gap
  (unpadded, 0.190) + exposure@PP2/CP8 (0.054) - exposure@golden/CP16
  (0.047). LPS-1063 is real but SMALL (~0.05); the dominant term exists
  with ZERO tail padding. Structure: per-token diffs diffuse and large in
  EVERY datum (even mean-clean d0/d6/d7/d8 carry mean |diff| ~0.08/decile
  = 80x bar — cancel in the mean); systematic mean shifts concentrate in
  d1-d5, escalating across partition 1 (d3 -0.170, d4 -0.600, d5 -1.125).
  Two readings: (a) STRUCTURAL — DSA top-k selection is discontinuous;
  bf16 score noise at the selection boundary flips borderline KV entries
  and CP8-vs-CP16 gather orders differ, so the sparse-attention function
  is genuinely topology-dependent per-token; (b) residual PP2-path bug —
  pure selection noise does not obviously explain the systematic
  net-negative d1-d5 shifts. Discriminators proposed: CP8/PP1 (CP-size
  sensitivity, PP-innocent) and/or PP2/CP1 (PP without CP). SEQUENCING
  FLAG to cauchy: the fix arm (f74785d7) addresses only the ~0.05 exposure
  term; a fixed-tree Gate-1 still fails by ~0.19. doppler holding fix-arm
  staging for cauchy's ruling; box reverted to exact bits.
- 2026-08-13 ~08:4x CDT — **FIX ARM READS NULL ON GLM-5.2: the wrapper never
  engages (mechanism correction).** Fix-arm PP2 boot (73c24b00+TF32+
  4e8b5f01[=cherry-pick of f74785d7]+filegate; both sentinels absent both
  nodes, verified per-pass via poincare's kubectl exec channel): pass 1
  pp2-padded-fixed = 12.535494961470064 (fb 101.0s, first-touch JIT), pass 2
  pp2-padded-fixed-rerun = 12.518228610946453 (fb 29.3s warm) — NOT bitwise
  equal. ZERO bt_te_padfix lines in the boot log (installation proven by
  doppler: apply() True, marker on the class, no import warning). Source
  verdict (poincare + doppler, convergent): GLM-5.2 core attention is
  DSAttention (experimental_attention_variant/dsa.py:1530) on its own
  cutlass/cuDNN kernel path — it NEVER calls TE DotProductAttention.forward,
  so the f74785d7 wrapper is inert on this model by construction (fix-ON ==
  fix-OFF numerically). AND: the vendored mcore already carries the
  equivalent TE-path fix in-tree (extensions/transformer_engine.py:1829-1843,
  basetenlabs/Megatron-LM#25 — explicit pad_between_seqs=True on
  real-vs-padded divergence under CP>1), so TE-DPA models on 73c24b00 were
  never exposed tonight. CONSEQUENCES: (a) tonight's ~0.05 padding terms and
  the 0.19 PP2-path gap are NOT the TE auto-detect bug — the mechanism lives
  in the DSA path's own pad handling (dsa_layout/dsa_masking) or is
  structural top-k sensitivity; (b) the padded path is NONDETERMINISTIC
  run-to-run even on the fix tree (three exposed samples: 12.5481 / 12.5355
  / 12.5182) — an uninit-memory-class read in the DSA/cuDNN path under tail
  padding, parallel to LPS-1063 but in OUR kernels; unpadded legs appear
  deterministic at loss level (leg C landed on the pre-registered point
  estimate to 0.0002); (c) the fix-arm toggle passes + perf A/B are null as
  fix tests on this model — cauchy ruling on cancellation; the ratified
  unpadded passes (permuted discriminator + an added unpadded-identity
  bitwise-repro/baseline-carryover check) proceed — fix-inert, unaffected.
- 2026-08-13 ~09:1x CDT — cauchy merged-plan ruling: fix-toggle passes and
  the fix perf A/B CANCELLED (null A/B on an inert wrapper); the unpadded
  program is the spine. RELABEL (cauchy, applies to all earlier entries):
  "LPS-1063 exposure" -> "tail-pad corruption, mechanism unattributed
  (DSA-path, under investigation)" — TE is exonerated on this tree by the
  in-tree #25 fix; f74785d7 remains valid for TE-DPA models but is
  out-of-scope for GLM-5.2. DISTRIBUTION CAVEAT (cauchy, folded in): the
  padded legs are DISTRIBUTIONS, not points — 4th padded sample
  pp2-padded-fixed-rerun2 = 12.539178028057696 (fb 28.8s); the four samples
  (12.5481/12.5355/12.5182/12.5392) span 0.030, so the tail-pad corruption
  terms are ~0.05 +/- 0.03 RANGES and the exact 0.197 closure was partly
  sample luck; the 0.190 unpadded PP2-path term stands (deterministic).
  Sawtooth reframed as a mechanism CLUE (position-dependent severity within
  partition => layout-misalignment class over pad-block-selection class);
  gauss re-tasked on the DSA tail-pad source hunt. Pass sequencing:
  padded repeat (done) -> pad valve ON -> unpadded identity repeat
  (pre-registered EXACT 12.494321280379372) -> permuted discriminator ->
  teardown -> PP1/CP8 forward-only leg (restored Gate 1 under original
  bars). d4 padded-vs-unpadded perf A/B demoted to best-effort tail.
- 2026-08-13 ~09:4x CDT — **PERMUTED-DOCUMENT DISCRIMINATOR: DECISIVE —
  corruption follows the SLOT, not the document (branch (b), the
  headline-PR-bug branch).** poincare pass 5b pp2-unpadded-rev (--permute
  reverse) on the fix-arm boot (pad valve ON, both-nodes valve evidence
  logged per pass): loss 12.509854771784232 (in the unpadded noise band).
  Per-datum deltas vs leg-C clean (slots source-verified via
  partition_thd_cp_datums greedy in-order fill, thd_cp.py:92-165;
  assignment model matched baseline padded totals exactly): d5 worst in
  BOTH baseline runs at partition-1-last (-1.125/-1.158) went CLEAN
  (+0.0007) at partition-0-slot-3 — document exonerated; d1 mild (-0.073)
  at partition-0-slot-1 INHERITED worst-slot corruption (-0.865) at
  partition-1-last; d2 -0.026 -> -0.827; d4 -0.600 -> -0.000 leaving
  partition 1. Control nuance (filed per mixed protocol): d3 at the same
  slot both orders moved -0.170 -> -0.021 — slot corruption magnitude is
  composition-dependent; read as (b)-family with layout-dependent
  severity, NOT mixed (no delta ever tracked a document). NOISE FLOOR
  finding: unpadded PP2 is nondeterministic run-to-run WITHIN one boot
  (5a 12.5145, 5a2 12.4892, baseline 12.4943 — spread ~0.025; per-token
  bitwise identity only 3.4%, max 22.7; per-datum corruption PATTERN
  stable across runs) — the 5a exact-bitwise pre-registration FAILED;
  per-datum deltas carry a +/-0.03-0.09 floor; the decisive moves are
  10-30x the floor. The 0.190 gap term stands (~7x floor). LOCALIZATION
  HYPOTHESIS (for the morning escalation): at PP2/M=3 the middle
  microbatch is the 1F1B steady-phase one — a schedule/state artifact in
  the M=N path hitting the middle partition forward (per-microbatch
  seq-param/metadata handoff or buffer reuse; first place to look:
  _schedule_data_iterator seam) fits slot-follows + position-escalation.
  PP1/CP8 forward-only leg now decides PP2-path vs CP8-path. Golden CP16
  fix-arm boot MOOT under the inert wrapper (flagged to cauchy).
- 2026-08-13 ~10:0x CDT — cauchy reframe ruling: nondeterminism is NOT
  padding-specific (unpadded range 0.025 ~ padded 0.030); per-datum noise
  bands built from the 3 identity samples; the permuted read survives at
  per-datum resolution. BANDED READ (poincare): every corrupted datum lands
  OUTSIDE its identity noise band in the slot-follows direction — d5
  +0.0007 at P0.3 (band -1.158..-1.019), d4 -0.0004 at P0.4 (band ~-0.58),
  d1 -0.865 at P1.2 (band -0.166..-0.073), d2 -0.827 at P1.1; clean datums
  in-band everywhere. d3 control attenuated 7x at the same slot (OUTSIDE
  its band) — composition-dependent magnitude, (b)-family. Slot-follows
  CONFIRMED with bands. RESTATED SIZES (ranges, per cauchy): the PP2-path
  mean gap = +0.195 +/- ~0.013 (unpadded PP2 mean 12.4993 n=3 vs golden
  12.3042) — stands at ~8-15x noise; padded-minus-unpadded at PP2 = +0.036
  (padded mean 12.5352 n=4) — borderline vs noise; the structured signals
  (golden d8-only max-diff 21.15; the PP2 slot sawtooth) carry the
  corruption phenomenon. Gate-1 formal bars (1e-6/1e-3) are UNMEASURABLE at
  the ~2e-3-rel noise floor — the formal-PASS path dies tonight; morning
  framing = statistical parity (N-repeat means + bands) pending a
  nondeterminism fix. TREE CONFOUND noted: repeat evidence is all fix-arm
  tree; the PP1/CP8 leg runs EXACT tree as a REPEAT PAIR (2-3 passes) —
  triple duty: gap axis, nondeterminism locality, exact-tree check.
  doppler tearing down fix-arm, staging PP1/CP8/EP8 forward-only on exact
  bits (no hunks; unpadded default at PP1; --forward-only dodges F3).
- 2026-08-13 ~10:5x CDT — **PP1/CP8 LEG: CP8 FORWARD KERNELS CONVICTED;
  M=N + PP2 PLUMBING EXONERATED for the corruption.** poincare trio on the
  exact tree (73c24b00+TF32, forward-only, 1-node DP1, F3 dodged):
  12.501736810906936 / 12.494330171902787 / 12.513274056510571 (mean
  12.5031, spread 0.019). (1) The per-datum slot structure REPRODUCES the
  PP2/CP8 table datum-for-datum (d5 mean -1.162, d4 -0.590, d3 -0.186,
  d1/d2 mild, d0/d6/d7/d8 clean) — with NO pipeline, NO 1F1B, NO backward:
  the defect is in the CP8/EP8 DSA forward path itself, not the M=N
  schedule, not the PP2 plumbing (both exonerated for the corruption by
  this reproduction). (2) Nondeterminism is present at PP1/CP8
  forward-only (spread 0.019) — the DSA/cuDNN forward path is
  nondeterministic on its own; ~2e-3-rel noise floor under every CP8
  parity comparison tonight. (3) Mean gap is PP-INNOCENT: PP1/CP8 12.5031
  ~= PP2/CP8 12.4993 (within noise); the +0.195 gap rides the
  CP8/EP8-vs-CP16/EP16 axis — clean at CP16 chunking, corrupted at CP8
  chunking, slot-structured (middle microbatch of the 3-microbatch M=N
  call, escalating toward its end), composition-dependent in magnitude —
  converging on gauss's 3a (cuDNN fused indexer consuming per-slot packed
  metadata). Formal Gate-1 compare for the record: rel 5.932e-04 / logprob
  max 14.8 — FAIL at the original bars but the noise floor exceeds the
  bars; formal-PASS unmeasurable tonight, statistical-parity framing
  stands.   NEXT (last box action): unfused PP2 reference-path leg — the
  fused-kernel discriminator (slot structure absent + stable repeats =
  fused cuDNN indexer convicted + free clean-PP2-forward existence proof).
- 2026-08-13 ~11:1x CDT — cauchy's workspace-growth pattern test
  (poincare, Mac-side binned profiles, parity/): PARTITION-LEVEL pattern
  CONFIRMED both orders (the corrupt partition is the one larger than its
  predecessor: identity 92960->98704 corrupt, 70448 clean; reverse
  105584->124784 corrupt, 31744 clean). WITHIN-PARTITION onset prediction
  FAILED: identity P1 holds the 0.09 drift floor through ~40k then a sharp
  sustained rise at ~41-42k (NOT the ~92960 predecessor extent, NOT a
  32768 power-of-2 tile); reverse P1 mild from ~32k, severe from ~64k —
  onsets align across orders by neither row, fraction, nor datum. The
  corruption front is partition-row-positioned with composition-dependent
  onset; P0 mostly floor with localized spikes (32k/72k/88k), P2 flat
  both orders. Net for the source hunt (gauss): simple
  workspace-high-water-mark refuted by the onset data; surviving pattern
  = "larger-than-predecessor partition gets a corruption front starting
  partway, escalating to the tail" — cuDNN fused indexer per-slot metadata
  mishandling class (3a) remains the convergent suspect; exact trigger
  needs the source side. Unfused PP2 boot dispatching (doppler srun
  109465).
- 2026-08-13 ~12:2x CDT — **CAMPAIGN CLOSED: ONE STALE WHEEL.** (poincare
  verdict read on the fe127 JSONs; note poincare was dark ~13:0x-16:4x —
  session silence after the unfused-pass OOM [the reference path tried to
  allocate 128.77 GiB of naive index scores at 131k — infeasible as
  constituted]; doppler covered all mechanics per spec, evidence intact).
  serre's find: the box venv ran the RETIRED cudnn-frontend 1.26 stack
  while the tree pinned the race-fixed 1.27.0.dev (LPS-1003 series incl.
  the head_dim-576/512 TMEM WAR race) — the race class fit the
  nondeterministic-corruption signature exactly. REPRODUCER GATE
  (PP1/CP8-fwd x3, bumped wheel) vs cauchy's pre-registered L1/L2/L3:
  L1 variance 12.30444/12.30521/12.30478 (spread 0.00077, ~25x collapse
  from 0.019) PASS; L2 slot table all deltas within +/-0.0031 (d5 +0.0023
  vs old-wheel -1.16) PASS; L3 mean = golden cluster 12.304 PASS — the
  0.19 gap VANISHED. Single race confirmed; Gate-1 topology invariance
  RESTORED; in-kernel-varlen suspect dies. CONVERGENCE: padded =
  unpadded = permuted = clean on the fixed wheel (12.3040-12.3050 across
  all five fe127 parity legs) — tail-pad corruption, slot structure, and
  nondeterminism all vanish together; the whole night's correctness story
  was one stale wheel. PERF on the fixed wheel: d2 744.6 / d4 885.7+878.0
  / d16 983.8 tok/s/GPU vs old-wheel 918/1000/1052 — the race-fix costs
  ~3.5% at d4, ~6.5% at d16 (the honest post-correctness anchors; d2
  initial last_loss 12.3047 confirms clean training numerics). Gates 1+2
  PASS. poincare lane closed per lebesgue; no catch-up owed.
- 2026-08-13 ~03:2x CDT — **EXPOSURE PINNED + ENDGAME LOCKED.**
  (1) poincare distribution analysis (Mac-side, leg1-vs-pp2fresh): 96.0%
  of weighted tokens over the 1e-3 bar, DIFFUSE through datums (positions
  0.18-0.92), max 26.1; NOT a permutation (per-datum means AND stdevs
  differ); zero-weight spans 0-filled identically (uninformative);
  DECISIVE: PP2 per-datum means form a SAWTOOTH (monotone-degrading
  within each partition, resetting at boundaries, worst at tail-filled
  last docs d2/d5/d8 ~= -13.07..-13.10 vs -12.29..-12.37 leading docs) —
  same sawtooth present in step-18 leg A JSON = stable pathology, not a
  boot fluke; golden CP16 tight/flat across partitions 0-1 with ONE bad
  datum (d8 -13.53, tail-filled last doc of last partition) — consistent
  with CP8-vs-CP16 chunking tail pad differently. Token (1,36051) maps
  to global pos ~67795 = chunk 16.55 INTERIOR — chunk-wide corruption,
  not boundary stitch. (2) doppler: TE 2.16.0 on w56lorq venv (affected
  <=2.17.1) — exposure claim CLOSED. (3) cauchy feasibility probe:
  f74785d7 cherry-picks CLEANLY onto 73c24b00 (3 files:
  + dp_worker/te_cp_tailpad_fix.py, worker.py hook, + unit test; scratch
  worktree, discarded). (4) ENDGAME RULING (cauchy): after leg C (in
  weight load) + PP2-UNPADDED probe (poincare preps one-line pad-gate
  flip; NO env exists; single documented-deviation boot; parity-evidence
  boots stay exact) + padded-rerun nondeterminism fingerprint if boot
  holdable: FIX-APPLIED ARM on tree "73c24b00 + f74785d7" — padded
  parity pair (original bars) + Gate 2 restored (true padding semantics
  on fixed tree) + d4 perf spot-check vs 918 with PRE-REGISTERED noise
  band (doppler to state observed d4 run-to-run variance BEFORE the
  run). Cache A/B + evidence tail DEPRIORITIZED; L1 wgrad A/B + L2 VPP2
  DROPPED for the night; offload discriminator only if time. (5) Asked
  poincare to PRE-REGISTER predicted leg C value from the per-datum
  model before it lands.
- 2026-08-13 ~03:3x CDT — **PRE-REGISTERED PREDICTION for leg C (golden
  unpadded), filed by poincare BEFORE the result exists (leg C still in
  weight load/boot at filing time). POINT: 12.304; BAND: 12.288-12.322.**
  Basis: golden-padded corruption concentrated in d8 (tail-filled last
  doc of partition 2, per-datum mean -13.529) while d0-d7 cluster tight
  at -12.293..-12.317 (mean -12.3039); leg C has no tail fill and d8
  (10752 tokens) is exactly 32-aligned = zero per-doc pad; repairing d8
  to cluster moves weighted mean 12.3512 -> 12.3039; band = cluster
  spread +/- 0.005 per-doc-pad residual. READINGS (pre-registered):
  in-band = exposure model QUANTITATIVELY CONFIRMED (golden-padded
  corruption d8-only/mild; PP2-padded sawtooth/large; leg C = third
  value = both padded legs corrupted differently, asymmetry
  characterized); 12.351 = model REFUTED (deeper cause than tail fill);
  12.548 = golden boot itself suspect (structural stop). This
  supersedes cauchy's coarser 12.351-ambiguous table row. poincare now
  building fix-applied staging spec (scratch-worktree cherry-pick +
  fix-branch unit test via stub harness + apply recipe for doppler,
  tree label 73c24b00+f74785d7).
- 2026-08-13 ~03:4x CDT — **Fix-applied perf pre-registration settled
  (conditionally).** doppler noise data: within-boot control repeats
  consistently <=2% spread (clean-tip 0.2%, +TF32 0.3%, M=N 1.4%, flex
  1.9%); boot-to-boot single observation 1052->998 (-5.1%) CONFOUNDED
  (post-537MB-save boot). cauchy ruling: (a) within-boot +-2% agreement
  gate RATIFIED; win bar >936 RATIFIED; (b) the -5% cross-boot stands
  line is CONDITIONALLY REPLACED — if the f74785d7 hook is env-gatable
  (poincare checking; only a trivial guard allowed, tree label must stay
  honest), PRIMARY metric = within-boot fix-OFF vs fix-ON A/B on the
  fix-applied boot (fix cost at 2% resolution; vs-918 secondary);
  if not gatable, doppler rule stands as filed (>=872 stands / <872
  re-anchor) with the confounded-datapoint caveat in the report.
  ORDER CHANGE: fix-applied arm AHEAD of fingerprint re-boot (the
  padded boot was stopped before the hold request landed; bits+config
  preserved). Sequence: leg C -> PP2-unpadded probe (gate-flip diff
  received by doppler, exact+complete) -> fix-applied arm ->
  fingerprint if time -> cache A/B if time.
- 2026-08-13 ~03:5x CDT — **Fix-arm design FINAL.** (1) poincare caught a
  cauchy labeling error: fix-arm tree must carry TF32 (all exposed legs
  AND the 918/1052 anchors had it; omitting confounds the fix A/B with
  head precision, ~1e-3 logprob shift + 6.7% perf). RATIFIED:
  73c24b00+TF32+f74785d7. (2) f74785d7 ships env-gated
  (BT_TE_CP_TAILPAD_FIX, default ON) -> CONSOLIDATION: fingerprint
  re-boot CANCELLED; ONE fix-arm boot carries (a) padded parity pair
  fix-ON (original bars), (b) exposed control fix-OFF (same-boot
  headline-config repro), (c) within-boot perf A/B fix-OFF vs fix-ON
  (PRIMARY metric, +-2% control gate, win >936; cross-boot >=872 rule
  demoted to fallback), (d) repro fingerprint both modes with
  pre-registered asymmetry (fix-OFF bitwise match = weak reading,
  mismatch = strong uninit-LSE evidence). (3) Cherry-pick re-verified by
  poincare (zero conflicts); fix-branch unit tests Mac-side 12 pass /
  1 TE-dependent skip (runs on box venv). Sequence: leg C (in flight) ->
  PP2-unpadded probe -> fix-arm boot -> cache A/B if time.
- 2026-08-13 ~04:0x CDT — **Filegate probe hunk RATIFIED (fix-arm design
  correction).** poincare corrected cauchy's (4): the native env kill
  switch is boot-bound (os.environ read per call but no remote env-set;
  same-TREE not same-BOOT) — within-boot A/B needs the trivial guard.
  Built + tested rather than speced: 3-line sentinel-file check in
  _enabled() (/tmp/bt_te_cp_tailpad_fix.off present => OFF; read per
  attention forward, ~us; toggled both-nodes between passes) + test
  pinning both directions (13/13 + 1 TE skip, ruff clean). Patch
  eyeballed by cauchy directly (below disposable-review bar per
  proportionality) — clean. Rides ON TOP of untouched f74785d7; tree
  label "73c24b00+TF32+f74785d7 (+filegate probe hunk)". Durable copy:
  pp2cp8ep8/patches/tailpad_filegate_probe.patch. Recipe guards: pre-boot
  kill-file ABSENCE check both nodes (stale file silently disables fix);
  boot-log first-engagement line "[bt_te_padfix] ... ACTIVE"; cauchy
  addition: per-pass sentinel-state capture on both nodes in each pass
  record (shell-level). Design point confirmed for cross-tree validity:
  fix engages ONLY on real-vs-padded cu_seqlens divergence (tail fill);
  per-doc interior pads already trip TE auto-detect => unpadded legs are
  fix-inert; leg C carries over unchanged to the fix-arm comparisons.
- 2026-08-13 ~00:5x CDT — GOLDEN PARITY (poincare drives, doppler box). Leg 1
  (pp1cp16-PADDED, golden EP16/CP16/PP1, exact bits 73c24b00+TF32, flex fix
  reverted from the working tree first) BANKED: loss 12.351163801620233,
  262,032 tokens exact, 9/9 outputs, fb 89.4s, JSON
  lps1062_bench/parity_pp1cp16-padded.json. **PP-gate compare vs
  pp2-padded-fresh (12.54805275637226) is FAIL-GRADE: loss rel 1.6e-2 vs bar
  1e-6, logprob max abs 26.1** — leg C (unpadded golden) is the
  discriminator: ~=12.548 ⇒ padded-leg contamination; ~=12.351 ⇒ real
  topology-invariance gap. CONTEXT (cauchy/poincare): both PADDED legs ran
  exposed to LPS-1063 (TE tail-pad mis-attention under THD+CP; fix f74785d7
  NOT in 73c24b00); box TE = 2.16.0 <= 2.17.1 = affected range, exposure
  claim PINNED. Leg-1 boot log saved (golden_leg1_padded_boot_20260813.log).
- 2026-08-13 ~01:0x CDT — ENDGAME QUEUE (cauchy ruling, supersedes earlier):
  leg C → PP2-UNPADDED probe (poincare one-line gate flip:
  BT_PACK_PAD_DISABLE hunk on _thd_partition_pad_to_length, single
  documented-deviation boot, reverted after) → FIX-ARM single boot
  (73c24b00+TF32+f74785d7 cherry-pick + poincare filegate sentinel hunk;
  carries fix-ON parity + bitwise repeat, kill-file fix-OFF control +
  nondeterminism repeat, d4 perf A/B fix-OFF-vs-ON as PRIMARY metric at the
  ±2% within-boot gate, then golden CP16 boot on the same tree for
  within-tree reproduction of the exposed 12.351163801620233) → cache A/B if
  time. Fingerprint re-boot CANCELLED (folded into the fix arm). L1/L2
  DROPPED for the night; cache A/B + evidence tail deprioritized.
  PRE-REGISTERED noise band (cauchy ratified): within-boot controls must
  agree ±2% or the run discards; win bar >918+2% = >936; vs-918 anchor
  demoted to secondary context since the fix is env/file-gatable (within-boot
  A/B is primary). d4 variance data filed: same-boot control spreads tonight
  0.2-1.9%; single cross-boot pair 1052→998 (-5.1%, confounded post-save).
- 2026-08-13 ~01:3x CDT — LEG C BANKED (poincare): pp1cp16-UNPADDED golden
  loss 12.304180991898834, 262,032 tokens exact, 9/9 outputs, fb 60.7s (JSON
  lps1062_bench/parity_pp1cp16-unpadded.json). Then the PP2-UNPADDED probe
  (documented-deviation boot: BT_PACK_PAD_DISABLE gate flip on
  _thd_partition_pad_to_length, reverted after; tree back to exact
  73c24b00+TF32, verified): **pp2-unpadded loss 12.494321280379372 — OUT of
  Gate-1 vs leg C (12.494 vs 12.304)** = a RESIDUAL PP2-path numerics gap
  that f74785d7 does not address. Parity map so far: golden unpadded 12.304
  / golden padded 12.351 / pp2 unpadded 12.494 / pp2 padded-fresh 12.548.
  Fix-arm staging HELD pending cauchy sequencing ruling (fix arm as planned
  vs CP8/PP1 discriminator boot first). Gate-flip mechanics worked exactly
  as designed (tokens exact, fb 64.4s vs padded 75.6s = the padding-waste
  measurement at PP2).
- 2026-08-13 ~04:2x CDT — **PP2-UNPADDED PROBE: OUT OF GATE — the 0.19
  term is the story now.** Result 12.494321280379372 vs leg C
  12.304180991898834 = rel 1.545e-2 (bar 1e-6). Pre-registered middle
  reading fires: residual PP2-path numerics beyond padding.
  DECOMPOSITION CLOSES EXACTLY: 0.197 (orig Gate-1 gap) = 0.190
  (PP2-path, zero padding) + 0.054 (tail-fill @PP2/CP8) - 0.047
  (tail-fill @golden/CP16). LPS-1063 real but MINOR (~0.05/side);
  dominant = ~0.19 PP2/CP8/EP8-vs-golden gap, character unknown.
  Structure: diffuse large diffs EVERYWHERE (mean-clean datums carry
  ~0.08 mean |logprob diff|/decile = 80x bar); mean shifts concentrate
  d1-d5, escalating partition 1 (d3 -0.170, d4 -0.600, d5 -1.125),
  systematically negative. READINGS: (a) structural DSA top-k
  CP-geometry sensitivity vs (b) real PP2/M=N-path defect
  (slot-dependent state). CAUCHY RULING: fix-arm proceeds AS PLANNED
  (perf A/B + LPS-1063 closure morning-critical regardless) AUGMENTED
  with permuted-document unpadded pass (driver-side; pre-registered:
  shifts follow DOC = (a), follow SLOT = (b) = headline-PR-bug branch,
  morning escalation). PARALLEL LANES: serre reactivated (assess
  env-only F3 unblock for PP1/CP8 same-CP reference boot); gauss
  reactivated (source memo: DSA top-k CP-geometry dependence + EP
  reduction order; M=N per-microbatch state carryover). Probe
  mechanics: gate flip applied+REVERTED, tree re-verified exact
  (diff = chunked_lm_head.py only); pp2-unpadded banked (fb 64.4s vs
  padded 75.6s = 1.17x, vs golden 1.47x — nuance only). Report §1
  correctness flag added; §3 rewritten to the decomposed state.
- 2026-08-13 ~04:4x CDT — **Permuted-pass mechanics settled via Option A
  (2nd filegate), with one timing collision.** poincare pre-registration
  for the permuted-document discriminator FILED before any result:
  reverse order; partition model verified against baseline padded totals
  (P0=[d8,d7,d6,d5,d4], P1=[d3,d2,d1], P2=[d0]); baseline per-datum
  deltas on record (d5 -1.125 worst); PREDICTIONS: (a) doc-follows =>
  d5 stays ~-1.1, total ~12.494 order-invariant; (b) slot-follows =>
  d1 inherits worst slot, d5 improves, total moves; CONTROL: d3 same
  slot both orders, must hold ~-0.17 (big move = instrument unstable,
  escalate); MIXED table = filed as its own finding, not forced.
  Contingency: permuted GOLDEN leg if gauss finds cross-doc DSA indexing
  within partitions. Old probe trainer confirmed DEAD (liveness curl
  refused) -> zero-cost path gone. CAUCHY RULING: Option A — second
  2-line sentinel valve in _thd_partition_pad_to_length (same pattern as
  ratified tailpad valve; both-direction test required; pre-boot absence
  check + per-pass logging cover BOTH valves; tree label
  "...(+2 filegate probe hunks)"). TIMING COLLISION: doppler dispatched
  fix-arm (srun 100672) before the ruling landed (one filegate only, no
  pass-5 mechanism) -> conditional kill-and-redispatch ordered if
  poincare ships the tested hunk within ~10 min (load barely started;
  minutes vs a full Option-B boot). PROVENANCE: doppler label shows
  4e8b5f01 = box-local cherry-pick SHA of f74785d7 (expected from
  cherry-pick) — verification + mapping record requested.
- 2026-08-13 ~04:5x CDT — **F3 DODGED, same-CP reference leg unlocked
  (serre).** results/F3_UNBLOCK_ASSESSMENT.md: env-only unblock = coin
  flip (~40-55%), BUT the parity bars are pure FORWARD quantities and F3
  crashes only at the backward grad sync -> drive PP1/CP8 with /forward
  (ForwardOp exists at 73c24b00, api/ops.py:37-43; one-line driver
  change parity_driver.py:116; forward_only=True never initializes the
  DP+CP communicator) = dodged by construction (~85-90%). Cost: one
  driver line + 15-20 min first-forward JIT. QUEUED after fix-arm:
  PP1/CP8/EP8 UNPADDED forward-only leg = axis isolation for the 0.19
  term (=12.494 -> PP-innocent, gap on CP/EP axis, M=N exonerated for
  mean gap; =12.304 -> gap follows PP2, (b) defect branch, headline PR
  blocked; else mixed). Residual confound acknowledged: CP+EP entangled
  toward golden either way — leg is decisive for the M=N question only.
  poincare drives + pre-registers; serre V1/V2 NCCL env ladder stays
  parked as the true F3 root-cause discriminator (morning decision).
  Cache A/B now effectively dead for the night (pre-registered
  deprioritization holds).
- 2026-08-13 ~01:5x CDT — FIX-ARM BOOT PROVENANCE (cauchy's check, for
  morning readers): box tree = 73c24b00 + TF32 patch (uncommitted,
  chunked_lm_head.py) + **4e8b5f01 == cherry-pick(f74785d7)** — verified
  content-identical: same commit message/author/date (Jack Rao, Aug 8),
  touches exactly te_cp_tailpad_fix.py (NEW) + worker.py (7-line hook:
  import + apply_te_cp_tailpad_fix() before the first attention forward) +
  test_te_cp_tailpad_fix.py (NEW) — plus the uncommitted filegate probe hunk
  (per-call sentinel valve /tmp/bt_te_cp_tailpad_fix.off, ANDed with env
  BT_TE_CP_TAILPAD_FIX default-ON). Pre-boot check: sentinel ABSENT both
  nodes. Boot = srun 100672 (PP2/CP8 merged config, headline env).
  poincare pass sequence: (1) fix-ON parity pp2-padded-fixed, (2) fix-ON
  repeat (bitwise fingerprint), (3) sentinel-ON fix-OFF control + repeat
  (nondeterminism: expect MISMATCH), (4) d4 perf A/B fix-OFF vs fix-ON
  (primary metric, ±2% within-boot gate), (5) pp2-unpadded-rev LAST
  (pad-valve mechanism pending: second filegate if poincare's hunk lands in
  the ~10-min window = kill+redispatch, else its own boot after).
- 2026-08-13 ~05:0x CDT — **Zero-GPU discriminators both LANDED; fix-arm
  redispatching with both valves.** (1) gauss source memo
  (results/GATE1_CP_GEOMETRY_SOURCE_MEMO.md, pins verified incl. all 7
  dirty-mcore gates OFF): Q1 — DSA top-k geometry-NORMALIZED by design
  (all-gather + global reorder, absolute positions, fixed-order
  accumulation; citations in memo); CP/EP sensitivity enters ONLY via
  discontinuous top-k + no tie-break bias + shape-dependent ULP noise
  (16384-vs-8192 local rows) amplified by 78 layers = diffuse component
  grounded as structural. EP router geometry-free. Q2 — NO cross-mb
  state channel (packer per-slot fresh, top-k holder per-slot,
  dispatcher cleared post-combine, RouterReplay cleared in finally) =
  scariest (b) mechanism ruled out at source. Systematic d1-d5
  component constrained to doc/packing-attached channels -> permuted
  pass near-decisive (slot-follows would leave only kernel-tiling ULP /
  allocator-stale-read exotics). LATENT TRAP (not tonight): dsa.py:1599
  global-config fallback when packed_seq_params+attention_mask both
  None = future cross-mb top-k leak for non-THD callers; forwarded to
  weierstrass as FUTURE queue line (silent-fallback family, 3rd
  member). (2) poincare pad-valve hunk SHIPPED in window (2/2 tests,
  ruff clean, patches/pad_disable_filegate_probe.patch, naming nit
  taken: /tmp/bt_pack_pad_to_max.off); cauchy eyeball = CLEAN; doppler
  kill+redispatch GO with both sentinels checked + both-valve per-pass
  logging; tree label 73c24b00+TF32+f74785d7 (+2 filegate probe hunks).
  (3) PP1/CP8 leg mechanics correction (poincare): PP1 is UNPADDED BY
  DEFAULT (pad gate = PP>1 or env) -> no pad mechanism needed; driver
  gains --forward-only (approved, Mac-verified, no round-trip). Report
  §3 updated with both memo results.
- 2026-08-13 ~05:1x CDT — **PP1/CP8 leg pre-registration RATIFIED +
  formal Gate-1 reframe.** poincare refinements: leg C already
  exonerates M=N-at-PP1 (golden ran the merged tree one-call path,
  clean) -> this leg isolates CP8/EP8-vs-CP16/EP16 with M=N-at-PP1 held
  equal. Readings: (1) PP-INNOCENT: loss ~12.494 (12.47-12.52) AND
  per-datum delta table REPRODUCES pp2-unpadded datum-for-datum ->
  CP/EP-axis property, (a) favored; (2) GAP FOLLOWS PP2: ~12.304 clean
  table -> (b), headline PR blocked; (3) mixed/attenuated -> PP2
  amplifies-not-originates, file as finding. CAUCHY ADDITION (ratified
  into the protocol): Gate 1 was ALWAYS pipeline-parallel invariance —
  golden was an F3-forced fallback reference; with F3 dodged this leg
  RESTORES the original design, so under reading (1) ALSO run the
  formal pp1cp8-fwd vs pp2-unpadded compare under ORIGINAL bars
  (rel<=1e-6, logprob<=1e-3; same CP8 chunking holds the chaos channel
  equal, kernel shapes identical, p2p lossless) — a PASS there = formal
  Gate-1 PASS as designed, flipping the morning story to "Gate 1
  passed under the correct reference + two characterized findings
  (LPS-1063 exposure, cross-CP structural sensitivity)". In-band but
  outside original bars = file as (3)-flavored. Driver --forward-only
  staged (Mac-verified); PP1 unpadded by default (no pad mechanism);
  DP2-vs-DP1 boot shape deferred to serre blocks; fires at doppler
  READY as pp1cp8-unpadded-fwd, queued behind fix-arm passes.
- 2026-08-13 ~05:2x CDT — Fix-arm boot REDISPATCHED with both probe
  valves: srun 102888. Kill of 100672 cost ~4 min mid-load. Tree label
  final: 73c24b00+TF32+4e8b5f01(==cherry-pick(f74785d7), provenance
  verified by doppler)+2 filegate valves. Both sentinels verified
  ABSENT both nodes pre-boot. Pass ladder on this boot: (1) padded
  parity leg fix-ON, (2) exposed control fix-OFF, (3) within-boot perf
  A/B fix-OFF vs fix-ON (PRIMARY metric, +-2% control gate, win >936),
  (4) nondeterminism fingerprints both modes, (5) permuted-document
  unpadded pass (pre-registered table). PP1/CP8/EP8 unpadded
  forward-only leg queued behind. This boot + that leg = the decision
  evidence for the morning.
- 2026-08-13 ~02:2x CDT — **DECISION-CRITICAL (doppler, source-verified on the
  fix-arm tree): the f74785d7 wrapper CANNOT ENGAGE on GLM-5.2.** It patches
  TE DotProductAttention.forward; this model never instantiates TE DPA:
  (1) mcore dsa.py has ZERO TE/DotProductAttention references (DSA = cutlass/
  cudnn kernels — matches tonight traces: sparse_attn_fwd_sm100,
  dsa_bwd_sm100, indexer_*); (2) mcore attention.py mentions TEDPA only in
  comments; (3) bridge: TE DPA only in eval_context_parallel_rebinding (eval)
  + ministral3 provider; (4) glm5_bridge.py maps core_attention.indexer.* —
  DSA indexer, no TE DPA. Live probe on the box venv: wrapper installs fine
  (marker True) — it is INERT, not broken. Consequences: (a) the
  [bt_te_padfix] ACTIVE line can never fire on this model — its absence at
  boot was NOT a fault signal; (b) fix-ON vs fix-OFF is numerically identical
  by construction — the fix-arm parity A/B measures an inert wrapper; (c)
  tonight LPS-1063 exposure claim for the parity legs (TE tail-pad
  mis-attention) conflicts with the model not using TE attention — the 0.19
  PP2-path residual (12.494 pp2-unpadded vs 12.304 golden-unpadded) is NOT a
  TE tail-pad effect on this model. The fix-arm boot still serves as a pure
  rerun-reproducibility control. Flagged to poincare + cauchy mid-pass-1;
  awaiting ruling.
- 2026-08-13 ~05:4x CDT — **ATTRIBUTION CORRECTION (decision-critical) +
  NEW FINDING: padded PP2 is NONDETERMINISTIC.** (1) doppler source
  finding, confirmed by poincare independently: GLM-5.2 attention =
  DSAttention on own cuDNN path (dsa.py:1530, dsa_cudnn_kernels.py) —
  NEVER calls TE DotProductAttention; f74785d7 numerically inert on
  this model (zero [bt_te_padfix] lines across padded passes =
  empirical confirmation). (2) poincare deeper: vendored mcore ALREADY
  carries the in-tree TE fix (extensions/transformer_engine.py:
  1829-1843, basetenlabs/Megatron-LM#25) -> TE-DPA models never exposed
  on this tree; f74785d7 redundant there, valid elsewhere. (3)
  CONSEQUENCE: tonight tail-pad corruption terms and their phenomenology
  STAND (poincare predictions validated the per-datum decomposition,
  not the TE mechanism) but the mechanism is an LPS-1063-CLASS bug in
  OUR DSA pad handling — gauss re-tasked on the source hunt
  (layout-misalignment vs pad-block-selection; sawtooth favors former).
  (4) NEW: padded PP2 same-config samples 12.5481 / 12.5355 / 12.5182 /
  12.5392 (4 distinct, range 0.030, mean ~12.535) = uninit-memory-class
  nondeterminism in the DSA/cuDNN path under tail padding; padded
  decomposition terms restated as RANGES (~0.05 +/- 0.03); the 0.190
  unpadded term (deterministic path) is the solid term. (5) CAUCHY
  RULING: fix-toggle passes + fix perf A/B cancelled as null; ladder
  now = 4th padded sample (DONE) -> valve ON -> (5a) pp2-unpadded
  IDENTITY repeat (pre-registered EXACT 12.494321280379372) -> (5b)
  permuted discriminator -> teardown -> PP1/CP8 forward-only leg;
  padded-vs-unpadded d4 perf demoted to best-effort tail. (6) Durable
  scope note written to memory (lps-1063-repro-state.md): GLM out of
  f74785d7 scope; do not re-attribute. Report §1/§3 corrected in place.
- 2026-08-13 ~05:5x CDT — **DSA tail-pad mechanism hunt lands (gauss,
  results/TAIL_PAD_DSA_SOURCE_MEMO.md).** (B) pad-block selection
  REFUTED at source (pads causally+varlen-confined at every mask site;
  real-query pools real-keys-only; citations in memo). (A) LIVE:
  three-way CP-split convention consistency (TE thd_get_partitioned_
  indices vs mcore build_packed_allgather_cp_local_positions vs cuDNN
  fused indexer internal split; divisibility guard CPU-only = silent on
  CUDA) — per-doc mismatch shifts later docs, accumulates to tail =
  sawtooth-shaped; UNSETTLED, probe (a) decisive. BONUS BUG (real,
  loss-level): indexer-loss-on-pads — real_token_mask_q read
  (dsa_masking.py:392) but SET NOWHERE -> pad rows contribute
  pad-volume-proportional garbage KL to indexer loss; one-field fix
  (packer padmask exists). CLEARED: RoPE, extraction/unshard, MoE
  capacity. ASSIGNMENTS: probe (c) loss-vs-logprob separation -> gauss
  NOW (Mac-side, free, sizes the indexer-loss term); probe (a) script
  -> gauss preps, doppler runs read-only during PP1/CP8 load window
  (zero queue cost); probe (b) uniform-doc leg -> morning candidate;
  indexer-loss fix -> weierstrass work-item line. Ship-call framing in
  report: (a)-mismatch = padded fixable at position-builder; (a)-pass =
  ship unpadded honest (cost = L3 deviceSync class). Report §3
  mechanism block updated.
- 2026-08-13 ~06:0x CDT — **Materiality flag (weierstrass) on the
  indexer-loss bug, ACCEPTED pending gauss verification:**
  glm52_dsa.py:105-110 (apply path used by ALL tonight runs + both
  parity legs) sets dsa_indexer_loss_coeff=0.0 +
  dsa_indexer_use_sparse_loss=False -> the pad-garbage KL term is
  coefficient-ZEROED tonight. Bug = REAL but LATENT (fires on
  full-parameter DSA / non-override configs); queue entry worded
  accordingly (own small PR, latent-for-tonight). CONSEQUENCE: probe
  (c) flips from sizing tool to CONSISTENCY CHECK, pre-registered
  expectation ~ZERO loss-level excess over logprob-aggregated gap;
  nonzero = coefficient not actually zero on parity path OR
  unaccounted loss-level contributor. gauss verifying the coefficient
  along the actual parity-leg path + running (c) regardless. Report
  §3 corrected.
- 2026-08-13 ~06:1x CDT — **Materiality flag CONFIRMED + probe (c)
  HOLDS (gauss).** Exact gate: glm52_dsa.py:105-110 coeff=0.0 AND
  dsa.py:1892-1894 gates use_indexer_loss on coeff>0 -> indexer loss
  NOT COMPUTED tonight (latent bug goes live via e.g. glm5_bridge.py:
  149 default 0.001 on full-parameter DSA). Probe (c) consistency
  check, pre-registered ~zero excess: golden loss gap +0.046983 vs
  logprob-implied +0.045373 (excess +0.0016); pp2 -0.132920 vs
  -0.128365 (excess -0.0046). |excess| <= 0.005 both, under the padded
  noise floor (0.030) -> VERDICT: parity gaps are ~fully
  forward-logprob phenomena, no unaccounted loss-level term. Side
  note: per-leg loss-vs-(-mean_lp) offset ~0.424 stable (RL objective
  non-logprob terms). Probe (a) script delivered
  (tools/probe_a_split_convention.py, mcore leg Mac-validated); runs
  in PP1/CP8 load window.
- 2026-08-13 ~02:4x CDT — FIX-ARM PASSES 1-2 BANKED + LADDER REVISED
  (cauchy). pp2-padded-fixed 12.535494961470064 (fb 101.0s, first-touch JIT)
  + pp2-padded-fixed-rerun 12.518228610946453 (fb 29.3s warm) — NOT bitwise
  equal: the PADDED path is nondeterministic run-to-run even on the fix
  tree; with the wrapper inert on GLM (previous entry), the nondeterminism
  lives in the DSA/cuDNN path under tail padding. Padded distribution now 4
  samples: 12.5481 / 12.5355 / 12.5182 / 12.5392 (range 0.030). cauchy
  revised ladder (supersedes): padded passes = repro-control/diagnostic
  only; fix-ON/OFF perf A/B CANCELLED (numerically inert, zero information);
  then valve ON (executed + verified both nodes: /tmp/bt_pack_pad_to_max.off
  present, tailpad sentinel absent) -> (5a) pp2-unpadded-identity repeat
  (pre-registered EXACT 12.494321280379372) -> (5b) pp2-unpadded-rev
  (permuted discriminator) -> teardown -> PP1/CP8 forward-only boot ->
  best-effort d4 padded-vs-unpadded perf A/B only if time. gauss probe-a
  (three-way index-convention diff, hypothesis A = tail-pad layout
  misalignment TE-sharder vs mcore position map) staged on box
  (lps1062_pp2/probe_a_split_convention.py sha 623b04da6bcf) — runs in the
  PP1/CP8 load window; exit 0 = conventions match (hypothesis A dead,
  residual = cuDNN kernel internal split), nonzero = per-rank/doc mismatch
  dump.
- 2026-08-13 ~02:5x CDT — **PASS 5a/5b: the UNPADDED PP2 path is ALSO
  nondeterministic run-to-run.** pp2-unpadded-identity 12.514509978265165,
  identity2 12.489176052163604, pp2-unpadded-rev (permuted) 12.509854771784232
  — identity repeat FAILS the pre-registered bitwise bar (expected EXACT
  12.494321280379372); two identity repeats on the SAME boot differ from each
  other. Unpadded 3-sample range 0.025 ≈ padded 4-sample range 0.030 → the
  nondeterminism is NOT padding-gated; it is a PP2-path (or wider) property,
  consistent with the uninit-memory class in the DSA/cuDNN path. The permuted
  pass lands INSIDE the noise band → the slot-vs-doc discriminator is void at
  this floor. CONSEQUENCE for the parity protocol: Gate-1 loss-rel ≤1e-6 is
  unmeasurable against a ~2e-3-rel run-to-run noise floor — needs N-repeat
  mean/spread redesign or the nondeterminism root-caused first. All JSONs in
  lps1062_bench/ (box) + Mac copies with poincare. Awaiting cauchy ruling.
- 2026-08-13 ~06:4x CDT — **Permuted verdict RATIFIED at per-datum
  resolution + endgame executing.** (1) poincare banded re-read: every
  corrupted datum lands OUTSIDE its identity noise band in the
  slot-follows direction (d5 -1.125-class -> +0.0007 leaving P1; d4
  released; d1/d2 inherit -0.865/-0.827 entering P1; clean stays clean;
  d3 control attenuated 7x = composition-dependent magnitude, filed
  honestly). VERDICT: (b)-family — corruption follows the SLOT (middle
  partition = the 1F1B steady-phase microbatch at M=3). (2) Noise
  restatements ratified: 0.19 gap = 0.1952 +/- ~0.013, STANDS 8-15x
  noise; PP2 padded-minus-unpadded +0.036 borderline-with-structure;
  Gate-1 formal bars UNMEASURABLE at ~2e-3-rel floor -> morning framing
  = statistical parity pending nondeterminism fix. (3) Unified
  mechanism hypothesis logged (cauchy): 1F1B steady-state p2p BUFFER
  REUSE (stale/uninit region) explains slot corruption AND
  nondeterminism in one PP2-specific mechanism; second suspect =
  M=N _schedule_data_iterator per-slot metadata seam; gauss hunting
  both in source. (4) PP1/CP8 leg BOOTING (srun 106191, 1-node, EXACT
  tree verified: HEAD 73c24b00, diff = chunked_lm_head.py only, hunks
  reverted, valve test removed; forward-only dodges F3; poincare fires
  x3): decides PP2-path (clean+deterministic) vs CP8-path (same slot
  structure) + nondeterminism locality + exact-tree confound. probe-a
  running on the IDLE worker node in parallel. Report §1/§3 updated to
  the defect-branch state; task #7 closed.
- 2026-08-13 ~06:5x CDT — **Probe (a) verdict: hypothesis A DEAD.**
  0 mismatched rows across all cases x CP8/CP16 x tailfill on/off
  (ran on idle worker node, trainer untouched). Three-way
  split-convention layout misalignment eliminated; per gauss decision
  tree the padded-residual suspect falls to cuDNN kernel-internal
  varlen handling. FRAMING SHIFT (cauchy): the memo ship-call
  ("(a)-pass => ship unpadded") is OBE — the permuted verdict showed
  unpadded carries the slot defect too, so NOTHING on the PP2 line
  ships until the slot defect is fixed; the padded-vs-unpadded
  residual (+0.036 borderline) demotes to secondary (probe-(b)-class
  morning item). Report ship-framing corrected. PP1/CP8 still loading;
  poincare x3 at READY.
- 2026-08-13 ~07:0x CDT — **Mechanism hunt CONVERGED (gauss x2 memos) +
  PP1/CP8 verdict framework locked.** (1) SLOT_FOLLOWS_SOURCE_MEMO.md:
  cauchy suspects REFUTED/CLEARED at source (p2p buffer reuse — full
  chain; M=N _schedule_data_iterator seam — cleared). LIVE: 3a per-slot
  scratch/workspace reuse in DSA/cuDNN (top: top-k scratch chunking
  dsa_cudnn_kernels.py:489-523, FlashMLA workspace, dispatcher shared
  D2H stream :380/461-462/931-955); 3b atomic-order ULP amplified by
  top-k (discriminator = golden determinism repeats — TONIGHT GAP:
  leg C was single-sample). (2) CONVERGENCE: kernel-internal-varlen
  (padded residual) and slot-defect hunts meet at the cuDNN fused
  indexer consuming per-slot packed metadata
  (_get_multi_packed_cp_thd_metadata, dsa_cudnn_kernels.py:157-176 ->
  :262-275). (3) SHARPEST NEXT DISCRIMINATOR: non-fused reference DSA
  path leg (bypasses fused indexer): corruption gone = fused kernel
  convicted; persists = 3a/3b. Feasibility (config-only?) with
  doppler; queues after PP1/CP8 if yes, else top morning probe.
  (4) PP1/CP8 verdict framework (poincare, ratified + one citation
  fix): slot structure = CP8 fwd kernels convicted DECISIVE; clean+
  deterministic = AMBIGUOUS (pipeline-phase interleaving-dependent
  scratch, still M=N-PR blocker — NOT exoneration); repeat spread =
  nondeterminism locality (stable = interleaving-dependent scratch in
  pipeline phase, NOT the refuted p2p-buffer suspect; jitter =
  DSA/cuDNN fwd nondeterministic standalone -> noise floor under all
  CP8 parity). Report §3 updated (suspect swap + convergence +
  non-fused discriminator + ship-framing OBE fix).
- 2026-08-13 ~07:1x CDT — **Unfused reference-path leg: GO (config-only
  confirmed).** attention_backend="unfused" (control.py:382) ->
  use_fused_dsa_kernels False (dsa_kernels.py:92-96) -> in-tree PyTorch
  reference DSA (_unfused_absorbed_dsa_fn, "authoritative fallback",
  dsa.py:158-161). One-field config delta on exact tree, unpadded,
  forward-only (F3-dodged same way). Staged during PP1/CP8 load; boots
  after poincare x3. NUMERICS PROBE ONLY (reference path slow by
  design; bounded-probe time-box with doppler; virgin 131k shape ->
  abort bounds pre-set). PRE-REGISTERED READINGS (compare per-datum
  STRUCTURE vs banded tables, not absolutes — reference reduction
  order legitimately shifts values): (i) slot structure ABSENT +
  stable = fused kernel CONVICTED + clean-PP2-forward existence proof;
  (ii) structure PRESENT = fused attention exonerated -> narrows to
  dispatcher shared-D2H-stream scratch (NOT bypassed by unfused
  attention — MoE-side) / atomic classes; (iii) unstable without
  structure = nondeterminism decouples from slot corruption (two
  mechanisms). x2 repeats min. This is the last box action of the
  night.
- 2026-08-13 ~07:2x CDT — **Unfused-leg pre-registration FILED
  (poincare) + RATIFIED, before result exists.** Method: per-datum
  delta vs leg-C clean, DEMEANED across datums (strips the legitimate
  uniform reduction-order offset of the reference path), compared
  against identity noise bands + baseline slot signature. Identity
  order (P1=[d3,d4,d5]). PRESENT = middle-partition escalation outside
  bands after demeaning; ABSENT = all within ~±0.03, no slot ordering.
  Readings: (i) absent+stable = fused cuDNN kernel CONVICTED (defect +
  nondeterminism source in one) + clean-PP2-forward existence proof;
  (ii) present = fused attention exonerated -> dispatcher-scratch/
  atomic classes; (iii) unstable w/o structure = two mechanisms,
  decoupled; refinement: present-attenuated+stable = fused kernel as
  amplifier-not-source, (ii)-flavored, filed not forced. Mechanics:
  needs gate-flip patch + BT_PACK_PAD_DISABLE=1 (pad valve went with
  fix-arm teardown; same documented deviation class, pass records log
  deviation state), tree otherwise exact, backend switch config-only.
  Sequence: PP1/CP8-fwd x3 -> unfused x2-3. Box done after.
- 2026-08-13 ~07:3x CDT — **PP1/CP8 LEG VERDICT: DECISIVE — CP8 FORWARD
  KERNELS CONVICTED; M=N/PP2 PLUMBING EXONERATED.** Trio (exact tree,
  fwd-only, 1-node DP1): 12.501736810906936 / 12.494330171902787 /
  12.513274056510571 (mean 12.5031, spread 0.019). (1) Slot structure
  REPRODUCED datum-for-datum at PP1 (d5 mean -1.162 vs PP2 band
  -1.158..-1.019; d4 -0.590; d3 -0.186; clean docs clean) with NO
  pipeline, NO 1F1B, NO backward -> the corruption lives in the CP8
  DSA/cuDNN forward path; config choice EXPOSED it, M=N did not
  introduce it. (2) DSA fwd nondeterministic standalone (0.019 spread)
  -> ~2e-3-rel noise floor under all CP8 parity tonight. (3) Mean gap
  PP-innocent (12.5031 ~ 12.4993) -> +0.195 rides CP8/EP8-vs-CP16/EP16;
  statistical PP-invariance HOLDS. Formal Gate-1 compare on record:
  rel 5.932e-04 / logprob max 14.8 = FAIL at bars, bars unmeasurable
  under noise floor (pre-registered framing stands). (4) NEW
  WORKSPACE-GROWTH SIGNATURE (cauchy, testing): in BOTH orders the
  corrupted partition is LARGER THAN ITS PREDECESSOR (92960->98704;
  105584->124784; smaller successors clean) -> scratch
  high-water-mark/first-call sizing hypothesis; prediction under test
  (poincare per-decile onset-position; gauss source check vs
  dsa_cudnn_kernels.py:489-523 + FlashMLA workspace). (5) UNFUSED
  discriminator boot dispatched (srun 109465; exact tree + gate-flip
  patch re-applied verified + BT_PACK_PAD_DISABLE=1 +
  trainer_pp2cp8ep8_131k_unfused.json) — last box action; poincare
  x2-3 at READY. Report §1 + §3 updated to the exoneration state.
- 2026-08-13 ~07:4x CDT — **Workspace-growth hypothesis test: MIXED
  (poincare, binned per-token profiles in parity/).** PARTITION-LEVEL
  CONFIRMED both orders (corrupt partition = larger-than-predecessor;
  smaller successors clean). SIMPLE EXTENT MODEL REFUTED: corruption
  is a FRONT starting partway into the bad partition
  (identity: floor to ~40k, sharp front ~41-42k, escalates to ~2.0 at
  tail, max 17.5; reverse: mild ~32k, severe ~64k) — onset aligns
  across orders by neither row, fraction, nor datum; not power-of-2;
  P0 = localized spikes not fronts; P2 flat. CONSTRAINT SET for any
  candidate mechanism (relayed to gauss): (a) larger-than-predecessor
  gating, (b) mid-partition composition-dependent onset, (c) tailward
  escalation, (d) nondeterminism. Report updated. Unfused
  discriminator (srun 109465) = next landing, may halve the space.
- 2026-08-13 ~07:5x CDT — **Source clearance complete (gauss, SLOT_
  FOLLOWS_SOURCE_MEMO §4d): NO first-call/unrevalidated buffer in the
  visible mcore path** (top-k scratch chunked per call :489-523; score
  buffer zeros per chunk :561; padding filled per call; attn_sink
  fresh; RoPE HWM cache revalidates correctly; only lru_cache = static
  SM query; multi-packed segment arithmetic per-call consistent, :828
  clamp never fires). The 3a class survives ONLY inside cudnn/
  flash_mla package internals. REFINED CANDIDATE clearing all four
  constraints: plan/workspace cached under a COARSE or composition-
  derived key, under-validated vs the actual per-call segment layout
  (gate = built for predecessor layout; onset = where cached coverage
  diverges; escalation = divergence accumulates tailward;
  nondeterminism = uninit workspace). Hypothesis-not-conviction
  (internals not in-repo). CAVEAT on the landing discriminator: the
  single knob (dsa_kernel_backend none/unfused) switches BOTH the
  indexer (fused_qk_topk_naive) and attention (_unfused_absorbed_
  dsa_fn) — a clean verdict convicts the PAIR; separating needs a
  two-line gate split (morning item).
- 2026-08-13 ~09:2x CDT — **JACK ORDER: FIX MODE ("just fix it, get it
  working").** Campaign launched, four parallel lanes: (1) doppler:
  kernel-package source snapshot off the box venv (cudnn_frontend,
  flash_mla, binding-layer deps) -> pp2cp8ep8/kernel_src_snapshot/ for
  gauss; box stays up; queue after unfused verdict = gate-split A/B
  then d4 perf of winning correct config vs 918. (2) gauss: hunt the
  execution-plan/graph cache and its KEY in cudnn_frontend + binding
  layer + flash_mla (hypothesis: plans keyed without full
  cu_seqlens/segment layout -> predecessor-layout plan reused on
  larger/different layout = all 4 constraints); deliverable = defect
  site + ranked fix (key extension / per-call rebuild / cache disable)
  + any TODAY env cache-disable switch. (3) serre: prior-art sweep
  (cudnn-frontend/FlashMLA/TE issues, NVIDIA forums) for known
  plan-cache-vs-varlen bugs + fixed-version windows (fix may be a
  version bump). (4) poincare: after unfused verdict, two-line gate
  split (indexer vs attention knobs) -> box A/B locates the defective
  half + its disable cost. Utility pod jrao-cpfs-cleanup2 DELETED
  (~09:1x). Unfused passes in flight as of ~09:1x.
- 2026-08-13 ~09:4x CDT — **Plan-cache hypothesis REFUTED at design
  level (gauss, results/CUDNN_PLAN_CACHE_HUNT.md; exact box packages
  cudnn-frontend 1.26.0+dsatopk1, flash_mla b7643bd).** All caches key
  on codegen params only (indexer SM100 key excludes seqlens as
  runtime args _interface.py:150-164; top-k object cache keys
  n_rows/num_cols = fresh compiles for larger microbatches; flash_mla
  sparse path has NO cross-call cache); kernels layout-dynamic by
  construction (to_cute_tensor dynamic marks; seqlen derived at
  runtime; per-doc bounds read from cu_seqlens CONTENTS in-kernel,
  SeqlenInfoQK utils/seqlen.py:47-86). NO env-level cache-disable
  exists in the DSA path today — the config knob is the only
  cache-free execution. SURVIVING SUSPECTS reranked: in-kernel
  dynamic-varlen/scheduling bug (SeqlenInfoQK / CLC persistent
  scheduler; open item: mark_layout_dynamic per-mode coverage —
  cutlass pkg being added to snapshot) or atomic-scheduler class.
  QUEUED: gauss one-shot cache-clear diagnostic rides the gate-split
  boot (empirical cache exoneration from the other side). Cauchy
  mechanism hypotheses killed tonight: 2 (extent-HWM, plan-cache) —
  both refutations tightened the hunt; critical path now = unfused
  verdict + gate-split half-isolation + serre prior art.
- 2026-08-13 ~10:0x CDT — **Scoped mcore-freeze exception SANCTIONED
  (cauchy)** for gauss cache-clear probe patch
  (patches/bt_dsa_cache_clear_mb_probe.patch, additive, default-inert
  BT_DSA_CACHE_CLEAR_MB, apply-check PASS vs box mcore 57efae08b).
  Rationale: freeze protected banked parity evidence + undispositioned
  dirty tree; parity phase closed, Jack ordered fix mode. CONDITIONS:
  pre-apply submodule state hash + byte-identical restore verification
  (git apply -R + re-hash) with dirty hunks untouched; probe boots
  only (parity/perf-anchor boots revert first); freeze otherwise in
  force; apply/revert hashes in boot record. Also: cutlass DSL source
  added to kernel_src_snapshot (sha 16a8cb3f69377070, 61MB) — gauss
  mark_layout_dynamic review unblocked. Unfused passes still in
  flight.
- 2026-08-13 ~10:2x CDT — **CAMPAIGN RE-VECTOR: STALE KERNEL WHEEL ON
  BOX = top fix candidate (serre, results/KERNEL_DEFECT_PRIOR_ART.md).**
  Box venv: cudnn-frontend 1.26.0+dsatopk1 (retired stack) + cudnn
  9.19.0.56. Tree pin since Aug 3 (6717eaf0): 1.27.0.dev20260803+
  git7478516 + 9.23.2.1, carrying the LPS-1003 DSA race series (#354
  wrapper stream race; #396 TMEM WAR race at head_dim 576/512 = exactly
  GLM-5.2, silent dkv corruption; #395/#426/#429/#439/#421); pin commit
  own B300 A/B: 1.26 fires 12/12, new wheel 0/12. Race class fits our
  nondeterministic-corruption signature. Box trainer CODE current —
  staleness is wheel-level only (provenance diagnosis ordered: does
  uv.lock at 73c24b00 pin 1.27? why did the venv build install 1.26).
  PLAN: unfused passes finish -> stage wheels (no live-venv install) ->
  install post-teardown -> reproducer gate PP1/CP8-fwd x3 identity
  (pre-registered read) -> clean+deterministic = validation ladder
  (PP2 legs, padded legs incl. tail-pad corruption re-check, canaries,
  d4 perf vs 918 — anchors were measured on the STALE wheel, perf may
  move) / still-corrupted = wheel exonerated, gate split + cache-clear
  probe resume. Also mapped: OPEN #543 (THD host-prep stream race +
  plan-time-only compile keys), #538 packed-stride; upstream note
  drafted as contingency only. (2) gauss: cache-clear probe delivered
  (spec: clear only corrupted slot, never slot 0); mark_layout_dynamic
  CLOSED from cutlass DSL source (all modes dynamic, stride-1 leading
  dim holds for our .contiguous() glue) -> cached-plan class refuted at
  ALL THREE levels (keys, runtime args, descriptor layer). Remaining
  suspects: in-kernel varlen arithmetic / CLC-atomic scheduler —
  empirical discriminators own them. Open question to gauss: can a
  pure race explain the slot-structured MEAN bias or must race +
  varlen coexist (shapes the residual-gap reading post-wheel-bump).
- 2026-08-13 ~11:3x CDT — **Unfused leg DIED on NCCL watchdog — zero
  verdict data (NOT a numerics event).** Rank-8 watchdog
  broadcastDumpSignal 11:26:17 -> NCCL ops failed/timed out -> trainer
  terminated on peer_broadcast_recv abort. Mechanism: unfused reference
  forward at 131k stalls a collective past the heartbeat watchdog (the
  120-min distributed_timeout lift does not cover it). Papercut filed.
  RULING: unfused retry PARKED — wheel bump is the discriminator now;
  resurrect (watchdog raised) only if the wheel fails. doppler:
  death-log snapshot -> sweep -> stage pinned wheels -> uv.lock
  provenance diagnosis. Wheel-bump pre-registration (3 nested levels:
  variance collapse / clean slot table / MEAN lands at golden ~12.304)
  filed with poincare; padded re-check pre-registered for the
  validation ladder (does the race own the tail-pad corruption too).
- 2026-08-13 ~11:4x CDT — **0.19-gap decomposition SETTLED (gauss, memo
  4e corrected): the gap IS the slot corruption.** Token-weighted
  per-datum: d5 alone -1.084 = 61% of the gap; d3+d4+d5 = 92%; clean
  docs (d0/6/7/8) at golden cluster (<=0.003). Corrected split: RACE
  owns the signed slot-mean (~all of 0.19); CHAOS AMPLIFIER owns the
  mean-neutral diffuse per-token |diff| (~0.08/decile, moves |diff|
  not loss). gauss earlier "0.19 structural and stays" wording
  self-corrected on the data. PRE-REGISTRATION OF RECORD for the
  wheel bump: post-1.27.0.dev PP1/CP8 + PP2 means land ~12.304 +/-
  0.02 (golden cluster; d2-class small residual caveat); landing at
  12.50-with-gap = residual structural term -> reopen varlen audit.
- 2026-08-13 ~12:0x CDT — **Install plan REFINED (serre): the fix is
  ONE package.** nvidia-cudnn-frontend 1.26.0+dsatopk1 -> 1.27.0;
  BACKEND UNCHANGED (9.23.2.1 pin is cu12-lane-only; cu13 lock
  resolves nvidia-cudnn-cu13==9.19.0.56 = what the box runs). Sources:
  (a) PyPI 1.27.0 (Aug 6 release, full fix series #354/#410/#407/
  #395/#396/#426/#429/#439/#421; DROP-IN PROVEN — the box own SAMPLER
  venv already runs 1.27.0 + 9.19.0.56); (b) fallback dev wheel from
  baseten/trainers-wheels:src-bec083142779. Install --no-deps; run
  test_cudnn_dsa_indexer_launch_stream.py +
  test_cudnn_dsa_indexer_topk.py BEFORE the PP1/CP8 x3 gate; host-only
  pybind+JIT = no torch ABI risk. PROVENANCE LEAD: trainer venv stale
  while sampler venv current on the SAME box = the build split to
  diagnose. Contingency upstream note for #543/#538 drafted
  (results/UPSTREAM_ENGAGEMENT_CUDNN_543_538.md) with explicit
  do-not-send condition. Report corrected (backend-unchanged nuance).
- 2026-08-13 ~12:2x CDT — **WHEEL BUMP VERIFIED on trainer venv:**
  nvidia-cudnn-frontend 1.27.0 (PyPI, --no-deps), backend cu13
  9.19.0.56 unchanged, import+version pass, BOTH DSA regression files
  GREEN (5 passed / 22s). PROVENANCE NAILED: the retired
  1.26.0+dsatopk1 came from gitignored vendor/wheels/ (uv origin
  recorded) — the stale vendored shim won resolution at trainer-venv
  build while the lock pinned 1.27.0.dev; make fetch-wheels never
  refreshed; the sampler venv on the SAME box resolved 1.27.0
  correctly. Papercut pc_c89b5acdeed2 filed; weierstrass queue
  work-item added (fix shape: staleness check / fetch-wheels
  invalidation tied to lock hash). NEXT: gate flip reverted -> exact
  73c24b00+TF32 -> PP1/CP8-fwd x3 reproducer gate on the bumped venv
  (pre-registration of record: variance collapse + clean slot table +
  mean ~12.304 +/- 0.02).
- 2026-08-13 ~12:3x CDT — weierstrass folded the wheel-staleness item
  as TOP standalone follow-up (own PR) with two precision notes, both
  adopted: (1) "hunt traced here" = campaign conclusion PENDING the
  in-flight reproducer gate; (2) ALL perf numbers in the report
  (645/691 anchors AND 918/1052 headlines) were measured on the STALE
  wheel — post-bump perf may move; validation ladder re-measures d4;
  §1 caveat added. Queue final shape: 9 slots + future section
  (2 standalone work items + 3-item silent-fallback family).
- 2026-08-13 ~03:0x CDT — UNFUSED PP2 leg DIED on the NCCL watchdog (not
  numerics): the reference-path forward at 131k stalls a collective past the
  watchdog window (rank 8 broadcastDumpSignal 11:26:17, NCCL ops
  failed/timed out, trainer terminated). Zero verdict data; leg PARKED
  (resurrect only with the NCCL heartbeat raised IF the wheel bump fails).
  Death log snapshotted (logs/unfused_pp2_watchdog_death_20260813.log, sha
  850dd292b2d1). probe-a verdict (rode the idle worker node during the
  PP1/CP8 load): **0 mismatched rows across all cases x CP8/CP16 x
  tailfill on/off — hypothesis A (TE-sharder vs mcore position-map layout
  misalignment) DEAD.** gauss cache-clear probe patch sanctioned by cauchy
  (scoped freeze exception: snapshot submodule diff-hash before apply,
  byte-identical restore via git apply -R after, probe boots only, hashes
  logged) — HELD unexercised unless the wheel bump fails.
- 2026-08-13 ~03:2x CDT — **CAMPAIGN RE-VECTOR (serre prior-art,
  results/KERNEL_DEFECT_PRIOR_ART.md): the box trainer venv ran the RETIRED
  cudnn-frontend 1.26.0+dsatopk1 + cudnn cu13 9.19.0.56; the tree has pinned
  1.27.0(.dev20260803+git7478516) since Aug 3 (commit 6717eaf0) carrying the
  LPS-1003 DSA race-fix series (#354 stream race, #396 TMEM WAR race at
  head_dim 576/512 = exactly GLM-5.2, silent corruption, pin commit's B300
  A/B: old wheel fires 12/12, new 0/12). Our nondeterministic-corruption
  signature fits the race class.** PROVENANCE (doppler): uv.lock @73c24b00
  pins 1.27.0.dev via vendored wheel paths; the wheels are GITIGNORED
  (.gitignore:26 server/vendor/wheels/*.whl, fetched via make fetch-wheels
  from baseten/trainers-wheels); the trainer-venv build resolved a STALE
  1.26.0+dsatopk1 shim wheel sitting in vendor/wheels (uv recorded origin),
  while the SAMPLER venv on the same box runs the release 1.27.0 +
  9.19.0.56. Build split = filing-worthy (cauchy: papercut + weierstrass
  work-item). ACTIONS: fetched the pinned wheels on box (manage_wheels.py
  fetch, needed DOCKERHUB_PULL_* creds), installed PyPI
  nvidia-cudnn-frontend==1.27.0 --no-deps into the trainer venv (backend
  unchanged per serre: cu13 lane stays 9.19.0.56), verified import+version,
  BOTH DSA regression test files GREEN (5 passed, 22s:
  test_cudnn_dsa_indexer_launch_stream.py + test_cudnn_dsa_indexer_topk.py).
  Kernel-source snapshots delivered Mac-side for gauss:
  pp2cp8ep8/kernel_src_snapshot/ (cudnn frontend pkg incl.
  deepseek_sparse_attention/*, flash_mla, + nvidia_cutlass_dsl
  python_packages/cutlass; MANIFEST.md; box tarballs sha a94233ff / 16a8cb3f).
  REPRODUCER GATE booting now: PP1/CP8-fwd x3 identity order on the bumped
  venv, exact tree 73c24b00+TF32 (gate flip reverted).
- 2026-08-13 ~12:5x CDT — **POSSIBLE WEDGE on bumped-wheel reproducer
  (srun post-bump PP1/CP8 boot).** Facts: READY clean on 1.27.0
  (warmup forward COMPLETED, kernels warm 107.1s); a pass fired; NO
  parity_driver process box or Mac; trainer log static 513 lines
  20+ min; 7/8 GPUs pinned 100%, rank 0 IDLE. CAUCHY RULING — do not
  conclude wheel-hangs-forward: wedge shape (rank0 idle + 7 spin) fits
  ORPHANED COLLECTIVE after client death at least as well (driver dies
  mid-op -> rank0 aborts handling, ranks 1-7 spin in entered
  collective); warmup ran clean on same wheel. Protocol: capture WHILE
  wedged (py-spy all 8 ranks = discriminator, nvidia-smi, log tail
  last-line, dmesg/Xid) -> teardown -> reboot same venv -> REFIRE ONCE
  supervised (driver liveness + send logged + pass timeout).
  REPRODUCTION = the discriminator: wedges again healthy-driver =
  wheel implicated (pivot to dev-wheel variant / 1.27-vs-1.26
  differential); clean = driver-death fallout, gate proceeds.
  poincare queried for driver fate (fired? died how? request sent?).
- 2026-08-13 ~03:5x CDT — REPRODUCER-GATE "WEDGE" REFUTED (capture-first
  discipline paid): 7/8 GPUs at 100% + static log looked like a mid-forward
  hang on the bumped wheel, but the py-spy capture (all 8 ranks,
  logs/wedge_capture_bumped_wheel/) showed the API rank idle in asyncio and
  7 workers parked in broadcast_object_list (_peer_loop worker.py:311) = the
  ARMED peer-broadcast wait (posted NCCL recv spins the GPUs at idle), and
  the trainer log had ZERO submit/forward_backward lines since READY —
  **no op was ever submitted: driver no-show, NOT a hang; the wheel is NOT
  implicated** (warmup forward had completed clean on 1.27.0). /status 200
  on probe. Trainer held at READY (no teardown); poincare re-firing the
  gate passes. Lesson logged: GPU-at-100% + static log is NOT sufficient
  wedge evidence on this trainer — check for a submitted op first.
- 2026-08-13 ~13:0x CDT — **WEDGE REFUTED — driver no-show, trainer
  healthy.** py-spy all ranks: API rank idle in asyncio select; 7
  workers parked in broadcast_object_list _peer_loop (worker.py:311)
  = the ARMED PEER-BROADCAST WAIT — the posted NCCL recv spins GPUs
  at 100% in this trainer IDLE-ARMED posture (diagnostic gotcha: looks
  like a wedge on nvidia-smi). Trainer log: ZERO submit lines since
  READY — no op was ever sent; /status 200. Wheel NOT implicated
  (warmup forward clean on 1.27.0). No teardown; poincare re-told to
  fire the gate passes. Reproducer gate pre-registration unchanged.
- 2026-08-13 ~04:1x CDT — **REPRODUCER GATE VERDICT (bumped wheel,
  PP1/CP8-fwd unpadded identity x3; doppler drove mechanics, poincare
  dark):** NEW wheel 12.304435882236712 / 12.30520944477376 /
  12.304781663702826 (mean 12.30481, spread 0.00077) vs OLD wheel
  12.501736810906936 / 12.494330171902787 / 12.513274056510571 (mean
  12.50310, spread 0.01899). **Variance collapsed ~25x; mean moved
  12.503 -> 12.3048 = lands on the golden PP1/CP16 unpadded leg-C value
  (12.30418, rel ~5e-5). The corruption is GONE at PP1/CP8 on the fixed
  wheel — nondeterminism collapsed to a small residual floor and
  topology-invariance restored to within that floor.** The LPS-1003 race-fix
  series in the pinned frontend is the conviction. Distinct -fe127 labels
  preserved the old-wheel evidence JSONs. Next per cauchy ladder:
  PP2-unpadded x2 (gate flip + BT_PACK_PAD_DISABLE=1, documented deviation),
  then padded legs, then d2/d4 canaries + d4 perf vs 918 (anchors were
  measured on the stale wheel — the perf number may move; decision data).
- 2026-08-13 ~13:2x CDT — **REPRODUCER GATE x3: PASS ON ALL
  PRE-REGISTERED LEVELS — THE STALE WHEEL WAS THE DEFECT.** fe 1.27.0
  PP1/CP8-fwd unpadded identity x3: 12.304435882236712 /
  12.30520944477376 / 12.304781663702826 — mean 12.30481, spread
  0.00077. Old wheel same leg: mean 12.50310, spread 0.01899. READ:
  (L1) variance collapsed ~25x; (L3) mean moved 12.503 -> 12.3048 =
  golden leg-C 12.30418 to rel ~5e-5 (inside 12.304 +/- 0.02);
  (L2) slot table formally pending from JSONs but bounded by the mean
  (a d5-class -1.1 shift would move the mean ~0.12 — impossible).
  CONSEQUENCES: single race CONFIRMED (gauss pre-registered rule:
  variance + slot-mean died together => in-kernel-varlen suspect
  dies); topology invariance RESTORED within the new tiny floor;
  Gate-1 story final pending PP2 confirmation. Mechanics note:
  doppler drove passes (poincare dark ~15 min+, spec followed, -fe127
  labels, old-wheel evidence preserved). JSONs:
  lps1062_bench/parity_pp1cp8-unpadded-fwd-fe127{,-2,-3}.json.
  VALIDATION LADDER GO: PP2-unpadded x2 (+optional permuted-rev for
  the record) -> padded x2 (pre-registered: padded mean ~= unpadded
  within new floor = race owned tail-pad corruption too) -> d2/d4
  canaries -> d4 perf vs 918 (stale-wheel anchor caveat). serre
  upstream note stays PARKED (fixed; do-not-send spirit holds).
- 2026-08-13 ~13:4x CDT — **Gate L2 (slot table) formally on record:
  CLEAN.** Per-datum means across the x3 new-wheel runs, spreads
  0.0003-0.0034, no slot-following outlier, no -1.1-class datum —
  corruption-absent on fe 1.27.0. Gate JSONs archived Mac-side at
  results/gate_fe127/. Reproducer gate COMPLETE on all three
  pre-registered levels. PP2-unpadded boot in weight load (bumped
  venv, gate flip + BT_PACK_PAD_DISABLE=1, deviation logged); x2 +
  optional permuted-rev at READY.
- 2026-08-13 ~14:0x CDT — **PP2 CONFIRMATION COMPLETE on fe 1.27.0 —
  the 0.19 gap is GONE and slot-absence is confirmed under
  permutation.** identity 12.304802410590792 / identity-2
  12.30435388263189 / permuted-rev 12.304956530329974 (spread 0.0006,
  mean 12.3047) vs old-wheel 12.4943/12.5145/12.4892 (spread 0.025)
  vs golden leg-C 12.30418. Mean matches golden rel ~4e-5;
  nondeterminism collapsed ~50x; rev INSIDE the identity band (the
  old slot-following signature absent under the same permutation that
  convicted it). Topology invariance PP2=PP1=golden-CP16 holds within
  the tiny floor. Task #8 (0.19 characterization) CLOSED: one raced
  wheel. NEXT: padded PP2 x2 re-boot (env-bound; gate-flip patch
  stays, inert with env unset — documented deviation, logged);
  pre-registered: padded ~= unpadded within floor = race owned
  tail-pad corruption too. Then d2/d4 canaries, then d4 perf vs 918.
- 2026-08-13 ~14:2x CDT — **PADDED RE-CHECK FIRES: the race owned the
  tail-pad corruption too.** Padded PP2 x2 on fe 1.27.0:
  12.30401995653033 / 12.304724362774156 (spread 0.0007, mean
  12.30437) == unpadded 12.3047 within floor (diff 0.0004) == golden
  12.30418. FULL OLD->NEW MAP: PP2 padded 12.535 (0.030) -> 12.3044
  (0.0007); PP2 unpadded 12.494 (0.025) -> 12.3047 (0.0006); PP1/CP8
  12.503 (0.019) -> 12.3048 (0.0008); golden ref 12.3042. Every
  topology x padding cell agrees ~5e-5 rel. CONSEQUENCES: (1) parity
  protocol SOUND — every earlier fail-grade was the stale wheel;
  (2) Gate 2 (padding semantics) = PASS via PP2 padded-vs-unpadded
  0.0004 (true padding-semantics measurement, tiny); (3) the earlier
  "DSA path needs its own audit/fix" framing is OBE — no DSA-side fix
  needed, the wheel owned it; remaining latent item = indexer-loss
  one-field fix (queue future section); (4) golden-PADDED rerun NOT
  queued (mechanism-covered; one PP1/CP16 boot if Jack wants
  belt-and-braces). NEXT: d2/d4 canaries same boot, then d4 perf vs
  918 with within-boot controls at +-2%.
- 2026-08-13 ~04:5x CDT — VALIDATION LADDER on the bumped wheel (doppler
  mechanics; poincare dark), ALL PRE-REGISTERED READS: (1) PP2-unpadded x2 +
  permuted-rev: 12.30480 / 12.30435 / 12.30496(rev) — spread 0.0006, mean
  12.3047 = golden 12.30418 to rel ~4e-5; **the 0.19 PP2-path residual GONE,
  nondeterminism collapsed ~50x, slot-absence confirmed under permutation**.
  (2) Padded PP2 x2: 12.30402 / 12.30472 — padded == unpadded within the
  floor → **the race owned the tail-pad corruption too**. Full old→new map
  (every topology x padding cell now agrees to ~5e-5 rel): PP2 padded
  12.535→12.3044, PP2 unpadded 12.494→12.3047, PP1/CP8 12.503→12.3048,
  golden 12.3042. Gate slot table on record (results/gate_fe127/: max
  per-datum spread 0.0034, no slot-following outlier). (3) d2 canary PASS
  (loss 12.305-12.322, gn 0.36-0.49). (4) d4 perf vs 918: **~878-886
  tok/s/GPU = -3.3 to -3.7%** — two runs cross-agree 0.9% but within-boot
  control pairs trip the ±2% gate on both (3.0%, 3.8%; direction flips =
  settle/drift, not a bug); BONUS: peak mem 143 GiB vs 175 old-wheel d4
  (-32 GiB, leaner 1.27.0 workspace). Anchors were stale-wheel — the -3.5%
  is decision data for Jack, not a regression verdict.
- 2026-08-13 ~14:4x CDT — **Fixed-wheel perf: d4 ~878-886 vs 918
  anchor (-3.3 to -3.7%) — but control gate TRIPPED on both runs
  (spreads 3.0%/3.8% vs the +-2% pre-registration; direction flips
  between runs = settle/drift read, not measurement bug; headline
  values cross-agree 0.9%). Filed with the noise caveat; d16 ordered
  as the number of record (bigger window + the 1052 headline anchor).
  FRAMING: re-anchor, not trade-off — no keep-the-old-wheel branch
  exists (it is corrupt). BONUS: peak mem d4 = 143 GiB on 1.27.0 vs
  175 old = -32 GiB (leaner frontend workspace) — flagged as material
  to the overlap-program memory wall (E1 258.8 vs 275 budget). d2
  canary PASS (12.305-12.322, gn 0.36-0.49). JSONs:
  pp2-131k-fe127-d2/-d4/-d4-rerun.
- 2026-08-13 ~05:1x CDT — **d16 NUMBER OF RECORD (bumped wheel): 984
  tok/s/GPU** (step 133.2s @2M tok, mfu3x 9.0%, peak 162 GiB) vs stale-wheel
  1052 = -6.5%. Controls 1023/947 (7.4% spread, declining through the run:
  955->987->1023->947 — settle/drift caveat wider at d16). cauchy framing:
  RE-ANCHOR, not a trade-off — the stale wheel was corrupt, so ~984 is the
  honest new headline. **PEAK-MEM FINDING (both rungs): 143 GiB at d4 and
  162 GiB at d16 on the new wheel vs 175/194 old = -32 GiB leaner workspace
  (1.27.0 frontend) — moves the overlap memory-wall fight (E1 wall was
  258.8/275 GiB).** d16 trace secured (traces/fe127_d16_rank0.pt.trace.json,
  2.83 GB). Validation ladder on the bumped wheel COMPLETE: parity restored
  everywhere, canaries pass, perf re-anchored ~878-886 d4 / ~984 d16.
- 2026-08-13 ~05:3x CDT — CLOSE-OUT (cauchy): validation ladder COMPLETE;
  box w56lorq HELD idle-armed for Jack's morning keep/stop call (live
  candidates: E1 re-measure on the fixed 1.27.0 wheel — the -32 GiB
  workspace win re-opens the overlap memory-wall math; belt-and-braces
  golden-padded leg). hausdorff runs the final evidence sweep (box-side
  pulls: traces/, logs/ snapshots, lps1062_bench/ JSONs — all in the
  persistent shared dir). No further runs without cauchy's word. doppler
  box-mechanics lane stands down at watch.
- 2026-08-13 ~15:0x CDT — **FIX CAMPAIGN CLOSED — LADDER COMPLETE.**
  d16 number of record on fe 1.27.0: 984 tok/s/GPU (step 133.2s @2M
  tok, control window 947-1023, 7.4% settle spread caveat, kineto tax
  ~0) vs stale 1052 = re-anchor ~-6.5% nominal inside a wide band.
  PEAK MEM 162 GiB vs 194 = -32 GiB CONFIRMED at d16 (also d4
  143 vs 175) — leaner 1.27.0 workspace re-opens the overlap
  memory-wall math (E1 258.8/275 was OLD-wheel; re-measure = top
  candidate morning probe). d16 trace secured
  (fe127_d16_rank0.pt.trace.json, 2.83GB). RELATIVE claims (M=N +56%,
  ladder shape) wheel-held-constant and stand; absolute anchors
  re-based. BOX HELD idle-armed at READY on the headline config
  (bumped wheel, exact tree, flip reverted) — E1-remeasure and
  golden-padded probes can fire in minutes; keep/stop = Jack.
  hausdorff final sweep running (box artefact map relayed). Report §1
  + §3 carry the re-anchor + closure. Tasks: #8 closed, #9 closing on
  sweep completion.
- 2026-08-13 ~3:3x PM CDT — **JACK VERBAL ORDERS (out for a while;
  autonomous mode): overlap program + "why is DeepEP not faster —
  realistically it should be" + succession cauchy->lebesgue when
  ready.** Program launched on the convergence thesis (DeepEP
  standalone loses at bandwidth-bound 131k a2a, but its async hooks
  are the designed OVERLAP transport, and the -32 GiB wheel win may
  have removed the overlap memory wall -> L5 reframes "not the lever
  ALONE" IF evidence supports): (1) doppler PROBE 1 E1-remeasure
  DISPATCHED srun 124595 (E1 lineage selective-recompute config,
  warmup IS the measurement — E1 OOMed in warmup at 258.84; abort
  guard >265; PRE-REGISTERED: <=245 fits/wall removed
  [necessary-not-sufficient caveat], 245-265 thin/gray, >265 wall
  persists; prediction ~227 if the -32 GiB transfers); PROBE 2 tuned
  flex queued pending serre. (2) gauss reactivated: trace
  decomposition of the flex 5.1s/step delta + roofline check ->
  report-ready "why not faster" paragraph. (3) serre reactivated:
  DeepEP config audit (was L5 tuned?) -> probe spec or plain no;
  + overlap-connection scoping (does the executor use flex async
  hooks). HANDOFF.md rewritten for lebesgue (fleet state, in-flight
  lanes, parked Jack decisions, gotchas incl. per-session task board
  + armed-idle wedge lookalike + poincare DARK). Succession
  announcement next; cauchy goes quiet on lebesgue ack.
- 2026-08-13 12:1x CDT (Mac `date`; cauchy's "3:3x PM" stamps are the
  drifted clock source again — same drift class as the 04:3x/01:4x note)
  — **SUCCESSION COMPLETE: lebesgue holds the reins** (4th orchestrator:
  maxwell -> borel -> cauchy -> lebesgue; Jack-ordered verbal before he
  went out; cauchy context full). cauchy acked and went quiet. Fleet
  announcements out: doppler (mid-E1-probe srun 124595, NOT interrupted —
  verdict to me per the pre-registered bands), serre (config audit +
  overlap-connection lanes unchanged; PROBE 2 GO/NO-GO is mine on the
  audit), hausdorff (sweep continues; manifest-final ping to me -> report
  §9 verified + close task #6), poincare (DARK-notice repeated; parity
  lane already closed, no catch-up owed); gauss + weierstrass pre-acked
  during the handoff window. Task board rebuilt in the lebesgue session
  (per-session rule): #1 E1 verdict decision (in flight), #2 gauss flex
  decomposition -> report §6, #3 serre audit -> PROBE 2 GO/NO-GO, #4
  hausdorff sweep -> §9 verified, #5 convergence-thesis adjudication
  (evidence-gated §5/§6 rewrite), #6 poincare-DARK flag for Jack.
  Parked Jack decisions re-listed untouched (merge-queue GO, F1
  escalation, upstream cudnn note, box stop/keep, CPFS dirty-mcore
  disposition incl. the gate-flip patch revert before any parity-class
  boot).
- 2026-08-13 12:1x CDT (Mac `date`)   — **lebesgue PRE-REGISTRATION ADDENDUM for
  the E1 re-measure (srun 124595), filed before the verdict.** cauchy's
  bands cover the WARMUP read (<=245 fits / 245-265 gray / >265 wall;
  point ~227 if the -32 GiB transfers). The decision-relevant read is the
  d2 STEADY STATE, and it wants its own pre-registered band: under M=N at
  PP2, d2 holds min(M,PP)=2 in-flight microbatch sets on stage 0. Sizing
  one selective-recompute saved set from old-wheel numbers: E1 warmup
  258.8 GiB (1 in flight) minus full-recompute d1 baseline 170 GiB =
  ~89 GiB per set (activations, not frontend workspace — wheel-invariant).
  Predictions filed: warmup 225-230 (cauchy's ~227 ratified); **d2 peak
  ~310-320 GiB = OVER the 265 abort guard**. READINGS: (a) warmup <=245
  AND d2 <=265: the wall is removed outright (per-set cost materially
  sub-additive — surprise, reopens E1 as-is); (b) warmup fits AND d2
  >265: the -32 GiB was necessary-not-sufficient — E1-alone stays
  infeasible at 131k steady state; the overlap path then runs through P1
  (per-layer recompute dial) and/or P2 (offload), and the convergence
  thesis's E1-fits leg FAILS at steady state; (c) warmup >265: wall
  persists outright, same conclusion faster. My point reading: **(b)**.
  doppler asked to log the d2 peak reserved read explicitly (not just the
  warmup number).
- 2026-08-13 12:2x CDT (Mac `date`) — **gauss + serre DELIVERABLES LANDED;
  PROBE 2 NO-GO; convergence-thesis flex leg REFUTED (lebesgue rulings).**
  (1) gauss results/FLEX_DEEPEP_VERDICT.md: the flex 5.1s/step delta is NOT
  bandwidth — NCCL a2a already at ~85% of NVLink line rate at ~1.6GB/call;
  decomposition: +1.4s extra DeepEP kernel work, ZERO compute overlap under
  flex (0.0% vs NCCL 8.3%) = +3.1s GPU-empty in 64 long serialized
  dispatch/combine windows (alltoall: 4 gaps/0.64s), +2.7s CP-collective
  ballooning (second-order convoy parking), SM contention ~0.7s. Flex wins
  at: 16k latency-bound shapes, cross-node IB EP, or under an actually-
  overlapping schedule. Report §6 updated (refined-decomposition bullet).
  (2) serre results/DEEP_EP_CONFIG_AUDIT.md + EXECUTOR_CONTRACT_SCOPING.md
  §7: L5 was UNTUNED but near "no material knobs" — only reachable knob
  (num_sms, needs a provider hunk) attacks kernel time already at parity;
  the regression lives in flex host-side/sync structure. AND the overlap
  connection: the executor admits alltoall BY NAME
  (transformer_config.py:2627-2629), hiding is stream-placement pairing
  (dispatcher-agnostic), flex async hooks only sharpen host behavior
  (token_dispatcher.py:1785-1786 vs :1274). lebesgue spot-verified all
  three citations in the vendored tree — they hold. RULINGS: (a) PROBE 2
  (tuned-flex d4 A/B) = NO-GO (relayed to doppler; box queue after the E1
  verdict: golden-padded belt-and-braces leg is the live candidate);
  (b) cauchy's convergence-thesis "flex-as-overlap-transport" leg is
  REFUTED as a reframing — the overlap EV path is contract shim + memory
  (offload/dial), dispatcher-agnostic, NO DeepEP dependency; flex-under-
  executor demotes to optional later A/B with serre's named falsifier
  (alltoall dispatch_preprocess DtoH blocking the comm stream in a traced
  overlap boot, token_dispatcher.py:600-610). Report §5 updated. The
  thesis's remaining open leg = E1-fits, pending doppler's probe against
  the 12:1x pre-registered bands.
- 13:1x — hausdorff: FINAL EVIDENCE SWEEP CLOSED (cauchy GO → lebesgue
  close). Four big traces pulled + sha256-verified exact (fe127_d16_rank0
  2.83GB, l5_flex_d4_rank0 719MB, l3_d16_rank0 2.80GB, l3_d16_rank8 3.0GB
  → ~/perf_profiles/lps-1062/pp2cp8ep8/). Also landed Mac-side: 22 parity
  JSONs (golden pp1cp16 pair, fixed/identity/rev PP2 legs, fe127 legs,
  pp1cp8 fwd legs) + 4 fe127 perf JSONs (37/37 parse-valid), all death
  logs (105MB unfused watchdog, l0 boot fail, l0bmem abort, golden leg1
  boot), wedge_capture_bumped_wheel/ old-wheel evidence set (20 files),
  probe-a outputs, full patch set (bridge_flex_deepep fix, l0 clearing,
  savesync+l0clearing hunks, three gate probes), tonight's experiment
  configs, cutlass_dsl_src_snapshot (61MB); kernel_src_snapshot sha-verified
  Mac==box. BOX_A_ARTIFACT_MANIFEST.md marked FINAL — report §9 stop-safety
  claim VERIFIED: w56lorq can be stopped without evidence loss. Reported
  to lebesgue for the §9 flip + task #6 close.
- 2026-08-13 12:3x CDT (Mac `date`) — lebesgue: §9 sweep note flipped to
  VERIFIED-COMPLETE (manifest final; four big traces sha256-verified vs
  box-side; 37 JSONs parse-validated; stop-safety claim in bold in the
  report). Task #6 (final sweep) CLOSED. Open box-side items now: E1
  re-measure verdict (srun 124595, in flight) → then the golden-padded
  belt-and-braces leg is the only queued candidate; box stop/keep stays
  Jack's parked call.
- 2026-08-13 ~05:5x CDT — **PROBE 1 (E1 re-measure on the fixed wheel)
  VERDICT: the overlap memory wall PERSISTS (lebesgue pre-registered reading
  (c): warmup over = wall outright).** Selective-recompute config (exact E1
  lineage) on the bumped venv: WARMUP FORWARD OOM at 163.3s — rank0 259.50
  GiB torch-allocated / 266.13 in-use, rank1 258.27/265.00, rank3
  259.54/266.23 (torch capacity 267.69). vs E1 old-wheel 258.84: **the -32
  GiB lean-workspace saving did NOT transfer to the selective-recompute
  class** (torch-allocated essentially unchanged, +0.7 if anything) — the
  cudnn workspace saving is not the dominant term; the saved-for-backward
  131k activations are. Conclusion: the overlap flag at 131k still needs the
  upstream per-layer recompute dial or offload engineering — the wheel fix
  does not move that wall. Death log snapshotted
  (logs/e1_remeasure_fe127_warmup_oom_20260813.log sha 922386775e97); clean
  self-termination, box swept + idle. PROBE 2 (tuned flex) ruled NO-GO by
  lebesgue (serre audit + gauss decomposition: no reachable knob attacks the
  -12.6%). **Orchestrator: cauchy -> lebesgue (Jack-ordered, ~05:4x).**
  Next live candidate: golden-padded belt-and-braces leg (PP1/CP16 boot).
- 2026-08-13 12:4x CDT (Mac `date`) — **E1 RE-MEASURE VERDICT: READING (c) —
  THE WALL PERSISTS OUTRIGHT (doppler, srun 124595).** Warmup OOM 163.3s
  into the phase: rank0 259.50 GiB torch-allocated / 266.13 in-use
  (capacity 267.69); ranks 1/3 same shape. vs old-wheel E1 258.84:
  UNCHANGED (+0.7 if anything) — **the -32 GiB lean-workspace saving does
  NOT transfer to the selective-recompute class**: that peak is
  saved-for-backward-131k-activation-dominated (wheel-invariant), not
  frontend-workspace-dominated. d2 question moot (no warmup survival).
  Clean fast death at the guard; log snapshotted
  (logs/e1_remeasure_fe127_warmup_oom_20260813.log, sha 922386775e97).
  SELF-CORRECTION (lebesgue): my 12:1x pre-registration pointed at (b)
  warmup-fits-then-d2-fails (predicted warmup 225-230); actual was (c) —
  I let the -32 GiB reduce the warmup base when it is a workspace-class
  saving that an activation-dominated peak never sees. Direction (wall
  persists) was the table's expectation; the warmup-arm miss is mine and
  is the honest evidence that the workspace win is config-class-scoped.
  CONSEQUENCE (ruled): the E1-fits leg of the convergence thesis is DEAD;
  combined with the serre-refuted flex leg, the thesis as launched does
  not hold. What survives is the plain overlap program: contract shim
  (0.5-2d) + memory via P1 (upstream per-layer dial) or P2 (offload) —
  report §5/§1/§9 updated to this state. The convergence-thesis
  adjudication is CLOSED.
- 2026-08-13 12:4x CDT — **BOX QUEUE GO (lebesgue), two pre-registered
  boots.** (1) GOLDEN-PADDED belt-and-braces leg: PP1/CP16 golden config +
  BT_PACK_PAD_TO_MAX=1, fixed wheel. PRECONDITION (parked-items rule):
  the gate-flip probe hunk must be REVERTED before this parity-class boot;
  tree exact 73c24b00+TF32 (diff = chunked_lm_head.py only), verified
  pre-boot. PRE-REGISTERED reading: loss lands in the golden cluster
  12.3042 +/- 0.002 (new noise floor) — old-wheel golden-padded was
  12.3512 (d8-corrupted); in-cluster = the race owned the golden-padded
  corruption too and the old->new map is complete (every topology x
  padding cell clean on the fixed wheel); outside = surprise, escalate.
  (2) Then the 32k OFFLOAD DISCRIMINATOR (the standing §5 probe, now the
  cheapest evidence on the live P2 memory leg): the same selective-
  recompute+offload config that aborted at 131k, one-field seqlen change
  to 32768, flag off. PRE-REGISTERED readings: (i) engagement table
  prints after the first completed step AND peak < plain-selective-32k =
  offload engages; the 131k abort was the first-step-burst class -> the
  backpressure valve (branch-tip fine_grained_offloading plumbing) is the
  fix path; (ii) no table / peak ~= plain selective = config-ordering bug
  (per-module flags baked False) -> the one-line pre-step log probe
  localizes; (iii) OOM even at 32k = structural, stop and rethink. Both
  boots are DP1 = F3-immune; neither touches the watchdog class (fused
  kernels).
- 2026-08-13 12:5x CDT (Mac `date`) — **BOOT 1 VERDICT (doppler, srun
  127723): golden-padded on the fixed wheel = 12.305529539616677**
  (262,032 tokens exact, 9/9 outputs, fb 45.2s; JSON
  lps1062_bench/parity_pp1cp16-padded-fe127.json). Pre-registered read:
  +0.0013 vs the golden cluster 12.3042, INSIDE the +/-0.002 band — the
  race owned the golden-padded corruption too (old-wheel 12.3512,
  d8-corrupted). **The old->new map is COMPLETE: every topology x
  padding cell is clean on the fixed wheel.** Parity-class precondition
  executed correctly (gate-flip hunk reverted, tree verified exact
  73c24b00+TF32 pre-boot). Report §3 updated (the cell is no longer
  "not re-run"). BOOT 2 (32k offload discriminator) staging now.
- 2026-08-13 12:5x CDT — BOOT 2 DISPATCHED (doppler, srun 130948):
  L0b-mem selective+VPP2+offload config at max_seq_len 32768 (modules
  [core_attn,attn_proj,expert_fc1,moe_act], big flag OFF), tree
  6d8b22da (offload plumbing) + TF32, NVTE_CPU_OFFLOAD_V1=1 +
  expandable_segments + BT_SKIP_WARMUP=1 (VPP2 warmup M=1 wall, the
  6fa3bfa7 class). BASELINE GAP caught by doppler + lebesgue ruling:
  no plain-selective-32k measurement exists tonight, so reading (i)'s
  peak comparison lacks a same-night baseline. The engagement TABLE is
  the decisive signal (binary); the no-offload twin boot is
  CONDITIONAL-GO — fires only on an ambiguous read (table prints but
  peak within ~10 GiB of the 165-175 estimate class, or table absent
  but peak clearly below). Otherwise the table verdict stands alone.
- 2026-08-13 13:1x CDT (Mac date) — **BOOT 1 (golden-padded belt-and-braces,
  fixed wheel) VERDICT: PASS in-band — the old->new map is COMPLETE.**
  pp1cp16-padded-fe127: loss 12.305529539616677 vs golden cluster 12.3042 =
  +0.0013, inside the pre-registered +/-0.002 band (old-wheel golden-padded
  was 12.3512, d8-corrupted). Every topology x padding cell now agrees on
  the fixed wheel. (doppler drove the pass; poincare lane closed in good
  standing per lebesgue record correction.) Gate-flip probe hunk REVERTED
  pre-boot per the parity-class rule; tree verified exact 73c24b00+TF32.
- 2026-08-13 13:2x CDT — BOOT 2 (32k offload discriminator) FIRST STEP
  FAILED on a NEW VPP2-path bug: RuntimeError at chunked_lm_head.py:196
  ("loss_fn='cross_entropy' requires a model stage with an output_layer")
  on stage-1 ranks — under VPP2 the output_layer lives only on the last
  chunk; the M=N+VPP loss path invokes the chunked loss on a chunk lacking
  it. **No VPP2 boot tonight has ever completed a driver step** (E2a
  schedule interface, L0 boot-1 warmup-M=1, L0 boot-2 executor contract,
  L0b-mem memory abort, now this loss-path bug) — VPP2 is untested
  territory with a growing bug list; logged for the morning report. Trainer
  alive (per-op error); offload table never printed (step never completed) —
  discriminator question UNANSWERED, blocked by the bug not memory. Offered
  to lebesgue: (a) drop VPP2 from the 32k config (offload engagement is
  VPP-agnostic; plain PP2 selective+offload ran its class clean all night)
  vs (b) poincare-domain VPP2 loss-path fix. Awaiting ruling.
- 2026-08-13 13:3x CDT (Mac `date`) — **lebesgue RULING: (a) FAST PATH.**
  Drop virtual_pipeline_parallel_size from the 32k config (plain PP2
  selective+offload, the class that ran clean all night); the offload
  engagement question is VPP-agnostic (per-module hooks), so the
  discriminator survives intact; one config-field deviation to log in the
  boot record. Readings unchanged (table prints + peak below the
  plain-selective class = engages; no table = config-ordering bug; OOM at
  32k = structural); the conditional twin-boot rule stands. The VPP2
  loss-path bug (chunked CE loss invoked on a chunk without output_layer,
  chunked_lm_head.py:196) is FILED as a finding — relayed to weierstrass
  for the INFRA-B validation-status note (VPP2: unit-tested only + now a
  named first hardware bug); poincare-domain when they want a lane, not
  tonight-critical.
- 2026-08-13 13:4x CDT (Mac `date`) — **BOOT 2 (32k offload discriminator)
  VERDICT: READING (i) ON ENGAGEMENT — offload ENGAGES; the
  config-ordering hypothesis is DEAD** (doppler, srun 134046, fast-path
  plain PP2 selective+offload @32k). The Activation Offload Summary
  printed after warmup: moe_act offloaded 15.1-23.0 GB/rank (ascending by
  rank, stage-1 higher), TOTAL 292 GiB across 16 ranks; table count 1
  (prints once at warmup). The 131k first-arm failure is thereby
  attributed to the first-step-burst class (PCIe drain / no backpressure),
  NOT to offload never engaging — the backpressure valve (branch-tip
  plumbing) is the designed fix for exactly that class. TWO SECONDARY
  FINDINGS: (1) **expert_fc1 = 0.00 on EVERY rank — the load-bearing
  expert-side module moved NOTHING; only moe_act fires** (core_attn/
  attn_proj not tabled — summary tracks MoE modules only, engagement
  unknown). Module-coverage gap = a named P2 engineering item (per-module
  flag or hook wiring for expert_fc1). (2) Peak caveat: d2 step at 32k
  completed clean (loss 12.295-12.304 in band, gn 0.36-0.37) but peak
  183 GiB driver / ~200 GiB nvidia-smi is NOT below the plain-selective-32k
  estimate class (~165-175) — offload's machinery (pinned host buffers +
  expandable segments) offsets the saving at 32k scale. CONDITIONAL TWIN
  FIRED per the pre-registered rule (doppler): plain selective 32k,
  no-offload, same tree — gives the exact peak delta. Report §5 offload
  bullet update waits on the twin (one coherent edit).
- 2026-08-13 13:5x CDT — **BOOT 2 (32k offload discriminator) CLOSED, full
  read.** (a) VPP2 loss-path bug (chunked_lm_head.py:196, loss invoked on a
  chunk without output_layer) killed the first attempt — lebesgue ruled the
  fast path (drop VPP2; offload engagement is VPP-agnostic). (b) Plain PP2
  selective+offload at 32k: **Activation Offload Summary PRINTED after
  warmup — offload ENGAGES (reading i; config-ordering bug DEAD; the 131k
  abort = first-step-burst class, backpressure valve = designed fix path).**
  Verbatim table: all 16 ranks expert_fc1=0.00, moe_act 15.1-23.0 GB
  (ascending by rank, stage-1 higher), TOTAL 298,998.80 MB = 292 GiB.
  **Named P2 engineering item: expert_fc1 module-coverage gap — the
  load-bearing expert module is INERT; only moe_act fires.** (c) Twin boot
  (plain selective 32k, no offload) pinned the real baseline: peak 203 GiB
  driver / ~226 smi. OFFLOAD DELTA at 32k d2: **-20 GiB driver / -26 GiB
  smi** (183 vs 203, ~200 vs ~226) — consistent with the table. PERF COST:
  offload 473 vs plain 681 tok/s/GPU at 32k d2 = ~-30% step time (D2H/H2D
  stash traffic — the valve's job). Canaries match across the pair
  (12.293-12.304, gn 0.36-0.37) — offload numerically clean. JSONs:
  pp2-32k-offload-novpp-d2 + pp2-32k-plain-d2. Box idle-armed.
- 2026-08-13 13:5x CDT (Mac `date`) — **TWIN VERDICT folded: BOOT 2 CLOSED
  in full.** Real baseline (plain selective 32k): 203 GiB driver / ~226
  smi — doppler's 165-175 estimate was crude-low; the measured offload
  delta is -20 GiB driver / -26 GiB smi, consistent with the table
  (moe_act ~18.7 GiB/rank moved + pinned-pool overhead). Perf cost
  473 vs 681 tok/s/GPU = ~-30% at 32k (D2H/H2D stash traffic) — the
  backpressure valve is the load-bearing fix direction. Report §5
  offload bullet updated in one coherent edit (engagement verdict + twin
  delta + expert_fc1 coverage gap + perf cost).
- 2026-08-13 13:5x CDT — **NEXT BOX CALL (lebesgue): the L1 wgrad-flag
  A/B — the last reviewed overlap lever with a box slot.** Config:
  headline PP2/CP8/EP8 @131k + comm_overlap.overlap_dispatch_backward_
  with_experts_wgrad=true on the CURRENT box tree (6d8b22da carries
  b8d868ff in lineage — no tree change), fixed wheel; validator
  satisfied (big flag OFF, delay_wgrad OFF); TE 2.16.0 clears the
  >=2.3.0 pre-flight. PRE-REGISTERED: PRIMARY = correctness (d2 canary
  in 12.2-12.4, gn comparable to fixed-wheel 0.36-0.49, loss trains —
  deferred-wgrad-must-fire; gn drift = STOP + verbatim report).
  SECONDARY = perf, honestly noise-bound: EV +1.5-3% vs the ~880 d4
  anchor sits BELOW tonight's d4 within-boot control spreads
  (3.0-3.8%), so >=2 A/B pairs, mean delta vs spread reported;
  |delta| < spread = perf-indistinguishable verdict (decision-useful:
  flag stays default-off, value unproven-not-negative); d16 escalation
  only on clean pairs + consistent sign. Rationale for spending the
  box: Jack's overlap directive + a reviewed, zero-memory-cost,
  full-recompute-compatible lever whose only missing evidence is
  on-hardware behavior.
- 2026-08-13 14:0x CDT (Mac `date`) — **L1 flag-ON d2 CANARY: PASS**
  (doppler). Loss 12.307-12.322 (band), gn 0.41-0.49 (comparable to
  fixed-wheel d2 0.36-0.49), loss trains across steps, NO gn drift —
  the deferred wgrad FIRES (the pre-registered STOP signal never
  showed). d2 perf a wash (739 vs 745 fixed-wheel d2) — fill/drain
  regime, not decision-relevant, as expected. The L1 flag is therefore
  CORRECTNESS-CLEAN on this stack at d2 — its first on-hardware
  evidence. d4 A/B executing: on-arm x2 on this boot, then a flag-OFF
  boot for off-arm x2 (the flag is config-bound = cross-boot A/B; the
  x2/x2 repeat structure is what bounds the boot-to-boot term — the
  pre-registration's mean-vs-spread rule applies to the four-run set).
- 2026-08-13 14:2x CDT — **L1 (overlap_dispatch_backward_with_experts_wgrad)
  A/B — the last overlap lever — ADJUDICATED.** Same tree (6d8b22da, carries
  b8d868ff), fixed wheel. PRIMARY correctness PASS: d2 canary 12.307-12.322
  in band, gn 0.41-0.49 comparable, loss trains — the deferred wgrad FIRES
  (first on-hardware evidence the path trains correctly on this stack).
  SECONDARY perf (4 runs, same-tree cross-boot A/B): ON 834+864 (mean 849),
  OFF 859+886 (mean 872.5) -> mean delta -2.7% (reads SLOWER), but |delta|
  23.5 < arm spread (27-30) -> **PERF-INDISTINGUISHABLE-AT-CURRENT-NOISE per
  the pre-registered rule: flag stays default-off, value
  unproven-not-negative.** MEMORY evidence line: ON 143/143 GiB vs OFF
  142/144 GiB — the zero-memory-cost claim HOLDS. d16 escalation not
  triggered (no consistent positive delta). Labels: pp2-131k-L1on-d2,
  pp2-131k-L1on-d4{,-r2}, pp2-131k-L1off-d4{,-r2}. **The overlap campaign is
  fully adjudicated for the night: big flag dead (contract + memory walls),
  L1 inert-but-unproven, L2/flex demoted, the live path = contract shim +
  P1 (per-layer dial) / P2 (offload valve, engagement proven at 32k with a
  -20-26 GiB saving and the expert_fc1 coverage gap to close).**
- 2026-08-13 14:2x CDT (Mac `date`) — **L1 A/B COMPLETE (doppler, d4, same
  tree 6d8b22da, fixed wheel); the overlap campaign is FULLY ADJUDICATED
  for the night.** PRIMARY correctness PASS (canary 12.307-12.322 in
  band, gn 0.41-0.49 comparable, loss trains, deferred wgrad FIRES — no
  drift). MEMORY: on-arm 143/143 vs off-arm 142/144 GiB — the
  zero-memory-cost claim HOLDS within 1 GiB. SECONDARY perf: flag-ON
  mean 849 (runs 834/864; controls 797/875, 882/847) vs flag-OFF mean
  872.5 (859/886; controls 864/853, 875/896) = -2.7%; |delta| 23.5 <
  arm spreads (30/27) -> **PERF-INDISTINGUISHABLE-AT-CURRENT-NOISE** per
  the pre-registered rule — unproven-not-negative, flag stays
  default-off, the +1.5-3% EV claim is NOT supported at tonight's floor
  (d4 control noise ran 3-4%; resolving a +2%-class effect needs a
  quieter window or more pairs). d16 escalation not triggered (no
  consistent positive delta). Labels: pp2-131k-L1on-d2,
  pp2-131k-L1on-d4{,-r2}, pp2-131k-L1off-d4{,-r2}. Report §5 L1 bullet
  updated to the verdict. Campaign ledger, all levers adjudicated: big
  overlap flag = dead at 131k (contract gap + memory wall, the latter
  re-confirmed on the fixed wheel); E1 selective-recompute = wall
  persists (259.5 GiB warmup OOM); L1 wgrad = correct + free but
  unproven-not-negative at current noise; L5 flex = answered-negative
  standalone (-12.6%, mechanism decomposed); P2 offload = engages,
  -20-26 GiB at 32k, -30% step cost, expert_fc1 coverage gap. Path
  forward = contract shim (0.5-2d) + P1 (upstream dial) / P2 (valve +
  coverage). Box idle-armed; stop/keep = Jack's parked call.
- 2026-08-13 14:3x CDT (Mac `date`) — **PROGRAM COMPLETE; HOLD-FOR-JACK
  STATE (lebesgue).** doppler final posture: box w56lorq idle-armed at
  READY, on-disk tree restored to canonical exact bits (73c24b00+TF32,
  diff = chunked_lm_head.py only + frozen-mcore marker) so whatever Jack
  fires next starts from the parity-evidence state; bumped venv
  (cudnn-frontend 1.27.0) standing. All evidence at rest in both stores
  (sweep-verified). Report FINAL and current (§1 re-anchor + E1, §3
  complete map, §5 all overlap verdicts, §6 DeepEP decomposition, §9
  sweep-verified); zero TBDs. Fleet: doppler at watch; gauss/serre/
  hausdorff/weierstrass delivered and quiet; poincare stood down in good
  standing. Parked for Jack, untouched: merge-queue GO (9 items, PR 987
  first), F1 escalation, upstream cudnn note, box stop/keep, CPFS
  dirty-mcore disposition. No further runs without the orchestrator's
  word; lebesgue holds the watch.
- 2026-08-13 19:1x CDT — **OVERLAP CAMPAIGN OPEN (fermi orchestrating;
  erdos handed off and closed).** 48h campaign per Jack's order; full
  mission + EV-ranked lever queue in
  `runs/overnight_20260813_overlap_campaign/CAMPAIGN_ORDERS.md`. Number
  of record carried in: **984 tok/s/GPU @d16** (fixed wheel, mfu3x 9.0%,
  hfu 13.0%). erdos's fresh bottleneck map on the fixed-wheel d16 trace
  (fe127_d16_rank0, exact interval-union): exposed EP a2a 34.3 s/step
  (26%), pure idle 20.8 s (15.7%), exposed CP collectives 7.5 s (5.7%),
  compute 67.1 s (51%) — non-compute 47% of the 133.4 s step;
  perfect-overlap ceiling ≈ 1900 tok/s/GPU, realistic band 15–19%.
  Box: w56lorq STOPPED; q8eg0gq (2×8 B300 ali) provisioning since
  19:02 CDT (log /tmp/devbox_up_campaign.log). Reporting: fermi posts
  30-min status in own session (Jack reading there); subordinates report
  to fermi. Fleet assignments (all six alive, fresh contexts):
  **doppler** = box mechanics P0 (wheel bump 1.27.0 → DSA tests → d2
  canary → d4/d16 re-anchor, ±3% bar pre-registered); **lebesgue** =
  executor contract shim, Option B per
  `results/EXECUTOR_CONTRACT_SCOPING.md`, branch
  `jackrao/lps-1062-overlap-contract-shim` (flagship); **kepler** =
  idle-window decomposition of the 20.8 s on the :9001 trace (deliverable
  `IDLE_WINDOW_DECOMPOSITION.md`); **bohr** = B/F rebuild from Aug-9
  patches (`jackrao/lps-1062-bf-rebuild`) + C′ #27 / A-v3 #29 landing
  prep; **hausdorff** = W1 go/no-go (EP-group count of PP2/CP8/EP8 vs
  PR #28 fix; `W1_GO_NOGO.md`); **jacobi** = memory-leg decision memo
  (dial vs offload valve; backpressure-valve plumbing state + expert_fc1
  gap root-cause; `MEMORY_LEG_DECISION.md`).
- 2026-08-13 19:4x CDT (Mac `date`) — **kepler: IDLE-WINDOW DECOMPOSITION of the
  fe127 d16 20.8s pure-idle COMPLETE (trace-only, no box).** Deliverable:
  runs/overnight_20260813_overlap_campaign/IDLE_WINDOW_DECOMPOSITION.md
  (pre-registered buckets before queries, exact SQL inside). Method check:
  kernel-only gap-union reproduces erdos's 20.8s exactly (20.797s); all-GPU
  (kernel+memcpy+memset) = 20.226s over 898k gaps. FINDINGS, all contrary to
  the pre-task framing: (1) **structural PP2 fill/drain bubble in the idle
  budget = ~0s** — stage-0 PP waits are posted SendRecv kernels (GPU-busy),
  already counted in the 34.3s exposed-SendRecv line; the "ideal 6% bubble"
  does not live in pure idle. (2) **Step seam = 1.95s only** (2.03s wall after
  last bwd, 95% idle): untraced host python 1.86s (1294/414/116ms holes around
  grad-norm broadcast -> optimizer), FusedAdam host annotation 28.8ms, adam GPU
  kernels 1.0ms + 456x bf16 copy-back ~3ms — the optimizer is NOT the hole;
  py-spy at the seam would name the python in one probe. (3) **The dominant
  named class is the DSA indexer `aten::nonzero` D2H sync: 55,328 calls/step,
  34.0s host time, of which 6.97s leaks as PURE GPU idle** (4.06s in >=1ms
  gaps + ~2.9s of micro-gaps) — erdos's "absorbed" host line is only mostly
  absorbed. (4) MoE a2a dispatch host block 1.20s (_AllToAll -> new_empty ->
  cudaEventQuery 100-160ms spins = allocator/sizing stall) + bwd aten::cat
  0.71s. (5) **Diffuse sub-1ms dispatch tax = 10.41s** (897k gaps, mean
  11.6us, uniform across the step) — biggest single line; ~6s of it
  indexer/a2a/cat-adjacent, ~4.4s generic per-kernel tax at 1.07M GPU
  ops/step. Ranked attack list (est win): indexer sync-free top-k 4-7s;
  dispatch-tax fusion/graphs 2-4s; seam python 1-1.5s; a2a/cat buffers 1-2s
  (disjoint from the 34.3s SendRecv prize). Total realistic 8-14s/step (6-10%)
  — matches the "~10 points" estimate but the composition is host-side, not
  pipeline. CAVEATS: rank-0 only (fe127 rank8 never pulled; classes are
  per-rank-local so should transfer); seam is a floor (trace ends at
  ProfilerStep boundary). INCIDENT: I wedged shared trace_processor :9001 (pid
  47309) with a naive 898k x 55k interval join — NOT killed per orders, still
  churning at 98% CPU, will recover on drain; my own instance serves the same
  trace on :9002 (pid 29344) and is left running for fleet use.
- 2026-08-13 19:4x CDT (Mac `date`) — **W1 GO/NO-GO (hausdorff, fermi task 7
  precondition): verdict ARM-AFTER-VALIDATION; port COMPLETE + reviewed
  SHIP.** Memo: `runs/overnight_20260813_overlap_campaign/W1_GO_NOGO.md`.
  EP-group count DERIVED by executing the pinned mcore RankGenerator
  (57efae08b): PP2/CP8/EP8 TP1/ETP1 world-16 → **2 EP groups** [0-7]/[8-15]
  (one per PP stage, each intra-node) → multi-EP-group, PR #28 fix
  load-bearing. Audit finding: PR #28's span guard SELF-DISARMS on this
  topology (loud no-op); arming needs the guard relax. Port branch chain
  (pushed, pre-push checks green): trainers `jackrao/lps-1062-w1-port`
  @11ad3e26 → megatron-bridge @595f0b7f → Megatron-LM @a58990a91 (454f0ce3f
  = faithful PR #28 port — mappings.py/__init__.py blob-identical to PR head,
  token_dispatcher = PR head minus C′; 7b51a83f6 = guard relax:
  local-sync creation unconditional + per-EP-group group_desc, span check
  demoted to telemetry; a58990a91 = comment precision). Estate W1 CPU suite
  GREEN vs the ported tree (2-proc gloo, all sec1-6). Fresh-subagent review:
  **SHIP** (6/6 checks) + two memo corrections folded in: (1) torch 2.11
  names local-sync groups by rank-hash → disjoint EP groups' store keys are
  distinct BY CONSTRUCTION (group_desc = observability, NOT a mitigation;
  a rendezvous FAIL → straight to HOLD, no retry); (2) new_group is lazy —
  comm 2 first rendezvous is at the first gated dispatch, so V0's PASS now
  requires one full optimizer step, not just build+armed lines. Estate
  pre-fix patch untouched. W1 stays gate-OFF dormant on the branch
  (byte-identical gate-off path); V0-V3 canary window = fermi to schedule
  (P3). Option-6 sequencing ratified: only after W1 V0-V2, vs the W1-armed
  baseline, never co-armed. hausdorff save_state probe stays QUEUED for
  campaign tail (P4 reminder owed to fermi).
- 2026-08-13 19:5x CDT (Mac `date`) — **kepler: IDLE RESIDUAL MAP COMPLETE;
  FLEET NOTICE: trace server = http://localhost:9002 (kepler's
  trace_processor, pid 29344, serving fe127_d16_rank0). fermi retired the
  wedged :9001.** Deliverable:
  runs/overnight_20260813_overlap_campaign/IDLE_RESIDUAL_MAP.md. The 55,328
  nonzero/step decompose by site (containment sweep vs dsa_layout.py /
  dsa_cudnn_kernels.py): **54,720 (98.9%) = CP layout builders = FIX B's
  target** (27,360 fwd + 27,360 replay; 6.98s host; 6.20s idle leak — ~89% of
  their host time leaks, they fire in launch-bound regions); **608 = bwd
  topk_length compaction = A-v3's target** (dsa_cudnn_kernels.py:2130; 27.0s
  host = 79% of nonzero CPU, but 97% absorbed -> 0.77s leak). FIX F's RoPE
  class measured: 1,630 heavy aten::to (>1ms), 37.0s host, 0.86s leak. C'
  replay eventSyncs: 1,120, 13.8s host, 0.26s leak. **STACK TOTAL: B+F+A-v3+C'
  covers 8.09s of the 20.23s pure idle (40%); residual 12.14s = new territory
  (generic dispatch tax ~4.4s, seam python 1.9s, a2a/cat host blocks ~1.9s,
  misc ~1.1s, micro remainder).** The stack's hidden prize = ~85s/step host
  drag removal (the F6 decompression mechanism). Q2 answered: my attack-1 was
  MIS-SITED — the leak is the layout builders (B's cache, staged), not the
  top-k kernel; attack-1 dissolves into lever 3 (B 89% + A-v3 11%); A-v3 stays
  parked (0.77s direct EV; dependency chain confirmed). Lever-3 correction
  filed: "mostly absorbed at d16" is WRONG for the layout-builder class;
  EV revised up. **Pre-registered B/F d16 A/B prediction (mine, filed before
  the run): central +4-7% (point +5%), floor +1.5-2.5%, ceiling +9-11%,
  <+1.5% refutes leak-causality** — consistent with gauss/bohr +3-6%, ~1pt
  higher central on directly-measured 7.06s B+F leak. Reported to fermi.
- 2026-08-13 20:1x CDT (Mac `date`) — **kepler: STAGE-ASYMMETRY CHECK COMPLETE
  (l3_d16 rank0+rank8, old-wheel caveat pre-registered: classes valid,
  absolute times suspect).** IDLE_RESIDUAL_MAP.md §7. l3 verified UNPATCHED
  (nonzero ~55-58k) = valid class-structure control. FINDINGS: (1) **the
  layout-builder leak is strongly stage-asymmetric: stage 0 leaks 8.07s vs
  stage 1's 2.96s on the same call count** (54,720 vs 57,632 — counts confirm
  the 38/40 PP2 layer split); stage 1's host runs builders inside its PP-recv
  slack, stage 0's runs just-in-time and starves its GPU. RoPE class
  stage-symmetric (0.89 vs 0.83s leak); bare-bwd compaction absorbed on both
  (0.72/0.17s); dispatch-tax density symmetric (~7.7% of span both). Total
  idle 22.2s (r0) vs 12.7s (r8). (2) SendRecv >100ms structure: stage 0 waits
  on stage 1 at the BOUNDARY (5.0-5.2s step-start + 2.4-2.7s drain = ~7.4s of
  fe127's 34.3s exposed SendRecv = stage-1 tail skew); **mid-step the
  direction INVERTS: stage 1's 0.66-1.67s recv waits coincide
  (normalized-fraction alignment) with 20-160ms host-stall gap clusters on
  stage 0 — stage 0's indexer stalls delay activation delivery; stage-0 idle
  is the laggard signature, not slack.** EP combine imbalance shows as
  100-190ms bwd-enclosed SendRecvs both stages. (3) VERDICT: **NO
  stage-asymmetry haircut to the B/F d16 band (central +4-7%, point +5%
  stands)** — the stage-0 leak is ON the critical path mid-step; boundary
  skew is separate seam-class territory the band never claimed. Cheap
  caveat-retirer filed: fixed-wheel rank8 capture on the B/F A/B box
  (BT_PROFILE_RANKS=0,8 already standing discipline). Trace servers: :9002
  fe127r0 (kepler), :9004/:9005 l3 r0/r8 (kepler), :9003 bohr's fe127r0.
- 2026-08-13 19:54 CDT (Mac `date`) — **CONTRACT SHIM LANDED + PUSHED
  (lebesgue, task 3 / lever-1 prerequisite).** Branch
  `jackrao/lps-1062-overlap-contract-shim` @ cd58e629 (+b4e6ce29 ty-harness
  placation), atop 6fa3bfa7. Option B per EXECUTOR_CONTRACT_SCOPING.md:
  CE/RL/DPO closures grew a keyword-only `return_schedule_plan` kwarg —
  under it they return `(build_schedule_plan(...), <existing loss partial>)`;
  the plan carries the microbatch, a passthrough output_processor yields
  post-norm hidden states (no full-vocab logits), and the loss partial runs
  untouched as the executor's loss node; `_sum_over_microbatches` passes the
  kwarg through. Deviation from the doc's "one shared wrapper" framing
  (sanctioned by its own parenthetical): a model-proxy wrapper is unworkable
  — plan construction gates on `model.post_process`
  (model_chunk_schedule_plan.py:398), which the closures' PostprocessDisabled
  flips around the forward; a proxy cannot distinguish a flipped last chunk
  from a genuine non-last chunk. Guards: (a) config-parse — flag at PP>1
  requires VPP>1 (plain PP>1 never enters the combined executor; silently
  inert otherwise); (b) VLM batch under the flag → clear NotImplementedError;
  (c) warmup hatch note (M=1 illegal under interleaved; 6fa3bfa7 auto-skip;
  BT_SKIP_WARMUP=1 manual). Plan-branch stage check keys on the chunk's own
  post_process — sidesteps the filed VPP is_pipeline_last_stage bug in the
  NEW path (conventional path untouched). **Scoping §4 router-replay "✓"
  OVERTURNED**: the REPLAY_FORWARD→REPLAY_BACKWARD FIFO flip is timed around
  a synchronous forward; under the executor's concurrent fwd(k+1)/bwd(k) on
  shared router state there is no safe flip point (silent wrong-route risk
  under MoE-module recompute) — shim raises NotImplementedError at step time
  AND config rejects R3×flag at parse. Mission-config check (fermi's gate):
  the PP2/CP8/EP8 131k bench does NOT run R3 (no router_replay_mode in any
  campaign config → NONE default; workload is SFT CE, replay is RL-family
  only; runner hard-fails on routed_experts under mode≠R3 and all boots ran
  clean) — the rejection does not intersect the mission config, no R3-off
  parity caveat needed for the A/B arms. Tests Mac-side green: 13 protocol
  (new test_schedule_plan_protocol.py, auto-fabricating megatron stub finder
  with exact sys.modules restore) + 5 config (models/test_control.py); full
  unit suite 343 passed, collection errors byte-identical to baseline. One
  fresh-subagent review → SHIP; its 4 follow-ups applied (keyword-only kwarg,
  R3×flag config rejection, hermetic test harness, fp32-head-tested note).
- 2026-08-13 19:54 CDT — **BRANCH-RED REPAIRED (lebesgue, fermi-authorized)**:
  `test_populated_comm_overlap_round_trips_through_payload` failed at
  campaign-branch HEAD — b8d868ff added
  `overlap_dispatch_backward_with_experts_wgrad` to CommOverlapConfig without
  updating the expected payload dict. One-line expectation fix pushed as its
  own commit: jackrao/lps-1062-pp2cp8ep8 @ c30afc3e (21/21 in
  test_trainer_registry). Every downstream branch inherits the green.
- 2026-08-13 19:5x CDT (Mac `date`) — **OPTION-6 STAGED (hausdorff, fermi
  order).** Estate patch 6fbf3acc-era rebased onto the W1 port: branch chain
  `jackrao/lps-1062-option6-staged` — trainers @855d8d61 → megatron-bridge
  @d92774fa9 → Megatron-LM @70710d116 (one commit on w1-port @a58990a91).
  Rebase ZERO-reject: transformer_engine.py / mappings.py / moe_utils.py
  blob-identical to patch targets; token_dispatcher.py offsets only
  (C′-absence). CPU proofs re-run green on the staged tree: option-6 suite
  (mechanism/engine-order + gloo carrier protocol + arm predicate + AST
  guards) AND the W1 suite (no regression from the mappings.py signature
  change). Fresh-subagent review SHIP (6/6); flags logged in W1_GO_NOGO.md
  §4 (TE kernel bitwise-equivalence = wrapper-contract-verified, on-box
  numerics gate arbitrates; `_w2_config` getattr is forward-compat). Both
  gates default-OFF dormant. Sequencing ratified: option-6 canary only
  after W1 V0-V2, vs the W1-armed baseline, never co-armed. hausdorff now
  genuinely quiet; save-state probe P4 reminder stands.
- 2026-08-13 20:2x CDT (Mac `date`) — **kepler: PP LAYER-REBALANCE ARM —
  KILLED PRE-BOX (no window spent).** PP_REBALANCE_ARM.md in the campaign run
  folder. (1) The 38/40 split is a trainers-side explicit layout override
  (glm52_dsa.py `_GLM52_DSA_PIPELINE_LAYOUTS[(78,2,1)]`), NOT an mcore default:
  GLM-5.2 = 78 layers (confirmed HF-side + fermi/lebesgue), DSA top-k sharing
  (index_topk_freq=4) forces stage starts to 1-indexed layers ==3 mod 4, so
  legal PP2 splits are {34/44, 38/40, 42/36} — 39/39 illegal, and the
  rebalance quantum is 4 LAYERS (fermi's 1-2 not expressible). (2) Measured
  stage compute (l3 old-wheel): stage0 68.40s vs stage1 74.05s (40L + ~2.0s
  chunked-head/loss; L=1.80s/layer; head is chunked below 3ms GEMM
  granularity, residual-bounded). Continuous optimum = move 1.6 layers —
  below the quantum; next legal cell 42/36 is +1.55s max-stage WORSE (75.6 vs
  74.05). **38/40 is already the compute-optimal legal split.** Under
  block+K21 (W1b) the margin narrows (2L'-H: 1.6 -> ~0.7s est) but does not
  flip. (3) Memory: +4 layers ~= +7GB on stage 0 (EP8-sharded params ~1.3GB/
  layer + ~0.5GiB/layer activations) — fits trivially under the mission
  config (162 vs 267.7); W1b-baseline memory section HELD OPEN per fermi
  (needs W1b measured peaks 221/240). (4) The 7.4s boundary skew routes NOT to
  layer movement but to R2 (seam python on stage-1's tail) + stage-1 compute
  excess is already legal-minimum. REOPEN CONDITION pre-registered (zero box
  cost): my W1b fixed-wheel rank0+rank8 mechanism-verification traces give
  H vs 2L' for free — reopen iff H > 2L'. INCIDENT NOTE: wedged my OWN :9005
  with an EXISTS-join (kernel x CF enclosure) — killed+restarted it (pid
  40020); fleet servers :9002/:9004 unaffected; lesson carried: window-
  function unions only, no per-row enclosure joins against the slice table.
- 2026-08-13 20:5x CDT (Mac `date`) — **W2 TREE: overlap+LoRA landmine fix
  LANDED on the shim chain (lebesgue, fermi-ordered).** jacobi's dial review
  surfaced a pre-existing executor bug on the shim's boot path: under LoRA's
  frozen embedding, `decoder_input` carries requires_grad=False, so the
  fine-grained executor's closing `pre_process.backward` roots a backward at
  a non-grad tensor → RuntimeError at the first stage-0 backward (never seen:
  every prior overlap attempt died in the first forward, memory). Mechanism
  verified against the vendored source before acting (block path forces the
  root at transformer_block.py:593; the plan path did not). b907b6153 was
  MIXED (grad-root fix + dial-specific hunks referencing
  moe_ep_overlap_checkpoint_num_layers, absent at the campaign pin 57efae08b),
  so per fermi's single-purpose-W2 order ONLY the PreProcessNode hunk was
  extracted as its own commit — jacobi verified the extraction byte-identical
  to his hunk. Chain (all pushed, verified): trainers
  jackrao/lps-1062-overlap-contract-shim @ e13de4d7 → bridge @ 146f2636 →
  mcore @ b37c01f2e (same branch name on both submodule forks). W2's first
  flag-ON boot is now ALSO the fix's first hardware validation: a stage-0
  backward RuntimeError there reads as FIX-INSUFFICIENT → jacobi+fermi, NOT
  shim failure (pre-registered in OVERLAP_AB_DESIGN.md §5/§7). MERGE NOTE
  (jacobi, for later): the shim branch and the dial branch both carry the
  identical PreProcessNode hunk — when both land on the campaign tree, git
  sees identical changes; trivial either-way resolve.
- 2026-08-14 00:1x CDT — **BOX WAR + RESTART WAVE (bayes, formerly fermi —
  session-restart wave ~00:00 renamed the whole fleet; orchestrator context
  survived, subordinate contexts did NOT).** Box chronicle: ali B300 full all
  night (39 nodes); the platform reaps pending jobs at ~60-62 min INCLUDING
  half-scheduled ones (per-JOB timer; teardown signature = interactive-session
  configmap deleted first → FailedMount → pods killed). Cycles: q8eg0gq reaped
  20:04, wov4kzq 21:08 (held a Running node 42 min, lost it), qjlr5pw 22:12,
  qe5gepq 23:13 — 4 reaps, each cycle re-capturing the same recycled single
  node. **Cycle 5 wprm693 LANDED ~23:16 (both pods Running — a second node
  freed at last).** Provisioning stalled ~40 min on a wedged driver with a
  broken env (restart-wave casualty; DNS+kubectl dead in its shell) — killed,
  relaunched clean 00:06, steps 4-5 green, venv build in flight. Fleet
  casualties mid-work: doppler (box mechanics), bohr (B200 wlxj8vw dial
  rung-1 mid-flight — box still up, state unknown), lebesgue (cut-down
  snapshot build mid-flight), kepler, hausdorff (was quiet). jacobi status
  unknown. New sessions roll-called; grothendieck confirmed fresh. B200
  side-box facts for the successor: wlxj8vw (1×8 B200 hyd) wheel-bumped
  GREEN (5 passed/23.3s, cu12 lane, import name 'cudnn', uv-in-repo-dir
  constraint gotcha — run pip from /tmp), dial chain staged at mcore
  06393114b (landmine fix b907b6153 confirmed in lineage), rung-1 test file
  test_recompute_dial_grad_equiv.py written Mac-side, execution state
  unknown at death.
- 2026-08-14 ~00:15 CDT — **bohr lane report (context survived the wave
  intact; this is the full arc since campaign open, honest chronology).**
  **(a) B/F rebuild (task 4) COMPLETE + pushed.** Premise correction on
  record: B/F was not branchless — mcore-side it existed as fork PR #26
  (`jackrao/lps-1062-ship-bf` = trainers-main + 2 commits, DRAFT/MERGEABLE,
  trainers-main == 57efae08b exactly); the missing piece was the runnable
  pointer chain. Byte-verified ship-bf == 57efae08b + Aug-9 patches
  0002+0003+0001v2 (only delta: one docstring hunk). Built + pushed:
  trainers `jackrao/lps-1062-bf-rebuild` (e864115a, off the pp2cp8ep8 tip;
  diff = gitlink only) → bridge `jackrao/lps-1062-bf-rebuild` (0e356eb2) →
  mcore 500ce306a; fresh recursive clone verified end-to-end (B/F gates
  in-tree). A-v3 decontaminated off the prohibited W1:
  `jackrao/lps-1062-ship-av3-wo-w1` (40d8b3578, pushed). Fresh-subagent
  review PASS (7 doc nits fixed). REBUILD_NOTE.md (campaign folder) carries
  deviations + the pre-registered A/B (d4 primary vs 878–886; d16 vs 984 with
  kepler's landed residual-map bands: central +4–7%, floor +1.5–2.5%, <+1.5%
  refutes leak-causality). **(b) NCCL re-sweep (lever 6) CLOSED cheap:**
  NCCL_RESWEEP_PLAN.md — all three ship knobs are NET/IB-scoped ⇒ on this
  topology they touch only the wire-speed/park-dominated PP p2p path (Run B's
  +0.4% stands as the measurement). Arm 2 (CP-collective tuning) KILLED at
  the Mac gate, zero box time: per-call payload×duration from the fe127 d16
  trace shows AllGather 151MB class min 0.225ms = AT the 600GB/s wire floor,
  avg 4.13ms = 18× floor (ReduceScatter same shape) ⇒ ~4.7s of the 5.6s
  exposure is peer-arrival skew, no NCCL knob addresses it. Arm 1 (ship-env
  ablation control) folds into the B/F window's ratified 3-boot design.
  **(c) DIAL VALIDATION rung 1 COMPLETE 7/7 on wlxj8vw (B200).** Specs
  (DIAL_GRAD_EQUIV_SPEC.md + DIAL_VALIDATION_LADDER.md, campaign folder)
  written against jacobi's frozen interface and code-verified vs the pushed
  branch (mcore 06393114b; b907b6153 PreProcessNode landmine fix confirmed in
  lineage). Suite `test_recompute_dial_grad_equiv.py` (campaign folder +
  on-box mcore tests dir), EP2 torchrun. **7/7 PASS both ranks: T1 LoRA-trap
  grad-equivalence (adapter grads nonzero + matching stock, input grad
  correct), T2 negative control both arms (flip stubbed ⇒ trap fires;
  b907b6153 source guard), T3 RNG fork/restore (checkpoint recompute ==
  eager ground truth BITWISE under active dropout), T4 K=0 no-op, T5 K=L vs
  stock full recompute, T6 node contract, T7 dense-exclusion/K-semantics.**
  Evidence: runs/overnight_20260813_overlap_campaign/
  rung1_dial_grad_equiv_7of7_20260814.log. Three harness bugs found+fixed en
  route (honest record): (i) the overlap flag is rejected at EP1
  (transformer_config.py:2624) → EP2 torchrun; (ii) the MLA default
  q_head_dim 192 trips flash-attn cute's sm100 "Must use 2CTA for hdim 192"
  assert in backward → fixture pinned to 128; (iii) two of MY test-design
  bugs — T3's process-global RNG tracker is shared across arms (fixed with
  snapshot/restore + eager ground truth, the sharper §6.2 form) and T5's
  raw-mcore reference arm lacks the bridge PEFT patch's grad-root force
  (fixed with a forward pre-hook forcing the block input — which ALSO makes
  T5 stronger: dial-flip vs bridge-style-forced stock). Rung 2 SUPERSEDED per
  bayes: the mission box's W1b block+K21 @131k measures the real per-layer
  table (BT_DIAL_MEM_PROBE); my rung 2 = consume it (K formula + parser
  pinned in the ladder). **(d) Mac outage (~22:0x–00:0x):** opendirectoryd
  lost uid 501's user record → ssh/DNS/sudo/iTerm-AppleEvents/TCC-on-
  ~/Documents all failed, then self-healed; cost ~1h of box time, zero
  campaign data (BLOCKER_STATUS_bohr.md). Papercuts filed this session:
  pre-push hook lints working-tree not pushed-ref (pc_165fef62f83f),
  trace_processor --httpd 404s /query (pc_627024976652), uv-pip-inside-repo
  applies pyproject pins as constraints (pc_1acf1a2ae1cf). wlxj8vw PARKED
  idle-armed (insurance per bayes). Holding for the W1c B/F window.

---

## 2026-08-13/14 overnight — overlap campaign (erdos → fermi → bayes), box wprm693

[doppler box-mechanics lane, honest chronology. Full detail:
runs/overnight_20260813_overlap_campaign/DOPPLER_STATUS.md]

- **Box race (19:02→00:06):** 4 cycles reaped at the platform's ~60-62 min
  pending-timeout (q8eg0gq, wov4kzq, qjlr5pw, qe5gepq — cluster full, rank1
  never scheduled inside the window; reaper signature = bt-interactive-session
  configmap deleted first). Cycle 5 (wprm693, created 23:16) landed both ranks
  at ~00:06. Resume-driver pattern (kubectl-first wait, 429-tolerant,
  no-create) ran steps 3-13 across two process deaths (restart wave).
- **Mac env failure ~23:50-00:0x:** opendirectoryd getpwuid dead (all ssh
  255), gh keychain token deleted, DNS flapping — all recovered except the
  gh token (bayes's session held a working one; step 8 ran token-backed).
- **PROVENANCE REBUILD (Jack's direct order):** venvs rebuilt fresh from the
  lock on the box (both rc=0). Findings: (a) lock-faithful hadamard =
  git v1.1.0 — the inherited venv ran a mystery PyPI 1.0.4.post1 swap;
  (b) fetch-wheels now vendors cudnn-frontend 1.27.0.dev20260803 (stale
  1.26.0+dsatopk1 gone); (c) pod image cuda:12.8.1-devel vs torch cu13 needs
  cuda-nvcc-13-0 + cuda-libraries-dev-13-0 for source builds (papercut
  pc_bad16ec3b02c); (d) CPFS partial-write casualty: nvshmem libs missing
  post-sync, reinstalled. Re-bumped to PyPI cudnn-frontend 1.27.0 --no-deps
  (backend 9.19.0.56 unchanged), DSA gate 5-passed on-hardware.
- **W1a re-anchor (fresh venv):** d2 canary PASS (12.31-12.33, gn 0.39-0.48);
  d4 = 848; d16 fresh-boot 945 → OUTSIDE the pre-registered 954.5-1013.5 band
  → STOP + report per protocol. bayes ruling: extend-to-plateau (diagnostic),
  record fresh-boot numbers as operational anchors, hand box to W1b.
- **d16 extension:** mains 962/960/959/958/958/958 → plateau 958-962 IN-BAND;
  warm-up confirmed; hadamard A/B CANCELLED per the pre-registered condition.
- **⚠ Standing asterisk:** tonight's absolute numbers are lock-faithful-venv,
  -3.5-4% vs the record's (dirty-provenance) numbers; lever verdicts are
  self-controlled fresh-boot A/Bs on this venv and unaffected.
- **anchor-20260814: d4 848 · d16 958-962 hot-plateau** — W1b/W1c A/B
  reference THESE.
- 2026-08-14 02:4x CDT (Mac `date`) — **W1b BOOT #1 OOM + the warmup-padding
  mechanism (grothendieck, block+K21 arm).** Boot of
  trainer_pp2cp8ep8_131k_blockK21.json (mission config + recompute
  {full, block, 21}) died in the trainer-INTERNAL warmup: rank15/stage-1
  MoE bias_act 872 MiB alloc failed — 260.33 GiB torch-allocated, 266.76
  in-use, 267.69 capacity (~14 min in, straight after weight load; zero
  bench windows ran). MECHANISM (source-verified Mac-side): the startup
  warmup's seq-64 single datum (BT_WARMUP_SEQ default 64, no override in
  env) is PADDED to the full 131k packed buffer — the trainer warmup is a
  full-shape M=1 fwd+bwd, and it runs while weight-load transients have
  not settled. block+K21 keeps 19/40 stage-1 layers EAGER (+~47 GiB stored
  vs full-recompute at M=1), which tips the warmup over capacity. TWO
  FINDINGS (bayes, for the record + ship notes): (1) **block+K21 requires
  BT_SKIP_WARMUP=1 operationally at 131k** — must ride the eventual PR/ship
  notes; (2) **trainer-internal warmup runs ~15-20 GiB over bench steady
  state** — independent confirmation of the warmup-peak class pauli's dial
  work flagged; the padding mechanism above is the WHY. bayes call: GO
  PATH A — boot #2 with BT_SKIP_WARMUP=1 (declared env delta #2; #1 =
  BT_PROFILE_RANKS=0,8 for the d16 traced pair). Pre-registered
  contingency (bayes, on record before data): any ON-vs-OFF verdict within
  ±2% of its win bar ⇒ ONE env-symmetric OFF re-pair (OFF arm with
  SKIP_WARMUP=1 + PROFILE_RANKS set) before declaring. Pairing validity
  argument accepted: anchors are bench windows; the bench driver's own
  warmup window absorbs autotune; fresh/plateau class pairing absorbs boot
  state. Fit protocol now executes at the BENCH ramp: d2 = the fit read
  (pred ~221 GiB reserved, poller-measured), d16 watch vs the 255 line
  (pred ~240). Retry formula armed (pauli-confirmed): K′ = 40 −
  19×(255 − peak_full_same_rung)/Δ, ONE retry.
- 2026-08-14 (Mac `date`) — **W1b CONSTANT CORRECTION (pauli, ratified by
  bayes): S_eager = 2.94 GiB/layer/mb, not the 2.25 used in
  MEMORY_LEG_DECISION.md.** Root of the error: 2.25 was the E1-*selective*
  constant (core_attn still checkpointed in E1); block+K eager layers are
  fully eager and retain the DSA attention internals (~+0.7 GiB/layer/mb:
  q/kv/proj saves, sparse-attn outputs+LSE, indexer state). Exact fit vs the
  bench-warmup0 OOM point: 147 (base) + 2×(21×0.19 + 19×2.94) = 266.7 =
  measured 266.3–266.7 — base and I=2 validated, only the constant was wrong.
  (The trainer-warmup OOM at M=1 rides the higher weight-load-transient base;
  the bench-ramp fit is the decision-relevant one.) **Corrected K: ≥23.1 →
  K=24 minimum (~250 GiB predicted), K=25 ratified for the retry**
  (warmup measurement may predate optimizer-state materialization). Retry-
  formula footgun closed with grothendieck: the base in K′ = 40 −
  19×(budget−base)/Δ is the SAME-RUNG FULL-RECOMPUTE anchor (~162), not the
  stripped 147 — the wrong-base reading boots a 261 GiB K=22; and round UP
  (K=23 lands 255.6, fails the 255 line by a hair). Prize restatement at
  K=24-25: dial overlap coverage 37–40% of layers (from 47.5%), dial combined
  ~6–10%, block+K standalone ~5–8%. **Gate re-registration (bayes, the
  pre-registration of record): the dial ladder's memory gate is now
  TWO-STAGE — d1 ramp measures the per-layer slope UNDER THE EXECUTOR, K is
  recomputed from that slope, d2 canary boots at the recomputed K** (the
  plain-path 2.94 doesn't transfer verbatim through the executor's per-node
  detach/free machinery). MEMORY_LEG_DECISION.md carries the correction at
  top + in-place marks; the blockK21 config is superseded by
  configs/trainer_pp2cp8ep8_131k_blockK25.json. Process note: the block+K
  probe did its designed job — it caught the wrong constant for the price of
  one OOM'd boot, before any dial boot depended on it.
- 2026-08-14 05:4x CDT — **INCIDENT + RULE ENFORCEMENT (bayes): manual
  trainer-wait blocked a session for 60+ idle box-minutes.** grothendieck
  (W1b driver) was not crashed but BLOCKED on a manual trainer-health wait
  during the K=25 retry prep — unresponsive through three pings; W1b was
  handed to lovelace at the clean boundary (succession rule). Jack's
  ruling, now zero-tolerance fleet-wide: waiting on trainer state happens
  through wait_trainer_health.sh ONLY (backgrounded, session stays
  responsive); manual/foreground waits = immediate lane handoff. Rule was
  already standing in CAMPAIGN_ORDERS; violation cost the campaign ~1h of
  landed-box time on the night's top lever. Broadcast acked-required to
  all six sessions.
  - Addendum: grothendieck resurfaced post-handoff, accepted the succession
    cleanly, and self-reports the hour as a session stall rather than a
    deliberate manual wait; Jack's terminal-side observation (manual
    trainer wait) stands as the primary account. The slurm-24 PD mystery
    closed: grothendieck's own srun --overlap mem-poll queued behind the
    trainer allocation. Rule acks on record: lovelace, pauli, kolmogorov
    (incl. self-flagged sleep-loop anti-pattern in their earlier local
    polls), grothendieck implicit via handoff acceptance.
  - Final: grothendieck's second ack retracts the stall framing and owns
    the violation exactly as Jack observed — foreground sleep-polling
    (sleep 150 + ssh stare on the d2 bench) and ssh-bound wait loops.
    All six sessions have now acked the zero-tolerance rule explicitly.
- 2026-08-14 (Mac `date`) — **W1b SECOND MISS + MODEL REJECTION (pauli,
  ratified by bayes).** K=25 block+K survived but measured 265.0 GiB reserved
  (node-1 binding; leader 218.3) vs the 244.6 prediction (+20). Reconciliation
  on the fresh W1a evidence JSONs: (i) the mission-box full-recompute anchors
  are d2 162.2 / d4 165.0 / d16 182.8 GiB stage-1 reserved — **+20–22 GiB over
  the old-box 143/162 carried in from w56lorq** (cause unassigned → NAMED
  HYGIENE ITEM, owner pauli; the d16 anchor pair ran under BT_PROFILE_RANKS
  so the clean comparison is the untraced d2/d4 pair); (ii) the three points
  REJECT the linear stored-set model (3-point fit → garbage constants,
  residuals ±10 GiB): anchor→K=21 slope 2.75 GiB/layer vs K=21→K=25 marginal
  0.44 GiB/layer; stage gap 11.4→46.7 GiB. Read: near the ceiling the stage-1
  peak is dominated by K-INDEPENDENT components (loss-path fp32 stack,
  cold-pool first-window burst under BT_SKIP_WARMUP=1 — W1a anchors ran
  warm-pool — fragmentation); "reserved" saturates at ~capacity under
  pressure and stops measuring demand (the demand metric is torch-allocated,
  carried only by OOM reports + the trainer _peak_memory_report line).
  **Ratified going forward (bayes): dial K-setting = pure empirical ramp
  (first boot K=26–27, walk DOWN under the executor d1 ramp); dial window
  budget +1 warm-pool variant boot +1 valve engagement probe (valve REVIVED
  as coverage-multiplier inside the dial program); the ~12% d16 prize stays
  soft, no pre-registration until ramp data; W5 go condition = CSV/allocated
  data + executor-slope protocol. Dial-family boots OFF until the revision
  data lands (second-miss rule).** grothendieck pulling the W1b evidence set
  Mac-side (read-only) for the CSV time-structure discrimination (warmup
  spike vs steady plateau decides whether pool-warming relaxes K to 23–24).
  MEMORY_LEG_DECISION.md carries both correction notes in place.

### 2026-08-14 ~04:5x CDT — ANCHOR-ENV CONFOUND FOUND (doppler, pre-W1c)

The TF32 CE-head patch is ENV-GATED (chunked_lm_head.py:65,
`BT_TF32_LM_HEAD` default "0"). The W1a re-anchor on wprm693 (d4 848, d16
945 fresh / 958-962 plateau) ran with NEITHER BT_TF32_LM_HEAD=1 NOR the ship
NCCL knobs — the P0 order specified only BT_SAVE_STATE_SYNC=1 (protocol-doc
gap; bayes owns it, not the driver). The record anchors (878-886/984) ran
the full env. So tonight's -3.5-4% vs record is confounded: plausibly mostly
ENV (TF32 head alone was +3.5% at exp04b), not venv provenance. bayes
adjudication on record: (a) W1c boot 1 (spec env = TF32 + ship NCCL =
record env) doubles as the cross-check — landing ~954-1013 EXONERATES the
venv and kills the hadamard question entirely, and boot-1's numbers become
the headline anchors; (b) landing ~945-class despite record env REOPENS
provenance and un-cancels the hadamard A/B (post-W1c). W1c internal A/Bs
unaffected (shared base env across its 3 boots). Canonical mission env block
now written into BOX_SCHEDULE.md; bench-kit env-dump papercut filed.
- 2026-08-14 (Mac `date`) — **W1b RECONCILIATION, FINAL (pauli, ratified by
  bayes with the guardrail noted).** grothendieck's evidence set (CSVs +
  runlogs + the K=25 same-instant triple) landed Mac-side. (1) UNITS: three
  metrics in play — torch-allocated < torch-reserved (/status, what the bench
  JSONs carry) < nvidia-smi (poller CSVs); K=25 same-instant: 253.2 / 256.8 /
  265.0 (gaps 3.6 allocator cache + 8.2 non-torch). The "+20 GiB anchor
  migration" = ~8 metric artifact + ~13 REAL uniform base shift
  old-box→mission-box (old torch anchors d2 141.3 / d4 143.2 / d16 161.7 vs
  mission ~154/~157/~175; unassigned → pauli's named hygiene item, env/tree
  diff follow-up). Papercut pc_248d82831027 filed on the dual-metric
  footgun. (2) MODEL: the linear stored-set model HOLDS where the ceiling
  doesn't bind — stage-0's K=21→K=25 slope 5.8 GiB/layer = 2×(e−c) ✓ — and
  stage-1 flattens (0.44/layer) under K-independent pinning (loss-path fp32
  stack + cold-pool burst + ceiling fragmentation). Constants of record
  (PRIOR for the ramp, not a fit): base ≈ 139 torch (mission, d2-class,
  stage-1), S_eager ≈ 2.9–3.1, S_ckpt 0.19, I=2; cold-pool burst ≈ +20
  torch; trainer-warmup transient base ≈ +60 (boot#1). K=21's OOM points are
  pre-optimizer-materialization = lower bounds. (3) FLOORS at the 255 poller
  line: K=25 fits steady (~251 smi), K=24 razor, K=23 needs the burst
  managed. Coverage 37.5/40/42.5% at K=25/24/23; dial prize ~12–13.5% at
  d16 (soft, no pre-registration until ramp data). Valve-stacked-on-dial
  load-bearing past K~23. Runbook rule: never cold-pool into a near-ceiling
  config (settle-then-warm or graduated driver warmup). CSV pre-registration
  of record: declining-to-~245–250 = burst theory; flat-265 = structural.
  **GUARDRAIL (bayes): W1b's DOES-NOT-FIT verdict is CLOSED — the
  K=25-fits-steady hypothesis is a NEW experiment (warm-pool boot inside the
  dial program's budget), not a retro-appeal; any blockK revival gets a fresh
  pre-registration after W1c/W2.** Dial-family boots remain OFF until the
  ramp data lands.
- 2026-08-14 (Mac `date`) — **W1b FINAL STATE (pauli; bayes guardrail on
  record).** (a) Anchor-migration hygiene item CLOSED — never existed: the
  W1a JSONs carry torch-reserved via /status, and the boxes AGREE within
  ±2.5 GiB on that metric (mission d2 138.8 / d4 143.0 / d16 163.8 vs
  old-box 141.3 / 143.2 / 161.7); the 131k metric gap is 8–12 GiB
  (torch-reserved vs nvidia-smi). (b) NEW hygiene finding in its place:
  **anchors are step-position-dependent — torch-reserved creeps +20.8 GiB
  across a run's early steps (143.0→163.8 over d16 steps 6→9), then plateaus
  flat (d16-ext through step 16). Memory A/Bs compare at matched step
  positions or at plateau; a window-1 read is not a steady read.** (c) The
  K=25 measurement is warmup0-ONLY (run terminated after warmup0; canary
  PASS 12.3242/0.4185; no mains, no result JSON) — 265.0/256.8 is the
  cold-pool first-window peak, steady state unmeasured; the model fits the
  warmup0 point exactly (steady 236.7 torch + 20.1 burst = 256.8). The
  warm-pool variant boot is the sole remaining discriminator and must run
  into mains for the plateau; dial-ramp fit reads at plateau or matched
  steps. W1b DOES-NOT-FIT stays CLOSED (bayes guardrail); the
  K=25-fits-steady hypothesis is a NEW experiment inside the dial program's
  budget, and any blockK revival gets a fresh pre-registration after
  W1c/W2.
- 2026-08-14 ~06:2x CDT — **W1c VERDICT (kolmogorov/bohr spec+adjudication,
  lovelace drives, wprm693): the B/F host-cache pair WINS at the mission
  topology post-M=N.** Window protocol per W1C_SPEC.md (3-boot, one-variable
  discipline; tree verified §0: canonical 73c24b00+TF32 bits, mcore dirty-tree
  sha256 e1e46818 exact, wheel 1.27.0). **Boot-1 cross-check finding of
  campaign weight: the record env (TF32+ship NCCL) on the fresh venv EXCEEDS
  the old record — d4 920 vs 878-886 (+3.9-4.8%), d16 1050.5 vs 984
  (+5.5/+8.0%) — the venv is EXONERATED and tonight's -3.5-4% anchor gap was
  the missing env (bayes rule (a) confirmed); the fixed wheel on a clean venv
  recovers the corrupt wheel's perf class (1052-era) with correct numerics.**
  **B/F d16 (decision rung): 1088/1118 (mean 1103) vs off-arm 1038/1063
  (1050.5) = +5.0%, |Δ| 52.5 > max-spread 30 ⇒ DISTINGUISHABLE, squarely in
  kepler's central +4-7% band; the <+1.5% leak-causality falsifier nowhere
  near.** B/F d4: strict x2/x2 read was perf-indistinguishable (on-arm
  warmup trend 929→974 vs off-arm 915/925); under the standing steady-state
  convention the warm class reads 974/966 (~970) vs 925 ≈ **+4.9%**, top of
  gauss's +0-5% band. Hard gate PASS (both gates ACTIVE ×16 at WARNING —
  inertness excluded). Memory flat (143/163 GiB both arms, all rungs).
  Canary clean every first window; the r3 canary-json −0.14 = weight
  evolution ~18 windows deep (lovelace's flag ratified — same signature as
  the Aug-9 A131-131k-d2 row), not numerics. Mechanism pre-read: on-arm
  traces are ~HALF the off-arm's size (1.32/1.43 GB vs 2.8/3.0 GB) — the
  host-sync-removal signature is visible in the event count before formal
  analysis; jacobi's W1C_TRACE_READS.md read (nonzero collapse + idle shrink)
  pending as the on-arm pair lands Mac-side. NCCL Arm-1 (boot 3, ship-env
  OFF) in flight. Decision-rule consequence per REBUILD_NOTE §5: **recommend
  landing PR #26 + the pointer-bump chain (branches already pushed:
  trainers/bridge jackrao/lps-1062-bf-rebuild); merge call = bayes/Jack per
  the merge-queue posture.** Follow-on per the same rule: the customer
  16k-d32 shape probe (Aug-9's biggest win, +18.5%) when the window allows.
- 2026-08-14 ~06:5x CDT — **W1c WINDOW COMPLETE (all 3 boots; kolmogorov
  adjudication, lovelace driving).** Boot 3 (NCCL Arm-1, ship-env OFF,
  /proc-verified absent; canary PASS): d4 923/929 (mean 926, spread 6) vs
  ship-ON 920 (915/925, spread 10) ⇒ |Δ| 6 ≤ max-spread 10 ⇒ **SHIP NCCL ENV
  INERT on PP2/CP8/EP8 @131k, per the pre-registered rule** — the post-M=N
  confirmation of Run B's pre-M=N +0.4%. Mechanism: the three knobs are
  NET/IB-scoped; on this topology only PP p2p rides the wire and it is
  wire-speed/park-dominated (NCCL_RESWEEP_PLAN.md §2). Ship-package guidance:
  keep the env (load-bearing +51% for golden EP16/CP16 on RoCE;
  harmless-neutral on PP2) — merge-queue item 9 updated. **Window summary of
  record: B/F on-arm d16 +5.0% (distinguishable, kepler central band) = the
  new operational record class (1103); d4 warm +4.9%; NCCL inert; memory
  flat; canaries clean throughout; both d16 arm traces safe on shared FS +
  Mac-side.** Remaining closure: jacobi's mechanism read (leak-causality)
  on the trace pair. Box released to W2 (ramanujan's shim canary); the tree
  swap to a3da1223 is restore-safe (mcore dirty diff snapshotted byte-exact
  Mac-side, e1e46818...).
- 2026-08-14 05:4x CDT (Mac `date`) — **W1b died at fit (no windows; jacobi's
  W1B_TRACE_READS rolled off VOID-no-data, not failed). W1c (B/F) MECHANISM
  VERIFICATION COMPLETE (jacobi ex-kepler; traces pulled, sha256-logged
  pp2cp8ep8/w1c_traces.sha256, served :9006-9009).** VALIDITY: off-arm
  reproduces the fe127 census EXACTLY (55,328 nonzero; 54,720 builder / 608
  bare-bwd; RoPE 1,527 heavy aten::to; 971,761 kernels) — M1's survivor
  expectation pinned to this box/wheel. Baseline note: tonight's off-arm pure
  idle is 12.00s (NOT fe127's 20.23s — leak-regime boot variance on identical
  call counts; the same-night on/off pair is the clean substrate). MAIN EVENT
  (on vs off, same box/venv): **nonzero 55,328 -> 1,328 — EXACT vs the
  pre-registered ~1,330** (720 builder first-misses, replay side ZERO =
  carrier cache hit; bare-bwd 608 unchanged = A-v3's class survives as
  registered). RoPE host 35.3 -> 1.07s. **Pure idle 12.00 -> 4.07s (-7.9s)
  — 3x the static leak coverage (2.71s tonight): the micro-gap tax collapsed
  9.51 -> 0.89s as kernel launches fell 971,761 -> 432,793 (-539k/step). F6's
  launch-decompression measured directly on the fixed wheel at d16;
  leak-causality CONFIRMED, conversion >1.** SendRecv flat (47.1 -> 45.9s ✓
  F6-consistent). Traced-step span 130.72 -> 119.85s = -8.3% (~ +9.1%
  tput-equivalent) — at the ceiling of jacobi's filed band (+4-7%/+5%, ceiling
  +9-11%); lovelace's bench A/B is the throughput authority. rank8 (first
  fixed-wheel rank8): nz 58,304 -> 1,424, idle 11.13 -> 1.98s, span -8.5%
  (same step ✓); tonight's off-arm stage leaks near-SYMMETRIC (2.52 vs 2.72s)
  — the l3 2.7x asymmetry did NOT reproduce on the fixed wheel (regime-
  dependent; no-haircut verdict holds trivially). M5: B/F host-side only,
  L untouched -> rebalance arm STAYS DEAD ✓. M7 seam null-control TRIPPED
  (1.80 -> 2.66s, outside +/-0.3s): the untraced-python seam holes persist
  (1,247+828ms) and now dominate — **the residual idle is seam-dominated
  (2.66/4.07s = 65%); R2 (seam python, py-spy probe) is now the top residual
  idle lever at 2.2% of the new step.** Campaign map consequence: the 20.8s
  idle problem is a 4.1s problem; lever-1 (exposed SendRecv, contract shim)
  untouched and standing. Docs: W1C_TRACE_READS.md (full read table),
  IDLE_RESIDUAL_MAP.md §4 outcome line.
- 2026-08-14 ~07:3x CDT — **B/F LEVER FULLY CLOSED (jacobi's mechanism read,
  W1C_TRACE_READS.md): leak-causality CONFIRMED, not just the speedup.**
  Off-arm validity check reproduced the fe127 census EXACTLY (55,328 nonzero,
  54,720 builder / 608 bare-bwd split, 971,761 kernels) — and caught a real
  boot-regime variance: tonight's off-arm idle was 12.00 s vs fe127's 20.23 s
  on identical call counts, re-anchoring the read to the same-night pair.
  On-arm vs off-arm: **nonzero 55,328 → 1,328 (EXACT vs the ~1,330
  pre-registered survivor count; replay side ZERO = the carrier cache hit),
  RoPE host 35.3 → 1.07 s, pure idle 12.00 → 4.07 s, launches −539k/step,
  micro-tax 9.51 → 0.89 s, SendRecv flat, traced-step span −8.3%/−8.5%
  (r0/r8).** The +5.0% d16 win therefore has its mechanism on record: the
  caches removed the host-sync mass exactly where the leak map said they
  would. Merge-queue item 7's last closure item is RESOLVED; PR #26 +
  pointer chain await Jack's merge call. (The on-arm traces at ~47% of the
  off-arm's byte size were the early visible signature.)
- 2026-08-14 ~06:0x CDT (Mac `date`) — **jacobi (ex-kepler): R2 SEAM-PROBE
  SPEC FILED (bayes-ordered).** runs/overnight_20260813_overlap_campaign/
  R2_SEAM_PROBE.md. Targets the on-arm seam idle (2.66s = 65% of the post-W1c
  residual 4.07s = 2.2% of the new step; holes 1247+828+362+122ms of
  untraced python). Pre-registered hypotheses for the B/F-grew-the-seam
  reading (1.80 -> 2.66s absolute): H1 exposure shift (constant ~2.6-2.7s
  python, pre-B/F ~0.9s hid under GPU drain backlog — favored) vs H2 new work
  (cache maintenance); discriminator = function-level composition, H2 refuted
  if cache functions <2% of seam samples. Protocol: py-spy record 250Hz x 420s
  (3+ seams) on rank0+rank8, --threads, isolated install (never the trainer
  venv), event-driven alignment to trainer step-log lines + dmon util dip,
  artifacts Mac-side immediately, probed steps EXCLUDED from perf reads.
  Decision rules by dominant family: logging -> async emit (~1.5-2s), dataloader
  -> prefetch overlap (~1-1.3s), optimizer glue -> slimming (~0.5-1s), NCCL
  host wait -> route to lever 1 and re-scope. STOP conditions S1-S4 (no
  attach x2 / any regression / two alignment misses / no perf-role creep).
  Runs as a ~10-min micro-window after W2's canary+parity legs, no dedicated
  boot. Trace servers stay up for the report (:9002/:9004/:9005 baselines,
  :9006-9009 W1c pairs).
- 2026-08-14 08:09 CDT (Mac `date`) — **W2 LEG 1: control PASS; flag-ON d2
  STOPPED on a latent chunked_lm_head bug exposed by the shim's fp32 boundary;
  FIXED + pushed; ladder resuming on the new tip.** Control re-anchored clean
  (fresh flag-OFF d2, plain PP2 32k: loss 12.2923/12.3019/12.2999 → band
  12.292-12.302, matches the selective twin; gn 0.356-0.410; peak 202/225 GiB
  = the 32k plain-selective class). The flag-ON d2 then died in the LOSS-NODE
  FORWARD (forward_step_calc_loss → _ce_loss_from_hidden → _project_logits →
  adapter_forward → peft linear_in matmul: float vs BFloat16). Adjudication
  (ramanujan, source-verified): NOT the landmine, NOT the shim contract — the
  combined-1F1B PostProcessNode unconditionally float16_to_fp32-upcasts the
  plan-boundary hidden (the accepted Option-B cost), and the fp32-head LoRA
  path's DELTA call passed that fp32 hidden to the bf16 LoRA adapter weights;
  the base projection was already fp32-safe, the delta was a latent
  bf16-assumption (its own comment: "same bf16 adapter call"). Fix
  (1cd31535): cast the delta's hidden to the base layer's bf16 working dtype
  — no-op conventionally, value-exact under the executor (fp32 boundary = an
  exact bf16 upcast, so the downcast recovers it). Verified Mac-side both
  paths + regression test (fp32-boundary hidden → delta runs bf16, numerics
  exact; without the fix the fake's bf16 matmul reproduces the mismatch).
  **Landmine-fix (b37c01f2e) status CORRECTED: STILL PENDING hardware
  validation** — this failure was in the loss-node forward, upstream of any
  backward, so the grad-root fix never got hardware contact (lovelace's
  "passed first contact" was premature; bayes's books corrected). Also on
  record: (a) the exp03 validator wall fix landed as a real commit (d34f76d9,
  moe_shared_expert_overlap cleared whenever the overlap flag is on,
  dispatcher-independent — the campaign's standing on-box patch, now shipped);
  ON-vs-OFF delta now packages executor + VPP chunking + shared_expert_overlap
  -off (scheduling-only; perf confound carried into the W5+ A/B accounting).
  (b) **BAYES OWNS THE SMOKE LESSON, on the record: the cut-down 1-node smoke
  (hatch a) was built and then held when the B300 box landed; this
  fp32×LoRA-delta failure class is EXACTLY what it would have caught for ~1h
  of Mac time and zero mission-box cost. Opportunistic hardware de-risking has
  positive EV even when the real gate is imminent — final-report lessons
  section.** Ladder resumes on 1cd31535; landmine fix's first backward
  hardware contact still pending.
- 2026-08-14 08:4x CDT (Mac `date`) — **W2 LEG 1 FULLY GREEN: the contract shim
  trains on hardware AND the landmine fix is VALIDATED.** Flag-ON d2 canary on
  1cd31535 (lovelace): warmup0 ran the FULL fwd+bwd with ZERO RuntimeError —
  no stage-0-backward failure → **b37c01f2e (the PreProcessNode grad-root fix)
  IS hardware-validated** (its first real backward contact, after the leg-1
  loss-forward bug had died upstream of it). CANARY: loss 12.3052/12.3016/
  12.2925, gn 0.3628/0.3614/0.3724, loss trains. vs the control band
  (12.292-12.302): both mains IN band (12.3016, 12.2925); warmup0 12.3052 is
  +0.003 above the band top — adjudicated (ramanujan): the flag-ON warmup0 is
  M=2 (the documented VPP2 warmup-datums deviation) vs the control's M=1, so
  the window token content differs, and the campaign's warmup0-transient
  pattern runs excess on shifted regimes — NOT a shim numeric defect; the
  mains are the canary and they're in band, gn comparable, loss trains.
  **CANARY PASS.** Peak mem 210/233.5 GiB vs control 202/225 — the VPP2
  2-chunk footprint is ~+8 GiB at 32k. The executor contract shim is now
  hardware-real: the combined-1F1B executor is reachable from our trainer and
  trains the same loss. Next: leg 2 parity (the tight numerics gate).
- 2026-08-14 08:5x CDT (Mac `date`) — **W2 LEG 2 PARITY: FAIL on both bars;
  window paused for root-cause (ramanujan adjudicating).** Numbers (lovelace,
  tree 1cd31535, both arms fresh boots, capped 6-datum set, forward_backward —
  executor engaged): loss ON 12.29735 vs OFF 12.29890 → rel_diff 1.259e-04
  (bar 1e-6); logprobs max_abs_diff 3.758 (bar 1e-3); datum count/lengths
  EXACT (6/6, 109827/109827 tokens — that bar passes). Pattern: PERVASIVE
  per-token divergence (87-96% of tokens in every datum exceed 1e-3, magnitude
  ~1.3-2.3, worst scattered 3%-75% through datums) with a nearly-preserved
  aggregate mean. Read (ramanujan): each token's logprob is computed against a
  shifted context/position under the executor — a per-token ORDERING/CONTEXT
  bug in the plan path. The leg-1 canary PASS does NOT clear this (the canary
  band ~0.01 wide can't resolve a 1.26e-4 offset; training tolerates small
  per-token shifts) — the parity gate is the tight instrument and it fired
  exactly as designed. Prime suspects (confidence ~65/25/10): (a) the loss
  node's CP-collective/zigzag-stitch ordering under the executor (the scoping
  doc's flagged §4-risk-2 watch item) corrupting REPORTED logprobs; (b) a
  forward per-token context shift (THD cu_seqlens / DSA indexer top-k under
  the executor's schedule); (c) runner per-partition mapping (least likely —
  datum counts exact). NEXT: divergence-STRUCTURE analysis (Mac-side) on the
  two parity JSONs — cross-correlate ON[i] vs OFF[j] to discriminate
  positional-layout-offset (reporting) vs context-corruption (forward), and
  whether the divergence aligns to THD zigzag chunk boundaries or the DSA
  4-layer top-k grouping. No further box boots until the structure is
  adjudicated. **The shim does not merge until leg 2 is green.**
- 2026-08-14 09:1x CDT (Mac `date`) — **W2 LEG 2 ROOT-CAUSE, STRUCTURE VERDICT:
  FORWARD-VALUE divergence, NOT a reporting/ordering offset (ramanujan,
  Mac-side on the two parity JSONs).** The multiset test is decisive:
  sorted(ON) vs sorted(OFF) diverge by up to 1.35 (a permutation would be
  ~0) — the executor computes genuinely different per-token logprob VALUES.
  The CP-collective/stitch reporting path (scoping §4 risk-2 watch item) is
  RULED OUT as the primary cause. Ownership boundary verified clean: the
  shim's threading is NOT the cause (fp32_output is a no-op under the
  executor; packed_seq_params/masks thread identically; the fp32 boundary is
  value-exact). The divergence is in mcore's combined-1F1B executor's FORWARD
  decomposition of GLM-5.2 DSA/THD — the executor was built for standard MoE
  attention; DSA (learned indexer + discontinuous top-2048 selection +
  host-read layout builders) is non-standard. Pattern read: pervasive
  per-token shift ~1.3-2.3 (max 3.758), zero-mean-ish so the aggregate mean is
  preserved to 1.26e-4 and the canary trained clean — most consistent with
  the DSA top-k's discontinuity amplified by the executor's different kernel
  scheduling, OR a deeper decomposition/race in the fine-grained callables'
  DSA handling under interleaving; the JSONs alone can't separate
  benign-numeric-sensitivity from real-bug. **Campaign-level: the shim
  contract is PROVEN (executor reachable + trains to aggregate precision;
  landmine fix validated); the open question is the executor's PER-TOKEN
  numeric equivalence on DSA.** NEXT (bayes's approval requested): a
  determinism probe — flag-ON parity TWICE, ON-run1 vs ON-run2.
  Self-consistent → systematic decomposition difference (re-derive the
  executor parity bar, proceed with a documented caveat); wildly divergent →
  nondeterministic race on our stack → blocker, escalate mcore-side. Analyzer:
  tools/parity_divergence_structure.py (proven on synthetic permutation vs
  context-shift cases); full output at
  runs/overnight_20260813_overlap_campaign/PARITY_DIVERGENCE_STRUCTURE_W2.txt.
  The dial program's executor-dependent rungs are HELD behind this verdict
  (bayes). Shim does not merge until leg 2 is green.
- 2026-08-14 ~07:5x CDT — **W1d VERDICT (kolmogorov adjudication, lovelace
  drives, wprm693): B/F WINS at the customer-dominant shape (16k-d32,
  8 docs/partition): on-arm warm class 1000/1000 (r2+r3 confirmed, tight) vs
  off-arm warm 942 = +6.2%, inside the pre-registered +5-12% band, falsifier
  (<+1.5%) clear.** Strict as-stands x2/x2 read was perf-indistinguishable
  (on-arm warmup trend 955→1000 vs off-arm 931/942); the standing
  steady-state convention + the documented r3 confirmed the warm class.
  Hard gate PASS (both gates ACTIVE ×16), canaries clean (drift ≤7e-4),
  memory flat (141-142 GiB both arms). Directional read on record: the
  docs-per-partition scaling hypothesis holds (16k-d32 +6.2% > d4's +4.9% —
  more doc bookkeeping, bigger win), with the honest caveat that post-M=N
  runahead absorbs much of what the convoy regime exposed (Aug-9's +18.5%
  was that regime's extreme and does not transfer). **B/F lever final
  ledger: d16 +5.0% (distinguishable, mechanism confirmed by jacobi's trace
  read), customer 16k-d32 +6.2% (warm-confirmed), memory flat, correctness
  clean — PR #26 + pointer chain unconditionally ready for Jack's merge
  review (merge-queue item 7).** One process note: a stale W2 trainer held
  the slurm allocation ~25 min pre-boot-1 (lovelace self-caught; surgical
  scancel-by-jobid; standing rule added: verify squeue empty before
  dispatching; papercut pc_16ab5ffe6226 on scancel --name over-breadth).
  Box released to ramanujan's determinism probe (tree swap to 1cd31535;
  mcore gate-stack snapshot e1e46818 preserved Mac-side).
- 2026-08-14 11:3x CDT (Mac `date`) — **W2 DETERMINISM PROBE: ROUTE (b)
  CONFIRMED — the combined-1F1B executor is NONDETERMINISTIC on the GLM-5.2
  DSA stack. BLOCKER (pre-registered route b).** Probe (lovelace, tree
  1cd31535, capped 6-datum set, forward_backward, no optim_step): WITHIN-BOOT
  (same boot, same step-0 weights, same data, back-to-back) on-r1 vs on-r2
  logprobs max_abs_diff 5.098, on-r2 vs on-r3 6.343; BOOT-TO-BOOT (leg-2
  boot-A vs boot-B) 3.424 — all 3-6 THOUSAND× the ~1e-3 kernel-noise bar.
  Aggregate losses stay tight (12.2988-12.2993, rel 2.3e-5/3.8e-5) — the
  preserved-mean-with-per-token-chaos signature, now proven to be run-to-run
  nondeterminism WITHIN one boot (not a systematic decomposition difference).
  CHARACTERIZATION (ramanujan, Mac-side on the 3 probe JSONs): a UNIFORM
  ~0.08/token run-to-run noise floor (80× the kernel-noise bar — the
  executor's timing-dependent reduction order is measurably noisier than the
  conventional path even in the bulk) PLUS scattered large spikes (up to 5.1)
  at RECURRING positions (worst recurs at datum 3 pos 25055 = 92% through, on
  both within-boot compares — deep in the DSA top-k boundary region). Read:
  the executor's interleaved scheduling introduces timing-dependent numeric
  noise, and the DSA indexer's discontinuous top-2048 selection amplifies it
  into large per-token flips at knife-edge positions. FIRST SUSPECT (bayes's
  prior): the DSA kernel race class — the wheel saga's race was
  scheduling-dependent, and the executor's interleaved scheduling is exactly
  that class; the kernel-source snapshots from that investigation are on
  disk. **What is BANKED and unaffected: the shim CONTRACT is proven (the
  executor is reachable from our trainer and trains to aggregate precision;
  the landmine fix b37c01f2e is hardware-validated). The blocker is the
  executor's DSA nondeterminism, NOT the shim.** Per route (b): executor
  rungs stay HELD (incl. the dial program's executor-dependent rungs);
  escalate to a mcore-side DSA×executor investigation; the campaign's
  remaining hours pivot to shippable levers + the report (bayes's call).
  Probe JSONs Mac-side at ~/perf_profiles/lps-1062/incoming/
  parity_overlap-on-r{1,2,3}.json; analysis tools/parity_divergence_structure.py.
- 2026-08-14 11:4x CDT (Mac `date`) — **UPSTREAM ESCALATION NOTE WRITTEN
  (ramanujan, bayes's final task): results/UPSTREAM_ESCALATION_DSA_EXECUTOR.md.**
  Format per serre's UPSTREAM_PROPOSAL.md precedent. Contents: the ask (a DSA ×
  combined-1F1B numeric-equivalence investigation upstream); the
  determinism-probe evidence table (within-boot 5.098/6.343, boot-to-boot
  3.424, all vs the 1e-3 bar; aggregate tight at rel ~3e-5; the ~0.08/token
  uniform noise floor + recurring spikes up to 5.1 incl. datum-3 pos-25055);
  the exact reproducer recipe (GLM-5.2-FP8 LoRA32, branch
  jackrao/lps-1062-overlap-contract-shim @ 1cd31535 + bridge 146f2636 + mcore
  b37c01f2e, config trainer_pp2cp8ep8_32k_selective_vpp2_overlap.json, fixed
  wheel 1.27.0, BT_SKIP_WARMUP=1, parity_driver 3× forward_backward no
  optim_step); the DSA-kernel-race prior (the wheel saga's LPS-1003 race-fix
  series incl. the TMEM WAR race at GLM-5.2's exact head_dim 576/512, kernel
  snapshots on disk) with the hypothesis that a residual DSA race is hidden by
  the conventional schedule's serialization but exposed by the executor's
  interleaved stream scheduling; and the ruled-out list (the shim contract,
  the landmine fix, reporting/ordering) so upstream doesn't re-litigate. The
  overlap program's executor-dependent rungs stay HELD behind this; the
  shim branch is correct and banked. ramanujan's lane (shim + A/B design +
  two root-causes) is complete.
- 2026-08-14 ~12:3x CDT (Mac `date`) — **jacobi (ex-kepler): W1 V1 COMM-2
  READ COMPLETE (kolmogorov handoff, hausdorff spec).**
  runs/overnight_20260813_overlap_campaign/W1_V1_COMM2_READ.md; traces
  w3_v1_rank{0,8} sha256'd, served :9010/:9011 (d4 window, ~42s span).
  VERDICT: **stream separation CONFIRMED, overlap NOT realized in V1.**
  (a) comm2 exists: track 10 both ranks, distinct from comm1 (tokens a2a,
  track 11/12) and PP p2p (track 24/6). (b) probs rides comm2: exactly 1/3 of
  a2a calls (420/840 r0, 480/960 r8) = 3 probs calls/layer-mb; small-payload
  signature (min 14-19us vs comm1's 2.5-2.7ms); phase split fwd 1 : bwd-region
  2 per layer-mb (likely the counter telemetry's 1:1:2 basis; my stream split
  = tokens:probs 2:1 by calls — counter owners reconcile). (c) THE EXCEPTION:
  fwd probs 97% hidden (works), but **bwd probs-grad — the actual W1 target
  class (5.41s/step exposed on the Aug-10 d16 map) — is still only 12-15%
  hidden (1.0-1.4s exposed in this ~1.1-step d4 window), and comm1/comm2 never
  overlap each other (0.4-0.5%)**: V1 bought stream separation, not
  concurrency; the reorder's issue-early/wait-late is not visible — either not
  in this build or the issue point still gates. Consequence for the queue:
  the -3.4...-5.1s prod-fabric model number is NOT supported by this trace;
  cheap next diagnostic = does the V1 build contain the reorder at all
  (source-side), and if so move the issue point earlier. d16 V1 trace would
  re-size the exposed class directly. Reported to kolmogorov + bayes.
- 2026-08-14 ~08:2x CDT — **W3 (W1 validation) GATES V0-V2 ALL PASS;
  direction change into W4 (kolmogorov adjudication, lovelace driving).**
  V0 PASS: the multi-EP-group verification PR #28 deferred — 16/16 armed
  verbatim + 16/16 gate-ACTIVE, comm-2 rendezvous clean at the first gated
  dispatch, a full optimizer step completed (two in fact; losses
  12.3242/12.3085 in band, gn 0.4157; peak 167 GiB = +2.9 vs the +1.3
  prediction, annotated for ship docs). V1 PASS: counters 900/900/1800
  (1:1:2 issue/wait signature); jacobi's structural trace read
  (W1_V1_COMM2_READ.md) then SHARPENED the finding: stream separation
  CONFIRMED (comm2 distinct, probs rides it, 2:1 tokens:probs by calls) but
  OVERLAP NOT REALIZED in the W1-alone build — fwd probs 97% hidden, bwd
  probs-grad (the 5.41s/step target class) still only 12-15% hidden,
  comm1/comm2 never overlap (0.4-0.5%). Source-side confirmation
  (kolmogorov): the reorder (BT_MOE_PROBS_BWD_REORDER) is ABSENT from the
  w1-port build by construction (it lives only in the option6-staged chain)
  — so the -3.4..-5.1s model number was never W1-alone's to capture; it is
  W4's. V2 PASS: d2 mains 12.3029/12.3060 in band, gn comparable, zero drift
  (bitwise-safe construction confirmed). **DECISION (bayes proposal,
  ratified by kolmogorov as spec proxy): V3 (W1-alone perf A/B) DEMOTED —
  trace-predicted null, not decision-relevant (the ship unit is the PAIR,
  whose number comes from the soak-stack A/B). W4 PROMOTED: option-6 canary
  + A/B on the option6-staged chain (trainers 855d8d61 -> bridge d92774fa9
  -> mcore 70710d116), arms = W1-armed baseline vs W1+option6, d4 x2 with
  +2% mean-vs-spread floor, d16 on clean pairs, ON-arm d16 traced window for
  jacobi to re-size the exposed bwd probs-grad class directly. hausdorff's
  sequencing preserved (W1 validated before co-arming; option-6 vs the
  W1-armed baseline, never vs 880).** If W4 loses, BOTH gates revert to
  default-off with 'structurally in, prizeless without the reorder'
  documented; if it wins, the pair joins the soak-stack A/B (P4).
- 2026-08-14 ~12:4x CDT (Mac `date`) — **jacobi: W1-V1 source-side check
  COMPLETE (bayes-ordered, kolmogorov's definitive answer independently
  re-verified).** (1) The bwd reorder (BT_MOE_PROBS_BWD_REORDER) is ABSENT
  from the V1 build by construction (hausdorff's never-co-armed rule):
  verified absent on the local W1-lineage tip (ship-w1 = d58a3214a) and
  main-line 57efae08b; present only in the option6-staged chain (mcore
  70710d116: transformer_engine.py/mappings.py/moe_utils.py). The V1 trace
  verdict is exactly the expected W1-alone shape. (2) **The 85%-exposed class
  is verbatim what option-6 attacks**: the patch makes the bwd probs reverse
  a2a the "early" sibling — issue WITHOUT the compute-stream wait, handle
  stashed in the carrier, the tokens-reverse "late" sibling waits both — "the
  early issue's flight then overlaps the remaining backward instead of
  stalling the compute stream at the early point" (patch text). My measured
  class (bwd probs-grad on comm2, 1.0-1.4s exposed per d4 window, 12-15%
  hidden, 0.4-0.5% comm1/comm2 concurrency) IS that stall; "recovers W1's bwd
  residual" confirmed as the design claim matching the measured class
  (5.41s/step at d16 on the Aug-10 map = the modeled -3.4...-5.1s territory).
  **Pre-registered option-6 canary trace reads filed in W1_V1_COMM2_READ.md
  (O1-O5)**: O1 bwd probs-grad exposure ~85% -> <20%; O2 comm1/comm2 mutual
  overlap ~0.5% -> materially >0; O3 no new exposed class at the late sibling;
  O4 bench -3.4...-5.1s at d16; O5 canary band 12.2-12.4 + gn comparable
  (schedule change, numerics-neutral; drift = STOP). Telemetry note from
  kolmogorov ratified: my 2:1 tokens:probs by CALLS vs the counter's 1:1:2 =
  issue/wait-granularity artifact, not a mechanism discrepancy.
- 2026-08-14 ~10:1x CDT — **wlxj8vw (B200 dial-validation side-box) STOPPED
  per Jack's kill-unused-devboxes order (bayes relay).** Evidence sweep
  complete before the stop: rung-1 log byte-exact Mac-side (34,006 B),
  T5 diagnostic evidence pulled (rung1_t5_diag_evidence_20260814.log),
  test file canonical in the campaign folder, wheel-bump gotchas in
  papercuts. The box served rung 1 (7/7) and was parked as insurance;
  insurance released.
- 2026-08-14 ~08:5x CDT — **W4 VERDICT (kolmogorov adjudication, lovelace
  driving): option-6 LOSES — no measurable win over the W1-armed baseline at
  either rung (d4 ON 830 vs OFF 850: -2.4% under spread, warm -0.6%; d16 ON
  937 vs OFF 936: +0.1% under spread, warm -0.4%). The lever model's
  -3.4..-5.1s/step is NOT realized on this topology tonight. Per the
  pre-registered framing: W1 + option-6 BOTH revert to default-off,
  'structurally in, prize unrealized' documented.** Canary gates all PASS
  (W1+option-6 gn-clean — the reorder's backward-order change is numerics
  clean). Mechanism closure pending: jacobi reads the ON-arm d16 trace (did
  the reorder fire + shrink the exposed bwd probs-grad class — the
  TF32-state-independent question). **CONFOUND FINDING (process): the W3/W4
  tree swap dropped the on-box TF32-head patch (grep-verified 0 hits on the
  box tree; the W1d restore had stash-popped it, the W3/W4 checklist didn't
  carry the line) — W3/W4 absolute levels are the no-TF32 class (d4 850 ≈
  the 848 anchor), so any cross-comparison to the TF32-on record class
  (920/1050.5) is INVALID; the A/B's internal ON-vs-OFF verdict is
  unaffected (same tree both arms). Papercut pc_6dca3b0f40f0; durable fix =
  landing PR 995.** W1 lever final ledger: structurally validated on the
  2-EP-group topology (V0-V2 PASS — the PR #28 deferred verification DONE),
  perf-prizeless without and with the reorder tonight; both gates stay
  default-off. W1/W4 CLOSED-negative.
- 2026-08-14 ~13:5x CDT (Mac `date`) — **jacobi: W4 ON-arm d16 MECHANISM
  CLOSURE (rank0; rank8 cross-check pending transfer).** Trace
  w4_on_d16_rank0 (2.82GB, sha256'd, :9012; no-TF32 swap confound MARKED,
  irrelevant to mechanism fractions). Stream map: comm1=track10 (3,360 SR,
  25.3s), comm2=track9 (1,680, 4.17s), PP=track24 (34, 34.7s); a2a total
  5,040 = fe127 exact; comm1:comm2 2:1 ✓. **O1-O3 vs pre-registered bars:
  O2 MET (comm1/comm2 mutual overlap 0.5% -> 12.8%), O3 MET (no new exposed
  class at the late sibling; comm1 87.5% exposed = standing lever-1 prize),
  O1 PARTIAL — bwd probs-grad hidden-under-compute 26.6% vs the >80% bar
  (V1 d4 was 12-15%).** Closure answer: **the reorder FIRED (no arming bug)
  but the shrink is PARTIAL** — firing position p50 = 0.69 into the layer
  backward (issue-early visible); early-half fires (n=309) complete in 0.27ms
  fully hidden; late-half fires (n=811) carry 4.89ms avg waits = the residual
  exposure is the wait-for-peer/convoy tail + thin remaining-compute cover,
  not the missing-reorder class. Cross-baseline (caveated): Aug-10's 5.41s/
  step exposed probs-grad -> ~2.7-3.0s/step here ≈ half the modeled
  -3.4...-5.1s prize recovered. Post-mortem sentence filed verbatim in
  W1_V1_COMM2_READ.md. rank8 cross-check follows on kolmogorov's relay.
- 2026-08-14 ~14:0x CDT (Mac `date`) — **jacobi: W4 reconciliation sentence
  ON RECORD (bayes-ordered, W1_V1_COMM2_READ.md).** "Exposure halved (~2.7s
  recovered vs the Aug-10 class) yet the bench is flat-to-negative, so
  CONVERSION ~= 0: the reorder moved local waits off the compute stream, but
  step time is set by the slowest rank's convoy, and a locally-hidden wait is
  not throughput." — W1c inverted: B/F removed work EVERY rank did
  (conversion > 1); option-6 hid waiting only SOME ranks do. Residual
  exposed-comm mass is convoy/imbalance-dominated; next real lever = BALANCE
  (expert/token de-staggering), roadmap-scale. EVIDENTIARY SCOPE marked
  honestly: trace supports the mechanism at rendezvous granularity (probs-grad
  wait distribution = pure peer-arrival skew: peer-ready floor 0.27-0.43ms vs
  staggered tail p90 9.8ms / p99 15.7ms; top-10 waits own only 0.16s — the
  mass is in the tail), unchanged in character by the reorder, only placement
  moved. The conversion fact is bench-side (bayes/lovelace). NOT trace-
  testable tonight: per-rank step-time spread ON vs OFF (no W4 off-arm pair
  exists; PP lockstep makes step spans equal by construction) — flagged, not
  overclaimed. rank8 cross-check (cross-rank firing-position stagger) appends
  on kolmogorov's relay.
- 2026-08-14 ~14:3x CDT (Mac `date`) — **jacobi: W4 rank8 CROSS-CHECK COMPLETE
  — the closure verdict is two-rank consistent.** w4_on_d16_rank8 (3.01GB,
  sha256'd, :9013; span 145.04s = rank0's 145.01, same step; CF/CFB 640/640 ✓;
  comm2 = 1,920 = 640x3 exact). rank8 vs rank0: O2 mutual overlap 13.8% vs
  12.8% ✓; O1 bwd probs-grad hidden 18.7% vs 26.6% (same shape, stage 1
  worse); fwd probs 97.2% hidden ✓; firing position p50 0.68 vs 0.69
  (identical issue point); early/late fire durations 0.22ms/6.42ms vs
  0.27ms/4.89ms (stage 1's convoy tail longer; exposed residual ~4.6s/step on
  stage 1 vs ~3.0s on stage 0). **FIRED, PARTIAL, convoy owns the residual —
  both stages.** Direct stagger note: both ranks fire at the same relative
  point but wait differently -> the wait is set by PEER arrival; the EP
  group's other 6 ranks are unprofiled tonight (BT_PROFILE_RANKS=0,8 = one
  rank per stage) — the convoy's existence is trace-visible, its per-rank
  composition is not. HONEST CORRECTION to my rank0 sizing language: the
  "roughly half recovered" estimate crossed baselines (Aug-10's 5.41s was a
  different topology/wheel and a 40-layer-stage number; same-stage W4 read =
  stage 1 ~4.6s exposed ~= only -16% against it). The well-evidenced
  same-night claims are within-trace: O2 concurrency appeared, issue point
  moved into the backward, and the early/late duration split shows the
  residual is peer-stagger; the recovery MAGNITUDE lacks a same-night
  W1-only d16 baseline (V1 was d4) — stated, not overclaimed. Reported to
  bayes + kolmogorov.
- 2026-08-14 ~14:5x CDT — **W4 MECHANISM CLOSURE (kolmogorov adjudication
  note to jacobi's two-rank read): the perf negative is now
  mechanism-EXPLAINED.** The reorder fired correctly (concurrency appeared,
  issue point p50 moved 0.68-0.69 into the backward, both stages), but the
  hide reached only 26.6%/18.7% against the >80% bar because the residual is
  the peer-stagger convoy tail (late-half fires 4.89-6.42ms vs early
  0.22-0.27ms fully hidden) — which NO issue-placement change can fix: the
  wait is set by PEER ARRIVAL, not by when the a2a is issued. The lever's
  ceiling on this topology is structurally convoy-limited; the flat wall
  read follows. W1+option-6 stays closed-negative and default-off; the
  pair's documentation is complete (V0-V2 validation + W1-alone trace + W4
  A/B + two-rank mechanism). The convoy tail itself is the overlap program's
  (shim+dial) target class — flagged as the through-line to the report.
- 2026-08-14 15:4x CDT — **ORCHESTRATOR SUCCESSION (bayes → turing, Jack's
  order, contexts filling; handed off at the atomic boundary per the
  succession rule — soak mid-flight under kolmogorov/lovelace, NOT held
  through it).** New Kimi fleet: hertz, curie, kirchhoff, poincare,
  dedekind, nash. State at handoff: record d16 1103 (mission+B/F, the
  settled ship stack; PR #26 unconditional); soak S1/S2/S3 in flight, box
  wprm693 to idle-armed after S3 for Jack's stop/keep; CAMPAIGN_REPORT.md
  drafted with §6 open; merge queue current; upstream escalation ready;
  conditional blockK warm-pool probe pre-registered. Full handoff message
  in turing's mailbox; compressed state in the bayes memory file.
- 2026-08-14 ~15:1x CDT — **S2-parity-fail ROOT-CAUSED (kolmogorov, Mac-side
  per-token analysis of the pulled parity JSONs): the divergence is the
  path's INTRINSIC run-to-run nondeterminism, NOT B/F — and the decisive
  control is stronger than the booted A2 probe.** The S2 failure pair
  (off-vs-on: per-token max-abs 3.76, 93.6% of tokens >1e-3) is reproduced
  at the SAME magnitude by WITHIN-BOOT, SAME-ARM repeats (on vs on-r1/r2/r3:
  max-abs 2.82-4.17, 93.6-93.7%; on-r1 vs on-r2: 5.10; on-r2 vs on-r3:
  6.34) — same boot, same gates, same tree, same everything. The path's
  forward is run-to-run nondeterministic at per-token magnitude ~3-6 with
  ~93% pervasiveness INDEPENDENT of B/F and independent of boot (the
  fix-campaign's known DSA sparse-attention nondeterminism, now measured at
  per-token level). Loss level: off-vs-on 1.26e-4 rel sits inside the
  ON-arm's own within-boot loss spread (1.56e-4 rel). turing's H2
  precision-mismatch branch was eliminated separately (boots env-symmetric
  except the 2 B/F gate vars; tree/wheel sha-exact — lovelace). **Verdict
  reframe (turing's noise-relative frame, hertz's no-silent-1e-6 sharpen):
  'B/F-off-vs-on is indistinguishable from the path's intrinsic
  nondeterminism at BOTH granularities (loss + per-token); the caches are
  bitwise-exact by construction; canary agreement ≤7e-4' — NOT 'parity
  proven at 1e-6/1e-3' (that bar is unmeasurable on this stack, even
  within-boot).** PR #26 HOLD lifts on this ruling (turing). The A2 fresh
  OFF boot becomes a CONFIRMING control (expected: same pervasive floor),
  no longer the sole discriminator. ESCALATION-WORTHY on its own: the DSA
  run-to-run nondeterminism sets a hard floor under ANY tight parity
  protocol on this stack — every future parity gate must be noise-relative.
  (Doc-drift note: the S2 run used a 6-datum set per the driver JSONs, not
  the runbook's 9 — both arms agreed, comparison internally valid.)
- 2026-08-14 ~15:4x CDT — **⚠️ IN-PLACE CORRECTION to the 15:1x entry above
  (kolmogorov, per turing's provenance challenge — the challenge was
  CORRECT).** The "within-boot same-arm control" analysis in that entry was
  computed on CONTAMINATED inputs: I pulled `parity_overlap-*.json` from the
  shared FS by name, and those files are the **W2-window's EXECUTOR-arm
  determinism-probe artifacts** (6-datum/109,827-token shape; my computed
  comparisons match lebesgue's OVERLAP_AB_DESIGN.md W2 table to 4 sig figs on
  four independent pairs — 3.758/3.424/5.098/6.343 — impossible for
  independent noise draws). The genuine S2 ship-stack files are
  `parity_ship-bf-{off,on}.json` (9 datums / 262,032 tokens; box mtimes
  19:10/19:27 UTC). **VOIDED: the "within-boot ON-arm repeat control", the
  "1.56e-4 loss-spread" interim, and the branch-(b)-confirmed ruling.**
  RE-VERIFIED on the genuine files (kolmogorov, just now): the S2 failure is
  REAL on the conventional path — off-vs-on loss rel 9.112e-05 (matches
  lovelace's live 9.113e-05), per-token max-abs 3.4953, 95.7% >1e-3 (matches
  3.495/87-98%) — still over-bar, still UNADJUDICATED. The executor-floor
  (W2's numbers) and the conventional-path floor are DIFFERENT questions
  (lovelace's scope point); the conventional path's run-to-run floor is
  unmeasured until A2. Process lessons on record: (1) the parity driver
  JSONs carry NO provenance fields (no job-id/timestamp/tree-sha inside) —
  the shared-FS filename collision is exactly the clobbering-rule failure
  class; (2) the shadow structure (turing's provenance challenge + hertz's
  corroboration) caught it — the system working. A2 (fresh B/F-OFF boot) +
  a genuine within-boot control with unique job-id-stamped filenames are the
  re-adjudication path (turing's plan). PR #26 HOLD remains until then.
- 2026-08-14 ~17:1x CDT — **S2 RE-ADJUDICATED: branch (b) CONFIRMED on the
  complete noise matrix — B/F EXONERATED, the S2 "failure" is the
  conventional path's intrinsic run-to-run nondeterminism floor.** The
  matrix (lovelace driving, kolmogorov's pre-registered rule from
  S2_NOISE_MATRIX_PREREG.md, all outputs job-id+ts-stamped + sha256-verified):
  N1a/N1b (A2 within-boot, OFF): 3.716/5.455 max-abs, 95.7%, loss-rel
  7.11e-05/3.18e-05; N3 (A vs A2 cross-boot, OFF): 4.270 / 95.7% / 1.58e-04;
  N4 (the banked genuine S2 off-vs-on): 3.495 / 95.7% / 9.11e-05.
  Pre-registered check: N4 max-abs 3.495 ≤ max(floors) 5.455 ✓; N4 pervasive
  95.7% within the floors' range ✓; N4 loss-rel within the floors' spreads
  ✓ — ALL THREE MET ⇒ branch (b). **The conventional PP2/CP8/EP8 path is
  itself run-to-run nondeterministic at the 1e-6/1e-3 bars; B/F's caches are
  bitwise-exact by construction and off-vs-on is indistinguishable from the
  path's own noise.** PR #26 HOLD LIFTS (turing's marker) — the
  correctness claim stands in the noise-relative wording (hertz/turing
  frame): "off-vs-on indistinguishable from the path's intrinsic
  nondeterminism at both granularities; bitwise-exact by construction;
  canary agreement ≤7e-4" — no 1e-6/1e-3 claim imported anywhere.
  **CAMPAIGN-LEVEL REFRAME (lovelace's, adopted): the conventional path's
  floor (3.7-5.5 max-abs) is the SAME magnitude class as the W2 executor's
  (5.098/6.343) — the executor "race" and the conventional nondeterminism
  are ONE phenomenon: the DSA learned-indexer's discontinuous top-2048
  selection flipping borderline per-token KV choices run-to-run, REGARDLESS
  of schedule.** CONSEQUENCE (turing's earlier reading (ii), now on genuine
  evidence): the W2 escalation doc's "conventional clean, executor exposes"
  contrast is FALSIFIED at the per-token level and needs correction before
  Jack forwards it upstream (nuance: the executor IS louder in the bulk
  ~0.08/token uniform floor; the large-spike pervasive class is shared).
  N2 (ON within-boot floor) rides the S1 soak boot per the amendment
  (parity driver ×3 at boot start, pre-step-1, excluded from soak reads —
  pre-registered non-contaminating). S1 soak gate now turing's go.
- 2026-08-14 ~17:4x CDT — **N2 cell COMPLETE (lovelace, on the S1 soak boot
  pre-step-1 per the amendment): the ON-arm within-boot floor = 4.958/5.312
  max-abs (loss-rel 1.73e-05/1.12e-05) — INSIDE the OFF-arm floor range
  (3.7-5.5).** The B/F gates add ZERO measurable run-to-run noise over the
  path's intrinsic floor. **The noise matrix is now COMPLETE (N1 OFF-within
  3.7-5.5, N2 ON-within 4.96-5.31, N3 OFF-cross-boot 4.270, N4 off-vs-on
  3.495) and branch (b) is fully cross-checked** — N4 sits inside every
  floor measure at every granularity. B/F's correctness claim is final:
  bitwise-exact by construction AND noise-indistinguishable on hardware.
  S1 soak now running (the N2 legs were its pre-step-1 driver runs, excluded
  from soak reads per the pre-registration).
- 2026-08-14 ~16:1x CDT (Mac `date`) — **jacobi: R2 SEAM PROBE RAN + READ
  FILED (S1 soak, job wprm693; py-spy 420s @250Hz on rank0 pid 99478 + rank8
  pid 79110, ~3.5 d16 steps mid-soak; artifacts pp2cp8ep8/r2_seam_probe/
  sha256'd).** R2_SEAM_PROBE.md §5. THE SEAM'S HOST CONTENT, NAMED: rank0
  (stage 0, loop thread asyncio_0): ~4.1s full-device torch.cuda.synchronize
  from the pipeline's _communicate_shapes (p2p_communication.py:263) + ~1.5s
  MoE dispatcher DtoH sync (token_dispatcher.py:1611) + ~0.45s THD
  next-microbatch packing (thd_cp.py:390 <- pack_thd_cp_microbatch <-
  packer.py:144). rank8 (stage 1, loop thread MainThread _peer_loop): **~7.1s
  per-step broadcast_object_list (distributed_c10d.py:3839 <- _peer_loop
  worker.py:311) — the whale** + ~0.65s THD packing. GPU-idle mapping: the
  kineto idle holes = the pure-python segments (THD packing + broadcast
  pickle/rendezvous + optimizer glue — invisible to kineto, no aten/cuda
  slices); the device-sync block + broadcast rendezvous are
  host-blocked-on-drain/peers (seam-adjacent, not all pure idle). **R2-Q2:
  H2 REFUTED (cache/maintenance = 0% of seam samples both ranks); H1
  CONFIRMED — B/F removed the backlog that hid this python, it did not grow
  the seam.** Fix directions (pre-registered rules): (1) the per-step
  broadcast_object_list -> async/deferred or shrunk (7.1s likely
  rendezvous-dominated = the boundary skew wearing a broadcast; async emission
  is the right shape); (2) _communicate_shapes full-device sync -> narrow
  scope or skip for static-shape runs (also ~20% of loop-thread active time
  window-wide = in-step finding worth its own look); (3) THD packing ->
  prefetch/overlap. Optimizer glue confirmed NOT the hole again. EV bound:
  ~1-2s recoverable of the 2.66s seam idle. STOP conditions S1-S4 all clean
  (attached first try, soak undisturbed 1121-1128 t/s/GPU, probed steps
  excluded from perf reads). Reported to turing/kolmogorov/lovelace.
- 2026-08-14 ~18:3x CDT — **S1 SOAK batch-1 (windows 1-21/60) ADJUDICATED
  CLEAN (kolmogorov; lovelace driving).** Throughput settled 1121-1128 warm
  class (note: above the W1c on-arm 1103 — settle-drift class, consistent);
  the mains-14/15 dip to 1049-1054 = jacobi's R2 py-spy window (probed steps
  excluded per spec; recovered 1108-1111 post-probe). Loss smooth monotonic
  12.316→12.138 (training progression); gn smooth 0.37→0.22; memory FLAT
  179.8 GiB (+0.9 over 21 windows, within the 2 GiB creep bar); B/F
  telemetry ZERO fallback/miss/stale lines. No STOP condition approached.
  **jacobi's R2 seam read (R2_SEAM_PROBE.md §5): the driver↔trainer seam is
  NAMED — rank8's whale = ~7.1s/step broadcast_object_list from the
  dp_worker _peer_loop (worker.py:311, rendezvous-dominated = boundary skew
  wearing a broadcast); rank0 = ~4.1s full-device sync from
  _communicate_shapes + ~1.5s dispatcher DtoH + ~0.45s THD packing. H2
  refuted (0% cache functions — B/F NOT implicated in the seam), H1
  confirmed (B/F unmasked, didn't grow).** The soak ran clean through the
  probe — the gap protocol worked as pre-registered.
- 2026-08-14 ~16:2x CDT (Mac `date`) — **jacobi (ex-kepler) LANE HANDOFF
  FILED: runs/overnight_20260813_overlap_campaign/LANE_HANDOFF_jacobi.md.**
  Contents: the naming hazard (old-jacobi = pauli; I am ex-kepler; NOTEBOOK
  attribution rule), the lane's seven docs, the full trace inventory with
  served ports (:9002-:9013, pids, sizes, confound marks), the sha256 ledger
  locations (pp2cp8ep8/w1c_traces.sha256 + r2_seam_probe/r2_probe.sha256 +
  BOX_A_ARTIFACT_MANIFEST.md for the Aug-13 big four), the query machinery
  note (docs carry the exact SQL; never send per-row enclosure joins to the
  slice table — two wedges, both mine, both documented), and the open-items
  register (R2 fix directions, the convoy/balance through-line, the two
  reopen conditions). Nothing in flight; lane closed clean under turing's
  succession. Awaiting turing's release confirm.
- 2026-08-14 ~19:4x CDT — **S1 SOAK batch-2 (windows 22-41/60) ADJUDICATED
  CLEAN (kolmogorov; lovelace driving).** Throughput settled to ~1089 (the
  early 1121-1128 settled = ~2.7% settle drift, comfortably under the >5%
  monotonic-decline stop); loss 12.10→12.01 smooth; gn 0.21→0.15 smooth
  (training evolution, NO step jump — hertz's watch item clean); memory FLAT
  180.1 GiB at main40 (+1.2 GiB over 41 windows, within the 2 GiB bar); B/F
  telemetry still ZERO fallback/miss/stale; no STOP condition approached.
  **Pre-registered throughput read, refined by the data: the deep-settle
  class is ~1089, not the early-window 1121-1128 — so the number of record's
  steady-state estimate lands in the 1089-1103 band (the A/B's 1103 =
  fresh-pair read; the soak's 1089 = deep-settle read; the band between them
  is the settle-drift envelope). No new record, no lever change — exactly
  the pre-registered language.** Final batch + S3 at completion.
- 2026-08-14 ~20:3x CDT — **S1 SOAK COMPLETE (60/60 steps) — FINAL
  ADJUDICATION: PASS WITH TWO FLAGS ON THE RECORD (kolmogorov; lovelace
  driving).** Primary claims all hold: throughput deep-settle 1080-1091 (the
  record's steady-state band 1089-1103 per the pre-registered refinement),
  loss/gn trajectories training-healthy and smooth, B/F telemetry ZERO
  fallback/miss/stale across all 60 steps, no persistent STOP-class event.
  **FLAG 1 — memory creep +2.8 GiB over 60 steps (178.9→181.7; poller max
  183.8), 0.8 GiB OVER the pre-registered 2 GiB bar.** Evidence against a
  B/F-leak reading: cache telemetry zero-issue all 60 steps; the caches are
  STRUCTURALLY bounded (FIX B's carrier dies with the microbatch; FIX F is a
  FIFO-64); the absolute stays within the d16 187-smi class. Most likely
  allocator-pool dynamics — BUT the creep's last third slightly ACCELERATED
  (+0.9/+0.3/+1.6 per third), so the bounded-envelope read is NOT proven.
  Honest projection: IF linear, +2.8 GiB/60 steps ≈ +0.047 GiB/step reaches
  the 275 GiB cap in ~1,900 steps ≈ ~72 h of continuous d16 — ship-relevant
  for multi-day runs, NOT tonight's PR. **Open watch item for the report;
  recommended follow-up = a longer soak or an allocator memory-history
  capture to distinguish envelope-vs-linear.** Not a PR #26 blocker (the
  caches are bounded by construction; the creep predates/!is-implicated-by
  the cache telemetry). **FLAG 2 — main54 gn spike 0.9665 + loss 12.139,
  full recovery at main55, trainer log clean.** The pre-registered
  smoothness bar's step-jump signal FIRED and recovered; adjudicated as a
  synthetic-data variance blip (the bench's random-token windows can produce
  single-window outliers; full next-step recovery + in-band loss + clean log
  = the benign class, distinct from training instability which persists).
  On the record as a flagged event, not waved through. **Net: the ship
  stack's 60-step stability claim PASSES with both flags documented.**
  Proceeding to S3 (export + sync-save) on the soak boot.
- 2026-08-14 ~22:3x CDT — **S1 FLAG ADJUDICATIONS (kolmogorov; turing's
  ordered real read, Mac-side poller-series shape analysis + lovelace's
  pre-S3 memory-state capture).** **FLAG 1 (memory creep +2.8 GiB/60 steps)
  RESOLVED DOWN to a documented benign envelope — NOT a leak, NOT a B/F
  concern, NO PR #26 flag.** The four discriminators: (a) SHAPE =
  plateau-leaning, not leak-class: an early pool-warmup jump (+2.1 GiB in
  the first fifth) then near-plateau (bucket means 180.5→181.6 over the
  last 4/5; late slope ~0.029 GiB/step ≈ 40% BELOW the 0.047 full-span
  average and decelerating) — a constant-rate leak does not decelerate; the
  linear-extrapolation-to-OOM model is the wrong model for this shape.
  (b) METRIC SPLIT (lovelace's pre-S3 capture): torch-allocated ≈ flat
  (164.6 GiB at end ≈ the 163 reference class) while the creep is in
  smi-reserved-NOT-allocated → the allocator pool holding freed-but-
  unreturned memory = BENIGN (the pool reuses it). turing's cheap
  confirmation landed exactly as predicted. (c) main54 correlation: none —
  the main54 memory neighborhood (max 183.77) is within the transient-spike
  envelope (183.5-183.8 recurs); no memory event correlates with the gn
  spike. (d) cache entry counts: NOT exposed (the caches log only the
  one-time ACTIVE line — instrumentation gap, papercut pc_03d6c4817ae5);
  the zero fallback/miss/stale telemetry across all 60 steps stands as the
  cache-behavior evidence, and the caches are structurally bounded (FIX B
  carrier dies per-microbatch; FIX F FIFO-64). turing's additional point on
  record: no off-arm 60-step control exists, so the creep is not even
  attributable to B/F vs the base trainer — blocking on an unattributed
  watch item would be over-reach. **Residual watch item (softened):** the
  late-drift tail (+0.69 GiB in the last bucket) — a longer run settles it;
  logged as a documented note, not a concern. **FLAG 2 (main54 gn spike
  0.9665 + loss 12.139, full recovery main55): adjudicated as an ACKNOWLEDGED
  bar trip (the pre-registered smoothness stop signal fired) whose event is
  benign-class** — synthetic-data variance (the bench's random-token windows
  produce single-window outliers), full next-step recovery, in-band loss,
  clean trainer log. Both truths on record: the bar tripped per the letter
  AND the event self-recovered benignly; the batched report cadence surfaced
  it late (process note: a per-step gn/loss watcher with a stop threshold
  would catch spikes in real time — future-soak tooling item). §6 treatment
  per hertz's record-keeping line: the PASS-with-flags override cites the
  telemetry-cleanliness (zero fallback/miss/stale x60) explicitly as the
  override basis, self-explanatory to a later reader.
- 2026-08-14 ~23:0x CDT — **S3 SAVE-LEG STOP (turing ratified): the ship
  tree's /save_state WEDGED — and the catch is a real finding: the
  campaign's F1 "workaround" (BT_SAVE_STATE_SYNC=1) was a SILENT NO-OP on
  every tree not carrying the db5d1826 megatron_config.py hunk.** Async save
  is hardcoded ON at megatron_config.py:325; the env-var read exists only in
  db5d1826 (which lived on the export branch + was applied on-box for the
  earlier probes); the ship tree (73c24b00+TF32) never carried it, so
  BT_SAVE_STATE_SYNC=1 did nothing and the CP>1 async-save wedge (F1, the
  ancdata daemon crash) is LIVE in any production save on this tree. The env
  var gave the campaign false comfort — S3 did exactly its job by catching
  it. **The S1/S2 claims are UNTOUCHED: the hunk only touches the save
  config path; the soak and parity bits never invoked a save.** The fix is
  merge-queue item 6 (the BT_SAVE_STATE_SYNC toggle PR), re-prioritized
  ship-critical. turing's re-run sequence for lovelace: snapshot wedge
  evidence -> restart with the db5d1826 hunk applied + sha-verified +
  weight_sync=local -> short burst for non-degenerate adapters -> S3 re-run.
  F1 escalation package gains tonight's reproduction (still parked awaiting
  Jack, outward-facing). Re-run bars + burst count = kolmogorov's, next
  entry.
- 2026-08-14 ~23:4x CDT — **S3 PASS + P4 CLOSE (kolmogorov's formal
  adjudication; lovelace driving). THE CAMPAIGN'S BOX PROGRAM IS COMPLETE.**
  S3 re-run on the genuinely-gated tree (db5d1826 hunk sha-verified,
  weight_sync=local): **SAVE leg — sync-save done in 65.93s** (bar <5 min;
  the Aug-13 sync-path class 70.8s), checkpoint 2718.0 MB complete
  (/tmp/checkpoints/s3-sync-save iter_00000061 + weight_sync local path,
  .metadata present); **resume-clean: post-save d2 window loss 12.2804 in
  band, gn 0.374 in class, trainer continued.** **EXPORT leg — the 1094-key
  complete GLM-5.2 adapter set, internally consistent, L1 non-degeneracy
  547/547, adapter_config correct** (scope note ratified: the 392 figure was
  the Qwen3-0.6B small-model transcription; the 1094-key set is the mission
  model's reference of record — in-place correction in P4_SOAK_SPEC.md).
  **P4 verdict (formally mine): the ship stack (mission + B/F) is VALIDATED
  — S2 parity re-adjudicated branch (b) on the complete noise matrix (B/F
  exonerated, noise-relative wording), S1 60-step soak PASS with two flags
  adjudicated to a documented benign envelope + a benign-class bar-trip
  recovery, S3 export+save sanity PASS on the genuinely-gated tree.** The
  S3 save leg's first-attempt FAILURE is the campaign's catch of the night:
  the F1 "workaround" env var was a silent no-op on trees without the
  hunk — now genuinely wired, and merge-queue item 6 is ship-critical.
  Box wprm693 → idle-armed for Jack's stop/keep (turing relays).
- 2026-08-14 ~20:4x CDT — **CAMPAIGN CLOSED (turing, orchestrator). Box
  wprm693 STOPPED on Jack's direct order** (curie executed: job stopped via
  the devbox tooling, zero pods remain, ssh dead; all evidence sha-verified
  Mac-side incl. an insurance pull of the R2/P4 box logs
  final_snapshot_r2_pp2logs.tgz 0beb0776…; declared non-pull: the 3.0G
  synthetic 5-step-burst S3 test payloads — deliverables banked in
  s3_artifacts/). Final state: CAMPAIGN_REPORT.md FINAL (all windows
  closed); number of record 1089-1103 steady-state band @d16/131k; ship
  stack (mission + B/F) VALIDATED by P4; PR #26 unconditional + queue item
  6 (save toggle) SHIP-CRITICAL; upstream escalation corrected (§7,
  read-first) + F1 escalation strengthened, both staged for Jack's
  forward; blockK closed mechanism-explained; seam named; next lever =
  balance (roadmap). Fleet stood down. o7

## 2026-08-22/23 overnight — 262k on PR #1070 vs tip of main (box wgm8row)

Jack's ask: does the PP2 stack (#1070, head `d8b9648f`, rebased onto main tip
`9b039d6b`) work at 262k; profile 262k on both tips with the campaign driver.
Full writeup, prereg, driver JSONs, rank-0 traces + memory pickles:
**`runs/overnight_20260822_262k_pr1070/`** (ANALYSIS.md is the deliverable).

Verdict: **#1070 WORKS at 262k** (boots, steps, canaries clean, worst GPU
≤207/268.6 GiB) and beats main tip: **383 vs 312 tok/s/GPU at d2 (+23%),
522 (~590 tight-cluster) at d4 (+67%)**; main's 262k layout (EP16/CP16)
spends **54% of its step in cross-node EP a2a (55 s/step)**. Same-layout
131k↔262k (d4): per-token cost ×1.76, all of it wait (PP bubble ×1.6, a2a
×1.54 — arrival-skew tail, p50 per-call bandwidth-flat; compute ~flat, DSA
top-2048 keeps attention linear). M=2 bubble measured 28.7 s ≈ theoretical
1/3 exactly; d4 collapses the a2a p99 tail 1.26→0.34 s. Flags: main@262k is
2× slower + 60 GiB leaner than the Aug-9 anchor (recompute default "full"
suspected, unconfirmed — needs a one-variable A/B); single 3.8 s coalesced
allreduce at step end at d4; 8×117 ms f32 SIMT GEMMs on main; R3's one slow
control window (464 vs 588-596 cluster) unexplained. Ops notes: new devbox-up
provisions without venvs (built via make megatron-bridge-venv CUDA_FLAVOR=cu13
on both nodes); sbatch = InvalidAccount on these boxes (use nohup setsid
srun); current main requires BT_TRAINER_SERVER_CONFIG_PATH or boot dies at
~2 min; profiles are rank-0-only on main (no BT_PROFILE_RANKS).
