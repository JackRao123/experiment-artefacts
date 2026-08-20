# PR DRAFT — INFRA package (gibbs lineage)

Drafted by weierstrass, 2026-08-13 night, for Jack's morning review.
Draft only — no PR created, nothing pushed. Source branch:
`jackrao/lps-1062-pp2cp8ep8` (pushed, tip b8d868ff), worktree
`~/Documents/wt-pp2-bringup`.

Terms used below: **PP** = pipeline parallelism (model layers split across
GPU groups); **CP** = context parallelism (each sequence split across GPUs);
**EP** = expert parallelism (MoE experts spread across GPUs); **VPP** =
virtual pipeline parallelism (each PP stage holds several non-contiguous
layer chunks, "interleaved" schedule); **DSA** = GLM-5.2's sparse-attention
variant ("deep sparse attention"; `experimental_attention_variant="dsa"`);
**THD** = the packed-variable-length sequence format the CP path uses;
**top-k rule** = GLM-5.2 DSA computes its sparse-attention top-k indices
only on layers whose 1-based number is ≤ 3 or `(layer − 3) % 4 == 0`, and
shares them with the next 3 layers — so every PP stage (or VPP chunk) must
*start* on a top-k-computing layer.

---

## Recommendation: split into 3 PRs, not 1

The five commits are three different review classes with three different
merge urgencies. One omnibus PR would force reviewers to hold the
gate-exemption evidence bar, a scheduling-feature design, and two
diagnostic toggles in their head at once — and would let the lowest-risk
commits be delayed by the highest-scrutiny one.

| PR | commits | review class | urgency |
|---|---|---|---|
| **INFRA-A: PP2 layout + DSA-only gate exemption** | 21d0c578 | production topology gate — needs the evidence review below | **ship-blocker** for any PP2 GLM-5.2 golden config |
| **INFRA-B: VPP config field + interleaved iterator fix** | 61c41d9e + 8c13ed31 | feature-enabling config + schedule fix | needed for the E2a/E2b overlap experiments |
| **INFRA-C: profiling/telemetry toggles** | ad39a97d + 760021be | env-gated diagnostic levers, default-unchanged | anytime; rubber-stamp class |

Justification for the boundaries:

- **21d0c578 stands alone** because it is the only commit with an explicit
  ship condition attached (Jack: gate exemptions ship only with validation
  evidence + the guard docstring update — both below). It is ~100 lines,
  self-contained, and the only commit tonight that changes what topologies
  the server *permits*.
- **61c41d9e + 8c13ed31 change together and must ship together**: the VPP
  config field is unreachable-but-broken without the iterator fix (a VPP2
  boot asserts at the first forward: the interleaved schedule wants one
  data iterator per model chunk), and the iterator fix is only reachable
  under VPP. House rule: things that change together live together.
- **ad39a97d + 760021be are the same shape of change**: a default-preserving
  env gate around an existing behavior, each with a unit test, neither
  touching the training path when unset. Bundling them keeps the
  "diagnostic lever" PR pattern uniform.
- Deliberately **excluded** from this package (lebesgue's lane or not
  ready): the M=N schedule fix + pad-to-131k (d1c939c3, 36c3c8f4,
  b34b7aed, 8b0ef108, merge 73c24b00) is the headline PR; b8d868ff
  (poincare's overlap-flag plumbing) is on the pushed branch tip but has
  **not run on hardware yet** (its first boot is queued in
  E2_REVISED_SPEC.md) — do not let it ride into any morning PR as if
  validated.

---

## INFRA-A — PP2 pipeline layout + DSA-only CP>1+PP>1 gate exemption

**Commit:** 21d0c578 — `feat(glm52): PP2 pipeline layout (38/40) + DSA
exemption from THD-CP PP=1 guard`

### What it does

1. Adds the `(78 layers, PP 2)` entry to `_GLM52_DSA_PIPELINE_LAYOUTS` in
   `glm52_dsa.py`: stage 0 = 38 layers + embedding, stage 1 = 40 layers +
   loss. Stage 1 starts at 1-based layer 39, and (39 − 3) % 4 == 0, so the
   DSA top-k sharing rule is satisfied. A new table-wide unit test proves
   *every* layout in the table (existing (8,2), (78,8), (78,16) included)
   satisfies that rule.
2. Relaxes `_validate_thd_context_parallelism` in `megatron_config.py`:
   the PP=1 restriction for THD CP now has a **DSA-only exemption**. The
   validator takes the provider and skips the PP>1 rejection only when
   `provider.experimental_attention_variant == "dsa"`. Every non-DSA
   attention stack still raises, exactly as before.

### The guard docstring update (Jack's ship condition — included)

The commit updates the validator's docstring in place:

> Reject unvalidated THD CP deployment topologies before serving traffic.
> We only implement CP for the validated golden configs (GLM 5.2 CP32,
> Nemotron Ultra CP4), PP=1. DP>=1 is supported.
>
> **GLM-5.2 DSA is exempt from the PP=1 restriction (LPS-1062 bring-up):
> its PP stage boundaries are constrained by the DSA top-k sharing rule and
> resolved through `_GLM52_DSA_PIPELINE_LAYOUTS`.**

So the guard still says "unvalidated topologies are rejected", names the
one exemption, and points at the mechanism that makes the exemption safe
(the layout table encodes the attention constraint). Non-DSA stacks keep
the hard error.

### Validation evidence (Jack's ship condition — tonight's runs)

The exempted topology — GLM-5.2-FP8 LoRA, PP2/CP8/EP8/TP1/DP1, 131,072-token
sequences, 2×8 B300 GPUs, box w56lorq — ran all night on this exact layout:

- **First light** 2026-08-12 ~13:00 CDT: 550 tok/s/GPU at 4 microbatches
  per step on a clean tip build. Loss canaries in the expected 12.2–12.4
  random-token band, gradient norms finite (0.40–0.50), matching prior
  nights' canary band.
- **Full sweep after the M=N schedule fix** (separate PR): 918 tok/s/GPU
  at 4 microbatches/step, 1000 at 8, 1052 at 16 — 1.42× to 1.63× the
  golden EP16/CP16 anchor of 645. (All tonight's throughput numbers,
  anchors included, were measured on the pre-bump cudnn-frontend wheel —
  absolute values may move after the wheel fix lands; see the merge
  queue's wheel-staleness follow-up. The topology validation below does
  not rest on absolute throughput.) Peak memory 170–194 GiB of 275 GiB
  across the post-M=N ladder (d1 170, d4 175, d8 182, d16 194), flat in
  microbatch count as the in-flight model predicts (2 in flight at PP2).
  No topology-related failure at any rung of the d1→d16 ladder.
- **Parity leg A** (fixed 9-document dataset, ~262k real tokens): the PP2
  leg produced loss 12.361400908911282 with all 9/9 datum outputs; a
  fresh-boot PP2 leg produced 12.54805275637226, 262,032 real tokens, 9/9
  datums. The reference legs on the golden EP16/CP16 topology are queued
  on box A (after the L0 probe) as this is written — box B was stood down
  on Jack's capacity order ~23:5x and never ran a workload — the
  three-leg verdict is the headline PR's evidence; what matters for
  *this* PR is that the exempted topology trains correctly and
  deterministically enough to participate in a parity protocol at all.
- **Constraint check:** the new unit test
  (`test_pipeline_layouts_satisfy_dsa_topk_constraint`) verifies all four
  table layouts against the top-k rule (the fifth, (78,2,2), arrives with
  the VPP commit in INFRA-B, which extends the same test);
  `test_glm52_dsa_pp2_layout` pins the (78,2) apply path.

What tonight did **not** validate (the exemption is deliberately narrower
than the old blanket ban, not broader than the evidence):

- Non-DSA attention stacks under PP>1+CP>1 — still hard-rejected.
- GLM-5.2's EP>1 grouped-expert *export* path — covered by the export
  branch's findings, not this PR. (Note tonight's PP2 did run one stage
  per node across the two nodes, so multi-node PP *training* is inside
  the validated envelope; the export branch's single-node tests are the
  ones that do not cover multi-node.)

### Relationship to the export branch's gate relaxations (read before merging)

The dedekind export branch (`jackrao/lps-1062-pp2-export`) carries its own
relaxations of the same two functions (G1: PP>1+CP>1 raise→warning; G2:
generic GPT providers allowed through CP). **Those stay test-branch-only;
this PR's DSA-only exemption is the mainline version.** The two rewrite
the same function signature and the same unit tests, so merging both is a
conflict *and* a semantic clash (warning-for-everyone vs
raise-for-everyone-except-DSA). Reviewers: merge this one; do not merge
the export branch's megatron_config.py hunks. See PR_DRAFT_EXPORT.md §"Gate
relationship".

---

## INFRA-B — VPP config field + interleaved iterator fix + VPP warmup auto-skip

**Commits:**
- 61c41d9e — `feat(glm52): PP2/VPP2 interleaved layout [18,20,20,20] + vpp
  config field`
- 8c13ed31 — `fix(server): replicate the data iterator per model chunk
  under VPP`
- 6fa3bfa7 — `fix(server): auto-skip the startup warmup under VPP>1`
  (poincare; added to this package after the initial draft — see below)

### What they do

- `TrainerControllerConfig` gains `virtual_pipeline_parallel_size`
  (default 1 = today's non-interleaved schedule; >1 interleaves that many
  layer chunks per PP stage). Backward-compatible by construction.
- The layout table is re-keyed to `(num_layers, pp, vpp)` and gains the
  (78, 2, 2) entry: chunks of 18/20/20/20 layers in mcore's interleaved
  flat order; every chunk start (19, 39, 59) satisfies the top-k rule.
  When vpp>1 the provider's `virtual_pipeline_model_parallel_size` is set
  (mcore also derives it from the layout; the two cross-check).
- The runner fix: the interleaved schedule indexes
  `data_iterator[model_chunk_id]`, so a single iterator asserts at the
  first forward ("each model chunk needs a data iterator"). Every chunk
  runs the same microbatch stream, so the runner replicates a fresh
  iterator per chunk. The non-VPP path is byte-identical to before.
- The warmup auto-skip (6fa3bfa7): the interleaved schedule requires
  microbatches ≥ PP (mcore `schedules.py:1141-1148`), and the single-datum
  startup warmup is always M=1 — so a VPP>1 config previously died at the
  warmup step unless the operator knew to set `BT_SKIP_WARMUP=1`. The
  backend now auto-skips the warmup under VPP>1 with a banner saying why
  (kernels compile on the first real forward_backward instead; the
  bounded-probe discipline is unchanged). `BT_SKIP_WARMUP` remains the
  manual lever for all other cases. Inert at VPP=1. It reads the
  `virtual_pipeline_parallel_size` field, so it hard-depends on 61c41d9e
  and cannot land alone. Tests: `test_startup_warmup.py` pins skip/run/
  env-lever both directions (50/50 in the backends dir via the Mac stub
  harness). Review status per policy: one disposable-subagent review,
  PASS-WITH-NITS, zero blocking; the commit-message wording nit is with
  poincare, and the PR-description nit is resolved by these two verbatim
  sentences from poincare, to be included in the PR body:
  "The warmup auto-skip keys on the REQUESTED config value
  (virtual_pipeline_parallel_size > 1). An operator setting VPP>1 on a
  (num_layers, PP, VPP) tuple absent from _GLM52_DSA_PIPELINE_LAYOUTS
  silently gets the non-interleaved schedule (pre-existing upstream gap)
  and still skips the warmup — conservative direction: a legal warmup is
  skipped, never an illegal one run." And: "Tests cover
  virtual_pipeline_parallel_size=1 (warmup runs) and =2 (warmup skips);
  the field is non-Optional int default 1, so None is unrepresentable."
  (Future follow-up, deliberately split from this hunk: fail-fast at
  apply time when a requested VPP>1 matches no layout tuple — the proper
  fix for that silent-fallback gap; same family as SELF_REVIEW.md
  bringup-finding 3.)

### Why they are one PR

The config field without the iterator fix boots into an assert; the
iterator fix without the config field is unreachable; the warmup auto-skip
reads the config field and is the difference between a VPP>1 boot that
works and one that dies at warmup with no obvious remedy. Tests:
`test_glm52_dsa_pp2_vpp2_layout` (layout + provider wiring),
`test_pipeline_layouts_satisfy_dsa_topk_constraint` (extended to vpp
keys), `test_schedule_data_iterator.py` (single-chunk passthrough,
per-chunk independence, multi-microbatch).

### Coordination seam with the headline M=N PR (lebesgue)

8c13ed31's call sites in `training_runner.py` are the exact lines the M=N
merge rewrote; the seam is marked in-tree by 8b0ef108's comment. The
merged form routes the M=N call through
`_schedule_data_iterator(microbatches, len(model_list))` — plain iterator
at VPP=1, per-chunk iterators under VPP. **Whichever of INFRA-B and the
M=N PR lands second reconciles at that marked seam**; the conflict is
small, intentional, and documented. Tonight's E2a/E2b experiments need
both, so order them adjacently in the queue (MORNING_MERGE_QUEUE.md).

### Validation status (honest)

Unit-tested and standalone-verified only — and now with a hardware
record, all negative: **no VPP2 boot has ever completed a driver step
tonight.** The attempts: the E2a schedule-interface failure, the L0
warmup-M=1 failure (the one 6fa3bfa7 in this PR fixes), the L0 executor
contract failure, the L0b memory abort, and — on the last attempt (32k
offload discriminator, 6d8b22da tree) — a **new, named VPP2-path bug**:
under VPP2 each stage holds 2 chunks, `output_layer` lives only on the
last chunk, and the M=N+VPP loss path invokes the chunked CE loss on a
chunk that lacks it (RuntimeError at `chunked_lm_head.py:196`, stage-1
ranks). That bug is unfixed and is poincare-domain when they want a
lane; it is not tonight-critical. This PR changes nothing when
`virtual_pipeline_parallel_size` is left at its default, so merging it
does not risk the validated PP2 path — but reviewers should know VPP2
itself is 0-for-5 on hardware and this PR is the substrate, not the
fix, for that.

---

## INFRA-C — diagnostic toggles

**Commits:**
- ad39a97d — `feat(models): BT_PROFILE_RANKS env for multi-rank kineto
  tracing`
- 760021be — `feat(server): BT_PEAK_MEM_REPORT=0 gates the world allreduce
  in optim_step`

### What they do

- `BT_PROFILE_RANKS`: `ProfilingConfig.from_env` hardcoded the trace set
  to rank 0, so on-demand profiling only ever traced one rank. PP stage
  diagnosis needs both stage leaders (ranks 0 and 8 at PP2). The env takes
  a comma-separated rank list; default unchanged (rank 0 only). Tonight's
  traces (`BT_PROFILE_RANKS=0,8`) are what made the stage-1 CPU-blockage
  analysis (results/TRACE_RANK8_ANALYSIS.md) possible.
- `BT_PEAK_MEM_REPORT=0`: `_peak_memory_report` embeds a world-wide MAX
  allreduce in *every* optim_step; on a skewed pipeline that collective is
  pure rendezvous wait — measured 6.5s/step on the PP2/CP8 stage-1 rank.
  Gated off, the metrics carry rank-local peaks instead of the world max.
  Default unchanged (allreduce on).

Both are env-gated, default-preserving, and covered by new unit tests
(`models/tests/test_profiling.py`,
`server/tests/unit/dp_worker/controller/test_peak_memory_report.py`).

---

## Test/CI status

- All new unit tests are written to run in CI (the megatron-dependent ones
  import-guard; the Darwin-verified ones are noted in the commit bodies).
- The bringup branch is 2 commits behind current origin/main
  (f51fae2e DSv4 repin, 05f3fafc CI permissions) — no interaction with
  this package; rebase at cut time.
- Self-review findings for this lineage: SELF_REVIEW.md (nothing blocking;
  two nits flagged for follow-up, listed there).
